"""FastAPI app: UI, indexing, search, WebDAV photo proxy."""

from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from PIL import Image, ImageOps
from webdav4.client import Client

load_dotenv()

__version__ = "0.2.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from clip_model import get_clip  # noqa: E402  — load env before heavy imports
from db import count_photos, get_connection, init_db  # noqa: E402
from indexer import run_index_job  # noqa: E402
from search import search_photos  # noqa: E402
from tag_stats import get_popular_tags, recompute_library_tags  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent


def _resolve_frontend_index() -> Path:
    """Support repo layout (backend/ + frontend/) and flat Docker layout (/app + /app/frontend/)."""
    here = BASE_DIR
    flat = here / "frontend" / "index.html"
    if flat.is_file():
        return flat
    return here.parent / "frontend" / "index.html"


FRONTEND_INDEX = _resolve_frontend_index()
FRONTEND_DIR = FRONTEND_INDEX.parent
_ASSETS_DIR = FRONTEND_DIR / "assets"

index_state: Dict[str, Any] = {
    "in_progress": False,
    "total": 0,
    "indexed_this_run": 0,
    "skipped": 0,
    "errors": 0,
    "last_error": None,
}
state_lock = threading.Lock()
db_lock = threading.Lock()

app = FastAPI(
    title="Nimbus",
    description="Semantic photo search over WebDAV (e.g. pCloud) with CLIP embeddings.",
)


def _basic_auth_credentials() -> tuple[str, str] | None:
    """If both user and password are set, require HTTP Basic Auth on every request."""
    user = os.environ.get("NIMBUS_AUTH_USER", "").strip()
    password = os.environ.get("NIMBUS_AUTH_PASSWORD", "")
    if not user or not password:
        return None
    return (user, password)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        creds = _basic_auth_credentials()
        if creds is None:
            return await call_next(request)

        expected_user, expected_password = creds
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Basic "):
            try:
                raw = base64.b64decode(auth[6:].strip(), validate=True)
                decoded = raw.decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            if ":" in decoded:
                u, p = decoded.split(":", 1)
                if len(u) == len(expected_user) and len(p) == len(expected_password):
                    if secrets.compare_digest(u, expected_user) and secrets.compare_digest(
                        p, expected_password
                    ):
                        return await call_next(request)

        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Nimbus"'},
        )


app.add_middleware(BasicAuthMiddleware)


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is not None and v != "":
        return v
    if default is not None:
        return default
    raise RuntimeError(f"Missing required environment variable: {name}")


def _normalize_webdav_path(path: str) -> str:
    """
    Reject path traversal and unsafe characters. Returns absolute WebDAV path
    starting with /.
    """
    raw = unquote(path).strip()
    if not raw or "\x00" in raw or "\r" in raw or "\n" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    if "\\" in raw or raw.startswith("//"):
        raise HTTPException(status_code=400, detail="Invalid path")
    if len(raw) > 8192:
        raise HTTPException(status_code=400, detail="Path too long")
    if not raw.startswith("/"):
        raw = "/" + raw
    for segment in raw.split("/"):
        if segment == "..":
            raise HTTPException(status_code=400, detail="Invalid path")
    return raw


@app.on_event("startup")
def startup() -> None:
    if (BASE_DIR.parent / "frontend" / "index.html").is_file():
        db_default = BASE_DIR.parent / "data" / "photos.db"
    else:
        db_default = BASE_DIR / "data" / "photos.db"
    db_path = os.environ.get("DB_PATH", str(db_default))
    os.environ.setdefault(
        "TORCH_HOME",
        str(BASE_DIR / ".cache" / "torch"),
    )
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    logger.info("Database ready at %s", db_path)
    if _basic_auth_credentials():
        logger.info("HTTP Basic authentication is enabled (set NIMBUS_AUTH_*).")
    # Warm CLIP once so first search/index is not cold
    get_clip()
    logger.info("Startup complete.")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + basic DB visibility for monitoring and the web UI."""
    payload: Dict[str, Any] = {
        "status": "ok",
        "version": __version__,
        "service": "nimbus",
    }
    try:
        with db_lock:
            payload["indexed"] = count_photos(get_connection())
        payload["database"] = "ok"
    except Exception as e:
        logger.warning("Health DB check failed: %s", e)
        payload["database"] = "error"
        payload["indexed"] = None
    return payload


@app.get("/", response_class=FileResponse)
def serve_index() -> FileResponse:
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(500, "frontend/index.html not found")
    return FileResponse(FRONTEND_INDEX)


@app.post("/index")
def start_index() -> JSONResponse:
    # EU default: ewebdav; non-EU regions: https://webdav.pcloud.com — see .env.example
    base = _env("PCLOUD_WEBDAV_URL", "https://ewebdav.pcloud.com")
    user = _env("PCLOUD_USERNAME")
    password = _env("PCLOUD_PASSWORD")

    def job() -> None:
        run_index_job(base, user, password, index_state, state_lock, db_lock)

    t = threading.Thread(target=job, name="indexer", daemon=True)
    t.start()
    return JSONResponse({"started": True})


@app.get("/index/status")
def index_status() -> Dict[str, Any]:
    with state_lock:
        snap = dict(index_state)
    with db_lock:
        indexed = count_photos(get_connection())
    return {
        "total": int(snap.get("total", 0)),
        "indexed": indexed,
        "in_progress": bool(snap.get("in_progress", False)),
        "errors": int(snap.get("errors", 0)),
        "skipped": int(snap.get("skipped", 0)),
        "indexed_this_run": int(snap.get("indexed_this_run", 0)),
        "last_error": snap.get("last_error"),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Natural language query"),
) -> Dict[str, Any]:
    results = search_photos(q, k=20)
    return {"query": q.strip(), "results": results}


@app.get("/tags/popular")
def tags_popular(limit: int = Query(24, ge=1, le=50)) -> Dict[str, Any]:
    """Concepts that match many photos vs a fixed CLIP prompt list (not EXIF tags)."""
    with db_lock:
        tags = get_popular_tags(get_connection(), limit=limit)
    return {"tags": tags}


@app.post("/tags/recompute")
def tags_recompute() -> JSONResponse:
    """Rebuild tag histogram (runs in background)."""

    def job() -> None:
        try:
            with db_lock:
                conn = get_connection()
                recompute_library_tags(conn)
        except Exception:
            logger.exception("Tag recompute job failed")

    threading.Thread(target=job, name="tags-recompute", daemon=True).start()
    return JSONResponse({"started": True})


def _thumb_bytes(data: bytes, max_px: int = 300) -> bytes:
    buf = BytesIO(data)
    img = Image.open(buf)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    out = BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


@app.get("/photo")
def photo_proxy(
    path: str = Query(..., max_length=8192, description="WebDAV path to the image"),
    thumb: bool = Query(False),
) -> StreamingResponse:
    raw_path = _normalize_webdav_path(path)

    base = _env("PCLOUD_WEBDAV_URL", "https://ewebdav.pcloud.com").rstrip("/")
    user = _env("PCLOUD_USERNAME")
    password = _env("PCLOUD_PASSWORD")

    client = Client(base, auth=(user, password), retry=True)
    buf = BytesIO()
    try:
        client.download_fileobj(raw_path, buf)
    except Exception as e:
        logger.warning("Photo fetch failed for %s: %s", raw_path, e)
        raise HTTPException(404, "Could not load image from WebDAV") from e

    data = buf.getvalue()
    if thumb:
        try:
            data = _thumb_bytes(data)
        except Exception as e:
            logger.warning("Thumbnail failed for %s: %s", raw_path, e)
            raise HTTPException(400, "Could not create thumbnail") from e

    if thumb:
        media = "image/jpeg"
    else:
        lower = raw_path.lower()
        if lower.endswith(".png"):
            media = "image/png"
        elif lower.endswith((".jpg", ".jpeg")):
            media = "image/jpeg"
        elif lower.endswith(".webp"):
            media = "image/webp"
        elif lower.endswith(".heic"):
            media = "image/heic"
        else:
            media = "application/octet-stream"
    headers: Dict[str, str] = {}
    if thumb:
        headers["Cache-Control"] = "public, max-age=86400, immutable"
    else:
        headers["Cache-Control"] = "private, max-age=120"
    return StreamingResponse(
        BytesIO(data),
        media_type=media,
        headers=headers,
    )


SW_PATH = FRONTEND_DIR / "sw.js"
_MANIFEST_PATH = _ASSETS_DIR / "manifest.webmanifest"


@app.get("/assets/manifest.webmanifest")
def serve_manifest() -> FileResponse:
    """Correct Content-Type for installability; registered before /assets mount."""
    if not _MANIFEST_PATH.is_file():
        raise HTTPException(404, "manifest not found")
    return FileResponse(
        _MANIFEST_PATH,
        media_type="application/manifest+json",
    )


@app.get("/sw.js")
def serve_service_worker() -> FileResponse:
    """Service worker at origin root so scope is `/`."""
    if not SW_PATH.is_file():
        raise HTTPException(404, "service worker not found")
    return FileResponse(
        SW_PATH,
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")
