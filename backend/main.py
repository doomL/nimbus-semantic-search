"""FastAPI app: UI, indexing, search, WebDAV photo proxy."""

from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from PIL import Image
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
from db import (  # noqa: E402
    count_photos,
    get_connection,
    get_index_roots,
    init_db,
    list_index_failures,
    list_recent_photos,
    set_index_roots,
)
from image_io import load_rgb_image  # noqa: E402
from indexer import list_immediate_subdirs, run_index_job  # noqa: E402
from search import search_photos  # noqa: E402
from tag_stats import (  # noqa: E402
    get_popular_tags,
    recompute_library_tags_background,
)

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
    "index_roots_effective": None,
}
state_lock = threading.Lock()
db_lock = threading.Lock()

_scheduler: Optional[Any] = None


def _parse_auto_index_interval_hours() -> float | None:
    raw = os.environ.get("NIMBUS_AUTO_INDEX_INTERVAL_HOURS", "").strip()
    if not raw:
        return None
    try:
        h = float(raw)
    except ValueError:
        logger.warning("Invalid NIMBUS_AUTO_INDEX_INTERVAL_HOURS=%r", raw)
        return None
    if h <= 0:
        return None
    return h


def _run_index_thread() -> None:
    """Start background WebDAV crawl + embedding job (same as POST /index)."""
    base = _env("PCLOUD_WEBDAV_URL", "https://ewebdav.pcloud.com")
    user = _env("PCLOUD_USERNAME")
    password = _env("PCLOUD_PASSWORD")

    def job() -> None:
        run_index_job(base, user, password, index_state, state_lock, db_lock)

    t = threading.Thread(target=job, name="indexer", daemon=True)
    t.start()


def _scheduled_index_tick() -> None:
    logger.info("Scheduled auto-index tick.")
    try:
        _run_index_thread()
    except RuntimeError as e:
        logger.warning("Scheduled index skipped (credentials or config): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _scheduler
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
    get_clip()
    logger.info("Startup complete.")

    hours = _parse_auto_index_interval_hours()
    _scheduler = None
    if hours is not None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.warning("apscheduler not installed; auto-index disabled.")
        else:
            try:
                delay_min = float(os.environ.get("NIMBUS_AUTO_INDEX_FIRST_DELAY_MINUTES", "5"))
            except ValueError:
                delay_min = 5.0
            start = datetime.now(timezone.utc) + timedelta(minutes=delay_min)
            _scheduler = BackgroundScheduler(timezone=timezone.utc)
            _scheduler.add_job(
                _scheduled_index_tick,
                IntervalTrigger(hours=hours, start_date=start),
                id="auto_index",
                replace_existing=True,
            )
            _scheduler.start()
            logger.info(
                "Auto-index every %s h (first run ~%s min after startup, UTC)",
                hours,
                delay_min,
            )

    yield

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


app = FastAPI(
    title="Nimbus",
    description="Semantic photo search over WebDAV (e.g. pCloud) with CLIP embeddings.",
    lifespan=lifespan,
)


def _basic_auth_credentials() -> tuple[str, str] | None:
    """If both user and password are set, require HTTP Basic Auth on every request."""
    user = os.environ.get("NIMBUS_AUTH_USER", "").strip()
    password = os.environ.get("NIMBUS_AUTH_PASSWORD", "")
    if not user or not password:
        return None
    return (user, password)


def _basic_auth_exempt_path(path: str) -> bool:
    """
    PWA bootstrap URLs: browsers often fetch manifest / SW without Authorization,
    so they must bypass Basic Auth. Only metadata + icons — the app UI and API
    stay protected.
    """
    if path == "/sw.js":
        return True
    if path == "/assets/manifest.webmanifest":
        return True
    if path in ("/assets/icons/icon-192.png", "/assets/icons/icon-512.png"):
        return True
    return False


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        creds = _basic_auth_credentials()
        if creds is None:
            return await call_next(request)

        if _basic_auth_exempt_path(request.url.path):
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


def _folder_basename(path: str) -> str:
    p = path.rstrip("/")
    return p.rsplit("/", 1)[-1] if p else ""


def _prepare_index_roots(raw: List[str]) -> List[str]:
    """Normalize roots; empty result means index the entire library."""
    seen: set[str] = set()
    out: List[str] = []
    for p in raw[:200]:
        if not isinstance(p, str) or not p.strip():
            continue
        n = _normalize_webdav_path(p.strip())
        n = n.rstrip("/") or "/"
        if n == "/":
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class IndexSettingsBody(BaseModel):
    """Empty ``index_roots`` indexes the entire WebDAV tree (default)."""

    index_roots: List[str] = Field(default_factory=list, max_length=200)


@app.get("/index/settings")
def get_index_settings() -> Dict[str, Any]:
    with db_lock:
        roots = get_index_roots(get_connection())
    return {
        "index_roots": roots,
        "entire_library": len(roots) == 0,
    }


@app.post("/index/settings")
def post_index_settings(body: IndexSettingsBody = Body(...)) -> Dict[str, Any]:
    roots = _prepare_index_roots(body.index_roots)
    with db_lock:
        conn = get_connection()
        set_index_roots(conn, roots)
        conn.commit()
    return {
        "index_roots": roots,
        "entire_library": len(roots) == 0,
    }


@app.get("/webdav/folders")
def webdav_folders(
    parent: str = Query(
        "/",
        max_length=8192,
        description="WebDAV directory to list (immediate children only)",
    ),
) -> Dict[str, Any]:
    raw = parent.strip() if parent else "/"
    if not raw:
        raw = "/"
    try:
        parent_path = _normalize_webdav_path(raw)
    except HTTPException:
        raise
    base = _env("PCLOUD_WEBDAV_URL", "https://ewebdav.pcloud.com").rstrip("/")
    user = _env("PCLOUD_USERNAME")
    password = _env("PCLOUD_PASSWORD")
    client = Client(base, auth=(user, password), retry=True)
    try:
        dirs = list_immediate_subdirs(client, parent_path)
    except Exception as e:
        logger.warning("webdav/folders failed for %s: %s", parent_path, e)
        raise HTTPException(502, "Could not list WebDAV folder") from e
    items = [{"path": d, "name": _folder_basename(d)} for d in dirs]
    return {"parent": parent_path, "folders": items}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + basic DB visibility for monitoring and the web UI."""
    payload: Dict[str, Any] = {
        "status": "ok",
        "version": __version__,
        "service": "nimbus",
    }
    ai = _parse_auto_index_interval_hours()
    payload["auto_index_interval_hours"] = ai
    sched = _scheduler
    if sched is not None:
        try:
            job = sched.get_job("auto_index")
            if job is not None and job.next_run_time is not None:
                payload["auto_index_next_utc"] = job.next_run_time.isoformat()
        except Exception:
            pass
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
    """Start indexing on the server; safe to close the browser — job runs in a background thread."""
    try:
        _run_index_thread()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return JSONResponse({"started": True})


@app.get("/index/status")
def index_status() -> Dict[str, Any]:
    with state_lock:
        snap = dict(index_state)
    with db_lock:
        indexed = count_photos(get_connection())
        roots_stored = get_index_roots(get_connection())
    return {
        "total": int(snap.get("total", 0)),
        "indexed": indexed,
        "in_progress": bool(snap.get("in_progress", False)),
        "errors": int(snap.get("errors", 0)),
        "skipped": int(snap.get("skipped", 0)),
        "indexed_this_run": int(snap.get("indexed_this_run", 0)),
        "last_error": snap.get("last_error"),
        "index_roots": roots_stored,
        "entire_library": len(roots_stored) == 0,
        "index_roots_effective": snap.get("index_roots_effective"),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Natural language query"),
) -> Dict[str, Any]:
    out = search_photos(q, k=20)
    return {"query": q.strip(), **out}


@app.get("/photos/recent")
def photos_recent(limit: int = Query(8, ge=1, le=48)) -> Dict[str, Any]:
    """Newest indexed paths (for home screen gallery)."""
    with db_lock:
        rows = list_recent_photos(get_connection(), limit=limit)
    items = [
        {"webdav_path": p, "filename": f, "indexed_at": t} for p, f, t in rows
    ]
    return {"items": items}


@app.get("/index/failures")
def index_failures_list(limit: int = Query(200, ge=1, le=2000)) -> Dict[str, Any]:
    with db_lock:
        rows = list_index_failures(get_connection(), limit=limit)
    return {
        "failures": [
            {"webdav_path": p, "reason": r, "failed_at": t} for p, r, t in rows
        ]
    }


@app.get("/index/failures.csv")
def index_failures_csv() -> Response:
    with db_lock:
        rows = list_index_failures(get_connection(), limit=5000)
    lines = ["webdav_path,reason,failed_at"]
    for p, r, t in rows:
        def esc(s: str) -> str:
            return '"' + s.replace('"', '""') + '"'

        lines.append(",".join([esc(p), esc(r), esc(t)]))
    body = "\r\n".join(lines) + "\r\n"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="nimbus-index-failures.csv"'},
    )


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
            recompute_library_tags_background(db_lock)
        except Exception:
            logger.exception("Tag recompute job failed")

    threading.Thread(target=job, name="tags-recompute", daemon=True).start()
    return JSONResponse({"started": True})


def _thumb_bytes(data: bytes, max_px: int = 300, *, source: str = "") -> bytes:
    img = load_rgb_image(data, source=source)
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    out = BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _download_webdav_bytes(client: Client, raw_path: str) -> bytes:
    buf = BytesIO()
    client.download_fileobj(raw_path, buf)
    data = buf.getvalue()
    if len(data) == 0:
        time.sleep(0.35)
        buf = BytesIO()
        client.download_fileobj(raw_path, buf)
        data = buf.getvalue()
    return data


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
    try:
        data = _download_webdav_bytes(client, raw_path)
    except Exception as e:
        logger.warning("Photo fetch failed for %s: %s", raw_path, e)
        raise HTTPException(404, "Could not load image from WebDAV") from e
    if thumb:
        try:
            data = _thumb_bytes(data, source=raw_path)
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
