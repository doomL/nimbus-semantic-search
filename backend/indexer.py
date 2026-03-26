"""WebDAV crawler + in-memory CLIP embedding indexer."""

from __future__ import annotations

import logging
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

from webdav4.client import Client, ResourceNotFound

from clip_model import encode_image_pil
from db import count_photos, get_connection, insert_photo, numpy_to_blob, path_exists
from image_io import load_rgb_image
from tag_stats import recompute_library_tags

logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    logger.warning("pillow-heif not available; HEIC support may be limited")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def _retry_call(
    fn: Callable[[], Any],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    last: Optional[BaseException] = None
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            logger.warning(
                "WebDAV call failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
    assert last is not None
    raise last


def _resolve_entry_path(parent: str, name: str) -> str:
    """
    webdav4's 'name' is usually path_relative_to(server root), so it can be
    'Photos/a.jpg' or '.Trash-1001/expunged'. Joining that with parent would
    duplicate segments. If 'name' has no slash, it is relative to `parent`.
    """
    n = name.strip()
    if not n:
        return ""
    if n.startswith("/"):
        return n
    if "/" in n:
        return "/" + n.lstrip("/")
    if parent in ("", "/"):
        return "/" + n
    return parent.rstrip("/") + "/" + n


def _basename(path: str) -> str:
    p = path.rstrip("/")
    return p.rsplit("/", 1)[-1] if p else ""


def _is_image_path(webdav_path: str) -> bool:
    lower = _basename(webdav_path).lower()
    dot = lower.rfind(".")
    if dot < 0:
        return False
    return lower[dot:] in IMAGE_SUFFIXES


def _should_skip_dir(webdav_path: str) -> bool:
    """Skip pCloud trash subtrees (avoids broken phantom folders under WebDAV)."""
    for part in webdav_path.split("/"):
        if part.startswith(".Trash"):
            return True
    return False


def _ls_detail(client: Client, path: str) -> List[Any]:
    """List directory; missing paths return [] (no multi-retry on 404)."""
    try:
        return client.ls(path, detail=True)
    except ResourceNotFound:
        logger.debug("WebDAV path not listable (skip): %s", path)
        return []


def _collect_image_paths(client: Client, path: str, out: List[str]) -> None:
    """Recursive PROPFIND listing; fills out with WebDAV paths to image files."""
    entries = _ls_detail(client, path)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        typ = entry.get("type", "")
        full = _resolve_entry_path(path, name)
        if not full:
            continue
        if typ == "directory":
            if _should_skip_dir(full):
                logger.debug("Skipping directory: %s", full)
                continue
            _collect_image_paths(client, full, out)
        elif typ == "file" and _is_image_path(full):
            out.append(full)


def _download_to_bytesio(client: Client, path: str) -> BytesIO:
    buf = BytesIO()
    _retry_call(lambda: client.download_fileobj(path, buf))
    buf.seek(0)
    return buf


def _encode_and_store(
    conn,
    client: Client,
    webdav_path: str,
    filename: str,
) -> None:
    buf = _download_to_bytesio(client, webdav_path)
    data = buf.getvalue()
    try:
        img = load_rgb_image(data, source=webdav_path)
    except Exception as e:
        logger.error("Cannot open image %s: %s", webdav_path, e)
        raise

    vec = encode_image_pil(img)
    blob = numpy_to_blob(vec)
    insert_photo(conn, webdav_path, filename, blob)


def run_index_job(
    base_url: str,
    username: str,
    password: str,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    db_lock: threading.Lock,
    commit_every: int = 50,
) -> None:
    """
    Background job: crawl WebDAV, skip existing paths, index new images in memory only.
    Updates state under state_lock.
    """
    with state_lock:
        if state.get("in_progress"):
            logger.info("Index already running; ignoring duplicate start.")
            return
        state["in_progress"] = True
        state["total"] = 0
        state["indexed_this_run"] = 0
        state["skipped"] = 0
        state["errors"] = 0
        state["last_error"] = None

    client = Client(
        base_url.rstrip("/"),
        auth=(username, password),
        retry=True,
    )

    paths: List[str] = []
    try:
        _retry_call(lambda: _collect_image_paths(client, "/", paths))
    except Exception as e:
        logger.exception("Crawl failed: %s", e)
        with state_lock:
            state["in_progress"] = False
            state["errors"] = int(state.get("errors", 0)) + 1
            state["last_error"] = str(e)
        return

    with state_lock:
        state["total"] = len(paths)

    logger.info("Found %s image files to consider.", len(paths))

    commits_since_batch = 0
    try:
        for webdav_path in paths:
            filename = webdav_path.rsplit("/", 1)[-1]
            with db_lock:
                conn = get_connection()
                try:
                    if path_exists(conn, webdav_path):
                        with state_lock:
                            state["skipped"] = int(state.get("skipped", 0)) + 1
                        continue
                    _encode_and_store(conn, client, webdav_path, filename)
                    commits_since_batch += 1
                    if commits_since_batch >= commit_every:
                        conn.commit()
                        commits_since_batch = 0
                    with state_lock:
                        state["indexed_this_run"] = (
                            int(state.get("indexed_this_run", 0)) + 1
                        )
                except Exception as e:
                    logger.exception("Failed to index %s: %s", webdav_path, e)
                    with state_lock:
                        state["errors"] = int(state.get("errors", 0)) + 1
                        state["last_error"] = str(e)
        with db_lock:
            conn = get_connection()
            conn.commit()
    finally:
        with state_lock:
            state["in_progress"] = False
        logger.info(
            "Indexing finished: indexed_this_run=%s skipped=%s errors=%s",
            state.get("indexed_this_run"),
            state.get("skipped"),
            state.get("errors"),
        )
        _schedule_library_tag_recompute(db_lock)


def _schedule_library_tag_recompute(db_lock: threading.Lock) -> None:
    """Recompute CLIP-based tag histogram in background (can take seconds on large libraries)."""

    def run() -> None:
        try:
            with db_lock:
                conn = get_connection()
                if count_photos(conn) == 0:
                    return
                logger.info("Updating library tag statistics…")
                recompute_library_tags(conn)
        except Exception:
            logger.exception("Library tag statistics update failed")

    threading.Thread(target=run, name="library-tags", daemon=True).start()
