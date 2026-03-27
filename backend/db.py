"""SQLite + sqlite-vec setup and helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Generator, List, Optional, Tuple

import sqlite_vec
from sqlite_vec import serialize_float32

EMBED_DIM = 512

_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[str] = None
# Avoid full-table COUNT(*) on every /index/status and /health poll (hot path).
_photo_count_cache: Optional[int] = None


def _refresh_photo_count_cache(conn: sqlite3.Connection) -> int:
    global _photo_count_cache
    _photo_count_cache = int(conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0])
    return _photo_count_cache


def _note_photo_inserted() -> None:
    global _photo_count_cache
    if _photo_count_cache is not None:
        _photo_count_cache += 1


def _load_extension(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables and vec0 virtual table; return a shared connection."""
    global _conn, _db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _db_path = db_path
    _conn = sqlite3.connect(db_path, check_same_thread=False)
    _load_extension(_conn)
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA foreign_keys=ON;")
    _conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webdav_path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            embedding BLOB NOT NULL,
            indexed_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(webdav_path);
        CREATE INDEX IF NOT EXISTS idx_photos_indexed_at ON photos(indexed_at DESC);

        CREATE TABLE IF NOT EXISTS index_failures (
            webdav_path TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            failed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tag_stats (
            tag TEXT PRIMARY KEY,
            count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    _conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_photos USING vec0(
            embedding float[512] distance_metric=cosine
        );
        """
    )
    _conn.commit()
    _refresh_photo_count_cache(_conn)
    return _conn


def get_connection() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("Database not initialized; call init_db first")
    return _conn


@contextmanager
def connection() -> Generator[sqlite3.Connection, None, None]:
    """Use the shared connection (caller should hold app-level lock for writes)."""
    conn = get_connection()
    try:
        yield conn
    finally:
        pass


def path_exists(conn: sqlite3.Connection, webdav_path: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM photos WHERE webdav_path = ? LIMIT 1", (webdav_path,)
    ).fetchone()
    return row is not None


def count_photos(conn: sqlite3.Connection) -> int:
    """Row count; uses an in-memory counter updated on insert (fast for status polls)."""
    global _photo_count_cache
    if _photo_count_cache is not None:
        return _photo_count_cache
    return _refresh_photo_count_cache(conn)


def insert_photo(
    conn: sqlite3.Connection,
    webdav_path: str,
    filename: str,
    embedding_f32: bytes,
    indexed_at: Optional[str] = None,
) -> int:
    """Insert a row into photos and vec_photos. embedding_f32 is float32 blob (512 dims)."""
    if indexed_at is None:
        indexed_at = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO photos (webdav_path, filename, embedding, indexed_at)
        VALUES (?, ?, ?, ?)
        """,
        (webdav_path, filename, embedding_f32, indexed_at),
    )
    pid = int(cur.lastrowid)
    cur.execute(
        "INSERT INTO vec_photos (rowid, embedding) VALUES (?, ?)",
        (pid, embedding_f32),
    )
    _note_photo_inserted()
    return pid


def parent_folder(webdav_path: str) -> str:
    """Parent directory as WebDAV path (``/`` for root-level files)."""
    p = str(webdav_path).strip().rstrip("/")
    if not p or p == "/":
        return "/"
    i = p.rfind("/")
    if i <= 0:
        return "/"
    return p[:i] or "/"


def list_photos_in_folder(
    conn: sqlite3.Connection,
    webdav_path: str,
    limit: int = 48,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Photos directly under ``parent_folder(webdav_path)``, excluding
    ``webdav_path`` itself. Returns ``(folder, [(webdav_path, filename, indexed_at), ...])``.
    """
    folder = parent_folder(webdav_path)
    exclude = webdav_path
    if folder == "/":
        sql = """
            SELECT webdav_path, filename, indexed_at
            FROM photos
            WHERE webdav_path GLOB '/*'
              AND webdav_path NOT GLOB '/*/*'
              AND webdav_path != ?
            ORDER BY indexed_at DESC, id DESC
            LIMIT ?
        """
        cur = conn.execute(sql, (exclude, limit))
    else:
        base = folder.rstrip("/")
        glob_one = base + "/*"
        glob_nested = base + "/*/*"
        sql = """
            SELECT webdav_path, filename, indexed_at
            FROM photos
            WHERE webdav_path GLOB ?
              AND webdav_path NOT GLOB ?
              AND webdav_path != ?
            ORDER BY indexed_at DESC, id DESC
            LIMIT ?
        """
        cur = conn.execute(sql, (glob_one, glob_nested, exclude, limit))
    rows = cur.fetchall()
    return folder, [(str(p), str(f), str(t)) for p, f, t in rows]


def search_similar_to_embedding(
    conn: sqlite3.Connection,
    query_embedding_f32: bytes,
    k: int = 20,
    exclude_path: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    Same as ``search_similar`` but optionally drops ``exclude_path`` from results
    and fills from extra neighbors (requests more from sqlite-vec then trims).
    """
    need = k + (1 if exclude_path else 0) + 5
    rows = conn.execute(
        """
        SELECT p.webdav_path, p.filename, v.distance
        FROM vec_photos AS v
        INNER JOIN photos AS p ON p.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        """,
        (query_embedding_f32, min(need, 100)),
    ).fetchall()
    out: List[Tuple[str, str, float]] = []
    for webdav_path, filename, dist in rows:
        if exclude_path and str(webdav_path) == exclude_path:
            continue
        out.append((str(webdav_path), str(filename), float(dist)))
        if len(out) >= k:
            break
    return out


def get_photo_indexed_at(
    conn: sqlite3.Connection, webdav_path: str
) -> Optional[str]:
    row = conn.execute(
        "SELECT indexed_at FROM photos WHERE webdav_path = ? LIMIT 1",
        (webdav_path,),
    ).fetchone()
    return str(row[0]) if row else None


def search_similar(
    conn: sqlite3.Connection,
    query_embedding_f32: bytes,
    k: int = 20,
) -> List[Tuple[str, str, float]]:
    """
    Return list of (webdav_path, filename, distance).
    Distance is sqlite-vec cosine distance (lower is more similar).
    """
    rows = conn.execute(
        """
        SELECT p.webdav_path, p.filename, v.distance
        FROM vec_photos AS v
        INNER JOIN photos AS p ON p.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        """,
        (query_embedding_f32, k),
    ).fetchall()
    out: List[Tuple[str, str, float]] = []
    for webdav_path, filename, dist in rows:
        out.append((str(webdav_path), str(filename), float(dist)))
    return out


_INDEX_ROOTS_KEY = "index_roots_json"


def get_index_roots(conn: sqlite3.Connection) -> List[str]:
    """
    Folder paths to crawl (each must start with /). Empty list or ["/"] means
    the entire WebDAV tree (default — same as before this setting existed).
    """
    row = conn.execute(
        "SELECT value FROM app_settings WHERE key = ?", (_INDEX_ROOTS_KEY,)
    ).fetchone()
    if not row:
        return []
    try:
        data = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for x in data:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def set_index_roots(conn: sqlite3.Connection, roots: List[str]) -> None:
    """Persist index roots; pass [] or [\"/\"] to index the full library."""
    payload = json.dumps(roots, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO app_settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_INDEX_ROOTS_KEY, payload),
    )


def list_recent_photos(
    conn: sqlite3.Connection, limit: int = 24
) -> List[Tuple[str, str, str]]:
    """Return [(webdav_path, filename, indexed_at_iso), ...] newest first."""
    rows = conn.execute(
        """
        SELECT webdav_path, filename, indexed_at
        FROM photos
        ORDER BY indexed_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [(str(p), str(f), str(t)) for p, f, t in rows]


def record_index_failure(
    conn: sqlite3.Connection, webdav_path: str, reason: str
) -> None:
    """Upsert a failed path (last failure wins)."""
    from datetime import datetime, timezone

    failed_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO index_failures (webdav_path, reason, failed_at)
        VALUES (?, ?, ?)
        ON CONFLICT(webdav_path) DO UPDATE SET
            reason = excluded.reason,
            failed_at = excluded.failed_at
        """,
        (webdav_path, reason[:2000], failed_at),
    )


def list_index_failures(
    conn: sqlite3.Connection, limit: int = 200
) -> List[Tuple[str, str, str]]:
    """Return [(webdav_path, reason, failed_at), ...] newest first."""
    rows = conn.execute(
        """
        SELECT webdav_path, reason, failed_at
        FROM index_failures
        ORDER BY failed_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [(str(p), str(r), str(t)) for p, r, t in rows]


def numpy_to_blob(vec) -> bytes:
    """Serialize a 1-D float32 vector to blob for storage and sqlite-vec."""
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBED_DIM:
        raise ValueError(f"Expected {EMBED_DIM} dims, got {arr.shape[0]}")
    return serialize_float32(arr)
