"""Natural language search via CLIP text encoder + sqlite-vec."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from clip_model import encode_text_query
from db import get_connection, numpy_to_blob, search_similar

# Matches below this similarity are returned as weak_results (UI can de-emphasize).
_STRONG_MIN = float(os.environ.get("NIMBUS_SEARCH_STRONG_MIN", "0.22"))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}


def _file_ext(path: str) -> str:
    dot = path.lower().rfind(".")
    return path[dot:].lower() if dot >= 0 else ""


def search_photos(
    query: str,
    k: int = 20,
    strong_min: float | None = None,
    media_type: str = "all",
) -> Dict[str, Any]:
    """
    Encode query text, run KNN, return results + weak_results split by score.

    Score is higher for better matches: (1.0 - cosine_distance), clamped to [0, 1].
    Top ``k`` neighbors are partitioned: ``results`` are at or above ``strong_min``;
    the rest of those k are ``weak_results`` (low confidence).
    ``media_type`` filters results: "all" | "image" | "video".
    """
    q = query.strip()
    if not q:
        return {"results": [], "weak_results": [], "strong_min": _STRONG_MIN}

    smin = strong_min if strong_min is not None else _STRONG_MIN
    vec = encode_text_query(q)
    blob = numpy_to_blob(vec)
    conn = get_connection()
    # Request a larger pool when filtering so we still return k results after pruning.
    fetch_k = k if media_type == "all" else min(k * 4, 500)
    rows = search_similar(conn, blob, k=fetch_k)
    mapped: List[Dict[str, Any]] = []
    for webdav_path, filename, distance in rows:
        ext = _file_ext(webdav_path)
        if media_type == "image" and ext not in _IMAGE_EXTS:
            continue
        if media_type == "video" and ext not in _VIDEO_EXTS:
            continue
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        mapped.append({"webdav_path": webdav_path, "filename": filename, "score": score})
        if len(mapped) >= k:
            break
    strong = [r for r in mapped if r["score"] >= smin]
    weak = [r for r in mapped if r["score"] < smin]
    return {"results": strong, "weak_results": weak, "strong_min": smin}
