"""Natural language search via CLIP text encoder + sqlite-vec."""

from __future__ import annotations

from typing import Any, Dict, List

from clip_model import encode_text_query
from db import get_connection, numpy_to_blob, search_similar


def search_photos(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """
    Encode query text, run KNN, return [{webdav_path, filename, score}, ...].
    Score is higher for better matches: (1.0 - cosine_distance), clamped to [0, 1].
    """
    q = query.strip()
    if not q:
        return []

    vec = encode_text_query(q)
    blob = numpy_to_blob(vec)
    conn = get_connection()
    rows = search_similar(conn, blob, k=k)
    results: List[Dict[str, Any]] = []
    for webdav_path, filename, distance in rows:
        # cosine distance in [0, 2] typically; similarity-style score
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            {
                "webdav_path": webdav_path,
                "filename": filename,
                "score": score,
            }
        )
    return results
