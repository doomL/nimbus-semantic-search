"""Natural language search via CLIP text encoder + sqlite-vec."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from clip_model import encode_text_query
from db import get_connection, numpy_to_blob, search_similar

# Matches below this similarity are returned as weak_results (UI can de-emphasize).
_STRONG_MIN = float(os.environ.get("NIMBUS_SEARCH_STRONG_MIN", "0.22"))


def search_photos(
    query: str, k: int = 20, strong_min: float | None = None,
) -> Dict[str, Any]:
    """
    Encode query text, run KNN, return results + weak_results split by score.

    Score is higher for better matches: (1.0 - cosine_distance), clamped to [0, 1].
    Top ``k`` neighbors are partitioned: ``results`` are at or above ``strong_min``;
    the rest of those k are ``weak_results`` (low confidence).
    """
    q = query.strip()
    if not q:
        return {"results": [], "weak_results": [], "strong_min": _STRONG_MIN}

    smin = strong_min if strong_min is not None else _STRONG_MIN
    vec = encode_text_query(q)
    blob = numpy_to_blob(vec)
    conn = get_connection()
    rows = search_similar(conn, blob, k=k)
    mapped: List[Dict[str, Any]] = []
    for webdav_path, filename, distance in rows:
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        mapped.append(
            {
                "webdav_path": webdav_path,
                "filename": filename,
                "score": score,
            }
        )
    strong = [r for r in mapped if r["score"] >= smin]
    weak = [r for r in mapped if r["score"] < smin]
    return {"results": strong, "weak_results": weak, "strong_min": smin}
