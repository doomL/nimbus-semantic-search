"""AI agent for natural language photo search via OpenRouter + CLIP."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = """\
You are an assistant that helps search a personal photo and video library using CLIP embeddings.

Given a natural language description from the user, extract 4-6 concise visual search terms \
that work well with CLIP (a model that matches images to text).

Rules:
- Each term must describe a visual scene, subject, mood, or object — concrete and visual
- Avoid abstract concepts, names, or dates (CLIP doesn't know "Leonardo" or "2022")
- Prefer short English phrases (3-7 words)
- No duplicates

Respond ONLY with a valid JSON array of strings, nothing else.
Example input: "le foto della gita in montagna con mio fratello"
Example output: ["mountain hiking trail", "family hiking outdoors", "alpine landscape summer", "people mountains outdoor", "green mountain valley"]
"""


def _call_openrouter(query: str) -> List[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    model = os.environ.get("NIMBUS_AGENT_MODEL", "google/gemini-2.5-flash-lite")
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{_OPENROUTER_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"].strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    terms = json.loads(content)
    if not isinstance(terms, list):
        raise ValueError(f"Expected JSON array, got: {type(terms)}")
    return [str(t).strip() for t in terms if str(t).strip()]


def agent_search(
    query: str,
    k: int = 20,
    media_type: str = "all",
) -> Dict[str, Any]:
    """Decompose query via LLM, run multi-term CLIP search, merge by best score."""
    from clip_model import encode_text_query
    from db import get_connection, numpy_to_blob, search_similar
    from search import _IMAGE_EXTS, _VIDEO_EXTS, _file_ext

    terms = _call_openrouter(query)
    logger.info("Agent expanded %r → %s", query, terms)

    conn = get_connection()
    best: Dict[str, Dict] = {}

    for term in terms:
        if not term:
            continue
        try:
            vec = encode_text_query(term)
            blob = numpy_to_blob(vec)
            rows = search_similar(conn, blob, k=15)
            for webdav_path, filename, distance in rows:
                ext = _file_ext(webdav_path)
                if media_type == "image" and ext not in _IMAGE_EXTS:
                    continue
                if media_type == "video" and ext not in _VIDEO_EXTS:
                    continue
                score = max(0.0, min(1.0, 1.0 - float(distance)))
                if webdav_path not in best or score > best[webdav_path]["score"]:
                    best[webdav_path] = {
                        "webdav_path": webdav_path,
                        "filename": filename,
                        "score": score,
                    }
        except Exception as e:
            logger.warning("CLIP search failed for term %r: %s", term, e)

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:k]
    strong_min = float(os.environ.get("NIMBUS_SEARCH_STRONG_MIN", "0.22"))
    return {
        "terms": terms,
        "results": [r for r in ranked if r["score"] >= strong_min],
        "weak_results": [r for r in ranked if r["score"] < strong_min],
        "strong_min": strong_min,
    }
