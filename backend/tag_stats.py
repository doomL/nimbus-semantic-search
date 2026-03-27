"""
Estimate which concepts appear most often in the indexed library.

CLIP does not output per-photo tags. We compare a fixed list of text prompts
to every image embedding (cosine similarity) and count matches above a
threshold — cheap as one matrix multiply once embeddings are loaded.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

from clip_model import encode_text_batch
from db import get_connection

logger = logging.getLogger(__name__)

# (short label shown in UI, text prompt for CLIP — descriptive phrases work well)
TAG_PROMPTS: List[Tuple[str, str]] = [
    ("nature", "nature and landscapes"),
    ("beach", "beach and ocean coastline"),
    ("mountains", "mountains and hills"),
    ("forest", "forest and trees"),
    ("snow", "snow and winter"),
    ("city", "urban city scene"),
    ("architecture", "buildings and architecture"),
    ("street", "street photography"),
    ("people", "people and persons"),
    ("portrait", "portrait of a person"),
    ("family", "family and group of people"),
    ("wedding", "wedding ceremony"),
    ("food", "food and meals"),
    ("drinks", "drinks and beverages"),
    ("coffee", "coffee and cafe"),
    ("animals", "animals and wildlife"),
    ("dogs", "dogs"),
    ("cats", "cats"),
    ("birds", "birds"),
    ("vehicles", "cars and vehicles"),
    ("night", "night time and evening"),
    ("sunset", "sunset and golden hour"),
    ("sunrise", "sunrise and dawn"),
    ("indoors", "indoor scene"),
    ("outdoors", "outdoor scene"),
    ("sports", "sports and athletics"),
    ("water", "water river lake"),
    ("sky", "sky and clouds"),
    ("flowers", "flowers and plants"),
    ("garden", "garden"),
    ("travel", "travel and vacation"),
    ("party", "party and celebration"),
    ("concert", "concert and live music"),
    ("baby", "baby and infant"),
    ("macro", "macro close-up detail"),
    ("black & white", "black and white photograph"),
    ("food close-up", "close-up of food"),
    ("sea", "sea and ocean water"),
    ("desert", "desert landscape"),
    ("rain", "rain and rainy weather"),
    ("autumn", "autumn fall colors"),
    ("spring", "spring blossoms"),
    ("home", "home interior"),
    ("work", "office and work"),
    ("pets", "pets"),
    ("boats", "boats and ships"),
    ("airplanes", "airplanes and aviation"),
    ("art", "art and artwork"),
    ("museum", "museum and gallery"),
    ("selfie", "selfie"),
    ("documents", "documents and screenshots of text"),
]

LABEL_TO_PROMPT: Dict[str, str] = {label: prompt for label, prompt in TAG_PROMPTS}


def _threshold() -> float:
    v = os.environ.get("TAG_SIMILARITY_THRESHOLD", "0.27").strip()
    try:
        return max(0.05, min(0.95, float(v)))
    except ValueError:
        return 0.27


def load_embedding_matrix(conn) -> np.ndarray:
    """(N, 512) float32 L2-normalized rows. Caller must hold db_lock."""
    rows = conn.execute("SELECT embedding FROM photos").fetchall()
    if not rows:
        return np.zeros((0, 512), dtype=np.float32)
    vecs = []
    for (blob,) in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != 512:
            continue
        vecs.append(v)
    if not vecs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.stack(vecs, axis=0).astype(np.float32)


def compute_tag_match_counts(X: np.ndarray) -> np.ndarray:
    """
    CLIP similarity vs fixed prompts; returns int counts per TAG_PROMPTS row.
    CPU/GPU heavy — call without holding db_lock.
    """
    prompts = [t[1] for t in TAG_PROMPTS]
    n = int(X.shape[0])
    if n == 0:
        return np.zeros(len(TAG_PROMPTS), dtype=np.int64)
    logger.info("Tag stats: computing matches for %s image embeddings…", n)
    T = encode_text_batch(prompts)
    sim = X @ T.T
    thr = _threshold()
    return (sim >= thr).sum(axis=0).astype(np.int64)


def replace_tag_stats(conn, counts: np.ndarray) -> int:
    """
    Replace tag_stats from per-label counts. Fast SQLite writes.
    Caller must hold db_lock.
    """
    labels = [t[0] for t in TAG_PROMPTS]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM tag_stats")
    nonzero = 0
    for label, c in zip(labels, counts):
        if int(c) > 0:
            conn.execute(
                "INSERT INTO tag_stats (tag, count, updated_at) VALUES (?, ?, ?)",
                (label, int(c), now),
            )
            nonzero += 1
    conn.commit()
    logger.info(
        "Tag stats: threshold=%.3f, %s tags with at least one match.",
        _threshold(),
        nonzero,
    )
    return nonzero


def recompute_library_tags(conn) -> int:
    """
    Replace tag_stats from current photos (single-threaded / tests).
    Caller must hold db_lock for the whole call.
    """
    X = load_embedding_matrix(conn)
    if X.shape[0] == 0:
        conn.execute("DELETE FROM tag_stats")
        conn.commit()
        return 0
    counts = compute_tag_match_counts(X)
    return replace_tag_stats(conn, counts)


def recompute_library_tags_background(db_lock: threading.Lock) -> int:
    """
    Rebuild tag_stats without holding db_lock during CLIP/numpy work.
    Other handlers can read the DB (e.g. GET /tags/popular) while this runs.
    """
    with db_lock:
        conn = get_connection()
        X = load_embedding_matrix(conn)
    if X.shape[0] == 0:
        with db_lock:
            conn = get_connection()
            conn.execute("DELETE FROM tag_stats")
            conn.commit()
        return 0
    counts = compute_tag_match_counts(X)
    with db_lock:
        conn = get_connection()
        return replace_tag_stats(conn, counts)


def get_popular_tags(conn, limit: int = 24) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT tag, count FROM tag_stats
        ORDER BY count DESC, tag ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        label = str(r[0])
        out.append(
            {
                "tag": label,
                "count": int(r[1]),
                "suggest": LABEL_TO_PROMPT.get(label, label),
            }
        )
    return out
