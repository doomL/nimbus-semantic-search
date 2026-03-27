"""Shared OpenCLIP model (ViT-B-32 / openai) for image and text encoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

_model: Optional["nn.Module"] = None
_preprocess = None
_tokenizer = None
_device: Optional[str] = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_clip() -> Tuple["nn.Module", object, object, str]:
    """Lazy-load model, preprocess, tokenizer, and device."""
    global _model, _preprocess, _tokenizer
    device = get_device()
    if _model is None:
        logger.info("Loading OpenCLIP ViT-B-32 (openai) on %s …", device)
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _model = _model.to(device)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("OpenCLIP ready.")
    assert _preprocess is not None and _tokenizer is not None
    return _model, _preprocess, _tokenizer, device


@torch.no_grad()
def encode_image_pil(image_rgb: Image.Image) -> np.ndarray:
    """Return L2-normalized float32 embedding vector (512,)."""
    model, preprocess, _, device = get_clip()
    tensor = preprocess(image_rgb.convert("RGB")).unsqueeze(0).to(device)
    emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_image_bytes(data: bytes, *, source: str = "") -> np.ndarray:
    """Decode image bytes (RGB), return L2-normalized float32 embedding (512,)."""
    from image_io import load_rgb_image

    img = load_rgb_image(data, source=source)
    return encode_image_pil(img)


@torch.no_grad()
def encode_text_query(text: str) -> np.ndarray:
    """Return L2-normalized float32 embedding for natural language query."""
    model, _, tokenizer, device = get_clip()
    tokens = tokenizer([text])
    tokens = tokens.to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_text_batch(texts: List[str]) -> np.ndarray:
    """Encode many prompts at once. Returns (N, 512) float32 L2-normalized."""
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)
    model, _, tokenizer, device = get_clip()
    batch_size = 32
    chunks: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        tokens = tokenizer(chunk)
        tokens = tokens.to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        chunks.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)
