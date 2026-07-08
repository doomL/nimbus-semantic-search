"""Shared helpers for opening image bytes (clearer errors than raw PIL)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Allow indexing/thumbnails for incomplete downloads or corrupt JPEGs from cloud storage.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _dms_to_dd(dms, ref: str) -> Optional[float]:
    """Convert EXIF DMS (degrees/minutes/seconds) tuple to decimal degrees."""
    try:
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
    except (TypeError, IndexError, ValueError):
        return None
    dd = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        dd = -dd
    return dd


def extract_gps_from_bytes(data: bytes) -> Optional[Tuple[float, float]]:
    """
    Return (latitude, longitude) decimal degrees from EXIF GPS tags, or None.
    Works with JPEG, HEIC, and other formats that embed EXIF GPS.
    """
    try:
        img = Image.open(BytesIO(data))
        exif = img.getexif()
        if exif is None:
            return None
        # 0x8825 = GPS IFD tag
        gps_ifd = exif.get_ifd(0x8825)
        if not gps_ifd:
            return None
        # GPS tags: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon
        lat = _dms_to_dd(gps_ifd.get(2), gps_ifd.get(1, ""))
        lon = _dms_to_dd(gps_ifd.get(4), gps_ifd.get(3, ""))
        if lat is None or lon is None:
            return None
        return (lat, lon)
    except Exception:
        return None


def extract_frame_from_video_bytes(data: bytes, *, source: str = "") -> Image.Image:
    """Extract one representative frame from video bytes via ffmpeg; returns RGB PIL image."""
    if not data:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"Empty video data{suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".video") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Try at 1 second first to skip potential black intro frames.
        result = None
        for seek in (["-ss", "1"], []):
            result = subprocess.run(
                ["ffmpeg", "-y", *seek, "-i", tmp_path,
                 "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                capture_output=True,
                timeout=60,
            )
            if result.stdout:
                break

        if not result or not result.stdout:
            suffix = f" ({source})" if source else ""
            raise ValueError(f"ffmpeg returned no frame data{suffix}")

        return Image.open(BytesIO(result.stdout)).convert("RGB")
    finally:
        os.unlink(tmp_path)


def load_rgb_image(data: bytes, *, source: str = "") -> Image.Image:
    """
    Open image bytes, apply EXIF orientation, return RGB.

    Raises ValueError if data is empty; UnidentifiedImageError with a useful
    message if bytes are not a decodable image (e.g. HTML error body).

    Truncated JPEG/PNG streams are loaded best-effort (Pillow may show artifacts
    at the bottom edge) so partially synced or damaged files still get embeddings.
    """
    if not data:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"Empty image data{suffix}")

    try:
        img = Image.open(BytesIO(data))
    except UnidentifiedImageError as e:
        suffix = f" ({source})" if source else ""
        raise UnidentifiedImageError(
            f"Not a valid image file{suffix}: received {len(data)} bytes that "
            "are not a recognized image format (wrong extension, corrupt file, "
            "or non-image response from server)."
        ) from e

    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
