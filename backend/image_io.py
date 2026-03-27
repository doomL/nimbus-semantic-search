"""Shared helpers for opening image bytes (clearer errors than raw PIL)."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Allow indexing/thumbnails for incomplete downloads or corrupt JPEGs from cloud storage.
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
