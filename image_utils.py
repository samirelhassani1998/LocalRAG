"""Helper utilities for preparing image uploads for multimodal requests."""

import base64
import io
import imghdr
from typing import Dict

from PIL import Image

SUPPORTED = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
MAX_IMAGE_MB = 10


def _ext_from_bytes(data: bytes) -> str:
    """Infer an image extension from raw bytes using ``imghdr``."""

    kind = imghdr.what(None, h=data)
    return "jpeg" if kind == "jpg" else (kind or "png")


def to_image_part(uploaded_file) -> Dict:
    """Convert a Streamlit uploaded file into a Chat Completions image part."""

    data = uploaded_file.getvalue()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"image > {MAX_IMAGE_MB} Mo")

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext not in SUPPORTED:
        ext = _ext_from_bytes(data)
        if ext not in SUPPORTED:
            raise ValueError("format non supporté")

    try:
        Image.open(io.BytesIO(data)).load()
    except Exception:
        # We intentionally swallow Pillow errors to fall back on raw bytes when possible.
        pass

    b64 = base64.b64encode(data).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:{SUPPORTED[ext]};base64,{b64}"}

