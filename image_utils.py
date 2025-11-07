import base64
import io
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


def _ext_and_mime_from_pillow(data: bytes):
    with Image.open(io.BytesIO(data)) as img:
        fmt = (img.format or "").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        mime = SUPPORTED.get(fmt)
        return fmt, mime


def to_image_part(uploaded_file) -> Dict:
    data = uploaded_file.getvalue()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"image > {MAX_IMAGE_MB} Mo")

    ext = (
        uploaded_file.name.rsplit(".", 1)[-1].lower()
        if "." in uploaded_file.name
        else ""
    )
    mime = SUPPORTED.get(ext)

    if mime is None:
        try:
            ext2, mime2 = _ext_and_mime_from_pillow(data)
            if mime2:
                ext, mime = ext2, mime2
        except Exception:
            pass

    if mime is None:
        raise ValueError("format d'image non supporté")

    b64 = base64.b64encode(data).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
