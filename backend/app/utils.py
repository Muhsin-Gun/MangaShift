from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_image_load(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def pil_to_png_bytes(image: Image.Image, optimize: bool = True) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=optimize)
    return out.getvalue()


def png_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return safe_image_load(image_bytes)


def flatten_texts(text_regions: Iterable[dict]) -> str:
    values = [str(item.get("text", "")).strip() for item in text_regions]
    return " | ".join(v for v in values if v)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp_bbox(x: int, y: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def timed_ms(start_time: float, end_time: float) -> int:
    return int((end_time - start_time) * 1000)
