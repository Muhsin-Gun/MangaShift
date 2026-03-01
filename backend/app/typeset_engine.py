from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont

from .utils import clamp_bbox, pil_to_png_bytes, safe_image_load


DEFAULT_FONT_CANDIDATES = [
    "Bangers-Regular.ttf",
    "KomikaText-Regular.ttf",
    "DejaVuSans.ttf",
]


def _load_font(fonts_dir: Path, size: int) -> ImageFont.ImageFont:
    for candidate in DEFAULT_FONT_CANDIDATES:
        path = fonts_dir / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    fonts_dir: Path,
    min_size: int = 10,
    max_size: int = 44,
) -> Tuple[str, ImageFont.ImageFont]:
    clean = " ".join(text.split())
    if not clean:
        return "", _load_font(fonts_dir, min_size)

    for size in range(max_size, min_size - 1, -1):
        font = _load_font(fonts_dir, size)
        for width in range(24, 5, -1):
            wrapped = textwrap.fill(clean, width=width)
            left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=2)
            tw = right - left
            th = bottom - top
            if tw <= int(box_w * 0.92) and th <= int(box_h * 0.9):
                return wrapped, font
    return clean, _load_font(fonts_dir, min_size)


def typeset_image(
    image: Image.Image,
    translated_regions: Iterable[dict],
    fonts_dir: Path,
) -> Image.Image:
    image = image.convert("RGB")
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w_img, h_img = out.size

    for region in translated_regions:
        text = str(region.get("translated_text", "")).strip()
        if not text:
            continue
        bbox = region.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x, y, w, h = clamp_bbox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), w_img, h_img)
        wrapped, font = _fit_text(draw, text, w, h, fonts_dir)
        if not wrapped:
            continue

        left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=2)
        tw, th = right - left, bottom - top
        tx = x + (w - tw) // 2
        ty = y + (h - th) // 2
        draw.multiline_text(
            (tx, ty),
            wrapped,
            font=font,
            fill="black",
            align="center",
            spacing=2,
            stroke_width=2,
            stroke_fill="white",
        )
    return out


def typeset_image_with_translations(
    image_bytes: bytes,
    translated_regions: list[dict],
    fonts_dir: Path,
) -> bytes:
    image = safe_image_load(image_bytes)
    out = typeset_image(image, translated_regions, fonts_dir=fonts_dir)
    return pil_to_png_bytes(out)
