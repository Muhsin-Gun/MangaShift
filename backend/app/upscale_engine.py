from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageFilter

from .config import get_settings
from .utils import pil_to_png_bytes, safe_image_load


def _quality_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@lru_cache(maxsize=1)
def _load_realesrgan_model():
    settings = get_settings()
    model_path = Path(settings.realesrgan_model_path)
    if not model_path.exists():
        logger.warning("Real-ESRGAN weights not found: {}", model_path)
        return None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        enhancer = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=512,
            tile_pad=24,
            pre_pad=0,
            half=False,
        )
        return enhancer
    except Exception as exc:
        logger.warning("Real-ESRGAN unavailable: {}", exc)
        return None


def _upscale_realesrgan(image: Image.Image, scale: float) -> Image.Image:
    enhancer = _load_realesrgan_model()
    if enhancer is None:
        raise RuntimeError("Real-ESRGAN unavailable")
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    output, _ = enhancer.enhance(bgr, outscale=scale)
    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _upscale_pil(image: Image.Image, scale: float) -> Image.Image:
    if scale <= 1.0:
        return image
    w, h = image.size
    up = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    # Conservative sharpening to keep line art crisp.
    return up.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))


def upscale_image(image: Image.Image, scale: float = 2.0, mode: str = "auto") -> Image.Image:
    out, _ = upscale_image_with_report(image=image, scale=scale, mode=mode)
    return out


def upscale_image_with_report(image: Image.Image, scale: float = 2.0, mode: str = "auto") -> tuple[Image.Image, str]:
    if scale <= 1.0:
        return image, "skipped"
    quality = _quality_score(image)
    selected = mode
    if mode == "auto":
        selected = "realesrgan" if quality < 180 else "pil"
    logger.info("Upscaler mode={} quality={:.2f}", selected, quality)

    if selected == "realesrgan":
        try:
            return _upscale_realesrgan(image, scale), "realesrgan"
        except Exception:
            return _upscale_pil(image, scale), "pil_unsharp"
    return _upscale_pil(image, scale), "pil_unsharp"


def upscale_image_bytes(image_bytes: bytes, scale: float = 2.0, mode: str = "auto") -> bytes:
    image = safe_image_load(image_bytes)
    out = upscale_image(image, scale=scale, mode=mode)
    return pil_to_png_bytes(out)
