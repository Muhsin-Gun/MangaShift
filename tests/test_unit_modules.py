from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.inpaint_engine import inpaint_image  # noqa: E402
from app.model_manager import ModelManager  # noqa: E402
from app.style_engine import apply_style  # noqa: E402
from app.translate_engine import detect_text_lang  # noqa: E402
from app.typeset_engine import typeset_image  # noqa: E402
from app.upscale_engine import upscale_image  # noqa: E402


def sample_image() -> Image.Image:
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 280, 140), radius=20, outline="black", width=2, fill="white")
    draw.text((90, 80), "hello", fill="black")
    return image


def test_detect_lang():
    assert detect_text_lang("\uc548\ub155\ud558\uc138\uc694") == "ko"
    assert detect_text_lang("\u3053\u3093\u306b\u3061\u306f") == "ja"
    assert detect_text_lang("hello") == "unknown"


def test_inpaint_fallback_runs():
    settings = get_settings()
    mm = ModelManager(settings)
    image = sample_image()
    out, mask = inpaint_image(image, [[80, 70, 100, 30]], model_manager=mm)
    assert out.size == image.size
    assert mask.size == image.size


def test_style_and_upscale():
    settings = get_settings()
    mm = ModelManager(settings)
    image = sample_image()
    styled = apply_style(image, "smooth", mm)
    upscaled = upscale_image(styled, scale=2.0, mode="pil")
    assert upscaled.size == (640, 480)


def test_typeset():
    settings = get_settings()
    image = sample_image()
    out = typeset_image(
        image,
        [{"bbox": [60, 50, 180, 80], "translated_text": "You are late, hurry up!"}],
        fonts_dir=settings.fonts_dir,
    )
    assert out.size == image.size
    before = io.BytesIO()
    after = io.BytesIO()
    image.save(before, format="PNG")
    out.save(after, format="PNG")
    assert before.getvalue() != after.getvalue()
