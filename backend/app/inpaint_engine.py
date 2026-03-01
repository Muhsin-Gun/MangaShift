from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from .model_manager import ModelManager
from .utils import clamp_bbox, pil_to_png_bytes, safe_image_load


def make_text_mask(size: Tuple[int, int], boxes: Iterable[List[int]], padding: int = 8) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        if len(box) != 4:
            continue
        x, y, w, h = [int(v) for v in box]
        x, y, w, h = clamp_bbox(x, y, w, h, width, height)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        mask[y1:y2, x1:x2] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return Image.fromarray(mask, mode="L")


def _fallback_cv2_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask.convert("L"))
    out = cv2.inpaint(image_np, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def inpaint_image(
    image: Image.Image,
    boxes: List[List[int]],
    model_manager: ModelManager,
    padding: int = 8,
) -> tuple[Image.Image, Image.Image]:
    out, mask, _ = inpaint_image_with_report(
        image=image,
        boxes=boxes,
        model_manager=model_manager,
        padding=padding,
    )
    return out, mask


def inpaint_image_with_report(
    image: Image.Image,
    boxes: List[List[int]],
    model_manager: ModelManager,
    padding: int = 8,
) -> tuple[Image.Image, Image.Image, str]:
    if not boxes:
        return image, Image.new("L", image.size, 0), "skipped"
    mask = make_text_mask(image.size, boxes, padding=padding)
    if np.array(mask).sum() == 0:
        return image, mask, "skipped"

    # On CPU, prefer deterministic OpenCV for latency. Keep diffusion inpaint for CUDA/final pipelines.
    prefer_diffusion = bool(model_manager.device == "cuda")
    if not prefer_diffusion:
        logger.info("Using OpenCV inpainting fallback (cpu_fast_mode)")
        return _fallback_cv2_inpaint(image, mask), mask, "opencv_telea"

    pipeline = model_manager.load_inpaint_pipeline()
    if pipeline is None:
        logger.info("Using OpenCV inpainting fallback")
        return _fallback_cv2_inpaint(image, mask), mask, "opencv_telea"

    try:
        result = pipeline(
            prompt="clean manga speech bubble background, preserve line art and screentones",
            image=image,
            mask_image=mask,
            guidance_scale=7.0,
            strength=0.98,
            num_inference_steps=20 if model_manager.device == "cuda" else 12,
        ).images[0]
        return result.convert("RGB"), mask, "diffusers_inpaint"
    except Exception as exc:
        logger.warning("Diffusion inpaint failed, fallback to OpenCV: {}", exc)
        return _fallback_cv2_inpaint(image, mask), mask, "opencv_telea"


def inpaint_image_bytes(
    image_bytes: bytes,
    boxes: List[List[int]],
    model_manager: ModelManager,
    padding: int = 8,
) -> bytes:
    image = safe_image_load(image_bytes)
    out, _ = inpaint_image(image, boxes, model_manager, padding=padding)
    return pil_to_png_bytes(out)
