from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from .model_manager import ModelManager


class Colorizer:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.series_palettes: Dict[str, dict] = {}

    @staticmethod
    def is_grayscale(image: Image.Image, threshold: float = 9.5) -> bool:
        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        sat = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
        return bool(sat < threshold)

    def register_series_palette(self, series_id: str, palette: dict) -> None:
        self.series_palettes[series_id] = palette

    def _palette_suffix(self, series_id: Optional[str]) -> str:
        if not series_id:
            return ""
        palette = self.series_palettes.get(series_id, {})
        hints = []
        for char_id, colors in palette.items():
            hair = colors.get("hair")
            outfit = colors.get("outfit")
            if hair:
                hints.append(f"{char_id} hair {hair}")
            if outfit:
                hints.append(f"{char_id} outfit {outfit}")
        return ", ".join(hints)

    def _fallback_colorize(self, image: Image.Image) -> Image.Image:
        gray = np.array(image.convert("L"), dtype=np.float32)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        detail = clahe.apply(np.clip(gray, 0, 255).astype(np.uint8)).astype(np.float32)
        n = np.clip(detail / 255.0, 0.0, 1.0)

        # Warm manhwa-friendly tonal ramp (avoids blue/purple cast in skin).
        shadow = np.array([42.0, 44.0, 50.0], dtype=np.float32)
        mid = np.array([176.0, 160.0, 142.0], dtype=np.float32)
        high = np.array([236.0, 225.0, 210.0], dtype=np.float32)

        t1 = (1.0 - n)[:, :, None]
        t2 = n[:, :, None]
        color = (t1 * t1) * shadow + (2.0 * t1 * t2) * mid + (t2 * t2) * high

        # Preserve line readability by darkening strong edges slightly.
        edges = cv2.Canny(detail.astype(np.uint8), 70, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        em = (edges > 0).astype(np.float32)[:, :, None]
        color = color * (1.0 - 0.18 * em)

        return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), mode="RGB")

    def colorize(self, image: Image.Image, series_id: Optional[str] = None) -> Image.Image:
        if not self.is_grayscale(image):
            return image
        if self.mm.device != "cuda":
            return self._fallback_colorize(image)
        pipe = self.mm.load_img2img_pipeline()
        if pipe is None:
            return self._fallback_colorize(image)

        palette_suffix = self._palette_suffix(series_id)
        prompt = "full-color manga panel, anime coloring, clean line preservation"
        if palette_suffix:
            prompt = f"{prompt}, {palette_suffix}"
        negative = "grayscale, monochrome, washed out, blurry"

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=image.convert("RGB"),
                strength=0.62,
                guidance_scale=8.0,
                num_inference_steps=16 if self.mm.device == "cuda" else 8,
            ).images[0]
            return result.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        except Exception as exc:
            logger.warning("Diffusion colorization failed: {}", exc)
            return self._fallback_colorize(image)
