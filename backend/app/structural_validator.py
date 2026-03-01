from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from PIL import Image


@dataclass
class StructuralReport:
    edge_overlap: float
    blur_score: float
    contrast_shift: float
    texture_shift: float
    should_rerender: bool
    note: str

    def to_dict(self) -> Dict[str, float | bool | str]:
        return {
            "edge_overlap": round(self.edge_overlap, 6),
            "blur_score": round(self.blur_score, 6),
            "contrast_shift": round(self.contrast_shift, 6),
            "texture_shift": round(self.texture_shift, 6),
            "should_rerender": self.should_rerender,
            "note": self.note,
        }


def _edge_map(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 80, 160)


def _laplacian_var(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _contrast_std(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(gray.std())


def analyze_structure(original: Image.Image, rendered: Image.Image) -> StructuralReport:
    e1 = _edge_map(original)
    e2 = _edge_map(rendered)
    e1_bin = (e1 > 0).astype(np.uint8)
    e2_bin = (e2 > 0).astype(np.uint8)
    intersection = int((e1_bin & e2_bin).sum())
    union = int((e1_bin | e2_bin).sum())
    edge_overlap = float(intersection / (union + 1e-9))

    blur_original = _laplacian_var(original)
    blur_rendered = _laplacian_var(rendered)
    blur_score = float(blur_rendered / (blur_original + 1e-9))

    contrast_shift = float(abs(_contrast_std(rendered) - _contrast_std(original)))

    # Texture proxy based on local binary edge density shift.
    texture_original = float(e1_bin.mean())
    texture_rendered = float(e2_bin.mean())
    texture_shift = float(abs(texture_rendered - texture_original))

    should_rerender = bool(
        edge_overlap < 0.52
        or blur_score < 0.45
        or contrast_shift > 42.0
        or texture_shift > 0.12
    )
    if should_rerender:
        note = "Structure drift detected; rerender recommended"
    else:
        note = "Structure within tolerance"
    return StructuralReport(
        edge_overlap=edge_overlap,
        blur_score=blur_score,
        contrast_shift=contrast_shift,
        texture_shift=texture_shift,
        should_rerender=should_rerender,
        note=note,
    )
