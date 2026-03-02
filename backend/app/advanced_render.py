from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.exposure import match_histograms

from .character_db import CharacterDatabase
from .ink_tone_refiner import InkToneParams, refine_ink_tone_pass
from .model_manager import ModelManager
from .quality_gate import QualityThresholds, evaluate_quality_gate
from .style_engine import get_style_package


@dataclass
class VariantParams:
    structure_strength: float
    structure_guidance: float
    structure_steps: int
    style_strength: float
    style_guidance: float
    style_steps: int
    control_line: float
    control_depth: float
    control_pose: float
    lora_scale: float
    line_overlay_alpha: float


@dataclass
class VariantResult:
    index: int
    seed: int
    params: VariantParams
    retries: int
    backend: str
    sketch_pass: Optional[Image.Image]
    structure_pass: Image.Image
    final: Image.Image
    quality: dict
    output_score: float


class QualityPathError(RuntimeError):
    """Raised when strict quality mode cannot execute the full SDXL path."""


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = str(value or "").strip().lstrip("#")
    if len(value) != 6:
        return (128, 128, 128)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _sha_seed(raw: bytes, salt: str = "") -> int:
    digest = hashlib.sha256(raw + salt.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _canny_map(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 80, 160)
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edge_rgb)


def _pseudo_depth_map(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)
    grad = cv2.Laplacian(blur, cv2.CV_32F)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = 255 - grad
    depth_rgb = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)


@lru_cache(maxsize=1)
def _midas_detector():
    from controlnet_aux import MidasDetector

    return MidasDetector.from_pretrained("lllyasviel/Annotators")


@lru_cache(maxsize=1)
def _openpose_detector():
    from controlnet_aux import OpenposeDetector

    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def _midas_depth_map(image: Image.Image, enabled: bool = True) -> Optional[Image.Image]:
    if not enabled:
        return None
    try:
        detector = _midas_detector()
        src = _to_rgb(image)
        result = detector(src)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, Image.Image):
            return result.convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(result).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
    except Exception:
        return None
    return None


def _pose_map(image: Image.Image, enabled: bool = True) -> Optional[Image.Image]:
    if not enabled:
        return None
    try:
        detector = _openpose_detector()
        src = _to_rgb(image)
        result = detector(src)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, Image.Image):
            return result.convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(result).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
    except Exception:
        return None
    return None


def _face_eye_hand_masks(
    image: Image.Image,
    character_db: CharacterDatabase,
) -> Dict[str, Image.Image]:
    w, h = image.size
    face_mask = Image.new("L", (w, h), 0)
    eye_mask = Image.new("L", (w, h), 0)
    hand_mask = Image.new("L", (w, h), 0)
    skin_mask = Image.new("L", (w, h), 0)
    hair_mask = Image.new("L", (w, h), 0)
    clothes_mask = Image.new("L", (w, h), 0)
    background_mask = Image.new("L", (w, h), 0)

    draw_face = ImageDraw.Draw(face_mask)
    draw_eye = ImageDraw.Draw(eye_mask)
    draw_skin = ImageDraw.Draw(skin_mask)
    draw_hair = ImageDraw.Draw(hair_mask)
    draw_clothes = ImageDraw.Draw(clothes_mask)

    try:
        faces = character_db.detect_faces(image)
    except Exception:
        faces = []

    rgb = _to_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dyn_thr = int(max(8, min(42, np.percentile(gray, 42) * 0.46)))
    fg_mask = (gray > dyn_thr).astype(np.uint8) * 255
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    primary_bbox: Optional[Tuple[int, int, int, int]] = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch >= max(1200, int(0.06 * w * h)):
            primary_bbox = (int(x), int(y), int(cw), int(ch))
    if primary_bbox is None:
        primary_bbox = (int(w * 0.12), int(h * 0.08), int(w * 0.76), int(h * 0.84))

    for _, (x, y, fw, fh) in faces:
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + fw))
        y2 = min(h, int(y + fh))
        draw_face.rectangle((x1, y1, x2, y2), fill=220)
        draw_skin.rectangle((x1, y1 + int(fh * 0.22), x2, y2), fill=200)
        draw_hair.rectangle((x1, y1, x2, y1 + int(fh * 0.35)), fill=170)

        # Approximate eyes from upper third of face.
        ey_h = max(4, int(fh * 0.18))
        ey_w = max(6, int(fw * 0.22))
        ey_y = y1 + max(2, int(fh * 0.2))
        lx = x1 + max(2, int(fw * 0.2))
        rx = x1 + max(2, int(fw * 0.58))
        draw_eye.ellipse((lx, ey_y, lx + ey_w, ey_y + ey_h), fill=255)
        draw_eye.ellipse((rx, ey_y, rx + ey_w, ey_y + ey_h), fill=255)

    if not faces and primary_bbox is not None:
        bx, by, bw, bh = primary_bbox
        fx1 = max(0, int(bx + bw * 0.16))
        fx2 = min(w, int(bx + bw * 0.84))
        fy1 = max(0, int(by + bh * 0.05))
        fy2 = min(h, int(by + bh * 0.63))
        if fx2 > fx1 and fy2 > fy1:
            draw_face.ellipse((fx1, fy1, fx2, fy2), fill=210)
            draw_skin.ellipse((fx1, int(fy1 + (fy2 - fy1) * 0.16), fx2, fy2), fill=200)
            draw_hair.ellipse((fx1, fy1, fx2, int(fy1 + (fy2 - fy1) * 0.42)), fill=185)
            ew = max(6, int((fx2 - fx1) * 0.2))
            eh = max(4, int((fy2 - fy1) * 0.12))
            ey = fy1 + int((fy2 - fy1) * 0.3)
            lx = fx1 + int((fx2 - fx1) * 0.22)
            rx = fx1 + int((fx2 - fx1) * 0.58)
            draw_eye.ellipse((lx, ey, lx + ew, ey + eh), fill=255)
            draw_eye.ellipse((rx, ey, rx + ew, ey + eh), fill=255)

    # Heuristic hand candidates from high-frequency contours in lower half.
    edges = cv2.Canny(gray, 70, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_hand = ImageDraw.Draw(hand_mask)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 1200 or area > (w * h * 0.08):
            continue
        if y < int(h * 0.35):
            continue
        aspect = cw / float(ch + 1e-6)
        if 0.25 <= aspect <= 2.8:
            draw_hand.rectangle((x, y, x + cw, y + ch), fill=180)

    # Heuristic skin extraction in HSV (kept conservative to avoid over-masking ink lines).
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([2, 20, 35], dtype=np.uint8)
    upper_skin = np.array([28, 200, 255], dtype=np.uint8)
    skin_hsv = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_hsv = cv2.morphologyEx(skin_hsv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin_hsv = cv2.dilate(skin_hsv, np.ones((3, 3), np.uint8), iterations=1)

    # Grayscale-friendly fallback skin gate: mid luminance inside detected face/subject area.
    skin_seed = np.array(skin_mask, dtype=np.uint8)
    face_arr = np.array(face_mask, dtype=np.uint8)
    fg_bin = (fg_mask > 0).astype(np.uint8) * 255
    luma_mid = ((gray >= 52) & (gray <= 205)).astype(np.uint8) * 180
    if face_arr.mean() > 0.0:
        skin_luma = cv2.bitwise_and(luma_mid, face_arr)
    else:
        skin_luma = cv2.bitwise_and(luma_mid, fg_bin)
    skin_luma = cv2.morphologyEx(skin_luma, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin_luma = cv2.morphologyEx(skin_luma, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    skin_arr = np.maximum.reduce([skin_seed, skin_hsv, skin_luma])
    skin_arr = cv2.bitwise_and(skin_arr, fg_bin)
    skin_mask = Image.fromarray(np.clip(skin_arr, 0, 255).astype(np.uint8), mode="L")

    # Coarse clothes band below facial region.
    if faces:
        for _, (x, y, fw, fh) in faces:
            x1 = max(0, int(x - fw * 0.35))
            x2 = min(w, int(x + fw * 1.35))
            y1 = min(h, int(y + fh * 0.95))
            y2 = min(h, int(y + fh * 3.8))
            if y2 > y1:
                draw_clothes.rectangle((x1, y1, x2, y2), fill=170)
    else:
        if primary_bbox is not None:
            bx, by, bw, bh = primary_bbox
            x1 = max(0, int(bx - bw * 0.16))
            x2 = min(w, int(bx + bw * 1.16))
            y1 = min(h, int(by + bh * 0.54))
            y2 = min(h, int(by + bh * 1.18))
            if y2 > y1 and x2 > x1:
                draw_clothes.rectangle((x1, y1, x2, y2), fill=140)
        else:
            draw_clothes.rectangle((0, int(h * 0.56), w, h), fill=110)

    # Hair emphasis from dark textured pixels in upper half, bounded by foreground.
    yy, xx = np.indices(gray.shape)
    if primary_bbox is not None:
        bx, by, bw, bh = primary_bbox
        hair_roi = (
            (xx >= int(max(0, bx - 0.06 * bw)))
            & (xx <= int(min(w - 1, bx + 1.06 * bw)))
            & (yy >= int(max(0, by - 0.04 * bh)))
            & (yy <= int(min(h - 1, by + 0.58 * bh)))
        )
    else:
        hair_roi = yy < int(h * 0.60)
    dark_thr = int(np.clip(np.percentile(gray, 32), 44, 96))
    hair_candidate = ((gray <= dark_thr) & hair_roi & (fg_mask > 0)).astype(np.uint8) * 170
    # Keep hair from flooding skin center.
    skin_soft = cv2.GaussianBlur((skin_arr > 0).astype(np.uint8) * 255, (0, 0), sigmaX=1.6)
    hair_candidate = cv2.bitwise_and(hair_candidate, cv2.bitwise_not(skin_soft))
    hair_arr = np.maximum(np.array(hair_mask, dtype=np.uint8), hair_candidate)
    hair_mask = Image.fromarray(np.clip(hair_arr, 0, 255).astype(np.uint8), mode="L")

    # Ensure face fallback has non-zero coverage for stylized/sketch portraits.
    face_arr = np.array(face_mask, dtype=np.uint8)
    if face_arr.mean() < 2.0 and primary_bbox is not None:
        bx, by, bw, bh = primary_bbox
        fx1 = max(0, int(bx + bw * 0.20))
        fx2 = min(w, int(bx + bw * 0.82))
        fy1 = max(0, int(by + bh * 0.08))
        fy2 = min(h, int(by + bh * 0.62))
        if fx2 > fx1 and fy2 > fy1:
            fallback_face = Image.new("L", (w, h), 0)
            ImageDraw.Draw(fallback_face).ellipse((fx1, fy1, fx2, fy2), fill=210)
            face_arr = np.maximum(face_arr, np.array(fallback_face, dtype=np.uint8))
            face_mask = Image.fromarray(face_arr, mode="L")
            skin_arr = np.maximum(skin_arr, (face_arr * 0.82).astype(np.uint8))
            skin_mask = Image.fromarray(np.clip(skin_arr, 0, 255).astype(np.uint8), mode="L")

    # Background is inverse subject foreground to avoid palette spill.
    fg = np.maximum.reduce(
        [
            np.array(face_mask, dtype=np.uint8),
            np.array(hand_mask, dtype=np.uint8),
            np.array(skin_mask, dtype=np.uint8),
            np.array(hair_mask, dtype=np.uint8),
            np.array(clothes_mask, dtype=np.uint8),
        ]
    )
    fg = np.maximum(fg, (fg_mask > 0).astype(np.uint8) * 84)
    bg = np.where(fg_mask > 0, np.clip(255 - fg, 0, 255), 255).astype(np.uint8)
    background_mask = Image.fromarray(bg, mode="L")

    return {
        "face": face_mask,
        "eyes": eye_mask,
        "hands": hand_mask,
        "skin": skin_mask,
        "hair": hair_mask,
        "clothes": clothes_mask,
        "background": background_mask,
    }


def _blend_region_fidelity(
    styled: Image.Image,
    structure_pass: Image.Image,
    masks: Dict[str, Image.Image],
    weights: Optional[Dict[str, float]] = None,
) -> Image.Image:
    out = _to_rgb(styled).astype(np.float32)
    base = _to_rgb(structure_pass).astype(np.float32)

    default_weights = {
        "face": 0.58,
        "eyes": 0.78,
        "hands": 0.66,
        "skin": 0.52,
        "hair": 0.45,
        "clothes": 0.34,
        "background": 0.08,
    }
    active_weights = dict(default_weights)
    if weights:
        active_weights.update({k: float(v) for k, v in weights.items()})

    alpha = np.zeros((styled.size[1], styled.size[0]), dtype=np.float32)
    for key, weight in active_weights.items():
        mask = np.array(masks.get(key, Image.new("L", styled.size, 0))).astype(np.float32) / 255.0
        alpha += np.clip(weight, 0.0, 1.0) * mask

    alpha = np.clip(alpha, 0.0, 0.96)
    alpha = alpha[:, :, None]

    mixed = out * (1.0 - alpha) + base * alpha
    return _to_pil(mixed)


def _apply_palette_governance(
    image: Image.Image,
    structure_pass: Image.Image,
    anchors: List[str],
    strength: float = 0.28,
    max_shift: float = 26.0,
) -> tuple[Image.Image, dict]:
    if not anchors:
        return image, {
            "applied": False,
            "anchors": [],
            "strength": 0.0,
            "max_shift": 0.0,
        }

    arr = _to_rgb(image).astype(np.float32)
    ref = _to_rgb(structure_pass).astype(np.float32)

    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    rgb_anchors = [_hex_to_rgb(c) for c in anchors[:3]]
    while len(rgb_anchors) < 3:
        rgb_anchors.append(rgb_anchors[-1])

    hsv_anchors = [
        cv2.cvtColor(np.uint8([[list(c)]]), cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
        for c in rgb_anchors
    ]

    q1, q2 = np.quantile(v, [0.33, 0.66])
    bands = [v <= q1, (v > q1) & (v <= q2), v > q2]

    for idx, mask in enumerate(bands):
        if not np.any(mask):
            continue
        ah, asat, _ = hsv_anchors[idx]
        hsv[:, :, 0][mask] = hsv[:, :, 0][mask] * (1.0 - strength) + ah * strength
        hsv[:, :, 1][mask] = hsv[:, :, 1][mask] * (1.0 - strength) + asat * strength

    recolored = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # Preserve luminance hierarchy against structure pass.
    recolored_lab = cv2.cvtColor(recolored.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    recolored_lab[:, :, 0] = match_histograms(recolored_lab[:, :, 0], ref_lab[:, :, 0])

    # Clamp color drift in LAB a/b channels.
    ab_shift = recolored_lab[:, :, 1:3] - ref_lab[:, :, 1:3]
    ab_shift = np.clip(ab_shift, -max_shift, max_shift)
    recolored_lab[:, :, 1:3] = ref_lab[:, :, 1:3] + ab_shift

    governed = cv2.cvtColor(np.clip(recolored_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(governed), {
        "applied": True,
        "anchors": anchors[:3],
        "strength": float(strength),
        "max_shift": float(max_shift),
    }


def _apply_region_color_grade(
    image: Image.Image,
    masks: Dict[str, Image.Image],
    anchors: List[str],
    quality_mode: bool,
) -> tuple[Image.Image, dict]:
    arr = _to_rgb(image).astype(np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_before = float(hsv[:, :, 1].mean())

    if sat_before < 36.0:
        sat_gain = float(np.clip(1.0 + (36.0 - sat_before) / 70.0, 1.0, 1.35))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 255)

    region_anchors = list(anchors[:3]) if anchors else ["#efcfb9", "#4a423a", "#b19f8b"]
    while len(region_anchors) < 3:
        region_anchors.append(region_anchors[-1])

    def _anchor_hsv(color_hex: str) -> np.ndarray:
        rgb = np.uint8([[list(_hex_to_rgb(color_hex))]])
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]

    skin_hsv = _anchor_hsv(region_anchors[0])
    hair_hsv = _anchor_hsv(region_anchors[1])
    cloth_hsv = _anchor_hsv(region_anchors[2])

    def _mask(name: str, sigma: float = 1.2) -> np.ndarray:
        m = np.array(masks.get(name, Image.new("L", image.size, 0)), dtype=np.float32) / 255.0
        if sigma > 0.0:
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma)
        return np.clip(m, 0.0, 1.0)

    face = _mask("face")
    skin = np.clip(np.maximum(_mask("skin"), face * 0.6), 0.0, 1.0)
    hair = _mask("hair")
    clothes = _mask("clothes")
    bg = _mask("background")

    base_strength = 0.34 if quality_mode else 0.28
    if sat_before < 20.0:
        base_strength += 0.20
    elif sat_before < 40.0:
        base_strength += 0.10

    strengths = {
        "skin": float(np.clip(base_strength + 0.14, 0.18, 0.72)),
        "hair": float(np.clip(base_strength + 0.08, 0.14, 0.62)),
        "clothes": float(np.clip(base_strength + 0.04, 0.10, 0.56)),
        "background": float(np.clip(0.10 + 0.08 * (1.0 if quality_mode else 0.0), 0.08, 0.22)),
    }

    # When face/skin masks are weak (common on grayscale sketches), bias away from cool spill.
    face_cov = float(face.mean())
    skin_cov = float(skin.mean())
    low_skin_lock = bool(face_cov < 0.012 and skin_cov < 0.09)
    if low_skin_lock:
        strengths["skin"] = float(np.clip(strengths["skin"] + 0.20, 0.22, 0.78))
        strengths["hair"] = float(np.clip(strengths["hair"] * 0.55, 0.08, 0.36))
        strengths["clothes"] = float(np.clip(strengths["clothes"] * 0.62, 0.08, 0.40))
        strengths["background"] = float(np.clip(strengths["background"] * 0.35, 0.03, 0.12))
        # Expand skin slightly from face prior for stable complexion.
        skin = np.clip(np.maximum(skin, face * 0.78), 0.0, 1.0)

    def _blend(
        mask: np.ndarray,
        target: np.ndarray,
        strength: float,
        v_gain: float,
        v_lift: float,
        *,
        shift_hue: bool = True,
        sat_scale: float = 0.92,
        sat_bias: float = 18.0,
    ) -> None:
        alpha = np.clip(mask * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
        if shift_hue:
            hsv[:, :, 0] = hsv[:, :, 0] * (1.0 - alpha) + target[0] * alpha
        target_sat = float(np.clip(target[1] * sat_scale + sat_bias, 0.0, 255.0))
        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - alpha) + target_sat * alpha
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + alpha * (v_gain - 1.0)) + alpha * v_lift, 0.0, 255.0)

    _blend(skin, skin_hsv, strengths["skin"], v_gain=1.08, v_lift=8.0, sat_scale=0.68, sat_bias=14.0)
    # Keep hair/clothes mostly tonal in fallback path; avoid hue pollution on face shadows.
    _blend(
        hair,
        hair_hsv,
        strengths["hair"],
        v_gain=0.96,
        v_lift=0.0,
        shift_hue=False,
        sat_scale=0.12,
        sat_bias=2.0,
    )
    _blend(
        clothes,
        cloth_hsv,
        strengths["clothes"],
        v_gain=1.02,
        v_lift=3.0,
        shift_hue=False,
        sat_scale=0.12,
        sat_bias=3.0,
    )
    # Keep background mostly neutral: lower saturation without forcing hue.
    _blend(
        bg,
        cloth_hsv,
        strengths["background"],
        v_gain=1.0,
        v_lift=0.0,
        shift_hue=False,
        sat_scale=0.18,
        sat_bias=2.0,
    )

    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # Anti-cool cast for skin: if skin region skews blue/cyan, nudge to warmer complexion.
    cool_cast_lock = False
    skin_mass = float(skin.sum())
    if skin_mass > 1.0:
        s = skin[:, :, None]
        r_mean = float((out[:, :, 0:1] * s).sum() / (skin_mass + 1e-6))
        b_mean = float((out[:, :, 2:3] * s).sum() / (skin_mass + 1e-6))
        if b_mean > r_mean + 5.0:
            warm_alpha = np.clip((skin * 0.82 + face * 0.28), 0.0, 1.0)[:, :, None]
            out[:, :, 0:1] = np.clip(out[:, :, 0:1] + 16.0 * warm_alpha, 0.0, 255.0)
            out[:, :, 1:2] = np.clip(out[:, :, 1:2] + 4.0 * warm_alpha, 0.0, 255.0)
            out[:, :, 2:3] = np.clip(out[:, :, 2:3] - 15.0 * warm_alpha, 0.0, 255.0)
            cool_cast_lock = True

    out = np.clip(out, 0, 255).astype(np.uint8)
    sat_after = float(cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[:, :, 1].mean())
    return Image.fromarray(out), {
        "applied": True,
        "anchors": region_anchors,
        "sat_before": round(sat_before, 6),
        "sat_after": round(sat_after, 6),
        "strengths": {k: round(v, 6) for k, v in strengths.items()},
        "low_skin_lock": bool(low_skin_lock),
        "cool_cast_lock": bool(cool_cast_lock),
        "coverage": {
            "face": round(face_cov, 6),
            "skin": round(float(skin.mean()), 6),
            "hair": round(float(hair.mean()), 6),
            "clothes": round(float(clothes.mean()), 6),
            "background": round(float(bg.mean()), 6),
        },
    }


def _rebalance_exposure(
    image: Image.Image,
    reference: Image.Image,
    masks: Dict[str, Image.Image],
    quality_mode: bool,
) -> tuple[Image.Image, dict]:
    arr = _to_rgb(image)
    ref = _to_rgb(reference)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean_now = float(gray.mean())
    mean_ref = float(max(1.0, ref_gray.mean()))
    ratio_now = float(mean_now / mean_ref)
    shadow_now = float((gray <= 8.0).mean())
    hi_now = float((gray >= 247.0).mean())

    if 0.80 <= ratio_now <= 1.22 and shadow_now <= 0.42 and hi_now <= 0.20:
        return image, {
            "applied": False,
            "luma_ratio_before": round(ratio_now, 6),
            "luma_ratio_after": round(ratio_now, 6),
            "shadow_clip_before": round(shadow_now, 6),
            "shadow_clip_after": round(shadow_now, 6),
            "highlight_clip_before": round(hi_now, 6),
            "highlight_clip_after": round(hi_now, 6),
        }

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    lch = lab[:, :, 0]
    target_ratio = 0.92 if quality_mode else 0.88
    target_luma = float(np.clip(mean_ref * target_ratio, 28.0, 220.0))
    gain = float(np.clip(target_luma / max(1.0, mean_now), 0.80, 1.34))
    lch = np.clip(lch * gain, 0.0, 255.0)

    if ratio_now < 0.84 or shadow_now > 0.45:
        norm = np.clip(lch / 255.0, 0.0, 1.0)
        gamma = float(np.clip(0.88 - max(0.0, 0.84 - ratio_now) * 0.55, 0.62, 0.94))
        lch = np.clip((norm ** gamma) * 255.0, 0.0, 255.0)
        shadow_gate = np.clip((92.0 - gray) / 92.0, 0.0, 1.0)
        skin = np.array(masks.get("skin", Image.new("L", image.size, 0)), dtype=np.float32) / 255.0
        lift = 9.0 + 20.0 * max(0.0, 0.84 - ratio_now)
        lch = np.clip(lch + shadow_gate * (1.0 + skin * 0.55) * lift, 0.0, 255.0)

    if ratio_now > 1.26 or hi_now > 0.20:
        lch = np.clip(lch * 0.95, 0.0, 255.0)

    lab[:, :, 0] = np.clip(lch, 0.0, 255.0)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    out_gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ratio_after = float(out_gray.mean() / mean_ref)
    shadow_after = float((out_gray <= 8.0).mean())
    hi_after = float((out_gray >= 247.0).mean())
    return Image.fromarray(out), {
        "applied": True,
        "luma_ratio_before": round(ratio_now, 6),
        "luma_ratio_after": round(ratio_after, 6),
        "shadow_clip_before": round(shadow_now, 6),
        "shadow_clip_after": round(shadow_after, 6),
        "highlight_clip_before": round(hi_now, 6),
        "highlight_clip_after": round(hi_after, 6),
        "gain": round(gain, 6),
    }


def _overlay_lineart(
    stylized: Image.Image,
    original: Image.Image,
    alpha: float = 0.9,
    mode: str = "multiply",
) -> Image.Image:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = _to_rgb(stylized).astype(np.float32)
    original_rgb = _to_rgb(original)
    line = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Keep line reinforcement without crushing mid-tones.
    ink = np.clip(line, 0.0, 1.0)
    ink_soft = np.clip(0.22 + 0.78 * ink, 0.0, 1.0)

    if mode == "darken":
        ink_rgb = np.repeat((ink_soft[:, :, None] * 255.0), 3, axis=2)
        out = np.minimum(base, ink_rgb * alpha + base * (1.0 - alpha))
    else:
        mul = (1.0 - alpha) + alpha * ink_soft[:, :, None]
        out = base * mul

    # Hard edge reinforcement: inject original Canny edges back into final render.
    edges = cv2.Canny(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY), 70, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    emask = (edges > 0).astype(np.float32)[:, :, None]
    edge_dark = 0.24 + (1.0 - alpha) * 0.12
    out = out * (1.0 - emask * edge_dark)

    return _to_pil(out)


def _score_variant(quality: dict) -> float:
    ssim = float(quality.get("structure_ssim", 0.0))
    lp = float(quality.get("lpips", 1.0))
    edge_r = float(quality.get("edge_recall", quality.get("edge_preservation", 0.0)))
    edge_p = float(quality.get("edge_precision", edge_r))
    ident = float(quality.get("identity_similarity", 1.0))
    style = float(quality.get("style_adherence", 1.0))
    line_sharpness = float(quality.get("line_sharpness", 0.0))
    line_continuity = float(quality.get("line_continuity", 0.0))
    worm = float(quality.get("worm_artifact_score", 1.0))
    luma_ratio = float(quality.get("luma_ratio", 1.0))
    shadow_clip = float(quality.get("shadow_clip", 0.0))
    highlight_clip = float(quality.get("highlight_clip", 0.0))
    reasons = list(quality.get("reasons", []))
    passed = bool(quality.get("passed", False))
    base = 100.0 * (
        0.27 * ssim
        + 0.20 * edge_r
        + 0.12 * edge_p
        + 0.12 * max(0.0, 1.0 - lp)
        + 0.10 * ident
        + 0.07 * style
        + 0.07 * line_sharpness
        + 0.05 * line_continuity
    )
    base -= 100.0 * 0.06 * worm
    base -= 78.0 * max(0.0, 0.56 - edge_r)
    base -= 62.0 * max(0.0, 0.52 - edge_p)
    base -= 46.0 * max(0.0, 0.78 - luma_ratio)
    base -= 22.0 * max(0.0, shadow_clip - 0.36)
    base -= 14.0 * max(0.0, highlight_clip - 0.16)
    base -= 2.8 * float(len(reasons))
    if passed:
        base += 8.0
    return float(base)


def _control_images(
    image: Image.Image,
    render_quality: str,
    device: str,
    pose_reference: Optional[Image.Image] = None,
    strict_quality: bool = False,
) -> Dict[str, Image.Image]:
    canny = _canny_map(image)

    # Keep preview/balanced lightweight and deterministic on CPU.
    use_heavy_depth = bool(render_quality in {"final", "quality"} and str(device) == "cuda")
    use_pose = bool(render_quality in {"final", "quality"} and str(device) == "cuda")

    depth = _midas_depth_map(image, enabled=use_heavy_depth) if use_heavy_depth else None
    if depth is None:
        if strict_quality and use_heavy_depth:
            raise QualityPathError("quality_requires_controlnet_depth_map")
        depth = _pseudo_depth_map(image)
    pose_src = pose_reference if pose_reference is not None else image
    pose = _pose_map(pose_src, enabled=use_pose)
    if strict_quality and use_pose and pose is None:
        raise QualityPathError("quality_requires_controlnet_pose_map")
    return {
        "canny": canny,
        "depth": depth,
        "pose": pose,
    }


def _fast_fallback_img2img(
    image: Image.Image,
    *,
    seed: int,
    strength: float,
    guidance: float,
    phase: str,
) -> Image.Image:
    arr = _to_rgb(image).astype(np.float32)
    rng = np.random.default_rng(seed)

    if phase == "structure":
        # Conservative cleanup: suppress tiny texture noise while keeping line intent.
        sigma = float(np.clip(22.0 + strength * 28.0, 18.0, 36.0))
        base = arr.astype(np.uint8)
        smooth = cv2.bilateralFilter(base, d=5, sigmaColor=sigma, sigmaSpace=5).astype(np.float32)
        mixed = cv2.addWeighted(arr, 0.82, smooth, 0.18, 0.0)
        out = mixed
    else:
        # Mild stylization to keep outputs non-identical even without diffusion models.
        contrast = 1.0 + 0.06 * float(np.clip(guidance, 1.0, 14.0) / 14.0)
        out = (arr - 127.5) * contrast + 127.5
        chroma = np.clip(strength - 0.25, 0.0, 0.5)
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.18 * chroma), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
        luma = lab[:, :, 0].astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(1.4 + 1.8 * chroma), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(luma)
        sat_now = float(hsv[:, :, 1].mean())
        if sat_now < 14.0:
            yy = np.indices((lab.shape[0], lab.shape[1]))[0].astype(np.float32) / max(1.0, float(lab.shape[0] - 1))
            lab[:, :, 1] = np.clip(lab[:, :, 1] + (0.5 - yy) * 5.5, 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + (yy - 0.5) * 8.5, 0, 255)
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        grain = rng.normal(0.0, 1.2 + 3.2 * chroma, size=out.shape).astype(np.float32)
        out = out + grain

    return _to_pil(out)


def _run_img2img(
    model_manager: ModelManager,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    strength: float,
    guidance: float,
    steps: int,
    *,
    allow_diffusion: bool = True,
    phase: str = "style",
    quality_mode: bool = False,
    strict_diffusion: bool = False,
) -> tuple[Image.Image, str]:
    if not allow_diffusion:
        if quality_mode or strict_diffusion:
            err = "quality_diffusion_path_disabled" if quality_mode else "final_diffusion_path_disabled"
            raise QualityPathError(err)
        return _fast_fallback_img2img(
            image,
            seed=seed,
            strength=strength,
            guidance=guidance,
            phase=phase,
        ), "fallback_fast"

    pipe = model_manager.load_img2img_pipeline(quality_mode=quality_mode)
    if pipe is None:
        if quality_mode or strict_diffusion:
            err = "quality_img2img_pipeline_unavailable" if quality_mode else "final_img2img_pipeline_unavailable"
            raise QualityPathError(err)
        return _fast_fallback_img2img(
            image,
            seed=seed,
            strength=strength,
            guidance=guidance,
            phase=phase,
        ), "fallback_fast"

    generator = None
    if model_manager.torch is not None:
        generator = model_manager.torch.Generator(model_manager.device).manual_seed(seed)

    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=float(np.clip(strength, 0.1, 0.9)),
            guidance_scale=float(np.clip(guidance, 1.0, 20.0)),
            num_inference_steps=int(np.clip(steps, 6, 90)),
            generator=generator,
        ).images[0]
        return out.convert("RGB").resize(image.size, Image.Resampling.LANCZOS), "diffusion_img2img"
    except Exception as exc:
        if quality_mode or strict_diffusion:
            code = "quality_img2img_inference_failed" if quality_mode else "final_img2img_inference_failed"
            raise QualityPathError(f"{code}:{exc.__class__.__name__}") from exc
        return _fast_fallback_img2img(
            image,
            seed=seed,
            strength=strength,
            guidance=guidance,
            phase=phase,
        ), "fallback_fast"


def _run_controlnet_img2img(
    model_manager: ModelManager,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    strength: float,
    guidance: float,
    steps: int,
    controls: Dict[str, Image.Image],
    control_scales: Dict[str, float],
    lora_reference: Optional[str] = None,
    lora_scale: float = 0.8,
    ip_adapter_reference: Optional[Image.Image] = None,
    ip_adapter_scale: float = 0.75,
    *,
    allow_diffusion: bool = True,
    phase: str = "style",
    quality_mode: bool = False,
    strict_diffusion: bool = False,
) -> tuple[Image.Image, str]:
    if not allow_diffusion:
        if quality_mode or strict_diffusion:
            err = "quality_controlnet_requires_diffusion" if quality_mode else "final_controlnet_requires_diffusion"
            raise QualityPathError(err)
        return _run_img2img(
            model_manager=model_manager,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            guidance=guidance,
            steps=steps,
            allow_diffusion=False,
            phase=phase,
            quality_mode=quality_mode,
            strict_diffusion=strict_diffusion,
        )

    info = None
    if hasattr(model_manager, "load_controlnet_img2img_pipeline"):
        info = model_manager.load_controlnet_img2img_pipeline(quality_mode=quality_mode)

    if not info or not isinstance(info, dict) or info.get("pipe") is None:
        if quality_mode or strict_diffusion:
            err = (
                "quality_controlnet_pipeline_unavailable"
                if quality_mode
                else "final_controlnet_pipeline_unavailable"
            )
            raise QualityPathError(err)
        return _run_img2img(
            model_manager=model_manager,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            guidance=guidance,
            steps=steps,
            allow_diffusion=True,
            phase=phase,
            quality_mode=quality_mode,
            strict_diffusion=strict_diffusion,
        )

    pipe = info.get("pipe")
    modes: List[str] = list(info.get("modes", []))
    if quality_mode:
        required_modes = {"canny", "depth", "pose"}
        missing_modes = sorted(required_modes - set(modes))
        if missing_modes:
            raise QualityPathError("quality_controlnet_missing_modes:" + ",".join(missing_modes))

    ctrl_images: List[Image.Image] = []
    ctrl_scales: List[float] = []
    missing_control_inputs: List[str] = []
    for mode in modes:
        if mode not in controls:
            missing_control_inputs.append(mode)
            continue
        img = controls[mode]
        if img is None:
            missing_control_inputs.append(mode)
            continue
        ctrl_images.append(img)
        ctrl_scales.append(float(control_scales.get(mode, 0.5)))

    if quality_mode and missing_control_inputs:
        raise QualityPathError(
            "quality_control_images_missing:" + ",".join(sorted(set(missing_control_inputs)))
        )

    if not ctrl_images:
        if quality_mode or strict_diffusion:
            err = "quality_control_images_missing" if quality_mode else "final_control_images_missing"
            raise QualityPathError(err)
        return _run_img2img(
            model_manager=model_manager,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            guidance=guidance,
            steps=steps,
            allow_diffusion=True,
            phase=phase,
            quality_mode=quality_mode,
            strict_diffusion=strict_diffusion,
        )

    generator = None
    if model_manager.torch is not None:
        generator = model_manager.torch.Generator(model_manager.device).manual_seed(seed)

    backend = "controlnet_img2img"

    lora_loaded = False
    if lora_reference:
        try:
            lora_path = Path(lora_reference)
            if not lora_path.is_absolute():
                lora_path = model_manager.settings.loras_dir / lora_path
            if lora_path.exists() and hasattr(pipe, "load_lora_weights"):
                pipe.load_lora_weights(str(lora_path))
                lora_loaded = True
                backend = "controlnet_img2img+lora"
            elif quality_mode and model_manager.settings.quality_require_full_path:
                raise QualityPathError("quality_lora_reference_missing_or_unloadable")
        except QualityPathError:
            raise
        except Exception:
            if quality_mode and model_manager.settings.quality_require_full_path:
                raise QualityPathError("quality_lora_load_failed")
            lora_loaded = False
    elif quality_mode and model_manager.settings.quality_require_full_path:
        raise QualityPathError("quality_lora_reference_required")

    ip_adapter_loaded = False
    if ip_adapter_reference is not None:
        try:
            repo = (
                model_manager.settings.quality_ip_adapter_repo
                if quality_mode
                else model_manager.settings.ip_adapter_repo
            )
            subfolder = (
                model_manager.settings.quality_ip_adapter_subfolder
                if quality_mode
                else model_manager.settings.ip_adapter_subfolder
            )
            weight_name = (
                model_manager.settings.quality_ip_adapter_weight_name
                if quality_mode
                else model_manager.settings.ip_adapter_weight_name
            )
            marker = "_mangashift_ip_adapter_loaded_quality" if quality_mode else "_mangashift_ip_adapter_loaded"
            if hasattr(pipe, "load_ip_adapter") and not getattr(pipe, marker, False):
                pipe.load_ip_adapter(
                    repo,
                    subfolder=subfolder,
                    weight_name=weight_name,
                    local_files_only=model_manager.settings.local_models_only,
                )
                setattr(pipe, marker, True)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(float(np.clip(ip_adapter_scale, 0.1, 1.0)))
            ip_adapter_loaded = hasattr(pipe, "load_ip_adapter")
            if ip_adapter_loaded:
                backend = f"{backend}+ipadapter"
        except QualityPathError:
            raise
        except Exception:
            if quality_mode and model_manager.settings.quality_require_full_path:
                raise QualityPathError("quality_ip_adapter_load_failed")
            ip_adapter_loaded = False
    elif quality_mode and model_manager.settings.quality_require_full_path:
        raise QualityPathError("quality_ip_adapter_reference_required")

    if quality_mode and model_manager.settings.quality_require_full_path:
        if not lora_loaded:
            raise QualityPathError("quality_lora_weights_not_loaded")
        if not ip_adapter_loaded:
            raise QualityPathError("quality_ip_adapter_not_loaded")

    try:
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "control_image": ctrl_images[0] if len(ctrl_images) == 1 else ctrl_images,
            "controlnet_conditioning_scale": ctrl_scales[0] if len(ctrl_scales) == 1 else ctrl_scales,
            "strength": float(np.clip(strength, 0.1, 0.9)),
            "guidance_scale": float(np.clip(guidance, 1.0, 20.0)),
            "num_inference_steps": int(np.clip(steps, 6, 90)),
            "generator": generator,
        }
        if lora_loaded:
            kwargs["cross_attention_kwargs"] = {"scale": float(np.clip(lora_scale, 0.0, 1.4))}
        if ip_adapter_loaded:
            kwargs["ip_adapter_image"] = ip_adapter_reference

        out = pipe(**kwargs).images[0]
        out = out.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    except Exception as exc:
        if quality_mode or strict_diffusion:
            code = (
                "quality_controlnet_inference_failed"
                if quality_mode
                else "final_controlnet_inference_failed"
            )
            raise QualityPathError(f"{code}:{exc.__class__.__name__}") from exc
        out, backend = _run_img2img(
            model_manager=model_manager,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            guidance=guidance,
            steps=steps,
            allow_diffusion=True,
            phase=phase,
            quality_mode=quality_mode,
            strict_diffusion=strict_diffusion,
        )
    finally:
        if lora_loaded and hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

    return out, backend


def _run_sketch_pass(
    *,
    model_manager: ModelManager,
    image: Image.Image,
    seed: int,
    controls: Dict[str, Image.Image],
    run_params: VariantParams,
    allow_diffusion: bool,
) -> tuple[Image.Image, str]:
    sketch_prompt = (
        "artist construction sketch, confident gesture lines, coherent anatomy, "
        "dynamic pose draft, clean graphite strokes"
    )
    sketch_negative = (
        "stiff mannequin, rectangle body blocks, geometric placeholder, low-detail limbs, bad anatomy"
    )
    sketch_strength = float(np.clip(run_params.structure_strength - 0.04, 0.15, 0.3))
    sketch_guidance = float(np.clip(run_params.structure_guidance + 0.4, 7.6, 9.4))
    sketch_steps = int(np.clip(run_params.structure_steps + 4, 20, 36))

    return _run_controlnet_img2img(
        model_manager=model_manager,
        image=image,
        prompt=sketch_prompt,
        negative_prompt=sketch_negative,
        seed=seed,
        strength=sketch_strength,
        guidance=sketch_guidance,
        steps=sketch_steps,
        controls=controls,
        control_scales={
            "canny": float(np.clip(run_params.control_line + 0.08, 1.0, 1.35)),
            "depth": float(np.clip(run_params.control_depth + 0.05, 0.5, 0.85)),
            "pose": float(np.clip(run_params.control_pose + 0.04, 0.35, 0.6)),
        },
        allow_diffusion=allow_diffusion,
        phase="structure",
        quality_mode=True,
        strict_diffusion=True,
    )


def _variant_grid(count: int) -> List[VariantParams]:
    base = [
        VariantParams(0.20, 7.2, 22, 0.35, 9.0, 28, 1.25, 0.65, 0.45, 0.75, 0.92),
        VariantParams(0.22, 7.5, 24, 0.40, 9.6, 30, 1.20, 0.62, 0.45, 0.80, 0.90),
        VariantParams(0.25, 7.8, 26, 0.45, 10.2, 32, 1.18, 0.60, 0.42, 0.85, 0.90),
        VariantParams(0.28, 8.2, 28, 0.50, 10.8, 34, 1.15, 0.58, 0.40, 0.90, 0.89),
        VariantParams(0.30, 8.6, 30, 0.55, 11.4, 36, 1.12, 0.56, 0.38, 0.95, 0.88),
        VariantParams(0.33, 8.9, 30, 0.60, 12.0, 38, 1.10, 0.55, 0.35, 1.00, 0.87),
    ]
    count = int(np.clip(count, 1, 32))
    if count <= len(base):
        return base[:count]

    out = list(base)
    while len(out) < count:
        last = out[-1]
        out.append(
            VariantParams(
                structure_strength=float(np.clip(last.structure_strength + 0.01, 0.15, 0.35)),
                structure_guidance=float(np.clip(last.structure_guidance + 0.2, 7.0, 9.0)),
                structure_steps=int(np.clip(last.structure_steps + 1, 20, 32)),
                style_strength=float(np.clip(last.style_strength + 0.02, 0.35, 0.65)),
                style_guidance=float(np.clip(last.style_guidance + 0.2, 9.0, 12.5)),
                style_steps=int(np.clip(last.style_steps + 1, 25, 42)),
                control_line=float(np.clip(last.control_line, 1.0, 1.3)),
                control_depth=float(np.clip(last.control_depth, 0.45, 0.75)),
                control_pose=float(np.clip(last.control_pose, 0.3, 0.5)),
                lora_scale=float(np.clip(last.lora_scale, 0.7, 1.0)),
                line_overlay_alpha=float(np.clip(last.line_overlay_alpha, 0.85, 0.95)),
            )
        )
    return out


def render_two_pass_variants(
    *,
    original: Image.Image,
    image_bytes: bytes,
    style_name: str,
    render_quality: str,
    model_manager: ModelManager,
    character_db: CharacterDatabase,
    variant_count: int = 6,
    palette_anchors: Optional[List[str]] = None,
    style_reference: Optional[Image.Image] = None,
    character_reference: Optional[Image.Image] = None,
    pose_reference: Optional[Image.Image] = None,
    line_weight: float = 1.0,
    hatch_density: float = 0.52,
    tone_intensity: float = 0.42,
    screentone_strength: float = 0.16,
    top_k: int = 3,
    persist_variants: bool = True,
) -> dict:
    package = get_style_package(style_name, render_quality=render_quality, settings=model_manager.settings)
    quality_mode = render_quality == "quality"
    strict_diffusion_mode = render_quality in {"final", "quality"}
    strict_quality_full_path = bool(quality_mode and model_manager.settings.quality_require_full_path)
    if strict_diffusion_mode and model_manager.device != "cuda":
        err = "quality_requires_cuda_gpu" if quality_mode else "final_requires_cuda_gpu"
        raise QualityPathError(err)
    if strict_quality_full_path:
        if style_reference is None and character_reference is None:
            raise QualityPathError("quality_ip_adapter_reference_required")
        info = model_manager.load_controlnet_img2img_pipeline(quality_mode=True)
        if not info or not isinstance(info, dict) or info.get("pipe") is None:
            raise QualityPathError("quality_controlnet_pipeline_unavailable")
        modes = set(info.get("modes", []))
        missing_modes = sorted({"canny", "depth", "pose"} - modes)
        if missing_modes:
            raise QualityPathError("quality_controlnet_missing_modes:" + ",".join(missing_modes))

    controls = _control_images(
        original,
        render_quality=render_quality,
        device=model_manager.device,
        pose_reference=pose_reference,
        strict_quality=strict_quality_full_path,
    )
    masks = _face_eye_hand_masks(original, character_db=character_db)

    anchors = [c for c in (palette_anchors or []) if isinstance(c, str) and c.strip()]
    if not anchors:
        anchors = list(package.palette_anchor[:3])

    ink_tone_params = InkToneParams(
        line_weight=float(np.clip(line_weight, 0.5, 2.0)),
        hatch_density=float(np.clip(hatch_density, 0.1, 1.0)),
        tone_intensity=float(np.clip(tone_intensity, 0.0, 1.0)),
        screentone_strength=float(np.clip(screentone_strength, 0.0, 1.0)),
        hatch_angle_variation=0.24 if quality_mode else 0.30,
        hatch_overlap=0.42 if quality_mode else 0.34,
        detail_strength=0.30 if quality_mode else 0.20,
        worm_suppression=0.66 if quality_mode else 0.44,
    )

    thresholds = QualityThresholds(
        structure_ssim_min=0.88 if quality_mode else (0.86 if strict_diffusion_mode else 0.85),
        lpips_max=0.24 if quality_mode else (0.26 if strict_diffusion_mode else 0.28),
        edge_recall_min=0.70 if quality_mode else (0.62 if strict_diffusion_mode else 0.58),
        edge_precision_min=0.56 if quality_mode else (0.48 if strict_diffusion_mode else 0.42),
        line_sharpness_min=0.16 if quality_mode else (0.15 if strict_diffusion_mode else 0.14),
        line_continuity_min=0.50 if quality_mode else (0.48 if strict_diffusion_mode else 0.46),
        worm_artifact_max=0.24 if quality_mode else (0.27 if strict_diffusion_mode else 0.30),
        luma_ratio_min=0.76 if quality_mode else (0.73 if strict_diffusion_mode else 0.70),
        luma_ratio_max=1.22 if quality_mode else (1.26 if strict_diffusion_mode else 1.30),
        shadow_clip_max=0.36 if quality_mode else (0.40 if strict_diffusion_mode else 0.45),
        highlight_clip_max=0.15 if quality_mode else (0.19 if strict_diffusion_mode else 0.22),
        identity_similarity_min=(
            float(model_manager.settings.quality_identity_similarity_threshold)
            if quality_mode
            else float(model_manager.settings.identity_similarity_threshold)
        ),
        style_adherence_min=(
            float(model_manager.settings.quality_style_adherence_threshold)
            if quality_mode
            else (0.58 if strict_diffusion_mode else 0.54)
        ),
    )

    variants: List[VariantResult] = []
    params_grid = _variant_grid(variant_count)

    cpu_fast_mode = model_manager.device != "cuda" and render_quality in {"preview", "balanced"}
    allow_diffusion = not cpu_fast_mode
    if strict_diffusion_mode:
        allow_diffusion = model_manager.device == "cuda"

    if model_manager.device != "cuda" and render_quality in {"preview", "balanced"}:
        hard_cap = 2 if render_quality in {"final", "balanced"} else 3
        params_grid = params_grid[: max(1, min(len(params_grid), hard_cap))]

    quality_scale = 1.0
    if render_quality == "preview":
        quality_scale = 0.30 if cpu_fast_mode else 0.45
    elif render_quality == "balanced":
        quality_scale = 0.42 if cpu_fast_mode else 0.70
    elif render_quality == "quality":
        quality_scale = 1.0

    if quality_scale < 1.0:
        scaled: List[VariantParams] = []
        for p in params_grid:
            scaled.append(
                VariantParams(
                    structure_strength=p.structure_strength,
                    structure_guidance=p.structure_guidance,
                    structure_steps=max(8, int(round(p.structure_steps * quality_scale))),
                    style_strength=p.style_strength,
                    style_guidance=p.style_guidance,
                    style_steps=max(10, int(round(p.style_steps * quality_scale))),
                    control_line=p.control_line,
                    control_depth=p.control_depth,
                    control_pose=p.control_pose,
                    lora_scale=p.lora_scale,
                    line_overlay_alpha=p.line_overlay_alpha,
                )
            )
        params_grid = scaled

    max_attempts = 1 if not allow_diffusion else (3 if model_manager.device == "cuda" else 2)

    for idx, params in enumerate(params_grid, start=1):
        seed = _sha_seed(image_bytes, salt=f"{style_name}:{idx}:{render_quality}")
        retries = 0
        backend = "fallback"
        sketch_pass: Optional[Image.Image] = None
        structure_pass = original.copy()
        final_img = original.copy()
        quality = {
            "passed": False,
            "structure_ssim": 0.0,
            "lpips": 1.0,
            "edge_recall": 0.0,
            "edge_precision": 0.0,
            "edge_preservation": 0.0,
            "identity_similarity": 0.0,
            "style_adherence": 0.0,
            "reasons": ["not_run"],
        }

        run_params = params
        for attempt in range(max_attempts):
            retries = attempt

            structure_input = original
            backend_sketch = "sketch_skip"
            if quality_mode:
                sketch_pass, backend_sketch = _run_sketch_pass(
                    model_manager=model_manager,
                    image=original,
                    seed=seed + 3 + 17 * attempt,
                    controls=controls,
                    run_params=run_params,
                    allow_diffusion=allow_diffusion,
                )
                structure_input = sketch_pass

            structure_prompt = (
                "structure-preserving manga redraw, keep silhouette, keep contour/form/texture line hierarchy, "
                "preserve primary light direction"
            )
            structure_negative = "style drift, anatomy change, silhouette drift, over-smoothing"
            structure_pass, backend_structure = _run_controlnet_img2img(
                model_manager=model_manager,
                image=structure_input,
                prompt=structure_prompt,
                negative_prompt=structure_negative,
                seed=seed + 11 * attempt,
                strength=run_params.structure_strength,
                guidance=run_params.structure_guidance,
                steps=run_params.structure_steps,
                controls=controls,
                control_scales={
                    "canny": run_params.control_line,
                    "depth": run_params.control_depth,
                    "pose": run_params.control_pose,
                },
                allow_diffusion=allow_diffusion,
                phase="structure",
                quality_mode=quality_mode,
                strict_diffusion=strict_diffusion_mode,
            )

            ip_adapter_reference = style_reference or character_reference
            style_pass, backend_style = _run_controlnet_img2img(
                model_manager=model_manager,
                image=structure_pass,
                prompt=package.prompt_template,
                negative_prompt=package.negative_prompt,
                seed=seed + 101 + 37 * attempt,
                strength=run_params.style_strength,
                guidance=run_params.style_guidance,
                steps=run_params.style_steps,
                controls=controls,
                control_scales={
                    "canny": run_params.control_line * 0.92,
                    "depth": run_params.control_depth,
                    "pose": run_params.control_pose,
                },
                lora_reference=package.lora_reference,
                lora_scale=run_params.lora_scale,
                ip_adapter_reference=ip_adapter_reference,
                ip_adapter_scale=0.78 if quality_mode else 0.66,
                allow_diffusion=allow_diffusion,
                phase="style",
                quality_mode=quality_mode,
                strict_diffusion=strict_diffusion_mode,
            )
            backend = (
                f"{backend_sketch}->{backend_structure}->{backend_style}"
                if quality_mode
                else f"{backend_structure}->{backend_style}"
            )
            if strict_diffusion_mode and "fallback_fast" in backend:
                err = "quality_fallback_backend_rejected" if quality_mode else "final_fallback_backend_rejected"
                raise QualityPathError(err)

            region_guarded = _blend_region_fidelity(
                style_pass,
                structure_pass,
                masks=masks,
                weights=(
                    {
                        "face": 0.72,
                        "eyes": 0.92,
                        "hands": 0.78,
                        "skin": 0.66,
                        "hair": 0.58,
                        "clothes": 0.48,
                        "background": 0.06,
                    }
                    if quality_mode
                    else {
                        "face": 0.56,
                        "eyes": 0.78,
                        "hands": 0.62,
                        "skin": 0.48,
                        "hair": 0.42,
                        "clothes": 0.30,
                        "background": 0.08,
                    }
                ),
            )
            if quality_mode:
                # Run a second strict region lock for face/eyes/hands in quality mode.
                region_guarded = _blend_region_fidelity(
                    region_guarded,
                    structure_pass,
                    masks=masks,
                    weights={
                        "face": 0.78,
                        "eyes": 0.95,
                        "hands": 0.82,
                        "skin": 0.70,
                        "hair": 0.62,
                        "clothes": 0.52,
                        "background": 0.04,
                    },
                )

            fallback_backend = "fallback_fast" in backend
            palette_strength = 0.22 if quality_mode else 0.28
            palette_shift = 18.0 if quality_mode else 26.0
            if fallback_backend and not quality_mode:
                palette_strength = 0.10
                palette_shift = 12.0

            governed, palette_report = _apply_palette_governance(
                region_guarded,
                structure_pass,
                anchors=anchors,
                strength=palette_strength,
                max_shift=palette_shift,
            )
            region_colored, region_color_report = _apply_region_color_grade(
                governed,
                masks=masks,
                anchors=anchors,
                quality_mode=quality_mode,
            )
            ink_tone_image, ink_tone_report = refine_ink_tone_pass(
                original=original,
                structure_pass=structure_pass,
                styled=region_colored,
                params=ink_tone_params,
            )
            line_mode = "multiply" if package.preserve_line else "darken"
            line_alpha = float(np.clip(run_params.line_overlay_alpha, 0.80, 0.92 if quality_mode else 0.90))
            line_img = _overlay_lineart(
                ink_tone_image,
                original,
                alpha=line_alpha,
                mode=line_mode,
            )
            final_img, exposure_report = _rebalance_exposure(
                line_img,
                reference=structure_pass,
                masks=masks,
                quality_mode=quality_mode,
            )

            quality_report = evaluate_quality_gate(
                original=original,
                structure_pass=structure_pass,
                rendered=final_img,
                thresholds=thresholds,
                use_lpips_model=bool(strict_diffusion_mode and model_manager.device == "cuda"),
                style_reference=style_reference,
                identity_reference=character_reference if character_reference is not None else original,
                character_db=character_db,
                fail_on_missing_faces=bool(strict_diffusion_mode and character_reference is not None),
            )
            quality = quality_report.to_dict()
            quality["palette"] = palette_report
            quality["line_overlay"] = {
                "mode": line_mode,
                "alpha": float(line_alpha),
            }
            quality["region_color"] = dict(region_color_report)
            quality["exposure_rebalance"] = dict(exposure_report)
            quality["ink_tone"] = dict(ink_tone_report)
            quality["worm_artifact_score"] = float(ink_tone_report.get("worm_artifact_score", 0.0))
            quality["sketch_first"] = {
                "enabled": bool(quality_mode),
                "backend": backend_sketch,
            }

            if quality_report.passed:
                break

            # Hard-gate retry: reduce denoise and increase structure lock.
            run_params = VariantParams(
                structure_strength=float(np.clip(run_params.structure_strength - 0.03, 0.15, 0.35)),
                structure_guidance=float(np.clip(run_params.structure_guidance + 0.2, 7.0, 9.0)),
                structure_steps=int(np.clip(run_params.structure_steps + 2, 20, 34)),
                style_strength=float(np.clip(run_params.style_strength - 0.06, 0.35, 0.65)),
                style_guidance=float(np.clip(run_params.style_guidance + 0.25, 9.0, 12.5)),
                style_steps=int(np.clip(run_params.style_steps + 2, 25, 44)),
                control_line=float(np.clip(run_params.control_line + 0.08, 1.0, 1.35)),
                control_depth=float(np.clip(run_params.control_depth + 0.03, 0.5, 0.8)),
                control_pose=float(np.clip(run_params.control_pose + 0.02, 0.35, 0.55)),
                lora_scale=float(np.clip(run_params.lora_scale, 0.7, 1.0)),
                line_overlay_alpha=float(np.clip(run_params.line_overlay_alpha + 0.01, 0.80, 0.93)),
            )

        variants.append(
            VariantResult(
                index=idx,
                seed=seed,
                params=run_params,
                retries=retries,
                backend=backend,
                sketch_pass=sketch_pass,
                structure_pass=structure_pass,
                final=final_img,
                quality=quality,
                output_score=_score_variant(quality),
            )
        )

    # Rank with hard quality-gate precedence.
    ranked = sorted(
        variants,
        key=lambda v: (
            bool(v.quality.get("passed", False)),
            float(v.output_score),
        ),
        reverse=True,
    )
    best = ranked[0]
    top_k = int(np.clip(top_k, 1, max(1, len(ranked))))
    top_selected = ranked[:top_k]

    run_id = hashlib.sha256(
        image_bytes + style_name.encode("utf-8") + str(time.time()).encode("utf-8")
    ).hexdigest()[:16]
    run_dir = model_manager.settings.cache_dir / "variant_runs" / run_id

    variant_payload = []
    if persist_variants:
        run_dir.mkdir(parents=True, exist_ok=True)
        original.save(run_dir / "original.png")

    for item in ranked:
        entry = {
            "index": int(item.index),
            "seed": int(item.seed),
            "retries": int(item.retries),
            "backend": item.backend,
            "params": {
                "structure_strength": float(item.params.structure_strength),
                "structure_guidance": float(item.params.structure_guidance),
                "structure_steps": int(item.params.structure_steps),
                "style_strength": float(item.params.style_strength),
                "style_guidance": float(item.params.style_guidance),
                "style_steps": int(item.params.style_steps),
                "control_line": float(item.params.control_line),
                "control_depth": float(item.params.control_depth),
                "control_pose": float(item.params.control_pose),
                "lora_scale": float(item.params.lora_scale),
                "line_overlay_alpha": float(item.params.line_overlay_alpha),
            },
            "quality": item.quality,
            "score": float(item.output_score),
            "paths": {},
        }
        if persist_variants:
            structure_path = run_dir / f"variant_{item.index:02d}_structure.png"
            final_path = run_dir / f"variant_{item.index:02d}_final.png"
            item.structure_pass.save(structure_path)
            item.final.save(final_path)
            entry["paths"] = {
                "structure": str(structure_path),
                "final": str(final_path),
            }
            if item.sketch_pass is not None:
                sketch_path = run_dir / f"variant_{item.index:02d}_sketch.png"
                item.sketch_pass.save(sketch_path)
                entry["paths"]["sketch"] = str(sketch_path)
            if any(t.index == item.index for t in top_selected):
                rank_idx = [t.index for t in top_selected].index(item.index) + 1
                entry["paths"]["top_rank"] = int(rank_idx)
        variant_payload.append(entry)

    metadata = {
        "run_id": run_id,
        "style_name": style_name,
        "render_quality": render_quality,
        "quality_mode": bool(quality_mode),
        "palette_anchors": anchors,
        "variant_count": len(variant_payload),
        "ink_tone_params": {
            "line_weight": float(ink_tone_params.line_weight),
            "hatch_density": float(ink_tone_params.hatch_density),
            "hatch_angle_variation": float(ink_tone_params.hatch_angle_variation),
            "hatch_overlap": float(ink_tone_params.hatch_overlap),
            "tone_intensity": float(ink_tone_params.tone_intensity),
            "screentone_strength": float(ink_tone_params.screentone_strength),
            "detail_strength": float(ink_tone_params.detail_strength),
            "worm_suppression": float(ink_tone_params.worm_suppression),
        },
        "variants": variant_payload,
        "best_index": int(best.index),
        "best_seed": int(best.seed),
        "best_score": float(best.output_score),
        "top_k": int(top_k),
        "top_indices": [int(v.index) for v in top_selected],
        "top_scores": [float(v.output_score) for v in top_selected],
    }

    if persist_variants:
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "best_image": best.final,
        "best_structure": best.structure_pass,
        "best_variant": {
            "index": int(best.index),
            "seed": int(best.seed),
            "params": {
                "structure_strength": float(best.params.structure_strength),
                "structure_guidance": float(best.params.structure_guidance),
                "structure_steps": int(best.params.structure_steps),
                "style_strength": float(best.params.style_strength),
                "style_guidance": float(best.params.style_guidance),
                "style_steps": int(best.params.style_steps),
                "control_line": float(best.params.control_line),
                "control_depth": float(best.params.control_depth),
                "control_pose": float(best.params.control_pose),
                "lora_scale": float(best.params.lora_scale),
                "line_overlay_alpha": float(best.params.line_overlay_alpha),
            },
            "backend": best.backend,
            "quality": best.quality,
            "score": float(best.output_score),
        },
        "top_variants": [
            {
                "index": int(v.index),
                "seed": int(v.seed),
                "score": float(v.output_score),
                "quality": dict(v.quality),
            }
            for v in top_selected
        ],
        "metadata": metadata,
        "run_dir": str(run_dir) if persist_variants else "",
    }
