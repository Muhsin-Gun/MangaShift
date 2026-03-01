from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def _to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


@dataclass
class InkToneParams:
    line_weight: float = 1.0
    hatch_density: float = 0.52
    hatch_angle_variation: float = 0.28
    hatch_overlap: float = 0.35
    tone_intensity: float = 0.42
    screentone_strength: float = 0.16
    detail_strength: float = 0.20
    worm_suppression: float = 0.45


def _normal_field(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx = gx / mag
    ny = gy / mag
    return nx, ny


def _shadow_map(structure_pass: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(_to_rgb(structure_pass), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return np.clip(1.0 - gray, 0.0, 1.0)


def _screentone_pattern(shape: tuple[int, int], strength: float) -> np.ndarray:
    h, w = shape
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    y, x = np.indices((h, w), dtype=np.float32)
    spacing = 7.0 - 3.0 * strength
    dots = (np.mod(x + y, spacing) < (1.0 + 1.4 * strength)).astype(np.float32)
    dots = cv2.GaussianBlur(dots, (0, 0), sigmaX=0.55 + strength * 0.35)
    return np.clip(dots, 0.0, 1.0) * strength


def _hatch_map(
    structure_pass: Image.Image,
    *,
    density: float,
    angle_variation: float,
    overlap: float,
    tone_intensity: float,
) -> np.ndarray:
    density = float(np.clip(density, 0.1, 1.0))
    angle_variation = float(np.clip(angle_variation, 0.0, 1.0))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    tone_intensity = float(np.clip(tone_intensity, 0.0, 1.0))

    shadow = _shadow_map(structure_pass)
    nx, ny = _normal_field(structure_pass)

    h, w = shadow.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    theta = np.arctan2(ny, nx) + np.pi * 0.5
    theta += angle_variation * 0.32 * np.sin((xx + yy) * 0.012)

    spacing = 18.0 - density * 12.0
    freq = (2.0 * np.pi) / max(2.0, spacing)
    proj_a = xx * np.cos(theta) + yy * np.sin(theta)
    line_a = (np.sin(proj_a * freq) > (0.62 + 0.18 * (1.0 - density))).astype(np.float32)

    # Cross hatching in deep shadows only.
    proj_b = xx * np.cos(theta + np.pi * 0.5) + yy * np.sin(theta + np.pi * 0.5)
    line_b = (np.sin(proj_b * freq) > (0.70 + 0.14 * (1.0 - density))).astype(np.float32)
    cross_gate = (shadow > (0.48 - 0.22 * overlap)).astype(np.float32)
    hatch = line_a + line_b * cross_gate * overlap

    hatch = cv2.GaussianBlur(hatch, (0, 0), sigmaX=0.42)
    hatch = np.clip(hatch, 0.0, 1.0)
    return hatch * shadow * tone_intensity


def _worm_artifact_score(edge_map: np.ndarray) -> float:
    edge_bin = (edge_map > 0).astype(np.uint8)
    total = float(edge_bin.sum())
    if total <= 1e-9:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)
    if num_labels <= 1:
        return 0.0
    small = 0.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 16.0:
            small += area
    return float(np.clip(small / total, 0.0, 1.0))


def _suppress_worms(
    image: np.ndarray,
    *,
    structure_pass: Image.Image,
    strength: float,
) -> Tuple[np.ndarray, float]:
    strength = float(np.clip(strength, 0.0, 1.0))
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    artifact = _worm_artifact_score(edge)
    if artifact <= 0.03 or strength <= 1e-6:
        return image, artifact

    edge_bin = (edge > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)
    if num_labels <= 1:
        return image, artifact

    tiny_mask = np.zeros_like(edge_bin, dtype=np.uint8)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 18:
            tiny_mask[labels == idx] = 1
    if tiny_mask.sum() <= 0:
        return image, artifact

    # Replace tiny disconnected worm-like segments with guided structure content.
    struct_rgb = _to_rgb(structure_pass).astype(np.float32)
    smooth = cv2.bilateralFilter(image.astype(np.uint8), d=5, sigmaColor=18, sigmaSpace=3).astype(np.float32)
    out = image.astype(np.float32)
    mask = cv2.dilate(tiny_mask, np.ones((2, 2), np.uint8), iterations=1).astype(np.float32)[:, :, None]
    mix = np.clip(0.38 + artifact * 0.6, 0.0, 0.92) * strength
    out = out * (1.0 - mask * mix) + (0.55 * smooth + 0.45 * struct_rgb) * (mask * mix)
    return np.clip(out, 0, 255), artifact


def _apply_line_weight(
    image: np.ndarray,
    *,
    original: Image.Image,
    line_weight: float,
) -> np.ndarray:
    line_weight = float(np.clip(line_weight, 0.5, 2.0))
    gray = cv2.cvtColor(_to_rgb(original), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 70, 155)
    k = int(np.clip(round(1 + (line_weight - 0.6) * 2.2), 1, 4))
    kernel = np.ones((k, k), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge_mask = (edge > 0).astype(np.float32)[:, :, None]
    dark = np.clip(0.18 + 0.24 * (line_weight - 0.5), 0.12, 0.42)
    out = image.astype(np.float32) * (1.0 - edge_mask * dark)
    return np.clip(out, 0, 255)


def refine_ink_tone_pass(
    *,
    original: Image.Image,
    structure_pass: Image.Image,
    styled: Image.Image,
    params: InkToneParams,
) -> tuple[Image.Image, Dict[str, float]]:
    base = _to_rgb(styled).astype(np.float32)
    hatch = _hatch_map(
        structure_pass,
        density=params.hatch_density,
        angle_variation=params.hatch_angle_variation,
        overlap=params.hatch_overlap,
        tone_intensity=params.tone_intensity,
    )
    shadow = _shadow_map(structure_pass)
    screentone = _screentone_pattern(shadow.shape, strength=params.screentone_strength)
    tone_layer = np.clip(hatch * 0.72 + screentone * shadow * 0.48, 0.0, 1.0)

    # Apply tone as multiplicative darkening, preserving highlights.
    darken = np.clip(tone_layer[:, :, None], 0.0, 0.75)
    toned = base * (1.0 - darken)

    # Detail recovery for skin/fabric micro-contrast without crushing lines.
    detail = float(np.clip(params.detail_strength, 0.0, 1.0))
    if detail > 1e-6:
        blur = cv2.GaussianBlur(toned, (0, 0), sigmaX=1.05)
        high = toned - blur
        toned = toned + high * (0.22 + detail * 0.35)

    weighted = _apply_line_weight(
        toned,
        original=original,
        line_weight=params.line_weight,
    )
    denoised, worm_score = _suppress_worms(
        weighted,
        structure_pass=structure_pass,
        strength=params.worm_suppression,
    )
    out = np.clip(denoised, 0, 255)

    report = {
        "line_weight": float(params.line_weight),
        "hatch_density": float(params.hatch_density),
        "hatch_angle_variation": float(params.hatch_angle_variation),
        "hatch_overlap": float(params.hatch_overlap),
        "tone_intensity": float(params.tone_intensity),
        "screentone_strength": float(params.screentone_strength),
        "detail_strength": float(params.detail_strength),
        "worm_suppression": float(params.worm_suppression),
        "worm_artifact_score": float(round(worm_score, 6)),
        "tone_coverage": float(round(float((tone_layer > 0.05).mean()), 6)),
        "tone_mean": float(round(float(tone_layer.mean()), 6)),
    }
    return _to_pil(out), report
