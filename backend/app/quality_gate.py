from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass
class QualityThresholds:
    structure_ssim_min: float = 0.85
    lpips_max: float = 0.28
    edge_recall_min: float = 0.90
    edge_precision_min: float = 0.72
    line_sharpness_min: float = 0.14
    line_continuity_min: float = 0.46
    worm_artifact_max: float = 0.30
    identity_similarity_min: float = 0.78
    style_adherence_min: float = 0.64
    luma_ratio_min: float = 0.72
    luma_ratio_max: float = 1.25
    shadow_clip_max: float = 0.40
    highlight_clip_max: float = 0.18


@dataclass
class QualityReport:
    structure_ssim: float
    lpips: float
    edge_recall: float
    edge_precision: float
    line_sharpness: float
    line_continuity: float
    worm_artifact_score: float
    luma_ratio: float
    shadow_clip: float
    highlight_clip: float
    identity_similarity: float
    style_adherence: float
    style_reference_used: bool
    identity_faces_compared: int
    passed: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, float | bool | List[str] | int]:
        edge_preservation = float(self.edge_recall)
        return {
            "structure_ssim": round(float(self.structure_ssim), 6),
            "lpips": round(float(self.lpips), 6),
            "edge_recall": round(float(self.edge_recall), 6),
            "edge_precision": round(float(self.edge_precision), 6),
            "edge_preservation": round(edge_preservation, 6),
            "line_sharpness": round(float(self.line_sharpness), 6),
            "line_continuity": round(float(self.line_continuity), 6),
            "worm_artifact_score": round(float(self.worm_artifact_score), 6),
            "luma_ratio": round(float(self.luma_ratio), 6),
            "shadow_clip": round(float(self.shadow_clip), 6),
            "highlight_clip": round(float(self.highlight_clip), 6),
            "identity_similarity": round(float(self.identity_similarity), 6),
            "style_adherence": round(float(self.style_adherence), 6),
            "style_reference_used": bool(self.style_reference_used),
            "identity_faces_compared": int(self.identity_faces_compared),
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
        }


def _to_rgb_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _to_gray_float(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(_to_rgb_uint8(image), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _canny_binary(image: Image.Image, low: int = 80, high: int = 160) -> np.ndarray:
    gray = cv2.cvtColor(_to_rgb_uint8(image), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, low, high)
    return (edge > 0).astype(np.uint8)


@lru_cache(maxsize=1)
def _lpips_model():
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex")
        model = model.to("cpu").eval()
        return model, torch
    except Exception:
        return None, None


def _lpips_proxy(original: Image.Image, rendered: Image.Image) -> float:
    o = cv2.resize(_to_rgb_uint8(original), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = cv2.resize(_to_rgb_uint8(rendered), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    levels = [1.0, 0.5, 0.25]
    score = 0.0
    wsum = 0.0
    for idx, scale in enumerate(levels):
        if scale != 1.0:
            size = (max(32, int(256 * scale)), max(32, int(256 * scale)))
            oo = cv2.resize(o, size, interpolation=cv2.INTER_AREA)
            rr = cv2.resize(r, size, interpolation=cv2.INTER_AREA)
        else:
            oo = o
            rr = r
        do = cv2.Laplacian(oo, cv2.CV_32F)
        dr = cv2.Laplacian(rr, cv2.CV_32F)
        w = 1.0 / float(idx + 1)
        score += w * float(np.mean(np.abs(do - dr)))
        wsum += w
    return float(score / max(wsum, 1e-9))


def compute_lpips(original: Image.Image, rendered: Image.Image, use_model: bool = False) -> float:
    if not use_model:
        return _lpips_proxy(original, rendered)
    model, torch = _lpips_model()
    if model is None or torch is None:
        return _lpips_proxy(original, rendered)

    o = cv2.resize(_to_rgb_uint8(original), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = cv2.resize(_to_rgb_uint8(rendered), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    o = (o / 127.5) - 1.0
    r = (r / 127.5) - 1.0
    t1 = torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0)
    t2 = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        val = model(t1, t2).item()
    return float(val)


def compute_structure_ssim(original: Image.Image, structure_pass: Image.Image) -> float:
    o = _to_gray_float(original)
    s = _to_gray_float(structure_pass)
    return float(structural_similarity(o, s, data_range=1.0))


def compute_edge_metrics(original: Image.Image, rendered: Image.Image) -> tuple[float, float]:
    e1 = _canny_binary(original)
    e2 = _canny_binary(rendered)
    kernel = np.ones((3, 3), np.uint8)
    e1_d = cv2.dilate(e1, kernel, iterations=1)
    e2_d = cv2.dilate(e2, kernel, iterations=1)

    total_original = float(e1.sum())
    total_rendered = float(e2.sum())
    overlap_recall = float(np.logical_and(e1 > 0, e2_d > 0).sum())
    overlap_precision = float(np.logical_and(e2 > 0, e1_d > 0).sum())

    recall = 1.0 if total_original <= 1e-9 else float(overlap_recall / total_original)
    precision = 1.0 if total_rendered <= 1e-9 else float(overlap_precision / total_rendered)
    return recall, precision


def compute_line_sharpness(rendered: Image.Image) -> float:
    gray = cv2.cvtColor(_to_rgb_uint8(rendered), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edge_mask = (edges > 0).astype(np.float32)
    if float(edge_mask.sum()) <= 1e-9:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) / 255.0
    val = float((mag * edge_mask).sum() / (edge_mask.sum() + 1e-9))
    return float(np.clip(val, 0.0, 1.0))


def compute_line_continuity(rendered: Image.Image) -> float:
    gray = cv2.cvtColor(_to_rgb_uint8(rendered), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edge_bin = (edges > 0).astype(np.uint8)
    total = int(edge_bin.sum())
    if total <= 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)
    if num_labels <= 1:
        return 1.0
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    tiny = sum(a for a in areas if a <= 16)
    continuity = 1.0 - tiny / float(total + 1e-9)
    return float(np.clip(continuity, 0.0, 1.0))


def compute_worm_artifact_score(rendered: Image.Image) -> float:
    gray = cv2.cvtColor(_to_rgb_uint8(rendered), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edge_bin = (edges > 0).astype(np.uint8)
    total = int(edge_bin.sum())
    if total <= 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)
    if num_labels <= 1:
        return 0.0
    small = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 16:
            small += area
    return float(np.clip(small / float(total + 1e-9), 0.0, 1.0))


def compute_exposure_metrics(original: Image.Image, rendered: Image.Image) -> tuple[float, float, float]:
    og = cv2.cvtColor(_to_rgb_uint8(original), cv2.COLOR_RGB2GRAY).astype(np.float32)
    rg = cv2.cvtColor(_to_rgb_uint8(rendered), cv2.COLOR_RGB2GRAY).astype(np.float32)
    o_mean = float(max(1.0, og.mean()))
    r_mean = float(rg.mean())
    luma_ratio = float(np.clip(r_mean / o_mean, 0.0, 3.0))
    shadow_clip = float((rg <= 8.0).mean())
    highlight_clip = float((rg >= 247.0).mean())
    return luma_ratio, shadow_clip, highlight_clip


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa <= 1e-9 or bb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (aa * bb))


def _visual_embedding(im: Image.Image) -> np.ndarray:
    arr = np.array(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256]).flatten()
    v = np.concatenate([hist_h, hist_s, edge_hist]).astype(np.float32)
    v /= max(1e-9, float(np.linalg.norm(v)))
    return v


def compute_style_adherence(style_reference: Optional[Image.Image], rendered: Image.Image) -> tuple[float, bool]:
    if style_reference is None:
        return 1.0, False
    ea = _visual_embedding(style_reference)
    eb = _visual_embedding(rendered)
    return max(0.0, min(1.0, _cosine(ea, eb))), True


def compute_identity_similarity(
    reference: Image.Image,
    rendered: Image.Image,
    character_db: Optional[object] = None,
) -> tuple[float, int]:
    if character_db is None:
        return 1.0, 0
    try:
        ref_faces = character_db.detect_faces(reference)
        dst_faces = character_db.detect_faces(rendered)
    except Exception:
        return 1.0, 0
    if not ref_faces or not dst_faces:
        return 1.0, 0
    pairs = min(len(ref_faces), len(dst_faces))
    if pairs <= 0:
        return 1.0, 0
    sims: list[float] = []
    for i in range(pairs):
        try:
            ref_emb = character_db.embed_face(ref_faces[i][0])
            dst_emb = character_db.embed_face(dst_faces[i][0])
            ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-9)
            dst_emb = dst_emb / (np.linalg.norm(dst_emb) + 1e-9)
            sims.append(float(np.dot(ref_emb, dst_emb)))
        except Exception:
            continue
    if not sims:
        return 1.0, 0
    return float(np.mean(sims)), len(sims)


def evaluate_quality_gate(
    original: Image.Image,
    structure_pass: Image.Image,
    rendered: Image.Image,
    thresholds: QualityThresholds | None = None,
    use_lpips_model: bool = False,
    style_reference: Optional[Image.Image] = None,
    identity_reference: Optional[Image.Image] = None,
    character_db: Optional[object] = None,
    fail_on_missing_faces: bool = False,
) -> QualityReport:
    cfg = thresholds or QualityThresholds()
    structure_ssim = compute_structure_ssim(original, structure_pass)
    lpips_score = compute_lpips(original, rendered, use_model=use_lpips_model)
    edge_recall, edge_precision = compute_edge_metrics(original, rendered)
    line_sharpness = compute_line_sharpness(rendered)
    line_continuity = compute_line_continuity(rendered)
    worm_artifact_score = compute_worm_artifact_score(rendered)
    luma_ratio, shadow_clip, highlight_clip = compute_exposure_metrics(original, rendered)
    style_adherence, has_style_ref = compute_style_adherence(style_reference, rendered)
    identity_score, identity_faces = compute_identity_similarity(
        reference=identity_reference or original,
        rendered=rendered,
        character_db=character_db,
    )

    reasons: List[str] = []
    if structure_ssim < cfg.structure_ssim_min:
        reasons.append(f"structure_ssim<{cfg.structure_ssim_min:.2f} ({structure_ssim:.4f})")
    if lpips_score > cfg.lpips_max:
        reasons.append(f"lpips>{cfg.lpips_max:.2f} ({lpips_score:.4f})")
    if edge_recall < cfg.edge_recall_min:
        reasons.append(f"edge_recall<{cfg.edge_recall_min:.2f} ({edge_recall:.4f})")
    if edge_precision < cfg.edge_precision_min:
        reasons.append(f"edge_precision<{cfg.edge_precision_min:.2f} ({edge_precision:.4f})")
    if line_sharpness < cfg.line_sharpness_min:
        reasons.append(f"line_sharpness<{cfg.line_sharpness_min:.2f} ({line_sharpness:.4f})")
    if line_continuity < cfg.line_continuity_min:
        reasons.append(f"line_continuity<{cfg.line_continuity_min:.2f} ({line_continuity:.4f})")
    if worm_artifact_score > cfg.worm_artifact_max:
        reasons.append(f"worm_artifact>{cfg.worm_artifact_max:.2f} ({worm_artifact_score:.4f})")
    if luma_ratio < cfg.luma_ratio_min:
        reasons.append(f"luma_ratio<{cfg.luma_ratio_min:.2f} ({luma_ratio:.4f})")
    if luma_ratio > cfg.luma_ratio_max:
        reasons.append(f"luma_ratio>{cfg.luma_ratio_max:.2f} ({luma_ratio:.4f})")
    if shadow_clip > cfg.shadow_clip_max:
        reasons.append(f"shadow_clip>{cfg.shadow_clip_max:.2f} ({shadow_clip:.4f})")
    if highlight_clip > cfg.highlight_clip_max:
        reasons.append(f"highlight_clip>{cfg.highlight_clip_max:.2f} ({highlight_clip:.4f})")
    if has_style_ref and style_adherence < cfg.style_adherence_min:
        reasons.append(f"style_adherence<{cfg.style_adherence_min:.2f} ({style_adherence:.4f})")
    if fail_on_missing_faces and identity_faces <= 0:
        reasons.append("identity_faces_missing")
    if identity_faces > 0 and identity_score < cfg.identity_similarity_min:
        reasons.append(f"identity_similarity<{cfg.identity_similarity_min:.2f} ({identity_score:.4f})")

    return QualityReport(
        structure_ssim=structure_ssim,
        lpips=lpips_score,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        line_sharpness=line_sharpness,
        line_continuity=line_continuity,
        worm_artifact_score=worm_artifact_score,
        luma_ratio=luma_ratio,
        shadow_clip=shadow_clip,
        highlight_clip=highlight_clip,
        identity_similarity=identity_score,
        style_adherence=style_adherence,
        style_reference_used=has_style_ref,
        identity_faces_compared=identity_faces,
        passed=len(reasons) == 0,
        reasons=reasons,
    )
