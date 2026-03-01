from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402


def _collect_images(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa <= 1e-9 or bb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (aa * bb))


def _clip_proxy_distance(img_a: Image.Image, img_b: Image.Image) -> float:
    # Deterministic visual embedding proxy (histogram+edges) when CLIP model is unavailable locally.
    def emb(im: Image.Image) -> np.ndarray:
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

    ea = emb(img_a)
    eb = emb(img_b)
    return float(1.0 - _cosine(ea, eb))


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch metrics report for MangaShift two-pass quality gate outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--style", default="seinen_gritty")
    parser.add_argument("--quality", default="balanced", choices=["preview", "balanced", "final", "quality"])
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--style-ref", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "cache" / "batch_metrics_report.json")
    args = parser.parse_args()

    images = _collect_images(args.input_dir)
    if not images:
        raise SystemExit(f"No image files in {args.input_dir}")

    target = max(1, int(args.max_samples))
    sample_paths: List[Path] = []
    i = 0
    while len(sample_paths) < target:
        sample_paths.append(images[i % len(images)])
        i += 1

    style_ref = None
    if args.style_ref and args.style_ref.exists():
        style_ref = Image.open(args.style_ref).convert("RGB")

    client = TestClient(app)

    structure_ssim: List[float] = []
    lpips_vals: List[float] = []
    edge_recall_vals: List[float] = []
    edge_precision_vals: List[float] = []
    identity_vals: List[float] = []
    style_adherence_vals: List[float] = []
    content_clip_dist: List[float] = []
    style_clip_dist: List[float] = []
    pass_flags: List[bool] = []
    errors: List[dict] = []

    for idx, path in enumerate(sample_paths, start=1):
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        files = {"file": (path.name, buf.getvalue(), "image/png")}
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": args.style,
            "render_quality": args.quality,
            "variant_count": str(args.variant_count),
            "upscale": "1.0",
            "colorize": "false",
            "series_id": "batch_metrics",
            "chapter_id": "batch",
            "page_index": str(idx),
        }

        resp = client.post("/process-image", files=files, data=data)
        if resp.status_code != 200:
            errors.append({"sample": str(path), "status": resp.status_code, "error": resp.text})
            continue

        report = json.loads(resp.headers.get("X-Process-Report", "{}"))
        q = report.get("planner_report", {}).get("quality_gate", {})

        structure_ssim.append(float(q.get("structure_ssim", 0.0)))
        lpips_vals.append(float(q.get("lpips", 1.0)))
        edge_recall_vals.append(float(q.get("edge_recall", q.get("edge_preservation", 0.0))))
        edge_precision_vals.append(float(q.get("edge_precision", 0.0)))
        identity_vals.append(float(q.get("identity_similarity", 1.0)))
        style_adherence_vals.append(float(q.get("style_adherence", 1.0)))
        pass_flags.append(bool(q.get("passed", False)))

        out_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        content_clip_dist.append(_clip_proxy_distance(img, out_img))
        if style_ref is not None:
            style_clip_dist.append(_clip_proxy_distance(style_ref, out_img))

    payload = {
        "config": {
            "input_dir": str(args.input_dir),
            "max_samples": int(args.max_samples),
            "style": args.style,
            "quality": args.quality,
            "variant_count": int(args.variant_count),
            "style_ref": str(args.style_ref) if args.style_ref else "",
        },
        "counts": {
            "requested": int(target),
            "processed": int(len(pass_flags)),
            "errors": int(len(errors)),
        },
        "metrics": {
            "mean_ssim": _mean(structure_ssim),
            "mean_lpips": _mean(lpips_vals),
            "mean_edge_recall": _mean(edge_recall_vals),
            "mean_edge_precision": _mean(edge_precision_vals),
            "mean_identity_similarity": _mean(identity_vals),
            "mean_style_adherence": _mean(style_adherence_vals),
            "mean_clip_distance_content_vs_output": _mean(content_clip_dist),
            "mean_clip_distance_style_vs_output": _mean(style_clip_dist),
            "quality_gate_pass_rate_pct": (100.0 * sum(1 for v in pass_flags if v) / max(1, len(pass_flags))),
        },
        "errors": errors[:50],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
