from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import adaptive_controller, app, settings  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: Dict[str, bool]


DEFAULT_VARIANTS: List[Variant] = [
    Variant(
        name="baseline",
        description="scene+episodic+planner+adaptive enabled",
        overrides={
            "enable_scene_memory": True,
            "enable_episodic_memory": True,
            "enable_repair_planner": True,
            "enable_adaptive_controller": True,
            "enforce_identity_consistency": True,
        },
    ),
    Variant(
        name="no_episodic",
        description="episodic memory disabled",
        overrides={"enable_episodic_memory": False},
    ),
    Variant(
        name="no_planner",
        description="repair planner disabled",
        overrides={"enable_repair_planner": False},
    ),
    Variant(
        name="no_adaptive",
        description="adaptive controller disabled",
        overrides={"enable_adaptive_controller": False},
    ),
    Variant(
        name="no_scene",
        description="scene memory disabled",
        overrides={"enable_scene_memory": False},
    ),
]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    if not vals:
        return float(default)
    return float(statistics.mean(vals))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _target_latency_ms(quality: str) -> float:
    mapping = {"preview": 2000.0, "balanced": 5500.0, "final": 9000.0}
    return mapping.get(quality, 5500.0)


def compute_overall_score(summary: Dict[str, float], target_ms: float) -> float:
    edge = _clamp(summary.get("edge_overlap_mean", 0.0), 0.0, 1.0)
    identity = _clamp(summary.get("identity_mean", 1.0), 0.0, 1.0)
    contrast = _clamp(summary.get("contrast_shift_mean", 0.0), 0.0, 80.0)
    texture = _clamp(summary.get("texture_shift_mean", 0.0), 0.0, 0.3)
    mean_ms = max(1.0, float(summary.get("latency_mean_ms", target_ms)))

    quality_score = 100.0 * (
        0.42 * edge
        + 0.32 * identity
        + 0.16 * (1.0 - contrast / 80.0)
        + 0.10 * (1.0 - texture / 0.3)
    )
    perf_score = 100.0 * _clamp(target_ms / mean_ms, 0.0, 1.0)
    overall = 0.72 * quality_score + 0.28 * perf_score
    return round(_clamp(overall, 0.0, 100.0), 4)


def summarize_records(records: List[Dict[str, object]], quality: str) -> Dict[str, object]:
    total = len(records)
    success = [r for r in records if int(r.get("status", 0)) == 200]
    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    pipeline_ms = [float(r.get("total_ms", 0.0)) for r in success if float(r.get("total_ms", 0.0)) > 0.0]
    edge = [float(r.get("edge_overlap", 0.0)) for r in success]
    blur = [float(r.get("blur_score", 1.0)) for r in success]
    contrast = [float(r.get("contrast_shift", 0.0)) for r in success]
    texture = [float(r.get("texture_shift", 0.0)) for r in success]
    identity = [float(r.get("identity_similarity", 1.0)) for r in success]
    planner_applied = [bool(r.get("planner_applied", False)) for r in success]
    adaptive_strength = [
        float(r.get("adaptive_strength", 0.0))
        for r in success
        if r.get("adaptive_strength") is not None
    ]
    adaptive_steps = [
        float(r.get("adaptive_steps", 0.0))
        for r in success
        if r.get("adaptive_steps") is not None
    ]
    style_backends: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    for row in records:
        code = str(row.get("status", "0"))
        status_counts[code] = status_counts.get(code, 0) + 1
        backend = str(row.get("style_backend", "unknown"))
        style_backends[backend] = style_backends.get(backend, 0) + 1

    summary = {
        "total_requests": total,
        "success_requests": len(success),
        "success_rate": round((len(success) / max(1, total)) * 100.0, 4),
        "latency_mean_ms": round(_safe_mean(latencies), 4),
        "latency_p50_ms": round(percentile(latencies, 0.5), 4),
        "latency_p95_ms": round(percentile(latencies, 0.95), 4),
        "pipeline_mean_ms": round(_safe_mean(pipeline_ms), 4),
        "edge_overlap_mean": round(_safe_mean(edge), 6),
        "blur_score_mean": round(_safe_mean(blur, default=1.0), 6),
        "contrast_shift_mean": round(_safe_mean(contrast), 6),
        "texture_shift_mean": round(_safe_mean(texture), 6),
        "identity_mean": round(_safe_mean(identity, default=1.0), 6),
        "planner_applied_rate": round((sum(planner_applied) / max(1, len(success))) * 100.0, 4),
        "adaptive_strength_mean": round(_safe_mean(adaptive_strength), 6),
        "adaptive_steps_mean": round(_safe_mean(adaptive_steps), 6),
        "status_counts": status_counts,
        "style_backend_counts": style_backends,
    }
    summary["overall_score"] = compute_overall_score(summary, target_ms=_target_latency_ms(quality))
    return summary


@contextmanager
def temporary_flags(overrides: Dict[str, bool]):
    tracked = {
        "strict_pro_mode": False,
        "strict_require_diffusion": False,
    }
    tracked.update(overrides)
    snapshot = {k: getattr(settings, k) for k in tracked.keys()}
    try:
        for key, value in tracked.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in snapshot.items():
            setattr(settings, key, value)


def _collect_images(input_dir: Path, limit_images: int) -> List[Path]:
    images = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if limit_images > 0:
        return images[:limit_images]
    return images


def _resolve_variants(names: List[str]) -> List[Variant]:
    if not names:
        return list(DEFAULT_VARIANTS)
    mapping = {v.name: v for v in DEFAULT_VARIANTS}
    variants: List[Variant] = []
    for name in names:
        if name not in mapping:
            raise SystemExit(f"Unknown variant '{name}'. Available: {', '.join(sorted(mapping.keys()))}")
        variants.append(mapping[name])
    return variants


def run_variant(
    client: TestClient,
    variant: Variant,
    images: List[Path],
    iterations: int,
    style: str,
    quality: str,
    upscale: float,
) -> Dict[str, object]:
    run_records: List[Dict[str, object]] = []
    with temporary_flags(variant.overrides):
        adaptive_controller.reset()
        request_idx = 0
        for iteration in range(iterations):
            for image_path in images:
                request_idx += 1
                files = {"file": (image_path.name, image_path.read_bytes(), "image/png")}
                data = {
                    "source_lang": "auto",
                    "target_lang": "en",
                    "style": style,
                    "render_quality": quality,
                    "colorize": "false",
                    "upscale": str(upscale),
                    "series_id": f"ablation_{variant.name}",
                    "chapter_id": f"iter_{iteration}",
                    "page_index": str(request_idx),
                }
                t0 = time.perf_counter()
                resp = client.post("/process-image", files=files, data=data)
                latency = (time.perf_counter() - t0) * 1000.0
                record: Dict[str, object] = {
                    "status": resp.status_code,
                    "latency_ms": latency,
                }
                if resp.status_code == 200:
                    payload = json.loads(resp.headers.get("X-Process-Report", "{}"))
                    struct = payload.get("structural_report", {})
                    identity = payload.get("identity_report", {})
                    planner = payload.get("planner_report", {})
                    engine = payload.get("engine_report", {})
                    adaptive = engine.get("adaptive_controller", {}).get("decision", {})
                    record.update(
                        {
                            "total_ms": payload.get("timings_ms", {}).get("total", 0),
                            "edge_overlap": struct.get("edge_overlap", 0.0),
                            "blur_score": struct.get("blur_score", 1.0),
                            "contrast_shift": struct.get("contrast_shift", 0.0),
                            "texture_shift": struct.get("texture_shift", 0.0),
                            "identity_similarity": identity.get("best_similarity", 1.0),
                            "planner_applied": planner.get("apply_rerender", False),
                            "style_backend": engine.get("style", {}).get("selected", "unknown"),
                            "adaptive_strength": adaptive.get("strength"),
                            "adaptive_steps": adaptive.get("steps"),
                        }
                    )
                run_records.append(record)

    summary = summarize_records(run_records, quality=quality)
    return {
        "name": variant.name,
        "description": variant.description,
        "overrides": variant.overrides,
        "summary": summary,
        "records": run_records,
    }


def _markdown_report(results: List[Dict[str, object]], preflight: dict) -> str:
    lines = []
    lines.append("# MangaShift Ablation Report")
    lines.append("")
    lines.append(f"- preflight tier: `{preflight.get('tier', 'unknown')}`")
    lines.append(f"- preflight score: `{preflight.get('readiness_score', 0)}`")
    lines.append("")
    lines.append("| Variant | Success % | Mean ms | Edge | Identity | Planner % | Adaptive Strength | Overall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        s = item["summary"]
        lines.append(
            (
                "| {name} | {succ:.2f} | {mean_ms:.1f} | {edge:.3f} | {ident:.3f} | "
                "{planner:.2f} | {strength:.3f} | {score:.2f} |"
            ).format(
                name=item["name"],
                succ=float(s["success_rate"]),
                mean_ms=float(s["latency_mean_ms"]),
                edge=float(s["edge_overlap_mean"]),
                ident=float(s["identity_mean"]),
                planner=float(s["planner_applied_rate"]),
                strength=float(s["adaptive_strength_mean"]),
                score=float(s["overall_score"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ablation experiments across pipeline feature flags.")
    parser.add_argument("--input-dir", type=Path, default=Path("output_export/phase2_rerun"))
    parser.add_argument("--limit-images", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--style", default="cinematic")
    parser.add_argument("--quality", choices=["preview", "balanced", "final"], default="balanced")
    parser.add_argument("--upscale", type=float, default=1.2)
    parser.add_argument("--variants", default="", help="Comma-separated variant names")
    parser.add_argument("--output-json", type=Path, default=Path("backend/cache/ablation_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("backend/cache/ablation_report.md"))
    args = parser.parse_args()

    variants = _resolve_variants([v.strip() for v in args.variants.split(",") if v.strip()])
    images = _collect_images(args.input_dir, limit_images=max(1, args.limit_images))
    if not images:
        raise SystemExit(f"No image files found in {args.input_dir}")

    client = TestClient(app)
    preflight = client.get("/preflight").json()

    results = []
    for variant in variants:
        result = run_variant(
            client=client,
            variant=variant,
            images=images,
            iterations=max(1, int(args.iterations)),
            style=args.style,
            quality=args.quality,
            upscale=float(args.upscale),
        )
        results.append(result)

    ranking = sorted(
        [{"name": r["name"], "overall_score": r["summary"]["overall_score"]} for r in results],
        key=lambda x: float(x["overall_score"]),
        reverse=True,
    )
    payload = {
        "config": {
            "input_dir": str(args.input_dir),
            "images_used": [str(p) for p in images],
            "iterations": int(args.iterations),
            "style": args.style,
            "quality": args.quality,
            "upscale": float(args.upscale),
            "variants": [v.name for v in variants],
        },
        "preflight": preflight,
        "results": results,
        "ranking": ranking,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown_report(results, preflight=preflight), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nMarkdown report saved to: {args.output_md}")


if __name__ == "__main__":
    main()
