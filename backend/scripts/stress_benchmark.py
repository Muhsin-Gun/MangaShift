from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import httpx


def _collect_images(input_dir: Path, limit: int) -> List[Path]:
    images = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )
    if limit > 0:
        return images[:limit]
    return images


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


def run_http_benchmark(
    base_url: str,
    images: List[Path],
    iterations: int,
    style: str,
    quality: str,
    upscale: float,
) -> Dict[str, object]:
    latencies: List[float] = []
    status_counts: Dict[str, int] = {}
    cache_hits = 0
    backend_counts: Dict[str, int] = {}

    with httpx.Client(timeout=300.0) as client:
        idx = 0
        for _ in range(iterations):
            for image_path in images:
                idx += 1
                files = {"file": (image_path.name, image_path.read_bytes(), "image/png")}
                data = {
                    "source_lang": "auto",
                    "target_lang": "en",
                    "style": style,
                    "render_quality": quality,
                    "upscale": str(upscale),
                    "colorize": "false",
                    "series_id": "benchmark_series",
                    "chapter_id": "benchmark_chapter",
                    "page_index": str(idx),
                }
                t0 = time.perf_counter()
                response = client.post(f"{base_url}/process-image", files=files, data=data)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)
                code = str(response.status_code)
                status_counts[code] = status_counts.get(code, 0) + 1
                if response.headers.get("X-Cache-Hit") == "1":
                    cache_hits += 1
                if response.status_code == 200:
                    report_raw = response.headers.get("X-Process-Report", "{}")
                    try:
                        report = json.loads(report_raw)
                    except Exception:
                        report = {}
                    style_backend = (
                        report.get("engine_report", {})
                        .get("style", {})
                        .get("selected", "unknown")
                    )
                    backend_counts[style_backend] = backend_counts.get(style_backend, 0) + 1

    total = len(latencies)
    success = status_counts.get("200", 0)
    return {
        "total_requests": total,
        "success_requests": success,
        "success_rate": round((success / max(1, total)) * 100.0, 4),
        "cache_hit_rate": round((cache_hits / max(1, total)) * 100.0, 4),
        "latency_ms": {
            "mean": round(float(statistics.mean(latencies)) if latencies else 0.0, 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(max(latencies) if latencies else 0.0, 3),
        },
        "status_counts": status_counts,
        "style_backend_counts": backend_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MangaShift process-image stress benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input-dir", type=Path, default=Path("output_export/phase2_rerun"))
    parser.add_argument("--limit-images", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--style", default="cinematic")
    parser.add_argument("--quality", default="balanced", choices=["preview", "balanced", "final"])
    parser.add_argument("--upscale", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("backend/cache/stress_benchmark.json"))
    args = parser.parse_args()

    images = _collect_images(args.input_dir, limit=max(1, args.limit_images))
    if not images:
        raise SystemExit(f"No image files found in {args.input_dir}")

    result = run_http_benchmark(
        base_url=args.base_url.rstrip("/"),
        images=images,
        iterations=max(1, int(args.iterations)),
        style=args.style,
        quality=args.quality,
        upscale=float(args.upscale),
    )
    payload = {
        "base_url": args.base_url,
        "input_dir": str(args.input_dir),
        "images_used": [str(p) for p in images],
        "style": args.style,
        "quality": args.quality,
        "upscale": float(args.upscale),
        "result": result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
