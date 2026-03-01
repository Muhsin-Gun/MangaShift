from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app, settings  # noqa: E402


def build_synthetic_panel(width: int = 640, height: int = 980) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 50, width - 40, 280), radius=36, fill="white", outline="black", width=4)
    draw.rounded_rectangle((70, 360, width - 90, 650), radius=40, fill=(242, 242, 242), outline="black", width=4)
    draw.rectangle((30, 730, width - 30, height - 40), fill=(220, 220, 220), outline="black", width=4)
    draw.text((90, 120), "annyeong haseyo", fill="black")
    draw.text((110, 430), "konnichiwa", fill="black")
    draw.text((130, 780), "this is a strict smoke test", fill="black")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@contextmanager
def strict_flags(
    strict_pro_mode: bool,
    strict_require_gpu: bool,
    strict_require_diffusion: bool,
    strict_require_ocr: bool,
    strict_require_translation_models: bool,
):
    keys = {
        "strict_pro_mode": strict_pro_mode,
        "strict_require_gpu": strict_require_gpu,
        "strict_require_diffusion": strict_require_diffusion,
        "strict_require_ocr": strict_require_ocr,
        "strict_require_translation_models": strict_require_translation_models,
    }
    prev = {name: getattr(settings, name) for name in keys}
    try:
        for name, value in keys.items():
            setattr(settings, name, value)
        yield
    finally:
        for name, value in prev.items():
            setattr(settings, name, value)


def run_smoke(
    image_bytes: bytes,
    style: str,
    render_quality: str,
    source_lang: str,
    colorize: bool,
    upscale: float,
) -> dict:
    client = TestClient(app)
    result: dict = {"status": "unknown", "ok": False}

    preflight = client.get("/preflight")
    result["preflight_status"] = preflight.status_code
    result["preflight"] = preflight.json() if preflight.status_code == 200 else {}

    warmup_payload = {
        "render_quality": render_quality,
        "style": style,
        "source_lang": source_lang,
        "colorize": colorize,
        "strict": True,
    }
    warmup = client.post("/warmup", json=warmup_payload)
    result["warmup_status"] = warmup.status_code
    result["warmup"] = warmup.json() if warmup.status_code == 200 else {}

    files = {"file": ("smoke_panel.png", image_bytes, "image/png")}
    form = {
        "source_lang": source_lang,
        "target_lang": "en",
        "style": style,
        "colorize": str(colorize).lower(),
        "upscale": str(upscale),
        "render_quality": render_quality,
        "series_id": "strict_smoke_series",
        "chapter_id": "strict_smoke_chapter",
        "page_index": "1",
    }
    started = time.perf_counter()
    process = client.post("/process-image", files=files, data=form)
    latency_ms = (time.perf_counter() - started) * 1000.0
    result["process_status"] = process.status_code
    result["process_latency_ms"] = round(latency_ms, 3)
    result["process_headers"] = dict(process.headers)
    if process.status_code == 200:
        result["output_bytes"] = len(process.content)
        report_raw = process.headers.get("X-Process-Report")
        result["process_report"] = json.loads(report_raw) if report_raw else {}
        result["output_png"] = process.content
    else:
        try:
            result["process_error"] = process.json()
        except Exception:
            result["process_error"] = {"raw": process.text}

    preflight_blockers = result.get("preflight", {}).get("blockers", [])
    warmup_ok = bool(result.get("warmup", {}).get("ok", False))
    process_ok = process.status_code == 200
    is_ok = bool(len(preflight_blockers) == 0 and warmup_ok and process_ok)
    result["ok"] = is_ok
    result["status"] = "ready" if is_ok else "blocked"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict smoke validation for MangaShift backend.")
    parser.add_argument("--input-image", type=Path, default=None, help="Optional input image for smoke run.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backend" / "cache" / "strict_smoke")
    parser.add_argument("--style", default="cinematic")
    parser.add_argument("--quality", default="final", choices=["preview", "balanced", "final"])
    parser.add_argument("--source-lang", default="auto", choices=["auto", "ja", "ko"])
    parser.add_argument("--colorize", action="store_true")
    parser.add_argument("--upscale", type=float, default=1.5)
    parser.add_argument("--strict-gpu", action="store_true")
    parser.add_argument("--strict-diffusion", action="store_true")
    parser.add_argument("--strict-ocr", action="store_true")
    parser.add_argument("--strict-translation", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.input_image is not None:
        if not args.input_image.exists():
            raise SystemExit(f"Input image not found: {args.input_image}")
        image_bytes = args.input_image.read_bytes()
    else:
        image_bytes = build_synthetic_panel()

    with strict_flags(
        strict_pro_mode=True,
        strict_require_gpu=bool(args.strict_gpu),
        strict_require_diffusion=bool(args.strict_diffusion),
        strict_require_ocr=bool(args.strict_ocr),
        strict_require_translation_models=bool(args.strict_translation),
    ):
        result = run_smoke(
            image_bytes=image_bytes,
            style=args.style,
            render_quality=args.quality,
            source_lang=args.source_lang,
            colorize=bool(args.colorize),
            upscale=float(args.upscale),
        )

    report_for_json = {k: v for k, v in result.items() if k != "output_png"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "strict_smoke_report.json"
    report_path.write_text(json.dumps(report_for_json, indent=2, ensure_ascii=False), encoding="utf-8")
    if "output_png" in result and isinstance(result["output_png"], (bytes, bytearray)):
        output_path = args.output_dir / "strict_smoke_output.png"
        output_path.write_bytes(result["output_png"])
        report_for_json["output_path"] = str(output_path)

    if args.json:
        print(json.dumps(report_for_json, indent=2, ensure_ascii=False))
    else:
        print("=== Strict Smoke ===")
        print(f"status: {report_for_json.get('status')}")
        print(f"ok: {report_for_json.get('ok')}")
        print(f"preflight_status: {report_for_json.get('preflight_status')}")
        print(f"warmup_status: {report_for_json.get('warmup_status')}")
        print(f"process_status: {report_for_json.get('process_status')}")
        print(f"process_latency_ms: {report_for_json.get('process_latency_ms')}")
        print(f"report: {report_path}")

    if not report_for_json.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
