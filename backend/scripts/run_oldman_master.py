from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_INPUT = Path(r"C:\Users\amadn\OneDrive\Pictures\Screenshots\Screenshot 2026-02-26 115905.png")
DEFAULT_CROP = (460, 120, 1260, 1110)
DEFAULT_LOCAL_INPUT = ROOT / "backend" / "cache" / "oldman_master" / "oldman_master_balanced_local.png"


def maybe_reexec_cuda_venv() -> None:
    if os.environ.get("MANGASHIFT_REEXEC_DONE") == "1":
        return
    if os.environ.get("MANGASHIFT_DISABLE_AUTO_PYTHON") == "1":
        return

    if os.name == "nt":
        candidate = ROOT / "backend" / ".venv_cuda" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / "backend" / ".venv_cuda" / "bin" / "python"
    if not candidate.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        target = candidate.resolve()
    except Exception:
        return
    if current == target:
        return

    should_switch = sys.version_info >= (3, 13)
    if not should_switch:
        try:
            import torch

            should_switch = not bool(torch.cuda.is_available())
        except Exception:
            should_switch = True
    if not should_switch:
        return

    env = dict(os.environ)
    env["MANGASHIFT_REEXEC_DONE"] = "1"
    proc = subprocess.run([str(target), __file__, *sys.argv[1:]], env=env, check=False)
    raise SystemExit(int(proc.returncode))


def parse_crop(raw: str) -> Tuple[int, int, int, int]:
    bits = [b.strip() for b in str(raw).split(",")]
    if len(bits) != 4:
        raise ValueError("crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = [int(v) for v in bits]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid crop coordinates")
    return x1, y1, x2, y2


def read_image(path: Path) -> Image.Image:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input image not found: {path}")
    return Image.open(path).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def maybe_ref_bytes(ref_path: Optional[Path]) -> Optional[bytes]:
    if ref_path is None:
        return None
    if not ref_path.exists() or not ref_path.is_file():
        raise SystemExit(f"Reference image not found: {ref_path}")
    return ref_path.read_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run old-man panel generation with production controls.")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_LOCAL_INPUT if DEFAULT_LOCAL_INPUT.exists() else DEFAULT_INPUT,
    )
    parser.add_argument("--crop", type=str, default="460,120,1260,1110")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backend" / "cache" / "oldman_master")
    parser.add_argument("--style", type=str, default="realistic")
    parser.add_argument(
        "--render-quality",
        type=str,
        default="final",
        choices=["preview", "balanced", "final", "quality"],
    )
    parser.add_argument("--shot-type", type=str, default="standing_full_body")
    parser.add_argument("--variant-count", type=int, default=24)
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument("--line-weight", type=float, default=1.08)
    parser.add_argument("--hatch-density", type=float, default=0.56)
    parser.add_argument("--tone-intensity", type=float, default=0.36)
    parser.add_argument("--screentone-strength", type=float, default=0.12)
    parser.add_argument(
        "--colorize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pre-style colorization pass (recommended for sketch/gray inputs).",
    )
    parser.add_argument("--source-lang", type=str, default="auto", choices=["auto", "ja", "ko"])
    parser.add_argument("--page-index", type=int, default=2)
    parser.add_argument("--pose-ref", type=Path, default=None)
    parser.add_argument("--style-ref", type=Path, default=None)
    parser.add_argument("--character-ref", type=Path, default=None)
    parser.add_argument("--use-crop-as-refs", action="store_true")
    parser.add_argument("--strict-gate", action="store_true", help="Fail if quality gate does not pass.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for non-quality runs (disabled by default to avoid slow/degraded outputs).",
    )
    return parser.parse_args()


def main() -> None:
    maybe_reexec_cuda_venv()
    args = parse_args()
    from app.main import app, settings  # noqa: E402

    input_img = read_image(args.input_image)
    # If source already looks like a panel crop, keep as-is; otherwise apply crop box.
    if input_img.width > 1100 and input_img.height > 1100:
        crop_box = parse_crop(args.crop)
        crop_img = input_img.crop(crop_box)
    else:
        crop_img = input_img
    crop_bytes = image_to_png_bytes(crop_img)

    pose_bytes = maybe_ref_bytes(args.pose_ref)
    style_ref_bytes = maybe_ref_bytes(args.style_ref)
    char_ref_bytes = maybe_ref_bytes(args.character_ref)
    if args.use_crop_as_refs:
        pose_bytes = pose_bytes or crop_bytes
        style_ref_bytes = style_ref_bytes or crop_bytes
        char_ref_bytes = char_ref_bytes or crop_bytes

    if (
        args.render_quality == "quality"
        and settings.quality_require_full_path
        and (style_ref_bytes is None and char_ref_bytes is None)
    ):
        raise SystemExit(
            "Quality full-path mode requires style or character reference. "
            "Provide --style-ref/--character-ref or --use-crop-as-refs."
        )
    if args.render_quality == "quality" and args.shot_type == "standing_full_body" and pose_bytes is None:
        raise SystemExit(
            "standing_full_body quality mode requires --pose-ref or --use-crop-as-refs."
        )

    client = TestClient(app)

    readiness = client.get("/quality/readiness")
    readiness_payload = readiness.json() if readiness.status_code == 200 else {"status_code": readiness.status_code}
    local_gpu_ready = bool(readiness_payload.get("local_gpu_ready", False))
    quality_ready = bool(readiness_payload.get("quality_ready", False))

    if args.render_quality in {"final", "quality"}:
        if not quality_ready:
            raise SystemExit(
                "Final/quality runtime not ready (strict GPU path). "
                f"Readiness: {json.dumps(readiness_payload, ensure_ascii=False)}"
            )
    elif not args.allow_cpu and not local_gpu_ready:
        raise SystemExit(
            "Local CUDA is required for non-quality oldman runs to prevent slow CPU fallback. "
            f"Readiness: {json.dumps(readiness_payload, ensure_ascii=False)}"
        )

    files = {"file": ("oldman_crop.png", crop_bytes, "image/png")}
    if pose_bytes is not None:
        files["pose_file"] = ("pose_reference.png", pose_bytes, "image/png")
    if style_ref_bytes is not None:
        files["style_ref_file"] = ("style_reference.png", style_ref_bytes, "image/png")
    if char_ref_bytes is not None:
        files["character_ref_file"] = ("character_reference.png", char_ref_bytes, "image/png")

    form = {
        "source_lang": args.source_lang,
        "target_lang": "en",
        "style": args.style,
        "colorize": str(bool(args.colorize)).lower(),
        "upscale": str(args.upscale),
        "inpaint_padding": "8",
        "render_quality": args.render_quality,
        "series_id": "oldman_master_series",
        "chapter_id": "oldman_master_chapter",
        "page_index": str(max(0, int(args.page_index))),
        "enforce_identity": "true",
        "shot_type": args.shot_type,
        "variant_count": str(max(1, min(32, int(args.variant_count)))),
        "line_weight": str(max(0.5, min(2.0, float(args.line_weight)))),
        "hatch_density": str(max(0.1, min(1.0, float(args.hatch_density)))),
        "tone_intensity": str(max(0.0, min(1.0, float(args.tone_intensity)))),
        "screentone_strength": str(max(0.0, min(1.0, float(args.screentone_strength)))),
    }
    response = client.post("/process-image", files=files, data=form)
    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = {"raw": response.text}
        raise SystemExit(f"process-image failed ({response.status_code}): {json.dumps(detail, ensure_ascii=False)}")

    report_raw = response.headers.get("X-Process-Report", "")
    report = json.loads(report_raw) if report_raw else {}
    gate = (
        report.get("engine_report", {})
        .get("planner", {})
        .get("quality_gate", {})
    )
    gate_passed = bool(gate.get("passed", False))
    cache_hit = str(response.headers.get("X-Cache-Hit", "0")) == "1"
    if args.strict_gate:
        if gate:
            if not gate_passed:
                raise SystemExit(f"Quality gate failed: {json.dumps(gate, ensure_ascii=False)}")
        elif cache_hit:
            raise SystemExit(
                "Strict gate requested but cached response did not include process report. "
                "Rerun with a different page_index/crop to bypass cache."
            )
        else:
            raise SystemExit("Strict gate requested but process report was missing from response.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"oldman_master_{args.render_quality}.png"
    report_path = out_dir / f"oldman_master_{args.render_quality}_report.json"
    crop_path = out_dir / "oldman_input_crop.png"
    output_path.write_bytes(response.content)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    crop_path.write_bytes(crop_bytes)

    print("=== Oldman Master Run ===")
    print(f"output_image: {output_path}")
    print(f"report: {report_path}")
    print(f"quality_gate_passed: {gate_passed}")
    print(f"remote_worker: {response.headers.get('X-Remote-Worker', '0')}")
    print(f"runtime_mode: {readiness_payload.get('mode', 'unknown')}")
    print(f"local_gpu_ready: {local_gpu_ready}")
    print(f"quality_ready: {quality_ready}")
    top_variants = report.get("engine_report", {}).get("two_pass", {}).get("top_variants", [])
    print(f"top_variants_count: {len(top_variants) if isinstance(top_variants, list) else 0}")


if __name__ == "__main__":
    main()
