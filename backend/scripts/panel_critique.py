from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402


def _default_panels() -> list[Path]:
    return [
        Path(r"C:\Users\amadn\OneDrive\Pictures\Screenshots\Screenshot 2026-02-26 120051.png"),
        Path(r"C:\Users\amadn\OneDrive\Pictures\Screenshots\Screenshot 2026-02-26 115905.png"),
        Path(r"C:\Users\amadn\OneDrive\Pictures\Screenshots\Screenshot 2026-02-25 150804.png"),
    ]


def _crop_for_panel(path: Path, image: Image.Image) -> Image.Image:
    name = path.name
    boxes = {
        "Screenshot 2026-02-26 120051.png": (520, 140, 1080, 1110),
        "Screenshot 2026-02-26 115905.png": (460, 120, 1260, 1110),
        "Screenshot 2026-02-25 150804.png": (610, 120, 1260, 1110),
    }
    box = boxes.get(name)
    if box is None:
        return image
    return image.crop(box)


def _decision(score: float, passed: bool, reasons: list[str]) -> tuple[str, str]:
    if passed and score >= 92:
        return "keep", "Strong fidelity and quality-gate pass"
    if passed:
        return "change", "Passes gate but still needs art-direction polish"
    if reasons:
        return "remove", f"Failed gate: {'; '.join(reasons)}"
    return "remove", "Failed quality gate"


def _triptych(original: Image.Image, structure: Image.Image, final: Image.Image) -> Image.Image:
    w = 420
    h = 620
    pad = 12
    label_h = 34

    def fit(im: Image.Image) -> Image.Image:
        return im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (pad * 4 + w * 3, pad * 2 + h + label_h), (17, 19, 26))
    draw = ImageDraw.Draw(canvas)
    items = [("original", fit(original)), ("structure", fit(structure)), ("final", fit(final))]
    for idx, (name, im) in enumerate(items):
        x = pad + idx * (w + pad)
        y = pad
        canvas.paste(im, (x, y))
        draw.rectangle((x, y + h, x + w, y + h + label_h), fill=(28, 31, 42))
        draw.text((x + 10, y + h + 10), name, fill=(235, 238, 245))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scored critique for target panels with 6 two-pass variants.")
    parser.add_argument("--style", default="seinen_gritty")
    parser.add_argument("--quality", default="quality", choices=["preview", "balanced", "final", "quality"])
    parser.add_argument("--variant-count", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_export")
    parser.add_argument("--panel", action="append", default=[])
    args = parser.parse_args()

    panel_paths = [Path(p) for p in args.panel] if args.panel else _default_panels()
    panel_paths = [p for p in panel_paths if p.exists()]
    if not panel_paths:
        raise SystemExit("No valid panel files found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_dir / f"critique_suite_{stamp}"
    originals_dir = out_root / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    critique: dict = {
        "generated_at": datetime.now().isoformat(),
        "style": args.style,
        "quality": args.quality,
        "variant_count": int(args.variant_count),
        "panels": [],
    }

    for panel_idx, src_path in enumerate(panel_paths, start=1):
        src_img = Image.open(src_path).convert("RGB")
        crop = _crop_for_panel(src_path, src_img)
        panel_name = f"panel_{panel_idx:02d}_{src_path.stem.replace(' ', '_')}"
        panel_dir = out_root / panel_name
        panel_dir.mkdir(parents=True, exist_ok=True)

        original_path = originals_dir / f"{panel_name}_original.png"
        crop.save(original_path)

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        files = {"file": (f"{panel_name}.png", buf.getvalue(), "image/png")}
        data = {
            "source_lang": "auto",
            "target_lang": "en",
            "style": args.style,
            "render_quality": args.quality,
            "variant_count": str(args.variant_count),
            "colorize": "false",
            "upscale": "1.0",
            "series_id": f"critique_{panel_name}",
            "chapter_id": "suite",
            "page_index": str(panel_idx),
        }

        resp = client.post("/process-image", files=files, data=data)
        if resp.status_code != 200:
            critique["panels"].append(
                {
                    "panel": panel_name,
                    "source": str(src_path),
                    "error": resp.text,
                }
            )
            continue

        report = json.loads(resp.headers.get("X-Process-Report", "{}"))
        two_pass = report.get("engine_report", {}).get("two_pass", {})
        metadata = two_pass.get("metadata", {})
        run_dir = Path(two_pass.get("run_dir", "")) if two_pass.get("run_dir") else None

        panel_entry = {
            "panel": panel_name,
            "source": str(src_path),
            "original": str(original_path),
            "variants": [],
        }

        if run_dir and run_dir.exists() and metadata:
            meta_out = panel_dir / "metadata.json"
            meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            for item in metadata.get("variants", []):
                idx = int(item.get("index", 0))
                q = dict(item.get("quality", {}))
                score = float(item.get("score", 0.0))
                passed = bool(q.get("passed", False))
                reasons = list(q.get("reasons", []))
                decision, reason = _decision(score=score, passed=passed, reasons=reasons)

                s_path = item.get("paths", {}).get("structure", "")
                f_path = item.get("paths", {}).get("final", "")
                if s_path and f_path and Path(s_path).exists() and Path(f_path).exists():
                    structure = Image.open(s_path).convert("RGB")
                    final = Image.open(f_path).convert("RGB")
                    trip = _triptych(crop, structure, final)
                    trip_path = panel_dir / f"variant_{idx:02d}_triptych.png"
                    trip.save(trip_path)
                else:
                    trip_path = None

                panel_entry["variants"].append(
                    {
                        "index": idx,
                        "seed": int(item.get("seed", 0)),
                        "score": score,
                        "decision": decision,
                        "reason": reason,
                        "quality": q,
                        "params": item.get("params", {}),
                        "structure_path": s_path,
                        "final_path": f_path,
                        "triptych": str(trip_path) if trip_path else "",
                    }
                )

            # Copy the selected best final for quick scan.
            best_idx = int(metadata.get("best_index", 1))
            best_final = run_dir / f"variant_{best_idx:02d}_final.png"
            if best_final.exists():
                shutil.copy2(best_final, panel_dir / "best_final.png")

        critique["panels"].append(panel_entry)

    critique_path = out_root / "scored_critique.json"
    critique_path.write_text(json.dumps(critique, indent=2), encoding="utf-8")
    print(f"saved: {critique_path}")


if __name__ == "__main__":
    main()
