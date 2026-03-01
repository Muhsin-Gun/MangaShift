from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "backend" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_weekly_lora_update as weekly  # noqa: E402


def _write_sample_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (128, 128), (220, 210, 200))
    img.save(path, format="PNG")


def test_collect_winning_images_and_build_job(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    run_dir = cache_dir / "variant_runs" / "run_a"
    final_path = run_dir / "variant_01_final.png"
    _write_sample_png(final_path)

    metadata = {
        "style_name": "realistic",
        "variants": [
            {
                "index": 1,
                "paths": {"final": str(final_path)},
            }
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rating = {
        "run_id": "run_a",
        "left_index": 1,
        "right_index": 2,
        "winner": "left",
        "timestamp": time.time(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "ab_ratings.jsonl").write_text(json.dumps(rating) + "\n", encoding="utf-8")

    selected = weekly.collect_winning_images(cache_dir=cache_dir, since_days=7, max_images=8)
    assert len(selected) == 1
    assert selected[0]["run_id"] == "run_a"
    assert selected[0]["variant_index"] == 1

    out_dir = weekly.build_weekly_job(
        selected,
        output_root=tmp_path / "weekly_jobs",
        style_name="weekly_quality_update",
        token="<weekly>",
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        resolution=1024,
        batch_size=1,
        learning_rate=5e-5,
        max_steps=400,
    )
    assert (out_dir / "dataset.jsonl").exists()
    assert (out_dir / "train_config.json").exists()
    assert (out_dir / "train_command.txt").exists()
    assert (out_dir / "selected_sources.json").exists()
