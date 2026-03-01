from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _winner_index(row: dict[str, Any]) -> int | None:
    winner = str(row.get("winner", "")).strip().lower()
    if winner == "left":
        try:
            return int(row.get("left_index"))
        except Exception:
            return None
    if winner == "right":
        try:
            return int(row.get("right_index"))
        except Exception:
            return None
    return None


def _resolve_variant_path(run_dir: Path, variant: dict[str, Any]) -> Path | None:
    paths = variant.get("paths", {}) if isinstance(variant, dict) else {}
    if not isinstance(paths, dict):
        return None
    final_path = str(paths.get("final", "")).strip()
    if not final_path:
        return None

    candidate = Path(final_path)
    if candidate.exists() and candidate.is_file():
        return candidate
    candidate = run_dir / final_path
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def collect_winning_images(
    cache_dir: Path,
    *,
    since_days: int = 7,
    style_filter: str = "",
    max_images: int = 256,
) -> list[dict[str, Any]]:
    ratings_path = cache_dir / "ab_ratings.jsonl"
    runs_dir = cache_dir / "variant_runs"
    rows = _load_jsonl(ratings_path)
    if not rows:
        return []

    now = dt.datetime.now(dt.timezone.utc).timestamp()
    cutoff = now - max(1, int(since_days)) * 86400
    style_filter = style_filter.strip().lower()
    out: list[dict[str, Any]] = []
    seen_hash: set[str] = set()

    for row in rows:
        ts = float(row.get("timestamp", 0.0) or 0.0)
        if ts < cutoff:
            continue
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        winner_idx = _winner_index(row)
        if winner_idx is None:
            continue

        run_dir = runs_dir / run_id
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue

        style_name = str(metadata.get("style_name", "")).strip().lower()
        if style_filter and style_name != style_filter:
            continue

        variants = metadata.get("variants", [])
        if not isinstance(variants, list):
            continue
        target = None
        for item in variants:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except Exception:
                idx = -1
            if idx == winner_idx:
                target = item
                break
        if target is None:
            continue

        image_path = _resolve_variant_path(run_dir, target)
        if image_path is None:
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen_hash:
            continue
        seen_hash.add(digest)
        out.append(
            {
                "run_id": run_id,
                "variant_index": int(winner_idx),
                "style_name": style_name,
                "image_path": str(image_path),
                "timestamp": ts,
            }
        )
        if len(out) >= max(1, int(max_images)):
            break

    return out


def build_weekly_job(
    selected: list[dict[str, Any]],
    *,
    output_root: Path,
    style_name: str,
    token: str,
    base_model: str,
    resolution: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    style_slug = style_name.strip().lower().replace(" ", "_")
    out_dir = output_root / f"{style_slug}_{stamp}"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    selected_payload: list[dict[str, Any]] = []
    for idx, item in enumerate(selected):
        src = Path(str(item.get("image_path", "")))
        if not src.exists() or not src.is_file():
            continue
        dst = data_dir / f"{idx:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        caption = (
            f"{token}, manhwa manga panel, coherent anatomy, stable identity, detailed clothes, "
            "high quality line fidelity"
        )
        dataset_rows.append({"file_name": dst.name, "text": caption})
        selected_payload.append(
            {
                "run_id": str(item.get("run_id", "")),
                "variant_index": int(item.get("variant_index", -1)),
                "style_name": str(item.get("style_name", "")),
                "source_path": str(src),
                "copied_path": str(dst),
            }
        )

    if not dataset_rows:
        raise SystemExit("No valid images were copied into the weekly LoRA dataset.")

    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in dataset_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_config = {
        "pretrained_model_name_or_path": base_model,
        "train_data_dir": str(data_dir),
        "output_dir": str(out_dir / "weights"),
        "resolution": int(resolution),
        "train_batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_train_steps": int(max_steps),
        "lr_scheduler": "cosine",
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 4,
        "validation_prompt": f"{token}, old man full body, manhwa manga style, accurate pose",
        "checkpointing_steps": 100,
        "rank": 8,
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    command = (
        "accelerate launch train_dreambooth_lora_sdxl.py "
        f"--pretrained_model_name_or_path='{base_model}' "
        f"--instance_data_dir='{data_dir}' "
        f"--output_dir='{out_dir / 'weights'}' "
        f"--instance_prompt='{token} manhwa manga style' "
        f"--resolution={int(resolution)} "
        f"--train_batch_size={int(batch_size)} "
        f"--learning_rate={float(learning_rate)} "
        f"--max_train_steps={int(max_steps)} "
        "--gradient_accumulation_steps=4 --mixed_precision=fp16 --rank=8"
    )
    (out_dir / "train_command.txt").write_text(command + "\n", encoding="utf-8")
    (out_dir / "selected_sources.json").write_text(
        json.dumps(selected_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare a weekly SDXL LoRA update job from accepted A/B winners."
    )
    parser.add_argument("--cache-dir", type=Path, default=root / "cache")
    parser.add_argument("--output-root", type=Path, default=root / "loras" / "weekly_updates")
    parser.add_argument("--style-name", type=str, default="weekly_quality_update")
    parser.add_argument("--style-filter", type=str, default="")
    parser.add_argument("--token", type=str, default="<mshift_weekly>")
    parser.add_argument("--since-days", type=int, default=7)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = collect_winning_images(
        cache_dir=args.cache_dir,
        since_days=args.since_days,
        style_filter=args.style_filter,
        max_images=args.max_images,
    )
    print(f"Selected winners: {len(selected)}")
    if len(selected) < max(1, int(args.min_images)):
        raise SystemExit(
            f"Not enough accepted outputs for weekly LoRA update: {len(selected)} < {int(args.min_images)}"
        )
    if args.dry_run:
        return
    out_dir = build_weekly_job(
        selected,
        output_root=args.output_root,
        style_name=args.style_name,
        token=args.token,
        base_model=args.base_model,
        resolution=args.resolution,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    print(f"Prepared weekly LoRA job: {out_dir}")
    print(f"Run command: {out_dir / 'train_command.txt'}")


if __name__ == "__main__":
    main()
