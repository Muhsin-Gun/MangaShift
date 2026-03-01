from __future__ import annotations

import argparse
import json
from pathlib import Path


def list_images(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a LoRA training dataset/config for MangaShift styles.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Reference image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output job directory.")
    parser.add_argument("--style-name", type=str, required=True, help="Name for output LoRA.")
    parser.add_argument("--token", type=str, default="<mshift_style>", help="Unique concept token.")
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=1800)
    args = parser.parse_args()

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    out = args.output_dir / args.style_name
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, img in enumerate(images):
        dst = data_dir / f"{idx:05d}{img.suffix.lower()}"
        if not dst.exists():
            dst.write_bytes(img.read_bytes())
        caption = f"{args.token}, manga panel, high detail, coherent character identity"
        records.append({"file_name": dst.name, "text": caption})

    dataset_jsonl = out / "dataset.jsonl"
    with dataset_jsonl.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_config = {
        "pretrained_model_name_or_path": args.base_model,
        "train_data_dir": str(data_dir),
        "output_dir": str(out / "weights"),
        "resolution": args.resolution,
        "train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_steps,
        "lr_scheduler": "cosine",
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 4,
        "validation_prompt": f"{args.token}, cinematic manga hero portrait",
        "checkpointing_steps": 200,
        "rank": 16,
    }
    (out / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    cmd = (
        "accelerate launch train_dreambooth_lora.py "
        f"--pretrained_model_name_or_path='{args.base_model}' "
        f"--instance_data_dir='{data_dir}' "
        f"--output_dir='{out / 'weights'}' "
        f"--instance_prompt='{args.token} manga style' "
        f"--resolution={args.resolution} "
        f"--train_batch_size={args.batch_size} "
        f"--learning_rate={args.learning_rate} "
        f"--max_train_steps={args.max_steps} "
        "--checkpointing_steps=200 --gradient_accumulation_steps=4 --mixed_precision=fp16 --rank=16"
    )
    (out / "train_command.txt").write_text(cmd + "\n", encoding="utf-8")

    print(f"Prepared LoRA job: {out}")
    print(f"Dataset file: {dataset_jsonl}")
    print("Run command from train_command.txt on a GPU machine with diffusers + accelerate installed.")


if __name__ == "__main__":
    main()
