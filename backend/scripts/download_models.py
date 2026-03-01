from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"huggingface-hub is required: {exc}")


MODEL_MAP: Dict[str, str] = {
    "manga_ocr": "kha-white/manga-ocr-base",
    "marian_ja_en": "Helsinki-NLP/opus-mt-ja-en",
    "marian_ko_en": "Helsinki-NLP/opus-mt-ko-en",
    "m2m100": "facebook/m2m100_418M",
    "sd_base": "runwayml/stable-diffusion-v1-5",
    "sd_inpaint": "runwayml/stable-diffusion-inpainting",
    "controlnet_canny": "lllyasviel/sd-controlnet-canny",
    "controlnet_depth": "lllyasviel/sd-controlnet-depth",
    "lama": "Sanster/LaMa",
}

REQUIRED_CORE_KEYS = [
    "manga_ocr",
    "marian_ja_en",
    "marian_ko_en",
    "m2m100",
    "sd_base",
    "sd_inpaint",
]

MODEL_ALLOW_PATTERNS = {
    "manga_ocr": ["*.json", "*.model", "*.bin", "*.safetensors", "*.txt"],
    "marian_ja_en": ["*.json", "*.bin", "*.model", "*.txt", "*.vocab", "*.spm", "*.safetensors"],
    "marian_ko_en": ["*.json", "*.bin", "*.model", "*.txt", "*.vocab", "*.spm", "*.safetensors"],
    "m2m100": ["*.json", "*.bin", "*.model", "*.txt", "*.spm", "*.safetensors"],
    "sd_base": ["*.json", "*.bin", "*.safetensors", "*.txt", "*.model", "*.md", "*.yaml"],
    "sd_inpaint": ["*.json", "*.bin", "*.safetensors", "*.txt", "*.model", "*.md", "*.yaml"],
    "controlnet_canny": ["*.json", "*.bin", "*.safetensors", "*.txt", "*.yaml", "*.md"],
    "controlnet_depth": ["*.json", "*.bin", "*.safetensors", "*.txt", "*.yaml", "*.md"],
    "lama": ["*.json", "*.bin", "*.pth", "*.yaml", "*.txt", "*.md"],
}


def _has_any_model_file(path: Path) -> bool:
    if not path.exists():
        return False
    for file in path.rglob("*"):
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix in {".bin", ".safetensors", ".pth", ".pt", ".model", ".json", ".yaml"}:
                return True
    return False


def _download_single(
    key: str,
    repo_id: str,
    output_dir: Path,
    hf_token: str | None,
    force: bool,
) -> tuple[bool, str]:
    if output_dir.exists() and _has_any_model_file(output_dir) and not force:
        logger.info("[{}] already present: {}", key, output_dir)
        return True, "skipped_existing"

    output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = MODEL_ALLOW_PATTERNS.get(key)
    logger.info("[{}] downloading {} -> {}", key, repo_id, output_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True,
            max_workers=4,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        logger.error("[{}] download failed: {}", key, exc)
        return False, f"download_error:{exc}"

    if not _has_any_model_file(output_dir):
        logger.error("[{}] download completed but model files not found in {}", key, output_dir)
        return False, "missing_model_files"
    return True, "ok"


def _resolve_keys(args: argparse.Namespace) -> List[str]:
    if args.required:
        return list(REQUIRED_CORE_KEYS)
    if args.all or not args.only:
        return list(MODEL_MAP.keys())
    return list(args.only)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MangaShift model weights with verification.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Destination directory for model folders.",
    )
    parser.add_argument("--all", action="store_true", help="Download all known models.")
    parser.add_argument("--required", action="store_true", help="Download strict core models only.")
    parser.add_argument("--only", nargs="*", default=[], help=f"Subset keys: {', '.join(MODEL_MAP.keys())}")
    parser.add_argument("--force", action="store_true", help="Redownload even when files already exist.")
    parser.add_argument(
        "--hf-token-env",
        default="HUGGINGFACE_HUB_TOKEN",
        help="Environment variable name holding HF token for gated/private models.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failed model.")
    args = parser.parse_args()

    keys = _resolve_keys(args)
    unknown = [k for k in keys if k not in MODEL_MAP]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}")

    token = os.environ.get(args.hf_token_env, "").strip() or None
    if token:
        logger.info("Using Hugging Face token from env var '{}'", args.hf_token_env)
    else:
        logger.info("No Hugging Face token found in env var '{}'", args.hf_token_env)

    args.models_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, str | bool]] = {}
    failures = 0
    for key in keys:
        ok, reason = _download_single(
            key=key,
            repo_id=MODEL_MAP[key],
            output_dir=args.models_dir / key,
            hf_token=token,
            force=args.force,
        )
        summary[key] = {"ok": ok, "reason": reason}
        if not ok:
            failures += 1
            if args.fail_fast:
                break

    logger.info("=== download summary ===")
    for key in keys:
        if key not in summary:
            continue
        item = summary[key]
        logger.info("{} -> ok={} reason={}", key, item["ok"], item["reason"])

    if failures > 0:
        logger.error("Model download finished with {} failure(s).", failures)
        sys.exit(1)
    logger.info("All requested model downloads completed successfully.")


if __name__ == "__main__":
    main()
