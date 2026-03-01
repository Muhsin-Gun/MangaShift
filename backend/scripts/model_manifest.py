from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_manifest import (  # noqa: E402
    REQUIRED_MODEL_DIRS,
    build_manifest,
    verify_manifest,
    write_manifest,
)


def _print_human(payload: dict) -> None:
    print("=== Model Manifest ===")
    print(f"manifest_path: {payload.get('manifest_path', '')}")
    print(f"matches: {payload.get('matches', False)}")
    missing = payload.get("missing_required_dirs", [])
    print(f"missing_required_dirs: {missing}")
    changed = payload.get("changed_dirs", {})
    print(f"changed_dirs: {list(changed.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/verify local model manifest for strict deployments.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Model directory containing downloaded folders.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "models" / "manifest.json",
        help="Manifest file path.",
    )
    parser.add_argument("--write", action="store_true", help="Write manifest from current model directory.")
    parser.add_argument("--verify", action="store_true", help="Verify current model directory against manifest.")
    parser.add_argument(
        "--include-file-hashes",
        action="store_true",
        help="Hash file contents (slower, stronger integrity).",
    )
    parser.add_argument(
        "--hash-max-mb",
        type=int,
        default=64,
        help="Skip content hashing for files larger than this size in MB.",
    )
    parser.add_argument(
        "--required",
        nargs="*",
        default=list(REQUIRED_MODEL_DIRS),
        help="Required model directories to track.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    if not args.write and not args.verify:
        parser.error("Select at least one action: --write or --verify")

    result: dict = {"models_dir": str(args.models_dir), "manifest_path": str(args.manifest_path)}

    if args.write:
        manifest = build_manifest(
            models_dir=args.models_dir,
            required_dirs=args.required,
            include_file_hashes=args.include_file_hashes,
            hash_max_mb=args.hash_max_mb,
        )
        write_manifest(manifest, args.manifest_path)
        result["write"] = {
            "ok": True,
            "generated_at": manifest.get("generated_at", ""),
            "required_dirs": manifest.get("required_dirs", []),
            "totals": manifest.get("totals", {}),
        }

    if args.verify:
        verification = verify_manifest(
            models_dir=args.models_dir,
            manifest_path=args.manifest_path,
            include_file_hashes=args.include_file_hashes,
            hash_max_mb=args.hash_max_mb,
        )
        result["verify"] = verification

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if args.write:
            print(f"Manifest written: {args.manifest_path}")
        if args.verify:
            _print_human(result["verify"])

    if args.verify and not result["verify"].get("matches", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
