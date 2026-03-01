from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.character_db import CharacterDatabase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-series character registry from pages.")
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=ROOT / "cache" / "character_registry.json")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    args = parser.parse_args()

    images = [p for p in args.input_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not images:
        raise SystemExit(f"No pages found in {args.input_dir}")

    pages = [Image.open(path).convert("RGB") for path in images]
    db = CharacterDatabase(db_path=args.db_path)
    db.build_series_registry(args.series_id, pages, min_cluster_size=args.min_cluster_size)
    print(f"Character DB updated: {args.db_path}")


if __name__ == "__main__":
    main()
