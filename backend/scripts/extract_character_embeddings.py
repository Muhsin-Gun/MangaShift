from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.character_db import CharacterDatabase  # noqa: E402


def list_images(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]


def fallback_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = int(min(w, h) * 0.45)
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    return image.crop((x1, y1, x1 + side, y1 + side)).resize((160, 160))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract character crops + embeddings from manga pages.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--series-id", type=str, default="default_series")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    args = parser.parse_args()

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    db = CharacterDatabase(db_path=args.output_dir / "character_registry.json")
    all_embs = []
    all_refs = []
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(tqdm(images, desc="Extracting faces")):
        page = Image.open(path).convert("RGB")
        faces = db.detect_faces(page)
        if not faces:
            faces = [(fallback_crop(page), (0, 0, page.width, page.height))]
        for j, (face, bbox) in enumerate(faces):
            emb = db.embed_face(face)
            crop_path = crops_dir / f"{path.stem}_{idx}_{j}.png"
            face.save(crop_path)
            all_embs.append(emb)
            all_refs.append({"image": str(path), "crop": str(crop_path), "bbox": bbox})

    if not all_embs:
        raise SystemExit("No embeddings generated")

    emb_array = np.vstack(all_embs).astype(np.float32)
    cluster = DBSCAN(eps=0.45, min_samples=args.min_cluster_size, metric="cosine")
    labels = cluster.fit_predict(emb_array)

    chars_dir = args.output_dir / "characters" / args.series_id
    chars_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"series_id": args.series_id, "characters": {}}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idxs = np.where(labels == label)[0]
        char_dir = chars_dir / f"character_{label}"
        char_dir.mkdir(parents=True, exist_ok=True)
        char_embs = emb_array[idxs]
        np.save(char_dir / "embeddings.npy", char_embs)
        refs = []
        for i, idx_ref in enumerate(idxs[:20]):
            dst = char_dir / f"ref_{i:03d}.png"
            Image.open(all_refs[idx_ref]["crop"]).save(dst)
            refs.append(str(dst))
        metadata["characters"][f"character_{label}"] = {
            "count": int(len(idxs)),
            "refs": refs,
            "embedding_path": str(char_dir / "embeddings.npy"),
        }

    (args.output_dir / "clusters.npy").write_bytes(labels.astype(np.int32).tobytes())
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved embeddings + clusters to: {args.output_dir}")


if __name__ == "__main__":
    main()
