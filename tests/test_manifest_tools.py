from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.model_manifest import (  # noqa: E402
    REQUIRED_MODEL_DIRS,
    build_manifest,
    verify_manifest,
    write_manifest,
)


def _seed_fake_models(models_dir: Path) -> None:
    for key in REQUIRED_MODEL_DIRS:
        folder = models_dir / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps({"id": key}), encoding="utf-8")
        (folder / "weights.bin").write_bytes(b"\x00\x01\x02" + key.encode("utf-8"))


def test_manifest_roundtrip(tmp_path: Path):
    models_dir = tmp_path / "models"
    manifest_path = models_dir / "manifest.json"
    _seed_fake_models(models_dir)

    manifest = build_manifest(models_dir=models_dir, include_file_hashes=False)
    write_manifest(manifest, manifest_path)

    verification = verify_manifest(
        models_dir=models_dir,
        manifest_path=manifest_path,
        include_file_hashes=False,
    )
    assert verification["manifest_exists"] is True
    assert verification["matches"] is True
    assert verification["missing_required_dirs"] == []


def test_manifest_detects_changes(tmp_path: Path):
    models_dir = tmp_path / "models"
    manifest_path = models_dir / "manifest.json"
    _seed_fake_models(models_dir)
    manifest = build_manifest(models_dir=models_dir, include_file_hashes=False)
    write_manifest(manifest, manifest_path)

    target = models_dir / REQUIRED_MODEL_DIRS[0] / "weights.bin"
    target.write_bytes(target.read_bytes() + b"delta")

    verification = verify_manifest(
        models_dir=models_dir,
        manifest_path=manifest_path,
        include_file_hashes=False,
    )
    assert verification["manifest_exists"] is True
    assert verification["matches"] is False
    assert REQUIRED_MODEL_DIRS[0] in verification["changed_dirs"]


def test_manifest_missing_file(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_manifest(
        models_dir=models_dir,
        manifest_path=models_dir / "manifest.json",
        include_file_hashes=False,
    )
    assert verification["manifest_exists"] is False
    assert verification["matches"] is False


def test_smoke_synthetic_image_builder():
    SCRIPT_DIR = ROOT / "backend" / "scripts"
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    import run_strict_smoke as strict_smoke  # noqa: E402

    image_bytes = strict_smoke.build_synthetic_panel()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    assert image.size[0] > 100
    assert image.size[1] > 100
