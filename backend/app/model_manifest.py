from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REQUIRED_MODEL_DIRS: tuple[str, ...] = (
    "manga_ocr",
    "marian_ja_en",
    "marian_ko_en",
    "m2m100",
    "sd_base",
    "sd_inpaint",
)

_SAMPLE_FILE_COUNT = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return ()
    return (item for item in path.rglob("*") if item.is_file())


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _quick_signature(files: Sequence[Path], model_dir: Path) -> str:
    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(model_dir).as_posix()
        stat = file_path.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _full_signature(
    files: Sequence[Path],
    model_dir: Path,
    hash_max_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    skipped = 0
    for file_path in files:
        rel = file_path.relative_to(model_dir).as_posix()
        stat = file_path.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        if stat.st_size > hash_max_bytes:
            skipped += 1
            digest.update(b"SKIPPED_LARGE_FILE")
            continue
        digest.update(_file_digest(file_path).encode("utf-8"))
    return digest.hexdigest(), skipped


def collect_model_inventory(
    models_dir: Path,
    required_dirs: Sequence[str] = REQUIRED_MODEL_DIRS,
    include_file_hashes: bool = False,
    hash_max_mb: int = 64,
) -> Dict[str, Any]:
    inventory: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    total_files = 0
    hash_max_bytes = int(hash_max_mb * 1024 * 1024)

    for model_key in required_dirs:
        model_dir = models_dir / model_key
        exists = model_dir.exists() and model_dir.is_dir()
        entry: Dict[str, Any] = {
            "path": str(model_dir),
            "exists": bool(exists),
            "file_count": 0,
            "total_bytes": 0,
            "total_mb": 0.0,
            "signature": "",
            "hash_mode": "full" if include_file_hashes else "quick",
            "skipped_large_files": 0,
            "sample_files": [],
        }
        if exists:
            files = sorted(_iter_files(model_dir), key=lambda p: p.as_posix())
            entry["file_count"] = len(files)
            size = sum(item.stat().st_size for item in files)
            entry["total_bytes"] = size
            entry["total_mb"] = round(size / (1024.0 * 1024.0), 3)
            entry["sample_files"] = [
                file_path.relative_to(model_dir).as_posix() for file_path in files[:_SAMPLE_FILE_COUNT]
            ]
            if include_file_hashes:
                signature, skipped = _full_signature(files, model_dir=model_dir, hash_max_bytes=hash_max_bytes)
                entry["signature"] = signature
                entry["skipped_large_files"] = skipped
            else:
                entry["signature"] = _quick_signature(files, model_dir=model_dir)
            total_bytes += size
            total_files += len(files)
        inventory[model_key] = entry

    missing = [name for name, payload in inventory.items() if not payload.get("exists", False)]
    return {
        "models_dir": str(models_dir),
        "required_dirs": list(required_dirs),
        "include_file_hashes": bool(include_file_hashes),
        "hash_max_mb": int(hash_max_mb),
        "generated_at": _utc_now(),
        "inventory": inventory,
        "totals": {
            "model_count": len(required_dirs),
            "missing_count": len(missing),
            "file_count": total_files,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / (1024.0**3), 3),
        },
    }


def build_manifest(
    models_dir: Path,
    required_dirs: Sequence[str] = REQUIRED_MODEL_DIRS,
    include_file_hashes: bool = False,
    hash_max_mb: int = 64,
) -> Dict[str, Any]:
    payload = collect_model_inventory(
        models_dir=models_dir,
        required_dirs=required_dirs,
        include_file_hashes=include_file_hashes,
        hash_max_mb=hash_max_mb,
    )
    return {
        "schema_version": 1,
        "generated_at": payload["generated_at"],
        "models_dir": payload["models_dir"],
        "required_dirs": payload["required_dirs"],
        "include_file_hashes": payload["include_file_hashes"],
        "hash_max_mb": payload["hash_max_mb"],
        "totals": payload["totals"],
        "inventory": payload["inventory"],
    }


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def diff_manifest(current: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    current_inventory = current.get("inventory", {})
    expected_inventory = expected.get("inventory", {})
    required_dirs = expected.get("required_dirs") or current.get("required_dirs") or []
    changed: Dict[str, Dict[str, Any]] = {}
    missing_required: List[str] = []

    for model_key in required_dirs:
        cur = current_inventory.get(model_key, {})
        exp = expected_inventory.get(model_key, {})
        cur_exists = bool(cur.get("exists", False))
        exp_exists = bool(exp.get("exists", False))
        issues: List[str] = []

        if exp_exists and not cur_exists:
            issues.append("missing_directory")
            missing_required.append(model_key)
        if cur_exists != exp_exists:
            issues.append("exists_mismatch")
        if cur.get("file_count") != exp.get("file_count"):
            issues.append("file_count_mismatch")
        if cur.get("total_bytes") != exp.get("total_bytes"):
            issues.append("size_mismatch")
        if cur.get("signature") != exp.get("signature"):
            issues.append("signature_mismatch")

        if issues:
            changed[model_key] = {
                "issues": sorted(set(issues)),
                "current": {
                    "exists": cur.get("exists", False),
                    "file_count": cur.get("file_count", 0),
                    "total_bytes": cur.get("total_bytes", 0),
                    "signature": cur.get("signature", ""),
                },
                "expected": {
                    "exists": exp.get("exists", False),
                    "file_count": exp.get("file_count", 0),
                    "total_bytes": exp.get("total_bytes", 0),
                    "signature": exp.get("signature", ""),
                },
            }

    return {
        "matches": len(changed) == 0,
        "required_dirs": list(required_dirs),
        "missing_required_dirs": sorted(set(missing_required)),
        "changed_dirs": changed,
        "current_generated_at": current.get("generated_at", ""),
        "expected_generated_at": expected.get("generated_at", ""),
    }


def verify_manifest(
    models_dir: Path,
    manifest_path: Path,
    include_file_hashes: bool = False,
    hash_max_mb: int = 64,
) -> Dict[str, Any]:
    expected = read_manifest(manifest_path)
    if expected is None:
        return {
            "manifest_exists": False,
            "manifest_path": str(manifest_path),
            "matches": False,
            "missing_required_dirs": [],
            "changed_dirs": {},
            "required_dirs": list(REQUIRED_MODEL_DIRS),
            "error": "manifest_not_found",
        }

    required_dirs = expected.get("required_dirs") or list(REQUIRED_MODEL_DIRS)
    current = build_manifest(
        models_dir=models_dir,
        required_dirs=required_dirs,
        include_file_hashes=include_file_hashes,
        hash_max_mb=hash_max_mb,
    )
    diff = diff_manifest(current=current, expected=expected)
    return {
        "manifest_exists": True,
        "manifest_path": str(manifest_path),
        "matches": bool(diff.get("matches", False)),
        "required_dirs": list(required_dirs),
        "missing_required_dirs": diff.get("missing_required_dirs", []),
        "changed_dirs": diff.get("changed_dirs", {}),
        "current_generated_at": diff.get("current_generated_at", ""),
        "expected_generated_at": diff.get("expected_generated_at", ""),
    }
