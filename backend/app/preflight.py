from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List

from .config import Settings
from .model_manager import ModelManager
from .model_manifest import verify_manifest


def _model_path_checks(settings: Settings) -> Dict[str, Dict[str, Any]]:
    checks = {
        "manga_ocr": settings.models_dir / "manga_ocr",
        "marian_ja_en": settings.models_dir / "marian_ja_en",
        "marian_ko_en": settings.models_dir / "marian_ko_en",
        "m2m100": settings.models_dir / "m2m100",
        "sd_base": settings.models_dir / "sd_base",
        "sd_inpaint": settings.models_dir / "sd_inpaint",
    }
    status: Dict[str, Dict[str, Any]] = {}
    for key, path in checks.items():
        exists = path.exists()
        status[key] = {
            "path": str(path),
            "exists": exists,
            "size_mb": round(_dir_size_mb(path), 2) if exists else 0.0,
        }
    return status


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024.0 * 1024.0)


def _ready_tier(
    blockers: List[str],
    modules: Dict[str, bool],
    model_paths: Dict[str, Dict[str, Any]],
    device: str,
) -> str:
    if blockers:
        return "blocked"
    has_ocr = modules.get("manga_ocr", False) or modules.get("easyocr", False)
    has_mt = modules.get("transformers", False)
    has_diff = modules.get("diffusers", False) and model_paths.get("sd_base", {}).get("exists", False)
    has_inpaint = modules.get("lama_cleaner", False) or model_paths.get("sd_inpaint", {}).get("exists", False)
    if has_ocr and has_mt and has_diff and has_inpaint and device == "cuda":
        return "full_pro"
    if has_ocr and has_mt:
        return "balanced_local"
    return "fallback_only"


def _score(blockers: List[str], warnings: List[str], tier: str) -> int:
    base = {
        "full_pro": 100,
        "balanced_local": 72,
        "fallback_only": 45,
        "blocked": 20,
    }.get(tier, 20)
    penalty = min(60, len(blockers) * 12 + len(warnings) * 3)
    return max(0, base - penalty)


def build_preflight_report(settings: Settings, model_manager: ModelManager) -> Dict[str, Any]:
    capabilities = model_manager.capabilities(refresh=True)
    modules: Dict[str, bool] = capabilities.get("modules", {})
    device = str(capabilities.get("device", "cpu"))
    gpu = capabilities.get("gpu", {"device": "cpu"})
    model_paths = _model_path_checks(settings)
    manifest_report = verify_manifest(
        models_dir=settings.models_dir,
        manifest_path=settings.model_manifest_path,
        include_file_hashes=False,
        hash_max_mb=64,
    )

    blockers: List[str] = []
    warnings: List[str] = []
    actions: List[str] = []

    if settings.strict_require_gpu and device != "cuda":
        blockers.append("GPU is required by strict settings but CUDA device is not available.")
    elif device != "cuda":
        warnings.append("Running on CPU. Final diffusion renders will be slower.")

    if not modules.get("torch", False):
        blockers.append("PyTorch is not available.")
    if not modules.get("transformers", False):
        blockers.append("Transformers is not available.")
    if not modules.get("manga_ocr", False) and not modules.get("easyocr", False):
        blockers.append("No OCR backend available (manga_ocr/easyocr missing).")

    if settings.local_models_only:
        missing_paths = [k for k, meta in model_paths.items() if not bool(meta.get("exists", False))]
        if missing_paths:
            blockers.append(
                "Local model folders missing: " + ", ".join(missing_paths)
            )
            actions.append(
                "python backend/scripts/download_models.py --all --models-dir backend/models"
            )
        if not manifest_report.get("manifest_exists", False):
            warnings.append("Model manifest not found. Generate it for deterministic production verification.")
            actions.append("python backend/scripts/model_manifest.py --write --models-dir backend/models")
        elif not manifest_report.get("matches", False):
            warnings.append("Model manifest mismatch detected. Run verification and sync models before release.")
            actions.append("python backend/scripts/model_manifest.py --verify --models-dir backend/models")
            if settings.strict_pro_mode:
                blockers.append("Model manifest mismatch in strict mode.")

    if not modules.get("diffusers", False):
        if settings.strict_require_diffusion:
            blockers.append("Diffusers is required by strict settings.")
        else:
            warnings.append("Diffusers unavailable. Style transfer will use fallback filters.")

    if not modules.get("realesrgan", False):
        warnings.append("Real-ESRGAN unavailable. Upscaling will use PIL fallback.")
    if not modules.get("lama_cleaner", False):
        warnings.append("LaMa unavailable. Inpainting will use OpenCV/SD fallback.")
    if not modules.get("facenet_pytorch", False):
        warnings.append("Face embedding backend unavailable. Identity enforcement will be limited.")
    if not settings.enable_episodic_memory:
        warnings.append("Episodic character memory is disabled.")
    if not settings.enable_repair_planner:
        warnings.append("Repair planner is disabled.")
    if not settings.enable_adaptive_controller:
        warnings.append("Adaptive controller is disabled.")

    tier = _ready_tier(blockers, modules, model_paths, device=device)
    readiness_score = _score(blockers, warnings, tier=tier)

    if settings.local_models_only and not actions:
        actions.append("Local model set is present. Keep backend/models persisted for offline runs.")
    if not settings.local_models_only:
        actions.append("Set LOCAL_MODELS_ONLY=true for deterministic offline production behavior.")

    report = {
        "app": settings.app_name,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "device": device,
        "gpu": gpu,
        "strict_mode": settings.strict_pro_mode,
        "strict_flags": {
            "strict_require_gpu": settings.strict_require_gpu,
            "strict_require_diffusion": settings.strict_require_diffusion,
            "strict_require_ocr": settings.strict_require_ocr,
            "strict_require_translation_models": settings.strict_require_translation_models,
        },
        "feature_flags": {
            "enable_scene_memory": settings.enable_scene_memory,
            "enable_episodic_memory": settings.enable_episodic_memory,
            "enable_repair_planner": settings.enable_repair_planner,
            "enable_adaptive_controller": settings.enable_adaptive_controller,
            "enforce_identity_consistency": settings.enforce_identity_consistency,
        },
        "local_models_only": settings.local_models_only,
        "modules": modules,
        "model_paths": model_paths,
        "model_manifest": manifest_report,
        "tier": tier,
        "readiness_score": readiness_score,
        "blockers": blockers,
        "warnings": warnings,
        "actions": actions,
    }
    return report
