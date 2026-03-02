from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REQUIRED_CORE = [
    "torch",
    "transformers",
    "manga_ocr",
    "easyocr",
    "diffusers",
    "controlnet_aux",
    "facenet_pytorch",
]

RECOMMENDED = [
    "lama_cleaner",
    "realesrgan",
    "xformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "opencv_python",
    "PIL",
]


@dataclass
class ModuleStatus:
    name: str
    available: bool


def has_module(module_name: str) -> bool:
    if module_name == "opencv_python":
        module_name = "cv2"
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    if module_name in {"realesrgan", "facenet_pytorch"}:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


def maybe_reexec_cuda_venv() -> None:
    if os.environ.get("MANGASHIFT_REEXEC_DONE") == "1":
        return
    if os.environ.get("MANGASHIFT_DISABLE_AUTO_PYTHON") == "1":
        return

    backend_dir = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        candidate = backend_dir / ".venv_cuda" / "Scripts" / "python.exe"
    else:
        candidate = backend_dir / ".venv_cuda" / "bin" / "python"
    if not candidate.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        target = candidate.resolve()
    except Exception:
        return
    if current == target:
        return

    # Prefer the known-good venv when accidentally launched with global Python (often 3.13+).
    if sys.version_info < (3, 13):
        return
    env = dict(os.environ)
    env["MANGASHIFT_REEXEC_DONE"] = "1"
    proc = subprocess.run([str(target), __file__, *sys.argv[1:]], env=env, check=False)
    raise SystemExit(int(proc.returncode))


def local_model_files_status() -> Dict[str, object]:
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        from app.config import get_settings
        from app.model_manifest import verify_manifest

        settings = get_settings()
        checks = {
            "manga_ocr": (settings.models_dir / "manga_ocr").exists(),
            "marian_ja_en": (settings.models_dir / "marian_ja_en").exists(),
            "marian_ko_en": (settings.models_dir / "marian_ko_en").exists(),
            "m2m100": (settings.models_dir / "m2m100").exists(),
            "sd_base": (settings.models_dir / "sd_base").exists(),
            "sd_inpaint": (settings.models_dir / "sd_inpaint").exists(),
        }
        missing = [name for name, ok in checks.items() if not ok]
        manifest = verify_manifest(
            models_dir=settings.models_dir,
            manifest_path=settings.model_manifest_path,
            include_file_hashes=False,
            hash_max_mb=64,
        )
        return {
            "local_models_only": bool(settings.local_models_only),
            "models_dir": str(settings.models_dir),
            "required_model_paths": checks,
            "missing_required_model_paths": missing if settings.local_models_only else [],
            "manifest": manifest,
        }
    except Exception as exc:
        return {
            "local_models_only": False,
            "models_dir": "",
            "required_model_paths": {},
            "missing_required_model_paths": [],
            "manifest": {},
            "error": str(exc),
        }


def python_status() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "executable": sys.executable,
    }


def gpu_status() -> Dict[str, str | bool]:
    result = {"cuda_available": False, "torch_cuda_version": "", "device_name": ""}
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda_version"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            result["device_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return result


def collect_status() -> Dict[str, object]:
    core = [ModuleStatus(name=m, available=has_module(m)) for m in REQUIRED_CORE]
    rec = [ModuleStatus(name=m, available=has_module(m)) for m in RECOMMENDED]
    model_files = local_model_files_status()
    quality_runtime = {
        "local_cuda_ready": bool(gpu_status().get("cuda_available", False)),
        "quality_cloud_worker_configured": False,
        "quality_cloud_worker_url": "",
        "quality_mode_ready": False,
    }
    try:
        backend_dir = Path(__file__).resolve().parents[1]
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from app.config import get_settings  # type: ignore

        cfg = get_settings()
        worker_url = str(cfg.quality_cloud_worker_url or "").strip()
        local_cuda_ready = bool(quality_runtime["local_cuda_ready"])
        quality_runtime.update(
            {
                "local_cuda_ready": local_cuda_ready,
                "quality_cloud_worker_configured": bool(worker_url),
                "quality_cloud_worker_url": worker_url,
                "quality_mode_ready": bool(local_cuda_ready or bool(worker_url)),
            }
        )
    except Exception as exc:
        quality_runtime["error"] = str(exc)
    return {
        "python": python_status(),
        "gpu": gpu_status(),
        "core_modules": [{"name": m.name, "available": m.available} for m in core],
        "recommended_modules": [{"name": m.name, "available": m.available} for m in rec],
        "missing_core": [m.name for m in core if not m.available],
        "missing_recommended": [m.name for m in rec if not m.available],
        "model_files": model_files,
        "quality_runtime": quality_runtime,
    }


def pip_install(packages: List[str]) -> int:
    if not packages:
        return 0
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print("Installing:", " ".join(packages))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def install_missing(status: Dict[str, object]) -> int:
    mapping = {
        "torch": ["torch", "torchvision", "torchaudio"],
        "transformers": ["transformers", "sentencepiece", "tokenizers", "accelerate"],
        "manga_ocr": ["manga-ocr"],
        "easyocr": ["easyocr"],
        "diffusers": ["diffusers", "safetensors"],
        "controlnet_aux": ["controlnet-aux"],
        "facenet_pytorch": ["facenet-pytorch", "--no-deps"],
        "realesrgan": ["realesrgan", "basicsr"],
    }
    failures = 0
    for module_name in status["missing_core"]:  # type: ignore[index]
        pkg_list = mapping.get(module_name, [module_name])
        code = pip_install(pkg_list)
        if code != 0:
            failures += 1
    return failures


def strict_ready(status: Dict[str, object]) -> bool:
    missing = status["missing_core"]  # type: ignore[index]
    model_files = status.get("model_files", {})
    missing_paths = (
        model_files.get("missing_required_model_paths", [])
        if isinstance(model_files, dict)
        else []
    )
    manifest = model_files.get("manifest", {}) if isinstance(model_files, dict) else {}
    manifest_matches = bool(manifest.get("matches", False)) if isinstance(manifest, dict) else False
    return len(missing) == 0 and len(missing_paths) == 0 and manifest_matches


def main() -> None:
    maybe_reexec_cuda_venv()
    parser = argparse.ArgumentParser(description="Verify/prepare MangaShift strict production stack.")
    parser.add_argument("--install-missing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    status = collect_status()
    if args.install_missing and status["missing_core"]:  # type: ignore[index]
        failures = install_missing(status)
        status = collect_status()
        status["install_failures"] = failures

    status["strict_ready"] = strict_ready(status)
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    print("=== MangaShift Pro Stack Verification ===")
    print(f"Python: {status['python']['python_version']} ({status['python']['executable']})")
    gpu = status["gpu"]
    print(
        "CUDA: "
        f"{gpu['cuda_available']} | "
        f"Torch CUDA: {gpu['torch_cuda_version']} | "
        f"Device: {gpu['device_name']}"
    )
    print("Missing core:", status["missing_core"])
    print("Missing recommended:", status["missing_recommended"])
    print("Quality runtime:", status.get("quality_runtime", {}))
    print("Strict ready:", status["strict_ready"])


if __name__ == "__main__":
    main()
