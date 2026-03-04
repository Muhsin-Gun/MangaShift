from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run_cmd(cmd: List[str], timeout: int = 12) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "cmd": cmd,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _find_nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found

    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
            Path(r"C:\Windows\System32\nvidia-smi.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return ""


def _torch_cuda_info() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "import_ok": False,
        "torch_version": "",
        "torch_cuda_version": "",
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "error": "",
    }
    try:
        import torch  # noqa: WPS433

        payload["import_ok"] = True
        payload["torch_version"] = str(torch.__version__)
        payload["torch_cuda_version"] = str(torch.version.cuda or "")
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count())
        names: List[str] = []
        for idx in range(int(torch.cuda.device_count())):
            try:
                names.append(str(torch.cuda.get_device_name(idx)))
            except Exception:
                names.append(f"cuda:{idx}")
        payload["device_names"] = names
    except Exception as exc:  # pragma: no cover
        payload["error"] = str(exc)
    return payload


def collect_report() -> Dict[str, Any]:
    torch_info = _torch_cuda_info()
    nvidia_smi_path = _find_nvidia_smi()
    smi_probe = (
        _run_cmd([nvidia_smi_path, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if nvidia_smi_path
        else {
            "ok": False,
            "cmd": [],
            "exit_code": -1,
            "stdout": "",
            "stderr": "nvidia-smi not found",
        }
    )

    colab_env = {
        "is_colab": bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")),
        "COLAB_GPU": str(os.environ.get("COLAB_GPU", "")),
        "COLAB_RELEASE_TAG": str(os.environ.get("COLAB_RELEASE_TAG", "")),
    }

    gpu_ready = bool(torch_info.get("cuda_available")) and int(torch_info.get("device_count", 0)) > 0
    suggestions: List[str] = []
    if not nvidia_smi_path:
        suggestions.append("Install NVIDIA drivers / CUDA runtime or run on a GPU-enabled machine.")
    if not gpu_ready:
        suggestions.append("Use render-quality=balanced with --allow-cpu for local CPU fallback.")
        suggestions.append("For strict quality/final path, configure QUALITY_CLOUD_WORKER_URL or use local CUDA GPU.")
    if colab_env["is_colab"] and not gpu_ready:
        suggestions.append("In Colab: Runtime -> Change runtime type -> GPU, then restart runtime.")

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "nvidia_smi": {
            "path": nvidia_smi_path,
            "probe": smi_probe,
        },
        "torch_cuda": torch_info,
        "colab": colab_env,
        "gpu_ready": gpu_ready,
        "suggestions": suggestions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CUDA/GPU runtime readiness for MangaShift.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=== CUDA Diagnose ===")
    print(f"python: {report['python']['version']} ({report['python']['executable']})")
    print(f"platform: {report['platform']['system']} {report['platform']['release']} {report['platform']['machine']}")
    print(f"nvidia_smi_path: {report['nvidia_smi']['path']}")
    print(f"torch_import_ok: {report['torch_cuda']['import_ok']}")
    print(f"torch_version: {report['torch_cuda']['torch_version']}")
    print(f"torch_cuda_version: {report['torch_cuda']['torch_cuda_version']}")
    print(f"cuda_available: {report['torch_cuda']['cuda_available']}")
    print(f"device_count: {report['torch_cuda']['device_count']}")
    print(f"device_names: {report['torch_cuda']['device_names']}")
    print(f"gpu_ready: {report['gpu_ready']}")
    if report["suggestions"]:
        print("suggestions:")
        for item in report["suggestions"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
