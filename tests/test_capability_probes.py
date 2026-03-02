from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
SCRIPTS_DIR = BACKEND_DIR / "scripts"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_pro_stack as verify_stack  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.model_manager import ModelManager  # noqa: E402


def test_model_manager_runtime_probe_blocks_broken_realesrgan(monkeypatch):
    settings = get_settings()
    mm = ModelManager(settings=settings)
    mm._capability_cache.clear()

    monkeypatch.setattr("app.model_manager.importlib.util.find_spec", lambda _: object())

    def _fake_import(module_name: str):
        if module_name == "realesrgan":
            raise ImportError("broken transitive dependency")
        return object()

    monkeypatch.setattr("app.model_manager.importlib.import_module", _fake_import)
    assert mm.module_available("realesrgan") is False


def test_verify_stack_runtime_probe_blocks_broken_realesrgan(monkeypatch):
    monkeypatch.setattr(verify_stack.importlib.util, "find_spec", lambda _: object())

    def _fake_import(module_name: str):
        if module_name == "realesrgan":
            raise ImportError("broken transitive dependency")
        return object()

    monkeypatch.setattr(verify_stack.importlib, "import_module", _fake_import)
    assert verify_stack.has_module("realesrgan") is False
    assert verify_stack.has_module("opencv_python") is True
