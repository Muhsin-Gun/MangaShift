#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "${ROOT_DIR}/.venv_cuda" ]; then
  VENV_DIR="${ROOT_DIR}/.venv_cuda"
else
  VENV_DIR="${ROOT_DIR}/.venv_pro"
fi

echo "[MangaShift] Pro bootstrap root: ${ROOT_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[MangaShift] Creating .venv_pro"
  python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[MangaShift] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[MangaShift] Installing base requirements"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install facenet-pytorch --no-deps

echo "[MangaShift] Dependency integrity check (advisory)"
python -m pip check || echo "[MangaShift] WARNING: pip check reported advisory conflicts; strict runtime gates still apply."

echo "[MangaShift] Installing/repairing strict-mode stack"
python "${ROOT_DIR}/scripts/verify_pro_stack.py" --install-missing

echo "[MangaShift] Final capability report"
python "${ROOT_DIR}/scripts/verify_pro_stack.py" --json

echo "[MangaShift] Generating and verifying model manifest"
python "${ROOT_DIR}/scripts/model_manifest.py" --write --verify --models-dir "${ROOT_DIR}/models"

echo "[MangaShift] Strict smoke (will fail if strict requirements are not met)"
python "${ROOT_DIR}/scripts/run_strict_smoke.py" --strict-ocr --strict-translation --quality balanced

echo "[MangaShift] To start server in strict mode:"
echo "  export STRICT_PRO_MODE=true"
echo "  uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir ${ROOT_DIR}"
