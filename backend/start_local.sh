#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODELS_DIR="${ROOT_DIR}/models"

echo "[MangaShift] Root: ${ROOT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "[MangaShift] Python not found in PATH"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[MangaShift] Creating virtual environment"
  python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[MangaShift] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[MangaShift] Installing Python dependencies"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install facenet-pytorch --no-deps

echo "[MangaShift] Dependency integrity check (advisory)"
python -m pip check || echo "[MangaShift] WARNING: pip check reported advisory conflicts. Continuing startup."

mkdir -p "${MODELS_DIR}"
mkdir -p "${ROOT_DIR}/cache"

echo "[MangaShift] Pre-downloading translation models (best effort)"
python - <<'PY'
from transformers import MarianMTModel, MarianTokenizer

MODELS = [
    "Helsinki-NLP/opus-mt-ja-en",
    "Helsinki-NLP/opus-mt-ko-en",
]

for model in MODELS:
    try:
        MarianTokenizer.from_pretrained(model)
        MarianMTModel.from_pretrained(model)
        print(f"[MangaShift] Downloaded: {model}")
    except Exception as exc:
        print(f"[MangaShift] WARNING: failed to pre-download {model}: {exc}")
PY

echo "[MangaShift] Starting API server on http://127.0.0.1:8000"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --app-dir "${ROOT_DIR}"
