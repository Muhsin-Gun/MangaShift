# MangaShift AI

Local-first manga/manhwa translation and art-enhancement system:
- `backend/`: FastAPI AI pipeline (OCR -> translation -> inpaint -> upscale -> color/style -> typeset)
- `extension/`: Chrome MV3 extension that intercepts reader images and swaps enhanced output in-place while scrolling
- `tests/`: backend API + module tests

## What Is Implemented

### Backend pipeline (working)
- OCR with fallback chain: `manga-ocr` -> `easyocr` -> `pytesseract`
- Translation with fallback chain: Marian JA/KO models -> M2M100 fallback -> identity fallback
- Text erasure (inpainting): diffusion inpainting if available, OpenCV fallback
- Upscaling: quality-based auto mode with Real-ESRGAN attempt + deterministic PIL fallback
- Colorization for grayscale pages with diffusion attempt + deterministic fallback
- Style transfer presets with diffusion attempt + deterministic style filters
- Typesetting engine with auto-fit and stroke rendering in speech bubble regions
- Disk LRU cache (SHA-256 keyed)
- API endpoints:
  - `GET /health`
  - `GET /capabilities`
  - `GET /preflight` (strict readiness, blockers, actions)
  - `GET /warmup` (on-demand model warmup/readiness for selected quality/style)
  - `POST /warmup` (structured warmup request body)
  - `GET /styles`
  - `GET /styles/full`
  - `GET /quality/readiness` (local GPU/cloud-worker quality runtime readiness)
  - `GET /memory/{series_id}` (episodic character memory summary)
  - `GET /memory` (all series memory summaries)
  - `GET /controller` (adaptive controller telemetry)
  - `GET /controller/{style_id}` (style-specific adaptive controller stats)
  - `POST /process-image` (multipart image upload)

### Extension (working)
- Detects likely manga panel images on page
- Queues processing through service worker with concurrency limit `2`
- Sends panel image to local backend
- Replaces original panel with processed image
- IndexedDB cache for processed panels
- Failure overlay with retry action
- Popup UI:
  - style preset selector
  - source language
  - auto-enhance toggle
  - per-series style memory toggle
  - upscale and colorization controls

## Quick Start (Windows)

1. Install Python 3.10 or 3.11 (recommended), CUDA if using GPU acceleration.
2. Open PowerShell in repo root:
   ```powershell
   cd backend
   py -3.10 -m venv .venv_cuda
   .\.venv_cuda\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
   .\.venv_cuda\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv_cuda\Scripts\python.exe -m pip install facenet-pytorch --no-deps
   $env:MANGASHIFT_REQUIRE_GPU="true"
   .\start_local.ps1
   ```
3. In Chrome:
   - Open `chrome://extensions`
   - Enable `Developer mode`
   - `Load unpacked`
   - Select `extension/`
4. Visit a manga/manhwa reader page, open extension popup, choose style, click `Apply to current tab`.

## Quick Start (Linux/macOS)

```bash
cd backend
chmod +x start_local.sh
python -m pip install facenet-pytorch --no-deps
./start_local.sh
```

Then load `extension/` as unpacked extension in Chromium/Chrome.

## Tests

Run from repository root:
```powershell
python -m pytest tests -q
```

All tests should pass on CPU. GPU-only acceleration paths are optional and lazy-loaded.

## Recommended Hardware

- Minimum dev: 6+ core CPU, 16 GB RAM
- Recommended for real-time style transfer: NVIDIA GPU with 12+ GB VRAM, 32 GB RAM
- High-performance target: 24 GB VRAM class GPU (RTX 4090/A5000)

## Model Preparation

Download model weights:
```powershell
cd backend
python scripts\download_models.py --all
python scripts\export_default_styles.py
```

See [MODELS.md](backend/MODELS.md) for exact model IDs and usage.

Generate or verify deterministic model manifest:
```powershell
python backend\scripts\model_manifest.py --write --models-dir backend\models --verify
```

Run strict deployment smoke gate (TestClient, no external server):
```powershell
python backend\scripts\run_strict_smoke.py --strict-diffusion --strict-ocr --strict-translation --quality final --style cinematic --output-dir backend\cache\strict_smoke
```

## Notes

- Default mode is local-first. For strict `render_quality=quality`, execution is GPU-only:
  - local CUDA GPU, or
  - forwarded to a configured cloud worker via `QUALITY_CLOUD_WORKER_URL`.
- Use `GET /quality/readiness` to verify quality runtime availability before long runs.
- Diffusion and heavy enhancement steps automatically degrade to deterministic fallbacks if models are missing.
- Set `STRICT_PRO_MODE=true` in backend `.env` to block fallback rendering and fail fast when pro dependencies are missing.
- For full-body quality jobs, set `shot_type=standing_full_body` and provide real pose/reference inputs.
- One-command strict old-man quality run:
  ```powershell
  .\backend\.venv_cuda\Scripts\python.exe backend\scripts\run_oldman_master.py --use-crop-as-refs --strict-gate
  ```
- There is no `oldman.py` runtime entrypoint. Use `backend/scripts/run_oldman_master.py` for strict old-man panel runs.
- Start with translation-only style (`style=original`, `upscale=1`) to validate OCR/translation quality, then enable heavier features.
- Run stress benchmark (server running):
  ```powershell
  python backend\scripts\stress_benchmark.py --input-dir output_export\phase2_rerun --iterations 3 --quality balanced
  ```
- Run ablation study (local TestClient, no external server needed):
  ```powershell
  python backend\scripts\run_ablation.py --input-dir output_export\phase2_rerun --iterations 2 --quality balanced --style cinematic
  ```
Sketch → Perfect Art: a complete, no-crumbs blueprint

(artist + senior ML engineer voice — deep, practical, research + product plan)
