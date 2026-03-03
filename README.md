# MangaShift AI

Local-first manga/manhwa translation and art-enhancement system:
- `backend/`: FastAPI AI pipeline (OCR -> translation -> inpaint -> upscale -> color/style -> typeset)
- `extension/`: Chrome MV3 extension that intercepts reader images and swaps enhanced output in-place while scrolling
- `tests/`: backend API + module tests

## Implemented

### Backend pipeline
- OCR fallback chain: `manga-ocr` -> `easyocr` -> `pytesseract`
- Translation fallback chain: Marian JA/KO -> M2M100 -> identity fallback
- Text erasure (inpainting): diffusion inpainting (when available) -> OpenCV fallback
- Upscaling: quality-based auto mode with Real-ESRGAN attempt + deterministic PIL fallback
- Colorization for grayscale pages with diffusion attempt + deterministic fallback
- Style transfer presets with diffusion attempt + deterministic style filters
- Typesetting engine with auto-fit and stroke rendering in speech bubble regions
- Two-pass variant rendering with quality-gate scoring
- Disk LRU cache (SHA-256 keyed)

### API endpoints
- `GET /health`
- `GET /capabilities`
- `GET /preflight`
- `GET /warmup`
- `POST /warmup`
- `GET /styles`
- `GET /styles/full`
- `GET /quality/readiness`
- `GET /memory/{series_id}`
- `GET /memory`
- `GET /controller`
- `GET /controller/{style_id}`
- `POST /process-image`

### Extension
- Detects likely manga panel images on page
- Queues processing through service worker with concurrency limit `2`
- Sends panel image to local backend
- Replaces original panel with processed image
- IndexedDB cache for processed panels
- Failure overlay with retry action
- Popup UI controls for style/source/upscale/colorization and series memory

## Quick Start (Windows, Recommended)

1. Install Python `3.10` or `3.11`.
2. Open PowerShell in repo root:
   ```powershell
   cd backend
   py -3.10 -m venv .venv_cuda
   .\.venv_cuda\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
   .\.venv_cuda\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv_cuda\Scripts\python.exe -m pip install facenet-pytorch --no-deps
   .\start_local.ps1
   ```
3. Load `extension/` in Chrome (`chrome://extensions` -> Developer mode -> Load unpacked).
4. Visit a manga/manhwa reader page and apply a preset from the extension popup.

## Quick Start (Linux/macOS)

```bash
cd backend
chmod +x start_local.sh
./start_local.sh
```

## Strict Quality Runtime Requirements

- `render_quality=final` and `render_quality=quality` are strict GPU paths.
- They require either:
  - Local CUDA GPU runtime, or
  - A configured cloud GPU worker via `QUALITY_CLOUD_WORKER_URL`.
- On CPU-only hosts, strict final/quality requests fail fast by design.

Check readiness:
```powershell
curl http://127.0.0.1:8000/quality/readiness
```

## Model Preparation

Download baseline required models:
```powershell
cd backend
python scripts\download_models.py --required
```

Download quality/full-path assets (SDXL + ControlNet + IP-Adapter):
```powershell
python scripts\download_models.py --quality
```

Generate/verify deterministic model manifest:
```powershell
python scripts\model_manifest.py --write --models-dir models --verify
```

## Validation Commands

Run all tests (from repo root):
```powershell
backend\.venv_cuda\Scripts\python.exe -m pytest tests -q
```

Run strict smoke gate:
```powershell
backend\.venv_cuda\Scripts\python.exe backend\scripts\run_strict_smoke.py --strict-diffusion --strict-ocr --strict-translation --quality final --style cinematic --output-dir backend\cache\strict_smoke
```

Run old-man master (strict gate):
```powershell
backend\.venv_cuda\Scripts\python.exe backend\scripts\run_oldman_master.py --use-crop-as-refs --strict-gate
```

CPU-safe validation run:
```powershell
backend\.venv_cuda\Scripts\python.exe backend\scripts\run_oldman_master.py --render-quality balanced --allow-cpu --variant-count 2 --use-crop-as-refs --strict-gate
```

## 4 Production Cells (Windows PowerShell)

Cell 1: deterministic environment setup
```powershell
cd backend
py -3.10 -m venv .venv_cuda
.\.venv_cuda\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv_cuda\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_cuda\Scripts\python.exe -m pip install facenet-pytorch --no-deps
cd ..
```

Cell 2: verify stack + model inventory
```powershell
python backend\scripts\verify_pro_stack.py --json
python backend\scripts\download_models.py --required
python backend\scripts\download_models.py --quality
python backend\scripts\model_manifest.py --write --models-dir backend\models --verify
```

Cell 3: quality runtime check + smoke
```powershell
curl http://127.0.0.1:8000/quality/readiness
python backend\scripts\run_strict_smoke.py --strict-diffusion --strict-ocr --strict-translation --quality final --style cinematic --output-dir backend\cache\strict_smoke
```

Cell 4: guarded git push (runs compile + tests + smoke before push)
```powershell
.\safe_push.ps1
```
Optional:
```powershell
.\safe_push.ps1 -Remote origin -Branch main
.\safe_push.ps1 -SkipSmoke
.\safe_push.ps1 -NoPush
```

## Notes

- Dependency checks (`pip check`) are run in bootstrap/start scripts as advisory diagnostics.
- Real-ESRGAN runs only when both package imports and model weights are available.
- There is no `oldman.py` entrypoint; use `backend/scripts/run_oldman_master.py`

## Git Push Stability (Port 443 SSH)

For more stable git pushes, configure SSH over port 443:

```powershell
# Test SSH connection
ssh -T -p 443 git@ssh.github.com

# Set remote to SSH
git remote set-url origin git@github.com:Muhsin-Gun/MangaShift.git

# Push via SSH (automatically uses port 443 via config)
git push origin main
```

SSH config (`~/.ssh/config`):
```
Host github.com
  HostName ssh.github.com
  Port 443
  User git
```
