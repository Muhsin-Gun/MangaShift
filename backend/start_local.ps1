Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreferredCudaVenv = Join-Path $Root ".venv_cuda"
$FallbackVenv = Join-Path $Root ".venv"
$Venv = if (Test-Path $PreferredCudaVenv) { $PreferredCudaVenv } else { $FallbackVenv }

Write-Host "[MangaShift] Root: $Root"
Write-Host "[MangaShift] Virtualenv: $Venv"

if (-not (Test-Path $Venv)) {
    Write-Host "[MangaShift] Creating virtual environment"
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        py -3.10 -m venv $Venv
    } else {
        python -m venv $Venv
    }
}

$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate

Write-Host "[MangaShift] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

Write-Host "[MangaShift] Installing dependencies"
python -m pip install -r (Join-Path $Root "requirements.txt")
python -m pip install facenet-pytorch --no-deps

Write-Host "[MangaShift] Dependency integrity check (advisory)"
python -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "[MangaShift] pip check reported advisory conflicts. Continuing startup."
}

Write-Host "[MangaShift] Torch/CUDA probe"
@'
import sys
try:
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    print(f"[MangaShift] python={sys.version.split()[0]} torch={torch.__version__} cuda={cuda_ok} torch_cuda={torch.version.cuda} device={device}")
    sys.exit(0 if cuda_ok else 42)
except Exception as exc:
    print(f"[MangaShift] CUDA probe failed: {exc}")
    sys.exit(43)
'@ | python -
$CudaProbeExit = $LASTEXITCODE
if (($env:MANGASHIFT_REQUIRE_GPU -eq "true") -and ($CudaProbeExit -ne 0)) {
    throw "MANGASHIFT_REQUIRE_GPU=true but CUDA is unavailable. Refusing CPU fallback."
}

New-Item -ItemType Directory -Force (Join-Path $Root "models") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root "cache") | Out-Null

Write-Host "[MangaShift] Pre-downloading translation models (best effort)"
@'
from transformers import MarianMTModel, MarianTokenizer
for model in ["Helsinki-NLP/opus-mt-ja-en", "Helsinki-NLP/opus-mt-ko-en"]:
    try:
        MarianTokenizer.from_pretrained(model)
        MarianMTModel.from_pretrained(model)
        print(f"[MangaShift] Downloaded: {model}")
    except Exception as exc:
        print(f"[MangaShift] WARNING: failed to pre-download {model}: {exc}")
'@ | python -

Write-Host "[MangaShift] Starting API server on http://127.0.0.1:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --app-dir $Root
