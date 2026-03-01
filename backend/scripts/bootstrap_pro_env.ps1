Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venv = Join-Path $Root ".venv_pro"

Write-Host "[MangaShift] Pro bootstrap root: $Root"

if (-not (Test-Path $Venv)) {
    Write-Host "[MangaShift] Creating .venv_pro"
    python -m venv $Venv
}

$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate

Write-Host "[MangaShift] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

Write-Host "[MangaShift] Installing base requirements"
python -m pip install -r (Join-Path $Root "requirements.txt")
python -m pip install facenet-pytorch --no-deps

Write-Host "[MangaShift] Installing/repairing strict-mode stack"
python (Join-Path $Root "scripts\verify_pro_stack.py") --install-missing

Write-Host "[MangaShift] Final capability report"
python (Join-Path $Root "scripts\verify_pro_stack.py") --json

Write-Host "[MangaShift] Generating and verifying model manifest"
python (Join-Path $Root "scripts\model_manifest.py") --write --verify --models-dir (Join-Path $Root "models")

Write-Host "[MangaShift] Strict smoke (will fail if strict requirements are not met)"
python (Join-Path $Root "scripts\run_strict_smoke.py") --strict-ocr --strict-translation --quality balanced

Write-Host "[MangaShift] To start server in strict mode:"
Write-Host "  `$env:STRICT_PRO_MODE='true'"
Write-Host "  uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir $Root"
