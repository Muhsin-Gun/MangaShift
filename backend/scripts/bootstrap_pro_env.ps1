Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PreferredCudaVenv = Join-Path $Root ".venv_cuda"
$FallbackVenv = Join-Path $Root ".venv_pro"
$Venv = if (Test-Path $PreferredCudaVenv) { $PreferredCudaVenv } else { $FallbackVenv }

Write-Host "[MangaShift] Pro bootstrap root: $Root"

if (-not (Test-Path $Venv)) {
    Write-Host "[MangaShift] Creating virtual environment: $Venv"
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        py -3.10 -m venv $Venv
    } else {
        python -m venv $Venv
    }
}

$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate

Write-Host "[MangaShift] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

Write-Host "[MangaShift] Installing base requirements"
python -m pip install -r (Join-Path $Root "requirements.txt")
python -m pip install facenet-pytorch --no-deps

Write-Host "[MangaShift] Dependency integrity check (advisory)"
python -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "[MangaShift] pip check reported advisory conflicts; strict runtime gates still apply."
}

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
