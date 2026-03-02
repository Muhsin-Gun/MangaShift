param(
    [string]$Remote = "origin",
    [string]$Branch = "",
    [switch]$SkipSmoke,
    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot "backend\.venv_cuda\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing backend\.venv_cuda\Scripts\python.exe. Run backend\start_local.ps1 setup first."
}

Write-Host "[MangaShift] Running compile checks"
& $Python -m compileall backend\app backend\scripts tests

Write-Host "[MangaShift] Running full test suite"
& $Python -m pytest tests -q

if (-not $SkipSmoke) {
    Write-Host "[MangaShift] Running strict smoke (final)"
    & $Python backend\scripts\run_strict_smoke.py `
        --strict-diffusion `
        --strict-ocr `
        --strict-translation `
        --quality final `
        --style cinematic `
        --output-dir backend\cache\strict_smoke

    if ($LASTEXITCODE -ne 0) {
        $ReportPath = Join-Path $RepoRoot "backend\cache\strict_smoke\strict_smoke_report.json"
        if (-not (Test-Path $ReportPath)) {
            throw "Strict smoke failed and report was not found: $ReportPath"
        }

        $report = Get-Content $ReportPath -Raw | ConvertFrom-Json
        $errorCode = $report.process_error.detail.error
        if ($errorCode -eq "strict_render_mode_requires_gpu") {
            Write-Warning "[MangaShift] No GPU/worker for strict final path. Running balanced strict-gate smoke fallback."
            & $Python backend\scripts\run_oldman_master.py `
                --render-quality balanced `
                --allow-cpu `
                --variant-count 2 `
                --use-crop-as-refs `
                --strict-gate `
                --page-index 97
        } else {
            throw "Strict smoke failed with non-GPU error: $errorCode"
        }
    }
}

if (-not $Branch.Trim()) {
    $Branch = (git rev-parse --abbrev-ref HEAD).Trim()
}

if (-not $Branch) {
    throw "Could not resolve current git branch."
}

if ($NoPush) {
    Write-Host "[MangaShift] NoPush enabled. Push target would be $Remote/$Branch. Skipping git push."
    exit 0
}
Write-Host "[MangaShift] Pushing $Branch to $Remote"
git push $Remote $Branch

Write-Host "[MangaShift] Push completed."
