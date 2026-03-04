param(
    [string]$Remote = "origin",
    [string]$Branch = "",
    [switch]$SkipSmoke,
    [switch]$NoPush,
    [int]$PushRetries = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-GitHubRepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteUrl
    )

    $raw = [string]$RemoteUrl
    $raw = $raw.Trim()
    if (-not $raw) {
        return ""
    }

    $patterns = @(
        "^git@github\.com:(.+?)(?:\.git)?$",
        "^ssh://git@(?:github\.com|ssh\.github\.com)(?::\d+)?/(.+?)(?:\.git)?$",
        "^https://github\.com/(.+?)(?:\.git)?$"
    )
    foreach ($pattern in $patterns) {
        if ($raw -match $pattern) {
            return "$($matches[1]).git"
        }
    }
    return ""
}

function Invoke-GitPushWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Target,
        [Parameter(Mandatory = $true)]
        [string]$Branch,
        [Parameter(Mandatory = $true)]
        [int]$Retries,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $Retries = [Math]::Max(1, [int]$Retries)
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        Write-Host "[MangaShift] Push attempt $attempt/$Retries via $Label -> $Target/$Branch"
        git push $Target $Branch
        if ($LASTEXITCODE -eq 0) {
            return $true
        }

        if ($attempt -lt $Retries) {
            $sleepSeconds = [Math]::Min(60, 5 * $attempt)
            Write-Warning "[MangaShift] Push failed via $Label. Retrying in $sleepSeconds seconds."
            Start-Sleep -Seconds $sleepSeconds
        }
    }
    return $false
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot "backend\.venv_cuda\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing backend\.venv_cuda\Scripts\python.exe. Run backend\start_local.ps1 setup first."
}

Write-Host "[MangaShift] Running compile checks"
& $Python -m compileall backend\app backend\scripts tests
if ($LASTEXITCODE -ne 0) {
    throw "Compile checks failed."
}

Write-Host "[MangaShift] Running full test suite"
& $Python -m pytest tests -q
if ($LASTEXITCODE -ne 0) {
    throw "Test suite failed."
}

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
            $fallbackPageIndex = Get-Random -Minimum 200 -Maximum 1000000
            & $Python backend\scripts\run_oldman_master.py `
                --render-quality balanced `
                --allow-cpu `
                --variant-count 2 `
                --use-crop-as-refs `
                --strict-gate `
                --page-index $fallbackPageIndex
            if ($LASTEXITCODE -ne 0) {
                throw "Balanced strict-gate fallback failed."
            }
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

$remoteUrl = (git remote get-url $Remote).Trim()
if (-not $remoteUrl) {
    throw "Could not resolve remote URL for '$Remote'."
}
$repoPath = Get-GitHubRepoPath -RemoteUrl $remoteUrl

$targets = @(
    @{ target = $Remote; label = "remote_name" }
)
if ($repoPath) {
    $ssh443Url = "ssh://git@ssh.github.com:443/$repoPath"
    $httpsUrl = "https://github.com/$repoPath"
    if ($ssh443Url -ne $remoteUrl) {
        $targets += @{ target = $ssh443Url; label = "ssh_443_url" }
    }
    if ($httpsUrl -ne $remoteUrl) {
        $targets += @{ target = $httpsUrl; label = "https_url" }
    }
}

$seenTargets = @{}
$pushOk = $false
foreach ($entry in $targets) {
    $target = [string]$entry.target
    if ($seenTargets.ContainsKey($target)) {
        continue
    }
    $seenTargets[$target] = $true

    $ok = Invoke-GitPushWithRetry `
        -Target $target `
        -Branch $Branch `
        -Retries $PushRetries `
        -Label ([string]$entry.label)
    if ($ok) {
        $pushOk = $true
        break
    }
    Write-Warning "[MangaShift] Push path failed for $target. Trying next transport."
}

if (-not $pushOk) {
    throw "git push failed for all available transports."
}

Write-Host "[MangaShift] Push completed."
