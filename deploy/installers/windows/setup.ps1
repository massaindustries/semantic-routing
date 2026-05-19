<#
.SYNOPSIS
  Brick → Claude Code installer for Windows 11 (Docker Desktop + NVIDIA GPU).

.DESCRIPTION
  Idempotent setup:
    1. Verify Docker Desktop is running with WSL2 backend.
    2. Verify the NVIDIA GPU is reachable from a CUDA container.
    3. Generate a random bearer token if none exists, persist to `.env`.
    4. Pull pre-built images (or build with -Build) and start the compose stack.
    5. Wait until both services report healthy.
    6. Set the user-scope ANTHROPIC_BASE_URL env var so the next `claude` shell
       picks up the proxy automatically.

  Run from an elevated PowerShell prompt (admin not strictly required, but
  Docker Desktop integration is smoother that way).

.PARAMETER Build
  Build images locally instead of pulling from GHCR. Use this if you've cloned
  the repo and want to test changes before they're published.

.PARAMETER HostPort
  Port exposed on localhost for mymodel. Default 18000.

.PARAMETER Tag
  Image tag to pull from GHCR. Default `latest`. Use `dev` for unreleased
  builds pushed manually by maintainers.

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -Build
  .\setup.ps1 -HostPort 19000
  .\setup.ps1 -Tag dev
#>
[CmdletBinding()]
param(
    [switch]$Build,
    [int]$HostPort = 18000,
    [string]$Tag = 'latest'
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeDir = Resolve-Path (Join-Path $here '..\..\docker-compose')
$composeFile = Join-Path $composeDir 'docker-compose.brick-cc.yml'
$envFile = Join-Path $composeDir '.env'

function Write-Section($msg) {
    Write-Host ""
    Write-Host "═══ $msg ═══" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host "✗ $msg" -ForegroundColor Red
    exit 1
}

function Ok($msg) { Write-Host "✓ $msg" -ForegroundColor Green }

# ── 1. Prerequisites ─────────────────────────────────────────────
Write-Section "Prerequisites"

try { docker --version | Out-Null } catch { Fail "Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop/" }
Ok "Docker installed"

try { docker compose version | Out-Null } catch { Fail "Docker Compose v2 plugin missing — update Docker Desktop." }
Ok "Docker Compose v2 available"

try { docker info --format '{{.OSType}}' | Out-Null } catch { Fail "Docker daemon not running. Start Docker Desktop and re-run." }
Ok "Docker daemon reachable"

# ── 2. GPU check ─────────────────────────────────────────────────
Write-Section "GPU"

$gpuOk = $false
try {
    $out = docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gpuOk = $true
        $gpuLine = ($out | Select-String 'NVIDIA-SMI' | Select-Object -First 1).ToString().Trim()
        Ok "GPU available — $gpuLine"
    }
} catch { }

if (-not $gpuOk) {
    Write-Host "✗ GPU not reachable from Docker." -ForegroundColor Red
    Write-Host "  Install/update the NVIDIA driver for Windows (it bundles CUDA-on-WSL):"
    Write-Host "    https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    Write-Host "  Then enable 'Use the WSL 2 based engine' in Docker Desktop → Settings → General."
    Fail "Re-run setup.ps1 once GPU passthrough works."
}

# ── 3. Bearer token ──────────────────────────────────────────────
Write-Section "Bearer token"

$token = $null
if (Test-Path $envFile) {
    $existing = Get-Content $envFile | Where-Object { $_ -match '^BRICK_CLASSIFIER_TOKEN=(.+)$' }
    if ($existing) {
        $token = ($existing -replace '^BRICK_CLASSIFIER_TOKEN=', '').Trim()
        Ok ".env exists, reusing existing token"
    }
}

if (-not $token) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = -join ($bytes | ForEach-Object { '{0:x2}' -f $_ })
    Ok "Generated new 64-char hex token"
}

@(
    "# brick-cc compose env — DO NOT COMMIT",
    "BRICK_CLASSIFIER_TOKEN=$token",
    "BRICK_CC_HOST_PORT=$HostPort",
    "BRICK_CC_TAG=$Tag"
) | Set-Content -Path $envFile -Encoding ASCII
Ok "Wrote $envFile"

# ── 4. Build or pull ─────────────────────────────────────────────
Write-Section "Images"

Push-Location $composeDir
try {
    if ($Build) {
        Write-Host "Building images locally (this can take 10+ min the first time)..."
        docker compose -f $composeFile build
    } else {
        Write-Host "Pulling images from GHCR..."
        docker compose -f $composeFile pull
    }
    if ($LASTEXITCODE -ne 0) { Fail "Image acquisition failed." }
    Ok "Images ready"

    # ── 5. Bring up ──────────────────────────────────────────────
    Write-Section "Stack"
    docker compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed." }
    Ok "Stack started"

    # ── 6. Wait for health ───────────────────────────────────────
    Write-Section "Health"
    $deadline = (Get-Date).AddSeconds(180)
    $url = "http://localhost:$HostPort/health"
    $healthy = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) { $healthy = $true; break }
        } catch { Start-Sleep 5 }
    }
    if (-not $healthy) {
        Write-Host "  Timed out waiting for $url to respond." -ForegroundColor Yellow
        Write-Host "  Check logs: docker compose -f $composeFile logs --tail 50"
        Fail "Stack did not become healthy in 3 minutes."
    }
    Ok "mymodel responding at $url"
} finally {
    Pop-Location
}

# ── 7. Claude Code env var ───────────────────────────────────────
Write-Section "Claude Code"

$baseUrl = "http://localhost:$HostPort"
[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', $baseUrl, 'User')
Ok "Set ANTHROPIC_BASE_URL=$baseUrl (User scope)"

Write-Host ""
Write-Host "All done." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Close and reopen your terminal (env var only takes effect in new shells)."
Write-Host "  2. Run: claude"
Write-Host "  3. Verify routing: mymodel claude status"
Write-Host ""
Write-Host "Stop:      docker compose -f docker-compose.brick-cc.yml down" -ForegroundColor DarkGray
Write-Host "Logs:      docker compose -f docker-compose.brick-cc.yml logs -f" -ForegroundColor DarkGray
Write-Host "Uninstall: .\\uninstall.ps1" -ForegroundColor DarkGray
