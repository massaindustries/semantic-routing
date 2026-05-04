<#
.SYNOPSIS
  Tear down the Brick → Claude Code stack on Windows.
#>
[CmdletBinding()]
param([switch]$KeepEnv)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeDir = Resolve-Path (Join-Path $here '..\..\docker-compose')
$composeFile = Join-Path $composeDir 'docker-compose.brick-cc.yml'

Push-Location $composeDir
try {
    Write-Host "Stopping containers..."
    docker compose -f $composeFile down -v
} finally {
    Pop-Location
}

if (-not $KeepEnv) {
    [Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', $null, 'User')
    Write-Host "Removed ANTHROPIC_BASE_URL user env var."
}

$envFile = Join-Path $composeDir '.env'
if (Test-Path $envFile) {
    Remove-Item $envFile
    Write-Host "Deleted $envFile."
}

Write-Host "Done. Reopen your terminal so the env change takes effect." -ForegroundColor Green
