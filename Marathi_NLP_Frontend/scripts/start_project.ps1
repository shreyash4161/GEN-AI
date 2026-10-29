$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$apiPort = 8000
$frontendPort = 5173
$logs = Join-Path $projectRoot 'results\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

function Test-ListeningPort([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1)
}

if (-not (Test-ListeningPort $apiPort)) {
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) { throw "Project virtual environment not found: $python`nCreate it with: py -3.12 -m venv .venv" }
    if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True' }
    Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','api:app','--host','127.0.0.1','--port',"$apiPort") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "api_$stamp.log") `
        -RedirectStandardError (Join-Path $logs "api_$stamp.err.log") | Out-Null
}

if (-not (Test-ListeningPort $frontendPort)) {
    $node = (Get-Command node -ErrorAction SilentlyContinue).Source
    $vite = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
    if (-not $node -or -not (Test-Path $vite)) { throw "Frontend runtime not found. Run 'npm install' in $projectRoot\frontend first." }
    Start-Process -FilePath $node -ArgumentList @($vite,'--host','127.0.0.1','--port',"$frontendPort") `
        -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "frontend_$stamp.log") `
        -RedirectStandardError (Join-Path $logs "frontend_$stamp.err.log") | Out-Null
}

$apiReady = $false
$frontendReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try { $apiReady = (Invoke-RestMethod "http://127.0.0.1:$apiPort/api/health" -TimeoutSec 2).status -eq 'ok' } catch { }
    try { $frontendReady = (Invoke-WebRequest "http://127.0.0.1:$frontendPort/" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { }
    if ($apiReady -and $frontendReady) { break }
    Start-Sleep -Seconds 1
}
if (-not $apiReady) { throw "API did not become healthy. Check $logs\api_$stamp.err.log" }
if (-not $frontendReady) { throw "Frontend did not become available. Check $logs\frontend_$stamp.err.log" }
Write-Host 'Marathi NLP Lab is running:' -ForegroundColor Green
Write-Host "  Frontend: http://127.0.0.1:$frontendPort/"
Write-Host "  API:      http://127.0.0.1:$apiPort/api/health"
Write-Host 'Both services are started hidden and remain available after this PowerShell window closes.'
