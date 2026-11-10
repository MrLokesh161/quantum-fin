$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot 'env\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'The project is not set up yet. Run .\setup.ps1 first.'
}
if (-not (Test-Path (Join-Path $ProjectRoot 'frontend\dist\index.html'))) {
    throw 'The frontend has not been built. Run npm.cmd install and npm.cmd run build in frontend\ once.'
}

Write-Host 'Starting the QETI backend and serving the frontend...' -ForegroundColor Cyan
Write-Host 'Open http://127.0.0.1:8000 in your browser.' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop the server.' -ForegroundColor Yellow
& $venvPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
