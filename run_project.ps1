$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot 'env\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'The project is not set up yet. Run .\setup.ps1 first.'
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw 'Node.js and npm are required. Install Node.js, then run .\setup.ps1.'
}

Write-Host '1/4 Refreshing daily market data...' -ForegroundColor Cyan
& $venvPython dataset.py

Write-Host '2/4 Training models and generating outputs...' -ForegroundColor Cyan
& $venvPython main.py

Write-Host '3/4 Building the frontend...' -ForegroundColor Cyan
Push-Location frontend
& npm.cmd run build
Pop-Location

Write-Host '4/4 Starting QETI...' -ForegroundColor Cyan
Write-Host 'Open http://127.0.0.1:8000 in your browser.' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop the server.' -ForegroundColor Yellow
& $venvPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
