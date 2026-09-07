$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host 'QETI setup starting...' -ForegroundColor Cyan

$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
if (-not $python) { throw 'Python 3.10+ is required. Install Python from https://www.python.org/downloads/ and run setup again.' }

$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) { throw 'Node.js 18+ and npm are required. Install Node.js from https://nodejs.org/ and run setup again.' }

if (-not (Test-Path 'env\Scripts\python.exe')) {
    Write-Host 'Creating Python virtual environment...' -ForegroundColor Yellow
    if ((Get-Command py -ErrorAction SilentlyContinue)) { & py -3 -m venv env } else { & python -m venv env }
}

$venvPython = Join-Path $ProjectRoot 'env\Scripts\python.exe'
Write-Host 'Installing Python dependencies...' -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host 'Installing frontend dependencies...' -ForegroundColor Yellow
Push-Location frontend
& npm.cmd install
Pop-Location

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Next: .\run_project.ps1' -ForegroundColor Cyan
