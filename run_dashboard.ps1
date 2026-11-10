$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot 'env\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'The project is not set up yet. Run .\setup.ps1 first.'
}

& $venvPython -m streamlit run dashboard/app.py --server.address 127.0.0.1
