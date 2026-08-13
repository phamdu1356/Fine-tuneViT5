$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Không tìm thấy Python trong venv: $pythonPath"
}

Set-Location -LiteralPath $projectRoot

Write-Host "[1/2] Chây evaluation.py..." -ForegroundColor Cyan
& $pythonPath ".\evaluation.py"
if ($LASTEXITCODE -ne 0) {
    throw "evaluation.py that bai voi exit code $LASTEXITCODE"
}

Write-Host "[2/2] Chay BERTscore.py..." -ForegroundColor Cyan
& $pythonPath ".\BERTscore.py"
if ($LASTEXITCODE -ne 0) {
    throw "BERTscore.py that bai voi exit code $LASTEXITCODE"
}

Write-Host "Đa hoan tat evaluation va BERTScore." -ForegroundColor Green
