# Fast, reproducible setup script for Chebyshev-Compressor on Windows PowerShell

$ErrorActionPreference = "Stop"

Write-Host "=== Chebyshev-Compressor: Windows Contributor Setup ===" -ForegroundColor Cyan

# Check Python
try {
    $pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "Found Python version: $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH."
}

# Create venv if not present
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing PyTorch CPU wheel (~180 MB)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing Chebyshev-Compressor in editable mode with development dependencies..." -ForegroundColor Yellow
pip install -e ".[dev]" --no-build-isolation

Write-Host "Verifying test suite (32 tests)..." -ForegroundColor Yellow
pytest tests/ -v

Write-Host "=== Setup Complete! You are ready to contribute. ===" -ForegroundColor Green
