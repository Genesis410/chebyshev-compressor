#!/usr/bin/env bash
# Fast, reproducible setup script for Chebyshev-Compressor on Linux/macOS

set -e

echo "=== Chebyshev-Compressor: Contributor Setup ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'"

# Create venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

echo "Upgrading pip and installing PyTorch CPU wheel..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing Chebyshev-Compressor in editable mode with development dependencies..."
pip install -e ".[dev]" --no-build-isolation

echo "Verifying test suite..."
pytest tests/ -v

echo "=== Setup Complete! You are ready to contribute. ==="
