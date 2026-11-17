# Multi-stage reproducible container for Chebyshev-Compressor
FROM python:3.11-slim

# Install system C compilers and build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    clang \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CPU PyTorch first to prevent downloading CUDA runtimes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy repository sources
COPY . /workspace

# Install project in editable development mode
RUN pip install --no-cache-dir -e ".[dev]" --no-build-isolation

# Default command: run full test suite with parity verification
CMD ["pytest", "tests/", "-v"]
