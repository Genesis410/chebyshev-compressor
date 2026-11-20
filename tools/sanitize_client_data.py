#!/usr/bin/env python3
"""Standalone client-side data sanitization utility for Chebyshev-Compressor.

Strips physical units, channel labels, and proprietary metadata before transmission.
Normalizes arbitrary multi-channel sensor arrays to zero-mean, unit-variance float32
arrays with division-by-zero protection on constant/dead sensor channels.

Zero external dependencies beyond NumPy. Run locally before sharing data.
"""

import argparse
from pathlib import Path
import numpy as np


def sanitize_sensor_data(input_path: str, output_path: str) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load input array (supports .npy and comma/space separated text/csv)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, delimiter="," if "," in path.read_text()[:200] else None)

    arr = np.asarray(arr, dtype=np.float32)

    # Flatten trailing dimensions if 3D (N, T, D) or 2D (T, D)
    flat_view = arr.reshape(-1, arr.shape[-1])
    mean = np.mean(flat_view, axis=0)
    std = np.std(flat_view, axis=0)

    # Guard against constant/zero-variance channels to prevent NaN/Inf
    std = np.where(std < 1e-8, 1.0, std)

    normalized = ((arr - mean) / std).astype(np.float32)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_p, normalized)

    print("=" * 60)
    print("CHEBYSHEV-COMPRESSOR: CLIENT DATA SANITIZATION SUCCESSFUL")
    print(f"  Input Shape   : {arr.shape} ({arr.dtype})")
    print(f"  Output Path   : {out_p.resolve()} ({normalized.nbytes:,} bytes)")
    print("  Sanitization  : Physical units & metadata stripped.")
    print("                  Zero-mean unit-variance normalized.")
    print("                  Zero-variance channel NaN guard applied.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize proprietary sensor logs for Chebyshev-Compressor")
    parser.add_argument("input", help="Path to raw .npy or .csv sensor data")
    parser.add_argument("-o", "--output", default="sanitized_sensor_data.npy", help="Path to sanitized output .npy")
    args = parser.parse_args()
    sanitize_sensor_data(args.input, args.output)
