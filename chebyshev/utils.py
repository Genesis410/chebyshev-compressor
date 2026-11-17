"""Reproducibility utilities, manifest generation, and artifact integrity verification."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, Union
import zlib
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducible operations where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hexadecimal digest of a file on disk."""
    path = Path(filepath)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_crc32(filepath: Union[str, Path]) -> str:
    """Computes CRC32 checksum of a file on disk formatted as 0xXXXXXXXXU."""
    path = Path(filepath)
    if not path.is_file():
        return ""
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            crc = zlib.crc32(chunk, crc)
    return f"0x{crc & 0xFFFFFFFF:08X}U"


def save_calibration(
    mean: np.ndarray,
    std: np.ndarray,
    out_path: Union[str, Path],
) -> Path:
    """Saves sensor calibration vectors (mean and std) to a .npy file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calib_arr = np.stack([mean.astype(np.float32), std.astype(np.float32)], axis=0)
    np.save(str(path), calib_arr)
    return path


def load_calibration(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Loads sensor calibration vectors (mean and std) from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found at {path}")
    calib_arr = np.load(str(path), allow_pickle=True)
    if isinstance(calib_arr, np.ndarray) and calib_arr.ndim == 2 and calib_arr.shape[0] >= 2:
        return calib_arr[0].astype(np.float32), calib_arr[1].astype(np.float32)
    elif isinstance(calib_arr, dict):
        return calib_arr["mean"].astype(np.float32), calib_arr["std"].astype(np.float32)
    else:
        raise ValueError(f"Unrecognized calibration format in {path}")


def generate_reproducibility_manifest(
    out_dir: Union[str, Path],
    seed: int = 42,
    hyperparameters: Optional[Dict[str, Any]] = None,
    benchmark_metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scans artifact directory and generates a comprehensive reproducibility manifest."""
    dir_path = Path(out_dir)
    manifest_path = dir_path / "reproducibility_manifest.json"

    # Identify compiler
    compiler_path = shutil.which("gcc") or shutil.which("clang") or shutil.which("cl") or "None"
    compiler_ver = "Unknown"
    if compiler_path != "None":
        try:
            res = subprocess.run([compiler_path, "--version"], capture_output=True, text=True, timeout=5)
            compiler_ver = res.stdout.splitlines()[0] if res.stdout else "Unknown"
        except Exception:
            pass

    # Collect tracked artifacts in out_dir
    tracked_filenames = [
        "teacher.pt",
        "student.pt",
        "calib.npy",
        "chebyshev_engine.h",
        "benchmark_results.json",
    ]
    artifacts_info: Dict[str, Any] = {}
    for fname in tracked_filenames:
        fpath = dir_path / fname
        if fpath.exists():
            artifacts_info[fname] = {
                "size_bytes": fpath.stat().st_size,
                "sha256": compute_file_sha256(fpath),
                "crc32": compute_file_crc32(fpath),
                "modified_iso": datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc).isoformat(),
            }

    manifest = {
        "project": "chebyshev-compressor",
        "version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "compiler": compiler_ver,
            "compiler_path": compiler_path,
        },
        "hyperparameters": hyperparameters or {},
        "benchmark_metrics": benchmark_metrics or {},
        "artifacts": artifacts_info,
        "metadata": extra_metadata or {},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
