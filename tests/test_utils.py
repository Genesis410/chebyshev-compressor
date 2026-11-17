"""Tests for reproducibility utilities, manifest generation, and calibration file handling."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from chebyshev.utils import (
    compute_file_crc32,
    compute_file_sha256,
    generate_reproducibility_manifest,
    load_calibration,
    save_calibration,
    set_seed,
)


def test_set_seed_determinism():
    """Verifies that set_seed ensures exact reproducibility across torch and numpy."""
    set_seed(1234)
    t1 = torch.randn(5, 5)
    n1 = np.random.randn(5, 5)

    set_seed(1234)
    t2 = torch.randn(5, 5)
    n2 = np.random.randn(5, 5)

    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)


def test_calibration_roundtrip(tmp_path: Path):
    """Verifies save_calibration and load_calibration round-trip fidelity."""
    mean_in = np.array([1.23, 4.56, 7.89], dtype=np.float32)
    std_in = np.array([0.12, 0.34, 0.56], dtype=np.float32)

    calib_path = tmp_path / "test_calib.npy"
    save_calibration(mean_in, std_in, calib_path)
    assert calib_path.exists()

    mean_out, std_out = load_calibration(calib_path)
    np.testing.assert_allclose(mean_in, mean_out, rtol=1e-6)
    np.testing.assert_allclose(std_in, std_out, rtol=1e-6)


def test_file_hashing_and_manifest(tmp_path: Path):
    """Verifies SHA-256, CRC32, and manifest generation."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Chebyshev Compressor Determinism Test", encoding="utf-8")

    sha = compute_file_sha256(sample_file)
    crc = compute_file_crc32(sample_file)

    assert len(sha) == 64
    assert crc.startswith("0x") and crc.endswith("U")

    # Generate test dummy artifacts
    (tmp_path / "teacher.pt").write_bytes(b"dummy_teacher")
    (tmp_path / "student.pt").write_bytes(b"dummy_student")

    manifest = generate_reproducibility_manifest(
        out_dir=tmp_path,
        seed=42,
        hyperparameters={"order": 8},
        benchmark_metrics={"speedup_x": 170.0},
    )

    manifest_file = tmp_path / "reproducibility_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["seed"] == 42
    assert "teacher.pt" in data["artifacts"]
    assert "student.pt" in data["artifacts"]
    assert data["hyperparameters"]["order"] == 8
    assert data["benchmark_metrics"]["speedup_x"] == 170.0
