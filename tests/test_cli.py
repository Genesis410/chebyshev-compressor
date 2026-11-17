from pathlib import Path
import pytest
import torch

from chebyshev.cli import main
from chebyshev.datasets import load_cmapss_fd001


def test_cmapss_dataset_loading_and_windowing(tmp_path: Path):
    X, y, mean, std = load_cmapss_fd001(
        data_dir=tmp_path,
        max_engines=10,
        window_size=20,
        stride=2,
    )
    
    assert X.ndim == 3
    assert X.shape[1] == 20  # window size
    assert X.shape[2] == 14  # 14 informative sensors
    assert y.shape == (X.shape[0], 1)
    assert mean.shape == (14,)
    assert std.shape == (14,)
    assert not torch.isnan(torch.from_numpy(X)).any()
    assert not torch.isnan(torch.from_numpy(y)).any()


def test_cli_subcommands_chain(tmp_path: Path):
    teacher_pt = tmp_path / "teacher.pt"
    student_pt = tmp_path / "student.pt"
    header_h = tmp_path / "chebyshev_engine.h"
    
    # 1. cheb train-teacher
    ret = main([
        "train-teacher",
        "--epochs", "2",
        "--smoke",
        "--out", str(teacher_pt),
    ])
    assert ret == 0
    assert teacher_pt.exists()
    
    # 2. cheb distill
    ret = main([
        "distill",
        "--teacher", str(teacher_pt),
        "--smoke",
        "--epochs", "2",
        "--out", str(student_pt),
    ])
    assert ret == 0
    assert student_pt.exists()
    
    # 3. cheb export
    ret = main([
        "export",
        "--model", str(student_pt),
        "--output", str(header_h),
    ])
    assert ret == 0
    assert header_h.exists()
    
    # 4. cheb benchmark
    ret = main([
        "benchmark",
        "--header", str(header_h),
        "--teacher", str(teacher_pt),
        "--iterations", "5000",
    ])
    assert ret == 0


def test_cli_smoke_test_flag(tmp_path: Path, monkeypatch):
    # Set output dir to tmp_path
    monkeypatch.setenv("CHEB_OUT_DIR", str(tmp_path / "cheb_out"))
    
    ret = main(["--smoke-test"])
    assert ret == 0
    
    out_dir = tmp_path / "cheb_out"
    assert (out_dir / "teacher.pt").exists()
    assert (out_dir / "student.pt").exists()
    assert (out_dir / "calib.npy").exists()
    assert (out_dir / "chebyshev_engine.h").exists()
    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "reproducibility_manifest.json").exists()
