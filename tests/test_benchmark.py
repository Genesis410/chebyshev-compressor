from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chebyshev.benchmark import (
    compare_benchmarks,
    generate_c_benchmark_harness_source,
    run_c_benchmark,
    run_pytorch_benchmark,
)
from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel


class SimpleTeacher(nn.Module):
    def __init__(self, in_features=4, hidden_dim=16, out_features=1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def test_generate_c_harness_source():
    src = generate_c_benchmark_harness_source("chebyshev_engine.h")
    assert '#include "chebyshev_engine.h"' in src
    assert "chebyshev_step(&state, raw_in, out);" in src
    assert "CHEB_LATENCY_US" in src
    assert "CHEB_THROUGHPUT_STEPS_SEC" in src


def test_run_c_benchmark_execution(tmp_path: Path):
    num_channels = 4
    order = 6
    hidden_dim = 16
    out_features = 1
    
    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    header_path = tmp_path / "chebyshev_engine.h"
    export_to_c_header(model, header_path)
    
    metrics = run_c_benchmark(header_path, iterations=10000, build_dir=tmp_path)
    
    assert metrics["iterations"] == 10000
    assert metrics["total_time_sec"] > 0
    assert metrics["latency_us"] > 0
    # C99 engine on modern CPU easily achieves sub-microsecond or near-microsecond latency
    assert metrics["latency_us"] < 10.0
    assert metrics["throughput_steps_sec"] > 100000.0
    assert metrics["ram_bytes"] == num_channels * order * 4


def test_run_pytorch_benchmark():
    teacher = SimpleTeacher(in_features=4, hidden_dim=16, out_features=1)
    metrics = run_pytorch_benchmark(teacher, num_channels=4, steps=200)
    
    assert metrics["steps"] == 200
    assert metrics["latency_us"] > 0
    assert metrics["throughput_steps_sec"] > 0


def test_compare_benchmarks_table_formatting():
    c_metrics = {
        "iterations": 100000,
        "total_time_sec": 0.05,
        "latency_us": 0.5,
        "latency_ns": 500.0,
        "throughput_steps_sec": 2000000.0,
        "ram_bytes": 448,
    }
    pt_metrics = {
        "steps": 1000,
        "total_time_sec": 0.5,
        "latency_us": 50.0,
        "throughput_steps_sec": 20000.0,
    }
    
    comparison = compare_benchmarks(c_metrics, pt_metrics, teacher_ram_bytes=32768)
    assert comparison["speedup_x"] == pytest.approx(100.0, rel=1e-2)
    assert "SPEEDUP" in comparison["summary_table"]
    assert "RAM REDUCTION" in comparison["summary_table"]
