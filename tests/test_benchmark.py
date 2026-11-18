from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chebyshev.benchmark import (
    compare_benchmarks,
    generate_c_benchmark_harness_source,
    run_c_benchmark,
    run_order_ablation,
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
    assert "CHEB_WINDOW_LATENCY_US" in src
    assert "CHEB_PERSISTENT_STATE_BYTES" in src
    assert "CHEB_STACK_SCRATCH_BYTES" in src


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

    metrics = run_c_benchmark(header_path, iterations=10000, seq_len=50, build_dir=tmp_path)

    assert metrics["iterations"] == 10000
    assert metrics["total_time_sec"] > 0
    assert metrics["latency_us"] > 0
    assert metrics["latency_us"] < 10.0
    assert metrics["throughput_steps_sec"] > 100000.0
    assert metrics["window_seq_len"] == 50
    assert metrics["window_latency_us"] > 0
    assert metrics["window_amortized_step_us"] > 0

    expected_persistent = num_channels * order * 4
    expected_stack = (order + hidden_dim) * 4
    assert metrics["ram_bytes"] == expected_persistent
    assert metrics["persistent_state_bytes"] == expected_persistent
    assert metrics["stack_scratch_bytes"] == expected_stack
    assert metrics["total_sram_bytes"] == expected_persistent + expected_stack


def test_run_pytorch_benchmark():
    teacher = SimpleTeacher(in_features=4, hidden_dim=16, out_features=1)
    metrics = run_pytorch_benchmark(teacher, num_channels=4, steps=200, seq_len=30, window_runs=20)

    assert metrics["steps"] == 200
    assert metrics["latency_us"] > 0
    assert metrics["throughput_steps_sec"] > 0
    assert metrics["window_seq_len"] == 30
    assert metrics["window_latency_us"] > 0
    assert metrics["window_amortized_step_us"] > 0


def test_compare_benchmarks_table_formatting():
    c_metrics = {
        "iterations": 100000,
        "total_time_sec": 0.05,
        "latency_us": 0.5,
        "latency_ns": 500.0,
        "throughput_steps_sec": 2000000.0,
        "window_seq_len": 50,
        "window_latency_us": 26.0,
        "ram_bytes": 448,
        "persistent_state_bytes": 448,
        "stack_scratch_bytes": 160,
        "total_sram_bytes": 608,
    }
    pt_metrics = {
        "steps": 1000,
        "total_time_sec": 0.5,
        "latency_us": 50.0,
        "throughput_steps_sec": 20000.0,
        "window_seq_len": 50,
        "window_latency_us": 1300.0,
    }

    comparison = compare_benchmarks(c_metrics, pt_metrics, teacher_ram_bytes=32768)
    assert comparison["speedup_x"] == pytest.approx(100.0, rel=1e-2)
    assert comparison["window_speedup_x"] == pytest.approx(50.0, rel=1e-2)
    assert "SPEEDUP" in comparison["summary_table"]
    assert "RAM REDUCTION" in comparison["summary_table"]
    assert "Persistent Recurrent State" in comparison["summary_table"]
    assert "Active Call-Stack Scratch" in comparison["summary_table"]
    assert "Total Estimated Runtime SRAM" in comparison["summary_table"]
    assert "MISRA-C" not in comparison["summary_table"]


def test_run_order_ablation_smoke(tmp_path: Path):
    results = run_order_ablation(
        orders=(4, 8),
        window_size=30,
        hidden_dim=16,
        epochs=5,
        seed=42,
        c_iterations=5000,
        max_engines=10,
        data_dir=tmp_path,
    )

    assert len(results) == 2
    assert results[0]["order_k"] == 4
    assert results[1]["order_k"] == 8
    # Higher order K=8 must strictly reduce signal reconstruction error over K=4
    assert results[1]["recon_rel_l2_error"] < results[0]["recon_rel_l2_error"]
    assert results[0]["persistent_state_bytes"] == 14 * 4 * 4
    assert results[1]["persistent_state_bytes"] == 14 * 8 * 4
    assert results[0]["c_latency_us"] > 0
    assert results[1]["student_val_rmse_cycles"] > 0
