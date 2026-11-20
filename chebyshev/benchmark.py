"""Standalone native C benchmark harness, end-to-end streaming audit, and order ablation utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import shutil
import subprocess
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn


def generate_c_benchmark_harness_source(header_filename: str = "chebyshev_engine.h") -> str:
    """Generates standalone C99 benchmark harness measuring both single-step and end-to-end window latency."""
    return f"""/*
 * Standalone High-Resolution C99 Benchmark Harness
 * Measures raw single-thread execution latency (single-step and full sequence window)
 * outside Python/FFI overhead.
 */

#if !defined(_WIN32)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "{header_filename}"

#if defined(_WIN32)
#include <windows.h>
static inline double get_time_sec(void) {{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}}
#else
static inline double get_time_sec(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}}
#endif

#ifndef CHEB_STACK_SCRATCH_BYTES
#define CHEB_STACK_SCRATCH_BYTES ((CHEB_ORDER + CHEB_HIDDEN_DIM) * (int)sizeof(float))
#endif

int main(int argc, char **argv) {{
    int iterations = 100000;
    int seq_len = 50;
    if (argc > 1) {{
        iterations = atoi(argv[1]);
        if (iterations <= 0) iterations = 100000;
    }}
    if (argc > 2) {{
        seq_len = atoi(argv[2]);
        if (seq_len <= 0) seq_len = 50;
    }}

    chebyshev_state_t state;
    chebyshev_init(&state);

    float raw_in[CHEB_NUM_CHANNELS];
    for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {{
        raw_in[d] = 1.0f + (float)d * 0.1f;
    }}
    float out[CHEB_NUM_OUTPUTS];

    /* Warmup 1,000 iterations */
    for (int i = 0; i < 1000; i++) {{
        chebyshev_step(&state, raw_in, out);
    }}

    /* 1. Single-step streaming microbenchmark (volatile sink prevents -O3 DCE of Readout MLP) */
    volatile float sink = 0.0f;
    double t_start = get_time_sec();
    for (int i = 0; i < iterations; i++) {{
        raw_in[0] += 0.00001f;
        chebyshev_step(&state, raw_in, out);
        sink += out[0];
    }}
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double sec_per_step = total_sec / (double)iterations;
    double us_per_step = sec_per_step * 1e6;
    double ns_per_step = sec_per_step * 1e9;
    double steps_per_sec = (double)iterations / total_sec;

    /* 2. End-to-end sequence window streaming benchmark (includes state init + seq_len raw steps) */
    int window_iters = iterations / seq_len;
    if (window_iters < 200) window_iters = 200;

    double w_start = get_time_sec();
    for (int w = 0; w < window_iters; w++) {{
        chebyshev_init(&state);
        for (int t = 0; t < seq_len; t++) {{
            raw_in[0] = 1.0f + (float)(t & 15) * 0.01f;
            chebyshev_step(&state, raw_in, out);
        }}
        sink += out[0];
    }}
    double w_end = get_time_sec();

    double w_total_sec = w_end - w_start;
    double w_us_per_window = (w_total_sec / (double)window_iters) * 1e6;
    double w_us_per_step = w_us_per_window / (double)seq_len;

    int persistent_bytes = (int)sizeof(chebyshev_state_t);
    int stack_scratch_bytes = (int)CHEB_STACK_SCRATCH_BYTES;
    int total_sram_bytes = persistent_bytes + stack_scratch_bytes;

    printf("CHEB_ITERATIONS=%d\\n", iterations);
    printf("CHEB_TOTAL_TIME_SEC=%.6f\\n", total_sec);
    printf("CHEB_LATENCY_US=%.4f\\n", us_per_step);
    printf("CHEB_LATENCY_NS=%.2f\\n", ns_per_step);
    printf("CHEB_THROUGHPUT_STEPS_SEC=%.1f\\n", steps_per_sec);
    printf("CHEB_WINDOW_SEQ_LEN=%d\\n", seq_len);
    printf("CHEB_WINDOW_ITERS=%d\\n", window_iters);
    printf("CHEB_WINDOW_LATENCY_US=%.4f\\n", w_us_per_window);
    printf("CHEB_WINDOW_AMORTIZED_STEP_US=%.4f\\n", w_us_per_step);
    printf("CHEB_RAM_BYTES=%d\\n", persistent_bytes);
    printf("CHEB_PERSISTENT_STATE_BYTES=%d\\n", persistent_bytes);
    printf("CHEB_STACK_SCRATCH_BYTES=%d\\n", stack_scratch_bytes);
    printf("CHEB_TOTAL_SRAM_BYTES=%d\\n", total_sram_bytes);
    printf("CHEB_CHECKSUM=0x%08X\\n", (unsigned int)CHEB_MODEL_CHECKSUM);
    printf("DUMMY_OUT=%.6f\\n", (float)sink);

    return 0;
}}
"""


def find_compiler() -> Tuple[str, str]:
    """Finds available host C compiler."""
    for name in ["gcc", "clang", "cl"]:
        path = shutil.which(name)
        if path:
            return path, name
    raise RuntimeError("No host C compiler (gcc, clang, cl) found in PATH.")


def run_c_benchmark(
    header_path: Union[str, Path],
    iterations: int = 100000,
    seq_len: int = 50,
    build_dir: Optional[Union[str, Path]] = None,
    compiler_path: Optional[str] = None,
) -> Dict[str, Union[int, float, str]]:
    """Compiles and executes the native C benchmark harness outside Python.

    Args:
        header_path: Path to chebyshev_engine.h.
        iterations: Number of continuous inference iterations to execute.
        seq_len: Sequence window length for end-to-end streaming window benchmark.
        build_dir: Directory for compiler outputs (temp dir if None).
        compiler_path: Explicit path to C compiler binary.

    Returns:
        Dictionary of single-step, end-to-end window, and SRAM footprint metrics.
    """
    header_path = Path(header_path).resolve()
    include_dir = header_path.parent

    cleanup_dir = False
    if build_dir is None:
        build_path = Path(tempfile.mkdtemp(prefix="cheb_bench_"))
        cleanup_dir = True
    else:
        build_path = Path(build_dir)
        build_path.mkdir(parents=True, exist_ok=True)

    try:
        if compiler_path is None:
            compiler_bin, compiler_type = find_compiler()
        else:
            compiler_bin = compiler_path
            compiler_type = "msvc" if "cl" in Path(compiler_bin).name.lower() else "gcc"

        src_path = build_path / "bench_harness.c"
        src_path.write_text(generate_c_benchmark_harness_source(header_path.name), encoding="utf-8")

        exe_suffix = ".exe" if subprocess.os.name == "nt" else ""
        exe_path = build_path / f"bench_harness{exe_suffix}"

        if compiler_type in ["gcc", "clang"]:
            cmd = [
                compiler_bin,
                "-std=c99",
                "-O3",
                f"-I{include_dir}",
                str(src_path),
                "-o",
                str(exe_path),
                "-lm",
            ]
        else:  # MSVC
            cmd = [
                compiler_bin,
                "/std:c11",
                "/O2",
                f"/I{include_dir}",
                str(src_path),
                f"/Fe:{exe_path}",
            ]

        compile_res = subprocess.run(cmd, capture_output=True, text=True)
        if compile_res.returncode != 0:
            raise RuntimeError(
                f"Benchmark harness compilation failed:\n{compile_res.stderr}\n{compile_res.stdout}"
            )

        run_res = subprocess.run(
            [str(exe_path), str(iterations), str(seq_len)], capture_output=True, text=True
        )
        if run_res.returncode != 0:
            raise RuntimeError(f"Benchmark harness execution failed:\n{run_res.stderr}")

        metrics: Dict[str, Union[int, float, str]] = {}
        int_keys = {
            "CHEB_ITERATIONS": "iterations",
            "CHEB_WINDOW_SEQ_LEN": "window_seq_len",
            "CHEB_WINDOW_ITERS": "window_iters",
            "CHEB_RAM_BYTES": "ram_bytes",
            "CHEB_PERSISTENT_STATE_BYTES": "persistent_state_bytes",
            "CHEB_STACK_SCRATCH_BYTES": "stack_scratch_bytes",
            "CHEB_TOTAL_SRAM_BYTES": "total_sram_bytes",
        }
        float_keys = {
            "CHEB_TOTAL_TIME_SEC": "total_time_sec",
            "CHEB_LATENCY_US": "latency_us",
            "CHEB_LATENCY_NS": "latency_ns",
            "CHEB_THROUGHPUT_STEPS_SEC": "throughput_steps_sec",
            "CHEB_WINDOW_LATENCY_US": "window_latency_us",
            "CHEB_WINDOW_AMORTIZED_STEP_US": "window_amortized_step_us",
        }

        for line in run_res.stdout.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if k in int_keys:
                    metrics[int_keys[k]] = int(v)
                elif k in float_keys:
                    metrics[float_keys[k]] = float(v)
                elif k == "CHEB_CHECKSUM":
                    metrics["checksum"] = v

        return metrics

    finally:
        if cleanup_dir:
            shutil.rmtree(build_path, ignore_errors=True)


def run_pytorch_benchmark(
    model: nn.Module,
    num_channels: int = 14,
    steps: int = 2000,
    seq_len: int = 50,
    window_runs: int = 200,
    device: str = "cpu",
) -> Dict[str, Union[int, float]]:
    """Benchmarks baseline PyTorch model under single-thread constraint for both 1-step and full window."""
    model = model.to(device)
    model.eval()

    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    try:
        x_step = torch.randn(1, 1, num_channels, device=device)
        x_window = torch.randn(1, seq_len, num_channels, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(30):
                _ = model(x_step)
            for _ in range(10):
                _ = model(x_window)

        # 1. Single-step timing (includes Python dispatch overhead)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                _ = model(x_step)
        t1 = time.perf_counter()

        total_sec = t1 - t0
        latency_us = (total_sec / steps) * 1e6
        throughput = steps / total_sec

        # 2. End-to-end sequence window timing (fair comparison across seq_len steps)
        w0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(window_runs):
                _ = model(x_window)
        w1 = time.perf_counter()

        window_total_sec = w1 - w0
        window_latency_us = (window_total_sec / window_runs) * 1e6
        window_amortized_step_us = window_latency_us / float(seq_len)

        return {
            "steps": steps,
            "total_time_sec": total_sec,
            "latency_us": latency_us,
            "throughput_steps_sec": throughput,
            "window_seq_len": seq_len,
            "window_runs": window_runs,
            "window_latency_us": window_latency_us,
            "window_amortized_step_us": window_amortized_step_us,
        }

    finally:
        torch.set_num_threads(orig_threads)


def compare_benchmarks(
    c_metrics: Dict[str, Union[int, float, str]],
    pytorch_metrics: Dict[str, Union[int, float]],
    teacher_state_bytes: int = 512,
    teacher_tflm_arena_bytes: int = 32768,
    teacher_ram_bytes: Optional[int] = None,
) -> Dict[str, Union[float, str]]:
    """Calculates audit comparison metrics and formats a transparent single-step + window summary table.

    Distinguishes:
      1. Pure mathematical recurrent state vector (LSTM 2-layer H=32: 2*2*32*4 = 512 B vs Chebyshev 14x8x4 = 448 B).
      2. Embedded runtime deployment SRAM (TensorFlow Lite Micro tensor arena: 32,768 B vs Chebyshev C99 engine: 608 B).
    """
    if teacher_ram_bytes is not None:
        teacher_tflm_arena_bytes = teacher_ram_bytes

    c_lat_us = float(c_metrics["latency_us"])
    pt_lat_us = float(pytorch_metrics["latency_us"])
    c_persistent = int(c_metrics.get("persistent_state_bytes", c_metrics.get("ram_bytes", 448)))
    c_stack = int(c_metrics.get("stack_scratch_bytes", 160))
    c_total_sram = int(c_metrics.get("total_sram_bytes", c_persistent + c_stack))

    speedup_x = pt_lat_us / max(1e-9, c_lat_us)
    state_reduction_pct = max(0.0, (1.0 - (c_persistent / teacher_state_bytes)) * 100.0)
    arena_reduction_pct = max(0.0, (1.0 - (c_total_sram / teacher_tflm_arena_bytes)) * 100.0)

    c_win_us = float(c_metrics.get("window_latency_us", c_lat_us * 50.0))
    pt_win_us = float(pytorch_metrics.get("window_latency_us", pt_lat_us * 50.0))
    seq_len = int(c_metrics.get("window_seq_len", pytorch_metrics.get("window_seq_len", 50)))
    window_speedup_x = pt_win_us / max(1e-9, c_win_us)

    summary_table = f"""
========================================================================================
                       CHEBYSHEV-COMPRESSOR AUDIT BENCHMARK
========================================================================================
Metric                                PyTorch LSTM (1-thread)   C99 Engine (Standalone)   Gain / Analysis
----------------------------------------------------------------------------------------
Single-Step Host Latency*             {pt_lat_us:12.2f} us          {c_lat_us:10.2f} us       {speedup_x:7.1f}x SPEEDUP
End-to-End Window Latency (T={seq_len:<2d})      {pt_win_us:12.2f} us          {c_win_us:10.2f} us       {window_speedup_x:7.1f}x SPEEDUP
Host Throughput (streaming steps/sec) {float(pytorch_metrics['throughput_steps_sec']):12.1f}             {float(c_metrics['throughput_steps_sec']):10.1f}
Persistent Recurrent State (.bss)     {teacher_state_bytes:12,d} bytes       {c_persistent:10,d} bytes   {state_reduction_pct:7.1f}% state reduction (512 vs 448 B)
Active Call-Stack Scratch (local)              N/A             {c_stack:10,d} bytes       next_s[8] + h1[32]
Total Estimated Runtime SRAM (vs TFLM){teacher_tflm_arena_bytes:12,d} bytes       {c_total_sram:10,d} bytes   {arena_reduction_pct:7.1f}% SRAM REDUCTION (32 KB arena vs 608 B)
Dynamic Heap Allocation (`malloc`)          Enabled                  0 bytes       Deterministic real-time execution
========================================================================================
* Note: Single-step PyTorch latency includes Python/dispatcher overhead; End-to-End
  Window Latency compares a full T={seq_len} sequence ingestion including state initialization.
  Baseline state math: 2 layers x 2 states (h, c) x 32 hidden x 4 bytes = 512 bytes.
  TFLM Tensor Arena: typical ~32 KB SRAM workspace required by TensorFlow Lite Micro LSTM op.
"""
    return {
        "speedup_x": speedup_x,
        "window_speedup_x": window_speedup_x,
        "state_reduction_pct": state_reduction_pct,
        "arena_reduction_pct": arena_reduction_pct,
        "ram_reduction_pct": arena_reduction_pct,  # Backwards compatibility alias
        "persistent_state_bytes": float(c_persistent),
        "stack_scratch_bytes": float(c_stack),
        "total_sram_bytes": float(c_total_sram),
        "summary_table": summary_table,
    }


def run_order_ablation(
    orders: Sequence[int] = (4, 8, 16, 32),
    window_size: int = 30,
    hidden_dim: int = 32,
    epochs: int = 15,
    seed: int = 42,
    c_iterations: int = 20000,
    max_engines: int = 25,
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Runs reproducible ablation across polynomial orders K in {4, 8, 16, 32}.

    Measures signal reconstruction L2 error, held-out C-MAPSS validation RMSE,
    persistent state & stack SRAM bytes, and native C99 streaming latency.
    """
    from chebyshev.datasets import create_dataloader, load_cmapss_fd001_splits
    from chebyshev.distill import distill, evaluate_rmse
    from chebyshev.exporter import export_to_c_header
    from chebyshev.math_ops import get_chebyshev_operators, reconstruct_chebyshev_history
    from chebyshev.models import StudentModel
    from chebyshev.teacher import CMapssLSTMTeacher, train_teacher
    from chebyshev.utils import set_seed

    set_seed(seed)

    # 1. Load strict engine-level held-out C-MAPSS FD001 split
    splits = load_cmapss_fd001_splits(
        data_dir=data_dir,
        max_engines=max_engines,
        window_size=window_size,
        stride=2,
        val_fraction=0.2,
        seed=seed,
    )
    train_loader = create_dataloader(splits["X_train"], splits["y_train"], batch_size=64, shuffle=True)
    val_loader = create_dataloader(splits["X_val"], splits["y_val"], batch_size=64, shuffle=False)
    rul_scale = splits["max_rul"]

    # 2. Train shared Teacher LSTM once on training engines
    teacher = CMapssLSTMTeacher(num_channels=14, hidden_dim=32, num_layers=2, out_features=1)
    train_teacher(teacher, train_loader, epochs=5, lr=3e-3, verbose=False)
    teacher_val_rmse = evaluate_rmse(teacher, val_loader, scale=rul_scale)

    # 3. Signal reconstruction reference setup (omega << K/theta)
    theta = float(window_size)
    dt_rec = 0.05
    t_steps = int(150.0 / dt_rec)
    t_arr = np.arange(t_steps, dtype=np.float64) * dt_rec
    u_arr = np.sin(2.0 * np.pi * t_arr / 20.0) + 0.5 * np.cos(2.0 * np.pi * t_arr / 12.0) + 0.25
    tau_grid = np.linspace(0.0, theta, 200)
    t_now = t_arr[-1] + 0.5 * dt_rec
    u_true = np.sin(2.0 * np.pi * (t_now - tau_grid) / 20.0) + 0.5 * np.cos(
        2.0 * np.pi * (t_now - tau_grid) / 12.0
    ) + 0.25

    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cheb_ablation_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for order in orders:
            set_seed(seed)

            # A. Signal reconstruction L2 error
            A_rec, B_rec = get_chebyshev_operators(order=order, window_size=theta, dt=dt_rec, dtype=np.float64)
            state = np.zeros(order, dtype=np.float64)
            for u_t in u_arr:
                state = A_rec @ state + B_rec * u_t
            u_rec = reconstruct_chebyshev_history(state, tau_grid, theta=theta)
            recon_l2_err = float(np.linalg.norm(u_true - u_rec) / np.linalg.norm(u_true))

            # B. Option A Distillation & Held-out ground-truth RUL RMSE
            student = StudentModel(
                num_channels=14,
                order=order,
                hidden_dim=hidden_dim,
                out_features=1,
                window_size=theta,
            )
            distill(
                teacher=teacher,
                student=student,
                dataloader=train_loader,
                val_dataloader=val_loader,
                channel_mean=splits["mean"],
                channel_std=splits["std"],
                epochs=epochs,
                lr=3e-3,
                distill_lambda=0.5,
                verbose=False,
            )
            val_rmse_cycles = evaluate_rmse(student, val_loader, scale=rul_scale)

            # C. Export C99 & Native Benchmark
            header_path = tmp_path / f"chebyshev_engine_k{order}.h"
            export_to_c_header(student, header_path)
            c_metrics = run_c_benchmark(
                header_path, iterations=c_iterations, seq_len=window_size, build_dir=tmp_path / f"build_k{order}"
            )

            results.append(
                {
                    "order_k": int(order),
                    "recon_rel_l2_error": recon_l2_err,
                    "teacher_val_rmse_cycles": float(teacher_val_rmse),
                    "student_val_rmse_cycles": float(val_rmse_cycles),
                    "persistent_state_bytes": int(c_metrics["persistent_state_bytes"]),
                    "stack_scratch_bytes": int(c_metrics["stack_scratch_bytes"]),
                    "total_sram_bytes": int(c_metrics["total_sram_bytes"]),
                    "c_latency_us": float(c_metrics["latency_us"]),
                    "c_window_latency_us": float(c_metrics["window_latency_us"]),
                }
            )

    return results
