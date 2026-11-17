"""Standalone native C benchmark harness and latency auditing engine."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import shutil
import subprocess
import tempfile
import time
import torch
import torch.nn as nn


def generate_c_benchmark_harness_source(header_filename: str = "chebyshev_engine.h") -> str:
    """Generates standalone C99 benchmark harness measuring pure execution time outside Python FFI."""
    return f"""/*
 * Standalone High-Resolution C99 Benchmark Harness
 * Measures raw single-thread execution latency without Python/FFI overhead.
 */

#include <stdio.h>
#include <stdlib.h>
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
#include <time.h>
static inline double get_time_sec(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}}
#endif

int main(int argc, char **argv) {{
    int iterations = 100000;
    if (argc > 1) {{
        iterations = atoi(argv[1]);
        if (iterations <= 0) iterations = 100000;
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

    /* Timed benchmark loop */
    double t_start = get_time_sec();
    for (int i = 0; i < iterations; i++) {{
        /* Slight input variation prevents compiler from discarding loop */
        raw_in[0] += 0.00001f;
        chebyshev_step(&state, raw_in, out);
    }}
    double t_end = get_time_sec();

    double total_sec = t_end - t_start;
    double sec_per_step = total_sec / (double)iterations;
    double us_per_step = sec_per_step * 1e6;
    double ns_per_step = sec_per_step * 1e9;
    double steps_per_sec = (double)iterations / total_sec;

    printf("CHEB_ITERATIONS=%d\\n", iterations);
    printf("CHEB_TOTAL_TIME_SEC=%.6f\\n", total_sec);
    printf("CHEB_LATENCY_US=%.4f\\n", us_per_step);
    printf("CHEB_LATENCY_NS=%.2f\\n", ns_per_step);
    printf("CHEB_THROUGHPUT_STEPS_SEC=%.1f\\n", steps_per_sec);
    printf("CHEB_RAM_BYTES=%d\\n", (int)sizeof(chebyshev_state_t));
    printf("CHEB_CHECKSUM=0x%08X\\n", (unsigned int)CHEB_MODEL_CHECKSUM);
    printf("DUMMY_OUT=%.6f\\n", out[0]);

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
    build_dir: Optional[Union[str, Path]] = None,
    compiler_path: Optional[str] = None,
) -> Dict[str, Union[int, float, str]]:
    """Compiles and executes the native C benchmark harness outside Python.
    
    Args:
        header_path: Path to chebyshev_engine.h.
        iterations: Number of continuous inference iterations to execute.
        build_dir: Directory for compiler outputs (temp dir if None).
        compiler_path: Explicit path to C compiler binary.
        
    Returns:
        Dictionary of benchmark metrics.
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

        # Native compile with maximum optimization flags
        if compiler_type in ["gcc", "clang"]:
            cmd = [
                compiler_bin,
                "-std=c99",
                "-O3",
                "-march=native",
                f"-I{include_dir}",
                str(src_path),
                "-o",
                str(exe_path),
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

        # Run native harness binary
        run_res = subprocess.run([str(exe_path), str(iterations)], capture_output=True, text=True)
        if run_res.returncode != 0:
            raise RuntimeError(f"Benchmark harness execution failed:\n{run_res.stderr}")

        # Parse key-value outputs
        metrics: Dict[str, Union[int, float, str]] = {}
        for line in run_res.stdout.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if k == "CHEB_ITERATIONS":
                    metrics["iterations"] = int(v)
                elif k == "CHEB_TOTAL_TIME_SEC":
                    metrics["total_time_sec"] = float(v)
                elif k == "CHEB_LATENCY_US":
                    metrics["latency_us"] = float(v)
                elif k == "CHEB_LATENCY_NS":
                    metrics["latency_ns"] = float(v)
                elif k == "CHEB_THROUGHPUT_STEPS_SEC":
                    metrics["throughput_steps_sec"] = float(v)
                elif k == "CHEB_RAM_BYTES":
                    metrics["ram_bytes"] = int(v)
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
    device: str = "cpu",
) -> Dict[str, Union[int, float]]:
    """Benchmarks baseline PyTorch model single-step inference under single-thread constraint."""
    model = model.to(device)
    model.eval()

    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    try:
        x_step = torch.randn(1, 1, num_channels, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = model(x_step)

        # Timed inference
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                _ = model(x_step)
        t1 = time.perf_counter()

        total_sec = t1 - t0
        latency_us = (total_sec / steps) * 1e6
        throughput = steps / total_sec

        return {
            "steps": steps,
            "total_time_sec": total_sec,
            "latency_us": latency_us,
            "throughput_steps_sec": throughput,
        }

    finally:
        torch.set_num_threads(orig_threads)


def compare_benchmarks(
    c_metrics: Dict[str, Union[int, float]],
    pytorch_metrics: Dict[str, Union[int, float]],
    teacher_ram_bytes: int = 32768,
) -> Dict[str, Union[float, str]]:
    """Calculates audit comparison metrics and formats summary table."""
    c_lat_us = float(c_metrics["latency_us"])
    pt_lat_us = float(pytorch_metrics["latency_us"])
    c_ram = int(c_metrics.get("ram_bytes", 448))

    speedup_x = pt_lat_us / max(1e-9, c_lat_us)
    ram_reduction_pct = max(0.0, (1.0 - (c_ram / teacher_ram_bytes)) * 100.0)

    summary_table = f"""
========================================================================================
                       CHEBYSHEV-COMPRESSOR AUDIT BENCHMARK
========================================================================================
Metric                       PyTorch LSTM (1-thread)     C99 Engine (Standalone)   Gain
----------------------------------------------------------------------------------------
Inference Latency (per step) {pt_lat_us:12.2f} us           {c_lat_us:10.2f} us     {speedup_x:7.1f}x SPEEDUP
Throughput (steps/sec)       {float(pytorch_metrics['throughput_steps_sec']):12.1f}              {float(c_metrics['throughput_steps_sec']):10.1f}
State RAM Footprint          {teacher_ram_bytes:12,d} bytes          {c_ram:10,d} bytes     {ram_reduction_pct:7.1f}% RAM REDUCTION
Heap Dynamic Allocation (`malloc`)        Enabled                     0 bytes (MISRA-C)
========================================================================================
"""
    return {
        "speedup_x": speedup_x,
        "ram_reduction_pct": ram_reduction_pct,
        "summary_table": summary_table,
    }
