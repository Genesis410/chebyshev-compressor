import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pytest
import torch

from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel


def find_c_compiler() -> Optional[Tuple[str, str]]:
    """Discovers available host C compiler (gcc, clang, cl)."""
    for name in ["gcc", "clang", "cl"]:
        path = shutil.which(name)
        if path:
            return path, name
    return None


C_RUNNER_TEMPLATE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "chebyshev_engine.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input: %s\n", argv[1]);
        return 2;
    }
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
        fclose(fin);
        fprintf(stderr, "Failed to open output: %s\n", argv[2]);
        return 3;
    }

    int num_steps = 0;
    if (fread(&num_steps, sizeof(int), 1, fin) != 1) {
        fclose(fin);
        fclose(fout);
        return 4;
    }

    chebyshev_state_t state;
    chebyshev_init(&state);

    float raw_in[CHEB_NUM_CHANNELS];
    float out[CHEB_NUM_OUTPUTS];

    for (int t = 0; t < num_steps; t++) {
        if (fread(raw_in, sizeof(float), CHEB_NUM_CHANNELS, fin) != CHEB_NUM_CHANNELS) {
            break;
        }
        chebyshev_step(&state, raw_in, out);
        fwrite(out, sizeof(float), CHEB_NUM_OUTPUTS, fout);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}
"""


def compile_c_runner(
    compiler_path: str,
    compiler_type: str,
    include_dir: Path,
    source_path: Path,
    output_exe: Path,
    extra_flags: Optional[List[str]] = None,
):
    """Compiles runner with strict C99 flags to verify zero warnings."""
    if compiler_type in ["gcc", "clang"]:
        cmd = [
            compiler_path,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            *(extra_flags or []),
            f"-I{include_dir}",
            str(source_path),
            "-o",
            str(output_exe),
        ]
    else:  # MSVC cl.exe
        cmd = [
            compiler_path,
            "/std:c11",
            "/W4",
            "/WX",
            "/O2",
            *(extra_flags or []),
            f"/I{include_dir}",
            str(source_path),
            f"/Fe:{output_exe}",
        ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"C compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_compiler_discovery():
    compiler = find_c_compiler()
    assert compiler is not None, "No C compiler (gcc, clang, cl) found in PATH for testing."


def test_strict_c99_compilation_no_warnings(tmp_path: Path):
    compiler = find_c_compiler()
    if not compiler:
        pytest.skip("No C compiler found")
    compiler_path, compiler_type = compiler

    model = StudentModel(num_channels=4, order=6, hidden_dim=16, out_features=1)
    header_path = tmp_path / "chebyshev_engine.h"
    export_to_c_header(model, header_path)

    runner_src = tmp_path / "test_compile.c"
    runner_src.write_text(C_RUNNER_TEMPLATE, encoding="utf-8")
    output_exe = tmp_path / "test_compile.exe"

    compile_c_runner(compiler_path, compiler_type, tmp_path, runner_src, output_exe)
    assert output_exe.exists()


def test_pytorch_c99_numerical_parity_1000_steps(tmp_path: Path):
    compiler = find_c_compiler()
    if not compiler:
        pytest.skip("No C compiler found")
    compiler_path, compiler_type = compiler

    num_channels = 14
    order = 8
    hidden_dim = 32
    out_features = 1

    # 1. Initialize PyTorch model with non-trivial parameters and baked normalization
    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    model.eval()

    mean = np.linspace(10.0, 100.0, num_channels, dtype=np.float32)
    std = np.linspace(2.0, 15.0, num_channels, dtype=np.float32)
    model.cell.set_normalization(mean, std)

    # 2. Export header and compile C runner
    header_path = tmp_path / "chebyshev_engine.h"
    export_to_c_header(model, header_path)

    runner_src = tmp_path / "runner.c"
    runner_src.write_text(C_RUNNER_TEMPLATE, encoding="utf-8")
    output_exe = tmp_path / "runner.exe"
    compile_c_runner(compiler_path, compiler_type, tmp_path, runner_src, output_exe)

    # 3. Construct 1,000 steps test sequence with normal, impulse, NaN, and zero phases
    T = 1000
    rng = np.random.RandomState(42)
    x = rng.randn(T, num_channels).astype(np.float32) * std + mean

    # Phase 2 (steps 400-500): Extreme outlier spikes triggering input clamping
    x[400:500] += 5000.0

    # Phase 3 (steps 600-650): Injected NaNs and Infs triggering NaN quarantine
    x[600:625, 0:5] = np.nan
    x[625:650, 5:10] = np.inf

    # Phase 4 (steps 700-800): All zeros
    x[700:800] = 0.0

    # Phase 5 (steps 800-1000): Resumed normal signal to test recovery

    # 4. Run PyTorch sequence inference
    with torch.no_grad():
        pytorch_out = model(
            torch.from_numpy(x).unsqueeze(0), return_sequence=True
        ).squeeze(0).cpu().numpy()

    # 5. Run compiled C executable
    input_bin = tmp_path / "input.bin"
    output_bin = tmp_path / "output.bin"

    with open(input_bin, "wb") as f:
        f.write(np.int32(T).tobytes())
        f.write(x.tobytes())

    res = subprocess.run([str(output_exe), str(input_bin), str(output_bin)], capture_output=True, text=True)
    assert res.returncode == 0, f"Runner execution failed: {res.stderr}"

    with open(output_bin, "rb") as f:
        c_out = np.frombuffer(f.read(), dtype=np.float32).reshape(T, out_features)

    # 6. Verify bit-accurate numerical parity (< 1e-4) across all 1,000 steps
    max_abs_error = np.max(np.abs(pytorch_out - c_out))
    mean_abs_error = np.mean(np.abs(pytorch_out - c_out))

    assert max_abs_error < 1e-4, f"Parity mismatch! max_error={max_abs_error:.6e}, mean_error={mean_abs_error:.6e}"


def test_multi_output_parity(tmp_path: Path):
    compiler = find_c_compiler()
    if not compiler:
        pytest.skip("No C compiler found")
    compiler_path, compiler_type = compiler

    num_channels = 6
    order = 4
    hidden_dim = 16
    out_features = 3
    T = 200

    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    model.eval()

    header_path = tmp_path / "chebyshev_engine.h"
    export_to_c_header(model, header_path)

    runner_src = tmp_path / "runner_multi.c"
    runner_src.write_text(C_RUNNER_TEMPLATE, encoding="utf-8")
    output_exe = tmp_path / "runner_multi.exe"
    compile_c_runner(compiler_path, compiler_type, tmp_path, runner_src, output_exe)

    x = np.random.randn(T, num_channels).astype(np.float32)

    with torch.no_grad():
        pytorch_out = model(torch.from_numpy(x).unsqueeze(0), return_sequence=True).squeeze(0).cpu().numpy()

    input_bin = tmp_path / "input_multi.bin"
    output_bin = tmp_path / "output_multi.bin"

    with open(input_bin, "wb") as f:
        f.write(np.int32(T).tobytes())
        f.write(x.tobytes())

    res = subprocess.run([str(output_exe), str(input_bin), str(output_bin)], capture_output=True, text=True)
    assert res.returncode == 0

    with open(output_bin, "rb") as f:
        c_out = np.frombuffer(f.read(), dtype=np.float32).reshape(T, out_features)

    max_abs_error = np.max(np.abs(pytorch_out - c_out))
    assert max_abs_error < 1e-4, f"Multi-output parity mismatch: max_error={max_abs_error:.6e}"


def test_fastmath_ieee754_nan_inf_quarantine_and_parity(tmp_path: Path):
    """Verifies IEEE-754 memcpy exponent bit-mask check quarantines NaN/+Inf/-Inf
    deterministically even when compiled with -O3 -ffast-math (-ffinite-math-only).
    """
    compiler = find_c_compiler()
    if not compiler:
        pytest.skip("No C compiler found")
    compiler_path, compiler_type = compiler

    torch.manual_seed(77)
    num_channels = 8
    order = 8
    hidden_dim = 16
    out_features = 1
    T = 120

    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    model.eval()
    mean = np.linspace(-5.0, 20.0, num_channels, dtype=np.float32)
    std = np.linspace(1.0, 5.0, num_channels, dtype=np.float32)
    model.cell.set_normalization(mean, std)

    header_path = tmp_path / "chebyshev_engine.h"
    export_to_c_header(model, header_path)

    runner_src = tmp_path / "runner_fastmath.c"
    runner_src.write_text(C_RUNNER_TEMPLATE, encoding="utf-8")
    output_exe = tmp_path / "runner_fastmath.exe"

    fastmath_flags = ["-O3", "-ffast-math"] if compiler_type in ["gcc", "clang"] else ["/O2", "/fp:fast"]
    compile_c_runner(
        compiler_path,
        compiler_type,
        tmp_path,
        runner_src,
        output_exe,
        extra_flags=fastmath_flags,
    )

    rng = np.random.RandomState(77)
    x = rng.randn(T, num_channels).astype(np.float32) * std + mean

    # Inject NaN, +Inf, -Inf, and extreme finite spikes mid-stream
    x[20:30, 0:3] = np.nan
    x[30:40, 2:5] = np.inf
    x[40:50, 4:7] = -np.inf
    x[50:60, :] = 1e6
    x[60:70, :] = -1e6

    with torch.no_grad():
        pytorch_out = model(
            torch.from_numpy(x).unsqueeze(0), return_sequence=True
        ).squeeze(0).cpu().numpy()

    input_bin = tmp_path / "input_fastmath.bin"
    output_bin = tmp_path / "output_fastmath.bin"
    with open(input_bin, "wb") as f:
        f.write(np.int32(T).tobytes())
        f.write(x.tobytes())

    res = subprocess.run([str(output_exe), str(input_bin), str(output_bin)], capture_output=True, text=True)
    assert res.returncode == 0, f"Fast-math runner execution failed: {res.stderr}"

    with open(output_bin, "rb") as f:
        c_out = np.frombuffer(f.read(), dtype=np.float32).reshape(T, out_features)

    # Zero state poisoning under -ffast-math
    assert np.all(np.isfinite(c_out)), "State poisoning detected: non-finite C output under -ffast-math!"
    assert np.all(np.isfinite(pytorch_out)), "Non-finite PyTorch output detected!"

    # Exact parity across normal, NaN, +Inf, -Inf, and extreme spike phases
    max_abs_error = float(np.max(np.abs(pytorch_out - c_out)))
    assert max_abs_error < 1e-4, f"Fast-math C/PyTorch parity mismatch: max_error={max_abs_error:.6e}"

