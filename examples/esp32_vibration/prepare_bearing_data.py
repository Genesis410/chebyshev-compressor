#!/usr/bin/env python3
"""Bearing Vibration Anomaly Compression & ESP32 Header Export.

Demonstrates compressing a 3-channel vibration recurrent network into a 96-byte
recurrent state Chebyshev engine (D_in=3, K=8).

Includes a deterministic synthetic bearing fault generator with periodic impact
harmonics mimicking CWRU/IMS bearing fault dynamics when external servers are offline.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from chebyshev.distill import distill
from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel
from chebyshev.teacher import CMapssLSTMTeacher, train_teacher
from chebyshev.datasets import create_dataloader


def generate_bearing_vibration_data(num_samples: int = 120, seq_len: int = 40, seed: int = 42):
    """Generates 3-axis vibration data (X: radial, Y: tangential, Z: axial).
    
    Simulates rotating machinery bearing degradation:
      - Channel 0 (Radial): Shaft rotation (30 Hz) + bearing race defect impulses.
      - Channel 1 (Tangential): Harmonics + increasing vibration envelope.
      - Channel 2 (Axial): Structural resonance + broadband noise.
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / 250.0  # 250 Hz sampling rate
    t = np.linspace(0, (seq_len - 1) * dt, seq_len, dtype=np.float32)

    X_list = []
    y_list = []

    for i in range(num_samples):
        # Health condition: 0.0 (healthy) -> 1.0 (severe bearing fault)
        denom = float(num_samples - 1) if num_samples > 1 else 1.0
        fault_severity = float(i) / denom

        # 30 Hz shaft rotation
        shaft = np.sin(2.0 * np.pi * 30.0 * t)

        # 105 Hz defect pass impact ringing
        defect_pulses = np.sin(2.0 * np.pi * 105.0 * t) * np.exp(-((t % 0.05) / 0.01))
        radial = shaft + (1.5 * fault_severity) * defect_pulses + 0.1 * rng.randn(seq_len)
        tangential = 0.8 * shaft + (2.0 * fault_severity) * np.sin(2.0 * np.pi * 60.0 * t) + 0.1 * rng.randn(seq_len)
        axial = 0.5 * shaft + (0.8 * fault_severity) * defect_pulses + 0.15 * rng.randn(seq_len)

        sample = np.stack([radial, tangential, axial], axis=-1).astype(np.float32)
        X_list.append(sample)
        # Target: Anomaly score in [0.0, 1.0]
        y_list.append(np.float32(fault_severity))

    X = np.stack(X_list, axis=0)  # (N, T, 3)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)  # (N, 1)
    return X, y


def generate_c_sample_header(sample_sequence: np.ndarray, output_path: Path):
    """Exports a 300-step test sequence into a C header for zero-hardware ESP32 evaluation."""
    T, D = sample_sequence.shape
    lines = [
        "/* Auto-generated sample vibration test vector (300 steps x 3 channels) */",
        "#ifndef VIBRATION_SAMPLE_DATA_H",
        "#define VIBRATION_SAMPLE_DATA_H",
        "",
        f"#define VIBRATION_TEST_STEPS {T}",
        f"#define VIBRATION_TEST_CHANNELS {D}",
        "",
        "static const float VIBRATION_TEST_DATA[VIBRATION_TEST_STEPS][VIBRATION_TEST_CHANNELS] = {",
    ]
    for row in sample_sequence:
        row_str = ", ".join(f"{val:.6g}f" for val in row)
        lines.append(f"    {{{row_str}}},")
    lines.extend([
        "};",
        "",
        "#endif /* VIBRATION_SAMPLE_DATA_H */",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    out_dir = Path(__file__).parent
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("CHEBYSHEV-COMPRESSOR: BEARING VIBRATION DEMO PIPELINE")
    print("=" * 70)

    # 1. Generate 3-channel vibration dataset
    print("[1/4] Generating 3-axis vibration dataset (N=160, T=40, D_in=3)...")
    X, y = generate_bearing_vibration_data(num_samples=160, seq_len=40, seed=seed)
    train_loader = create_dataloader(X[:120], y[:120], batch_size=30, shuffle=True)
    val_loader = create_dataloader(X[120:], y[120:], batch_size=40, shuffle=False)

    # Fit channel normalization on train split
    flat_train = X[:120].reshape(-1, 3)
    mean = np.mean(flat_train, axis=0).astype(np.float32)
    std = np.std(flat_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)

    # 2. Train LSTM baseline teacher
    print("[2/4] Training baseline PyTorch LSTM teacher (3 channels, H=16, 2 layers)...")
    teacher = CMapssLSTMTeacher(num_channels=3, hidden_dim=16, num_layers=2, out_features=1)
    train_teacher(teacher, train_loader, epochs=15, lr=5e-3, verbose=False)

    # 3. Distill into Chebyshev Student (Option A, order K=8 -> 96 B state RAM!)
    print("[3/4] Distilling to Chebyshev Student (D_in=3, K=8, persistent state = 96 Bytes)...")
    student = StudentModel(num_channels=3, order=8, hidden_dim=16, out_features=1, window_size=30.0)
    student.cell.set_normalization(mean, std)
    distill(teacher=teacher, student=student, dataloader=train_loader, epochs=25, lr=5e-3, distill_lambda=0.5, verbose=False)

    # 4. Export standalone C99 engine & sample data header
    engine_h = out_dir / "chebyshev_engine.h"
    data_h = out_dir / "vibration_sample_data.h"
    print(f"[4/4] Exporting strict C99 single-header engine -> {engine_h.name}")
    export_to_c_header(student, engine_h)

    # Generate 300 steps continuous vibration stream for ESP32 demo (~3.6 KB .rodata)
    demo_stream, _ = generate_bearing_vibration_data(num_samples=1, seq_len=300, seed=77)
    generate_c_sample_header(demo_stream[0], data_h)
    print(f"      Exported test vector ({data_h.stat().st_size:,} bytes) -> {data_h.name}")

    print("=" * 70)
    print("SUCCESS: ESP32 Vibration demo assets generated!")
    print(f"  Persistent State RAM : 96 Bytes (3 channels x 8 order x 4 bytes)")
    print(f"  Flash Test Vector    : ~3.6 KB (safe for ESP32 1.2 MB default app partition)")
    print("=" * 70)


if __name__ == "__main__":
    main()
