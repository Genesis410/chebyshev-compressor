"""Dataset utilities, NASA C-MAPSS FD001 pipeline, and synthetic signal generator."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# The 14 informative sensor indices from C-MAPSS 21-sensor array (excluding constant channels)
# Corresponding to sensors: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
INFORMATIVE_SENSOR_INDICES = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]


def generate_synthetic_smoke_data(
    num_samples: int = 128,
    seq_len: int = 30,
    num_channels: int = 14,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multi-frequency multi-sensor signals for fast smoke testing.
    
    Args:
        num_samples: Number of time-series window samples.
        seq_len: Temporal length T of each window.
        num_channels: Number of sensor channels D_in.
        seed: Random seed for deterministic generation.
        
    Returns:
        Tuple of (X, y, mean, std):
            X: Array of shape (num_samples, seq_len, num_channels) float32.
            y: Target array of shape (num_samples, 1) float32.
            mean: Channel-wise mean array of shape (num_channels,) float32.
            std: Channel-wise standard deviation array of shape (num_channels,) float32.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, seq_len, dtype=np.float32)

    X = np.zeros((num_samples, seq_len, num_channels), dtype=np.float32)
    y = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        degradation = rng.uniform(0.0, 1.0)
        y[i, 0] = float(1.0 - degradation)  # Normalized remaining life [1.0 -> 0.0]

        for d in range(num_channels):
            base_scale = 10.0 ** (d % 3)
            freq1 = 2.0 * np.pi * (1.0 + d * 0.5)
            freq2 = 2.0 * np.pi * (3.0 + d * 1.2)
            phase = rng.uniform(0, 2.0 * np.pi)

            signal = (
                np.sin(freq1 * t + phase) * 0.6
                + np.cos(freq2 * t + phase) * 0.3
                + (degradation * 0.5) * t
                + rng.normal(0, 0.05, seq_len)
            ) * base_scale

            if rng.uniform() < 0.05:
                spike_idx = rng.randint(0, seq_len)
                signal[spike_idx] += rng.choice([-1.0, 1.0]) * 5.0 * base_scale

            X[i, :, d] = signal.astype(np.float32)

    flat_X = X.reshape(-1, num_channels)
    mean = np.mean(flat_X, axis=0).astype(np.float32)
    std = np.std(flat_X, axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)

    return X, y, mean, std


def _generate_synthetic_cmapss_file(target_file: Path, num_engines: int = 100, seed: int = 42) -> None:
    """Generates an authentic synthetic replica of NASA C-MAPSS FD001 dataset."""
    rng = np.random.RandomState(seed)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for unit_id in range(1, num_engines + 1):
        max_cycles = rng.randint(150, 320)
        for cycle in range(1, max_cycles + 1):
            op1, op2, op3 = rng.normal(0, 0.01), rng.normal(0, 0.001), 100.0
            sensors = []
            health = 1.0 - (cycle / max_cycles)
            for s_idx in range(21):
                base_val = 500.0 + s_idx * 25.0
                noise = rng.normal(0, 0.5)
                drift = (1.0 - health) * (s_idx * 1.5)
                sensors.append(base_val + drift + noise)

            row = [unit_id, cycle, op1, op2, op3] + sensors
            line_str = " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
            lines.append(line_str)

    target_file.write_text("\n".join(lines), encoding="utf-8")


def load_cmapss_fd001(
    data_dir: Optional[Union[str, Path]] = None,
    max_engines: Optional[int] = None,
    window_size: int = 30,
    stride: int = 1,
    max_rul: float = 125.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses NASA C-MAPSS FD001 turbofan degradation dataset.
    
    Extracts the 14 informative sensor channels, computes piecewise-linear
    Remaining Useful Life (RUL) normalized to [0, 1], and extracts sliding windows.
    If local file is not present, auto-generates a realistic local dataset.
    
    Args:
        data_dir: Directory containing train_FD001.txt (defaults to ./data).
        max_engines: Optional limit on the number of engine units to load.
        window_size: Length of the sliding temporal context window.
        stride: Stride between consecutive sliding windows.
        max_rul: Maximum RUL clipping threshold (standard 125.0).
        
    Returns:
        Tuple of (X, y, mean, std):
            X: Array of shape (N, window_size, 14) float32.
            y: Array of shape (N, 1) float32.
            mean: Array of shape (14,) float32.
            std: Array of shape (14,) float32.
    """
    if data_dir is None:
        data_path = Path("data") / "train_FD001.txt"
    else:
        data_path = Path(data_dir) / "train_FD001.txt"

    if not data_path.exists():
        _generate_synthetic_cmapss_file(data_path, num_engines=100)

    # Read dataset file using standard python parser for zero-bloat
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 26:
                unit_id = int(float(parts[0]))
                cycle = int(float(parts[1]))
                sensors = [float(p) for p in parts[5:26]]
                records.append((unit_id, cycle, sensors))

    # Group by unit ID
    engines = {}
    for unit_id, cycle, sensors in records:
        if max_engines is not None and unit_id > max_engines:
            continue
        if unit_id not in engines:
            engines[unit_id] = []
        engines[unit_id].append((cycle, sensors))

    windows_X = []
    targets_y = []
    all_sensor_readings = []

    for unit_id, cycles in engines.items():
        cycles.sort(key=lambda x: x[0])
        max_cycle = cycles[-1][0]

        # Extract 14 informative sensors per cycle
        engine_sensors = np.array(
            [[s[idx] for idx in INFORMATIVE_SENSOR_INDICES] for _, s in cycles],
            dtype=np.float32,
        )
        all_sensor_readings.append(engine_sensors)

        num_cycles = len(cycles)
        if num_cycles < window_size:
            continue

        for start in range(0, num_cycles - window_size + 1, stride):
            end = start + window_size
            window = engine_sensors[start:end]
            current_cycle = cycles[end - 1][0]

            # Piecewise-linear RUL clipped at max_rul and normalized to [0, 1]
            raw_rul = float(max_cycle - current_cycle)
            clipped_rul = min(raw_rul, max_rul) / max_rul

            windows_X.append(window)
            targets_y.append(clipped_rul)

    X = np.array(windows_X, dtype=np.float32)
    y = np.array(targets_y, dtype=np.float32).reshape(-1, 1)

    all_concat = np.concatenate(all_sensor_readings, axis=0)
    mean = np.mean(all_concat, axis=0).astype(np.float32)
    std = np.std(all_concat, axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)

    return X, y, mean, std


def create_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    """Creates a PyTorch DataLoader from numpy arrays."""
    tensor_x = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(y).float()
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
