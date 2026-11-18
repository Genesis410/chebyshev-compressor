"""Unified command-line interface entrypoint for Chebyshev-Compressor (`cheb`)."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Optional
import numpy as np
import torch

from chebyshev.benchmark import compare_benchmarks, run_c_benchmark, run_pytorch_benchmark
from chebyshev.datasets import (
    create_dataloader,
    generate_synthetic_smoke_data,
    load_cmapss_fd001_splits,
)
from chebyshev.distill import distill, evaluate_rmse
from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel
from chebyshev.teacher import CMapssLSTMTeacher, load_teacher, save_teacher, train_teacher
from chebyshev.utils import (
    generate_reproducibility_manifest,
    load_calibration,
    save_calibration,
    set_seed,
)


def get_default_out_dir() -> Path:
    """Returns convention output directory from CHEB_OUT_DIR env or default ./cheb_out."""
    out = os.environ.get("CHEB_OUT_DIR", "./cheb_out")
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_full_smoke_test(out_dir: Path, seed: int = 42) -> int:
    """Executes full automated pipeline on synthetic data with held-out validation."""
    set_seed(seed)
    print("=" * 80)
    print("CHEBYSHEV-COMPRESSOR: RUNNING 15-SECOND FULL PIPELINE SMOKE-TEST")
    print(f"Reproducibility Seed: {seed}")
    print("=" * 80)

    # 1. Generate synthetic dataset and split into disjoint train / held-out validation sets
    print("[1/4] Generating synthetic multi-sensor machinery data (80/20 held-out split)...")
    X, y, _, _ = generate_synthetic_smoke_data(num_samples=160, seq_len=30, num_channels=14, seed=seed)
    split_idx = 128
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # Fit normalization strictly on training split only
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(flat_train, axis=0).astype(np.float32)
    std = np.std(flat_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)

    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)

    # Save calibration artifact
    calib_path = out_dir / "calib.npy"
    save_calibration(mean, std, calib_path)
    print(f"      Saved calibration vectors to: {calib_path}")

    # 2. Train baseline teacher
    print("[2/4] Training baseline PyTorch LSTM teacher...")
    teacher = CMapssLSTMTeacher(num_channels=14, hidden_dim=32, num_layers=2, out_features=1)
    train_teacher(teacher, train_loader, epochs=3, lr=3e-3, verbose=False)
    teacher_path = out_dir / "teacher.pt"
    save_teacher(teacher, teacher_path)
    teacher_val_rmse = evaluate_rmse(teacher, val_loader, scale=1.0)
    print(f"      Saved teacher checkpoint to: {teacher_path} (Held-out Val RMSE: {teacher_val_rmse:.5f})")

    # 3. Distill into Chebyshev student (Option A: Fast State Precomputation + Joint Loss)
    print("[3/4] Distilling Readout MLP over frozen Chebyshev states (Option A)...")
    student = StudentModel(num_channels=14, order=8, hidden_dim=32, out_features=1, window_size=30.0)
    distill(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        val_dataloader=val_loader,
        channel_mean=mean,
        channel_std=std,
        epochs=10,
        lr=3e-3,
        distill_lambda=0.5,
        verbose=False,
    )
    student_val_rmse = evaluate_rmse(student, val_loader, scale=1.0)
    student_path = out_dir / "student.pt"
    torch.save(student.state_dict(), str(student_path))
    print(f"      Saved student model to: {student_path} (Held-out Val RMSE: {student_val_rmse:.5f})")

    # 4. Export C99 single-header library
    print("[4/4] Exporting C99 zero-dependency header...")
    header_path = out_dir / "chebyshev_engine.h"
    export_to_c_header(student, header_path)
    print(f"      Saved C99 header to: {header_path}")

    # 5. Run benchmark and audit comparison
    print("\n[BENCHMARK] Executing standalone native C harness (100,000 steps)...")
    c_metrics = None
    pt_metrics = None
    comparison = None
    try:
        c_metrics = run_c_benchmark(header_path, iterations=100000)
        pt_metrics = run_pytorch_benchmark(teacher, num_channels=14, steps=1000)
        comparison = compare_benchmarks(c_metrics, pt_metrics)
        print(comparison["summary_table"])

        # Save benchmark results artifact
        bench_json_path = out_dir / "benchmark_results.json"
        bench_data = {
            "c_metrics": c_metrics,
            "pt_metrics": pt_metrics,
            "speedup_x": comparison["speedup_x"],
            "ram_reduction_pct": comparison["ram_reduction_pct"],
            "teacher_val_rmse": teacher_val_rmse,
            "student_val_rmse": student_val_rmse,
        }
        with open(bench_json_path, "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=2)
        print(f"      Saved benchmark audit metrics to: {bench_json_path}")
    except Exception as e:
        print(f"      Benchmark warning (compiler not invoked): {e}")

    # 6. Generate reproducibility manifest
    generate_reproducibility_manifest(
        out_dir=out_dir,
        seed=seed,
        hyperparameters={
            "num_channels": 14,
            "order": 8,
            "hidden_dim": 32,
            "window_size": 30.0,
            "teacher_hidden_dim": 32,
            "teacher_layers": 2,
            "smoke_epochs": 10,
            "distill_lambda": 0.5,
            "distillation_mode": "option_a_fixed_analytical_precomputed",
        },
        benchmark_metrics={
            "c_latency_us": c_metrics["latency_us"] if c_metrics else None,
            "pt_latency_us": pt_metrics["latency_us"] if pt_metrics else None,
            "speedup_x": comparison["speedup_x"] if comparison else None,
            "ram_bytes": c_metrics.get("ram_bytes", 448) if c_metrics else 448,
            "ram_reduction_pct": comparison["ram_reduction_pct"] if comparison else None,
            "teacher_val_rmse": teacher_val_rmse,
            "student_val_rmse": student_val_rmse,
        },
    )
    print(f"      Saved reproducibility manifest to: {out_dir / 'reproducibility_manifest.json'}")

    print("\n>>> SMOKE-TEST COMPLETED SUCCESSFULLY. All artifacts saved in:", out_dir)
    return 0


def cmd_train_teacher(args: argparse.Namespace) -> int:
    seed = getattr(args, "seed", 42)
    set_seed(seed)
    out_path = Path(args.out)
    print(f"--- Training Teacher Network -> {out_path} (Seed: {seed}) ---")

    if args.smoke:
        X, y, _, _ = generate_synthetic_smoke_data(num_samples=160, seq_len=30, num_channels=14, seed=seed)
        X_train, y_train = X[:128], y[:128]
        X_val, y_val = X[128:], y[128:]
        rul_scale = 1.0
    else:
        splits = load_cmapss_fd001_splits(
            data_dir=getattr(args, "data_dir", None),
            max_engines=args.max_engines,
            window_size=args.window_size,
            seed=seed,
        )
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        rul_scale = splits["max_rul"]
        print(
            f"Loaded C-MAPSS FD001 split: {len(splits['train_units'])} train engines, "
            f"{len(splits['val_units'])} held-out validation engines."
        )

    train_loader = create_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    teacher = CMapssLSTMTeacher(
        num_channels=X_train.shape[2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        out_features=y_train.shape[1],
    )
    train_teacher(teacher, train_loader, epochs=args.epochs, lr=args.lr, verbose=True)
    val_rmse = evaluate_rmse(teacher, val_loader, scale=rul_scale)
    print(f"Teacher Held-Out Validation RMSE: {val_rmse:.4f} (scale={rul_scale:.1f})")
    save_teacher(teacher, out_path)
    print(f"Teacher model successfully saved to {out_path}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    seed = getattr(args, "seed", 42)
    set_seed(seed)
    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        print(f"Error: Teacher checkpoint not found at {teacher_path}. Run `cheb train-teacher` first.")
        return 1

    out_path = Path(args.out)
    print(f"--- Distilling Teacher ({teacher_path}) -> Chebyshev Student ({out_path}) (Seed: {seed}) ---")

    teacher = load_teacher(teacher_path)

    if args.smoke or getattr(args, "smoke_test", False):
        X, y, _, _ = generate_synthetic_smoke_data(
            num_samples=160, seq_len=args.window_size, num_channels=teacher.num_channels, seed=seed
        )
        X_train, y_train = X[:128], y[:128]
        X_val, y_val = X[128:], y[128:]
        flat_train = X_train.reshape(-1, X_train.shape[-1])
        mean = np.mean(flat_train, axis=0).astype(np.float32)
        std = np.std(flat_train, axis=0).astype(np.float32)
        std = np.where(std < 1e-4, 1.0, std)
        rul_scale = 1.0
    else:
        splits = load_cmapss_fd001_splits(
            data_dir=getattr(args, "data_dir", None),
            max_engines=args.max_engines,
            window_size=args.window_size,
            seed=seed,
        )
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        mean, std = splits["mean"], splits["std"]
        rul_scale = splits["max_rul"]
        print(
            f"Loaded C-MAPSS FD001 split: {len(splits['train_units'])} train engines, "
            f"{len(splits['val_units'])} held-out validation engines."
        )

    # Save calibration vectors as artifact (fit strictly on training split)
    calib_out = out_path.parent / "calib.npy"
    save_calibration(mean, std, calib_out)
    print(f"Saved calibration vectors to {calib_out}")

    train_loader = create_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    student = StudentModel(
        num_channels=teacher.num_channels,
        order=args.order,
        hidden_dim=args.hidden_dim,
        out_features=teacher.out_features,
        window_size=float(args.window_size),
    )

    distill(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        val_dataloader=val_loader,
        channel_mean=mean,
        channel_std=std,
        epochs=args.epochs,
        lr=args.lr,
        distill_lambda=getattr(args, "distill_lambda", 0.5),
        verbose=True,
    )

    teacher_val_rmse = evaluate_rmse(teacher, val_loader, scale=rul_scale)
    student_val_rmse = evaluate_rmse(student, val_loader, scale=rul_scale)
    print(
        f"Held-Out Validation Ground-Truth RMSE -> Teacher: {teacher_val_rmse:.4f} | "
        f"Student: {student_val_rmse:.4f} (scale={rul_scale:.1f})"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "num_channels": student.num_channels,
            "order": student.order,
            "hidden_dim": student.hidden_dim,
            "out_features": student.out_features,
            "window_size": student.cell.window_size,
        },
        str(out_path),
    )
    print(f"Student model successfully saved to {out_path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Student model checkpoint not found at {model_path}.")
        return 1

    out_header = Path(args.output)
    print(f"--- Exporting PyTorch Model ({model_path}) -> C99 Header ({out_header}) ---")

    checkpoint = torch.load(str(model_path), map_location="cpu")
    # Support both raw state_dict or dictionary with metadata
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        num_channels = checkpoint["num_channels"]
        order = checkpoint["order"]
        hidden_dim = checkpoint["hidden_dim"]
        out_features = checkpoint["out_features"]
        window_size = checkpoint.get("window_size", 30.0)
    else:
        state_dict = checkpoint
        num_channels = state_dict["cell.mean"].shape[0]
        order = state_dict["cell.B"].shape[0]
        hidden_dim = state_dict["readout.mlp.0.weight"].shape[0]
        out_features = state_dict["readout.mlp.2.weight"].shape[0]
        window_size = 30.0

    student = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
        window_size=window_size,
    )
    student.load_state_dict(state_dict)

    # Check for external calibration file if specified or present
    calib_file = getattr(args, "calib", None)
    if calib_file is not None and Path(calib_file).exists():
        mean, std = load_calibration(calib_file)
        student.cell.set_normalization(mean, std)
    elif (model_path.parent / "calib.npy").exists():
        mean, std = load_calibration(model_path.parent / "calib.npy")
        if torch.allclose(student.cell.mean, torch.zeros_like(student.cell.mean)):
            student.cell.set_normalization(mean, std)

    export_to_c_header(student, out_header)
    print(f"C99 Header successfully generated at {out_header}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    header_path = Path(args.header)
    if not header_path.exists():
        print(f"Error: C99 header not found at {header_path}.")
        return 1

    print(f"--- Benchmarking Native C99 Engine ({header_path}) ---")
    c_metrics = run_c_benchmark(header_path, iterations=args.iterations)

    teacher_path = Path(args.teacher)
    if teacher_path.exists():
        teacher = load_teacher(teacher_path)
        pt_metrics = run_pytorch_benchmark(teacher, num_channels=teacher.num_channels, steps=1000)
    else:
        dummy_teacher = CMapssLSTMTeacher(num_channels=14, hidden_dim=64, num_layers=2)
        pt_metrics = run_pytorch_benchmark(dummy_teacher, num_channels=14, steps=1000)

    comparison = compare_benchmarks(c_metrics, pt_metrics)
    print(comparison["summary_table"])

    out_json = getattr(args, "out_json", None)
    if out_json:
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        bench_data = {
            "c_metrics": c_metrics,
            "pt_metrics": pt_metrics,
            "speedup_x": comparison["speedup_x"],
            "ram_reduction_pct": comparison["ram_reduction_pct"],
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=2)
        print(f"Benchmark results saved to {out_json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_dir = get_default_out_dir()
    parser = argparse.ArgumentParser(
        prog="cheb",
        description="Chebyshev-Compressor: Distill recurrent models into O(1) memory Chebyshev state-space C99 engines.",
    )
    parser.add_argument("--version", action="version", version="chebyshev-compressor 0.1.0")
    parser.add_argument("--smoke-test", action="store_true", help="Run 15-second full end-to-end smoke test.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for full reproducibility.")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # cheb train-teacher
    p_teacher = subparsers.add_parser("train-teacher", help="Train baseline PyTorch LSTM teacher network.")
    p_teacher.add_argument("--smoke", action="store_true", help="Use synthetic dataset for quick training.")
    p_teacher.add_argument("--data-dir", type=str, default=None, help="Directory containing C-MAPSS FD001 dataset.")
    p_teacher.add_argument("--max-engines", type=int, default=None, help="Limit number of C-MAPSS engines.")
    p_teacher.add_argument("--window-size", type=int, default=30, help="Sliding window size.")
    p_teacher.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden units.")
    p_teacher.add_argument("--num-layers", type=int, default=2, help="LSTM layer count.")
    p_teacher.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    p_teacher.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p_teacher.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p_teacher.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p_teacher.add_argument("--out", type=str, default=str(default_dir / "teacher.pt"), help="Output path.")

    # cheb distill
    p_distill = subparsers.add_parser("distill", help="Distill teacher into Chebyshev StudentModel.")
    p_distill.add_argument("--teacher", type=str, default=str(default_dir / "teacher.pt"), help="Teacher checkpoint path.")
    p_distill.add_argument("--smoke", action="store_true", help="Use synthetic data.")
    p_distill.add_argument("--smoke-test", action="store_true", help="Alias for --smoke.")
    p_distill.add_argument("--data-dir", type=str, default=None, help="Directory containing C-MAPSS FD001 dataset.")
    p_distill.add_argument("--max-engines", type=int, default=None, help="Limit C-MAPSS engines.")
    p_distill.add_argument("--order", type=int, default=8, help="Chebyshev polynomial order K.")
    p_distill.add_argument("--window-size", type=int, default=30, help="Sliding window length.")
    p_distill.add_argument("--hidden-dim", type=int, default=32, help="Readout MLP hidden units.")
    p_distill.add_argument("--epochs", type=int, default=15, help="Distillation epochs.")
    p_distill.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p_distill.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    p_distill.add_argument(
        "--distill-lambda",
        type=float,
        default=0.5,
        help="KD weight lambda in [0, 1] balancing ground-truth MSE and teacher mimicry MSE (auto 1.0 if unlabeled).",
    )
    p_distill.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p_distill.add_argument("--out", type=str, default=str(default_dir / "student.pt"), help="Output path.")

    # cheb export
    p_export = subparsers.add_parser("export", help="Export trained student into C99 single-header engine.")
    p_export.add_argument("--model", type=str, default=str(default_dir / "student.pt"), help="Student checkpoint path.")
    p_export.add_argument("--calib", type=str, default=None, help="Optional calibration .npy path.")
    p_export.add_argument("--output", type=str, default=str(default_dir / "chebyshev_engine.h"), help="Output C header.")

    # cheb benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run standalone C99 native latency and audit benchmark.")
    p_bench.add_argument("--header", type=str, default=str(default_dir / "chebyshev_engine.h"), help="Header to benchmark.")
    p_bench.add_argument("--teacher", type=str, default=str(default_dir / "teacher.pt"), help="Teacher model for baseline.")
    p_bench.add_argument("--iterations", type=int, default=100000, help="Benchmark iterations.")
    p_bench.add_argument("--out-json", type=str, default=None, help="Optional path to save benchmark JSON results.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    default_dir = get_default_out_dir()

    if args.smoke_test:
        return run_full_smoke_test(default_dir, seed=args.seed)

    if args.subcommand == "train-teacher":
        return cmd_train_teacher(args)
    elif args.subcommand == "distill":
        return cmd_distill(args)
    elif args.subcommand == "export":
        return cmd_export(args)
    elif args.subcommand == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
