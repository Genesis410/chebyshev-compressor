"""Scientific Benchmark Suite: Legendre vs Chebyshev, Lambda Distillation Ablation, and C-MAPSS Evaluation."""

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chebyshev.datasets import load_cmapss_fd001_splits
from chebyshev.models import StudentModel
from chebyshev.distill import distill, evaluate_rmse
from chebyshev.math_ops import continuous_chebyshev_operators, discretize_tustin
from chebyshev.teacher import train_teacher, CMapssLSTMTeacher


def get_legendre_operators(order: int, theta: float = 30.0, dt: float = 1.0):
    """Constructs continuous and Tustin-discretized HiPPO-Leg (Legendre) memory operators."""
    A = np.zeros((order, order), dtype=np.float64)
    B = np.zeros((order, 1), dtype=np.float64)
    for n in range(order):
        c_n = math.sqrt(2 * n + 1)
        B[n, 0] = c_n / theta
        for k in range(order):
            if n > k:
                c_k = math.sqrt(2 * k + 1)
                A[n, k] = -(c_n * c_k) / theta
            elif n == k:
                A[n, n] = -(n + 1) / theta
    A_bar, B_bar = discretize_tustin(A, B, dt=dt)
    return A_bar, B_bar


def run_experiment():
    print("=== Loading C-MAPSS FD001 Data ===")
    data = load_cmapss_fd001_splits(window_size=30, stride=2, max_engines=100)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    mean, std = data["mean"], data["std"]
    max_rul = data["max_rul"]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128, shuffle=False)

    print("\n=== 1. Training/Evaluating Baseline Teacher LSTM ===")
    torch.manual_seed(42)
    teacher = CMapssLSTMTeacher(num_channels=14, hidden_dim=64, num_layers=2, out_features=1)
    train_teacher(teacher, train_loader, epochs=15, lr=1e-3, verbose=False)
    teacher_val_rmse = evaluate_rmse(teacher, val_loader, scale=max_rul)
    print(f"Teacher LSTM Final Val RMSE: {teacher_val_rmse:.2f} cycles")

    results = {
        "teacher_lstm_val_rmse": teacher_val_rmse,
        "lambda_ablation": {},
        "basis_ablation": {},
        "order_ablation": {}
    }

    # 2. Lambda Distillation Ablation (K=8, H=32)
    print("\n=== 2. Distillation Loss Ablation (lambda in {0.0, 0.5, 1.0}) ===")
    for lam in [0.0, 0.5, 1.0]:
        torch.manual_seed(42)
        student = StudentModel(num_channels=14, order=8, hidden_dim=32, out_features=1)
        student.cell.set_normalization(mean, std)
        
        distill(
            teacher=teacher,
            student=student,
            dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=20,
            lr=2e-3,
            distill_lambda=lam,
            verbose=False,
        )
        val_rmse = evaluate_rmse(student, val_loader, scale=max_rul)
        results["lambda_ablation"][f"lambda_{lam}"] = val_rmse
        print(f"  lambda={lam:.1f}: Val RMSE = {val_rmse:.2f} cycles")

    # 3. Basis Comparison: Chebyshev vs Legendre (HiPPO-Leg) at K=8, K=16
    print("\n=== 3. Basis Comparison: Chebyshev vs Legendre (HiPPO-Leg) ===")
    for K in [8, 16]:
        # Chebyshev
        torch.manual_seed(42)
        cheb_student = StudentModel(num_channels=14, order=K, hidden_dim=32, out_features=1)
        cheb_student.cell.set_normalization(mean, std)
        distill(teacher, cheb_student, train_loader, val_dataloader=val_loader, epochs=20, lr=2e-3, distill_lambda=0.5, verbose=False)
        cheb_rmse = evaluate_rmse(cheb_student, val_loader, scale=max_rul)

        # Legendre (HiPPO-Leg)
        torch.manual_seed(42)
        leg_student = StudentModel(num_channels=14, order=K, hidden_dim=32, out_features=1)
        leg_student.cell.set_normalization(mean, std)
        A_leg, B_leg = get_legendre_operators(K, theta=30.0, dt=1.0)
        leg_student.cell.A.copy_(torch.from_numpy(A_leg).float())
        leg_student.cell.B.copy_(torch.from_numpy(B_leg.ravel()).float())

        distill(teacher, leg_student, train_loader, val_dataloader=val_loader, epochs=20, lr=2e-3, distill_lambda=0.5, verbose=False)
        leg_rmse = evaluate_rmse(leg_student, val_loader, scale=max_rul)

        results["basis_ablation"][f"K={K}"] = {
            "chebyshev_rmse": cheb_rmse,
            "legendre_rmse": leg_rmse,
            "chebyshev_advantage": leg_rmse - cheb_rmse
        }
        print(f"  K={K}: Chebyshev RMSE = {cheb_rmse:.2f} cycles | Legendre RMSE = {leg_rmse:.2f} cycles (Advantage = {leg_rmse - cheb_rmse:+.2f})")

    # 4. Order Ablation across K in {4, 8, 16, 32}
    print("\n=== 4. Polynomial Order Ablation (K in {4, 8, 16, 32}) ===")
    for K in [4, 8, 16, 32]:
        hidden = 16 if K == 4 else (32 if K <= 16 else 64)
        torch.manual_seed(42)
        student = StudentModel(num_channels=14, order=K, hidden_dim=hidden, out_features=1)
        student.cell.set_normalization(mean, std)
        distill(teacher, student, train_loader, val_dataloader=val_loader, epochs=20, lr=2e-3, distill_lambda=0.5, verbose=False)
        rmse = evaluate_rmse(student, val_loader, scale=max_rul)
        results["order_ablation"][f"K={K}"] = {
            "rmse": rmse,
            "hidden_dim": hidden,
            "sram_bytes": 14 * K * 4 + (K + hidden) * 4
        }
        print(f"  K={K} (H={hidden}): RMSE = {rmse:.2f} cycles, SRAM = {14*K*4 + (K+hidden)*4} B")

    Path("scientific_benchmark_summary.json").write_text(json.dumps(results, indent=2))
    print("\nSaved results to scientific_benchmark_summary.json")

if __name__ == "__main__":
    run_experiment()
