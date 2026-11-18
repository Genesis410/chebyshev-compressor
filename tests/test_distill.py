import time
import numpy as np
import pytest
import torch
import torch.nn as nn

from chebyshev.datasets import create_dataloader, generate_synthetic_smoke_data
from chebyshev.distill import (
    TeacherStateHook,
    distill,
    evaluate_rmse,
    precompute_chebyshev_states,
)
from chebyshev.models import StudentModel


class ToyLSTMTeacher(nn.Module):
    def __init__(self, in_features=14, hidden_dim=32, out_features=1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ToyNonRecurrentTeacher(nn.Module):
    def __init__(self, in_features=14, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x.mean(dim=1))


def test_synthetic_smoke_data_generation():
    num_samples = 64
    seq_len = 25
    num_channels = 14

    X, y, mean, std = generate_synthetic_smoke_data(
        num_samples=num_samples,
        seq_len=seq_len,
        num_channels=num_channels,
        seed=42,
    )

    assert X.shape == (num_samples, seq_len, num_channels)
    assert y.shape == (num_samples, 1)
    assert mean.shape == (num_channels,)
    assert std.shape == (num_channels,)
    assert np.all(std > 0)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()

    loader = create_dataloader(X, y, batch_size=16, shuffle=True)
    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape == (16, seq_len, num_channels)
    assert batch_y.shape == (16, 1)


def test_teacher_hook_lstm():
    teacher = ToyLSTMTeacher()
    hook = TeacherStateHook(teacher)

    assert hook.has_recurrent_layer

    x = torch.randn(4, 20, 14)
    _ = teacher(x)
    trajectory = hook.get_trajectory()

    assert trajectory is not None
    assert trajectory.shape == (4, 20, 32)
    hook.remove()


def test_teacher_hook_non_recurrent_fallback():
    teacher = ToyNonRecurrentTeacher()
    hook = TeacherStateHook(teacher)

    assert not hook.has_recurrent_layer

    x = torch.randn(4, 20, 14)
    _ = teacher(x)
    trajectory = hook.get_trajectory()
    assert trajectory is None
    hook.remove()


def test_fast_state_precomputation_matches_student_forward():
    """Verifies precomputed Chebyshev states feed into ReadoutMLP identically to full StudentModel forward."""
    torch.manual_seed(42)
    X, y, mean, std = generate_synthetic_smoke_data(num_samples=40, seq_len=15, num_channels=14, seed=7)
    loader = create_dataloader(X, y, batch_size=16, shuffle=False)

    student = StudentModel(num_channels=14, order=8, hidden_dim=32, out_features=1, window_size=15.0)
    student.cell.set_normalization(mean, std)
    student.eval()

    S, teacher_out, y_true = precompute_chebyshev_states(student, loader, teacher=None)
    assert teacher_out is None
    assert y_true is not None
    assert S.shape == (40, 14 * 8)

    with torch.no_grad():
        out_from_precomputed = student.readout(S)
        out_from_full = student(torch.from_numpy(X).float())

    torch.testing.assert_close(out_from_precomputed, out_from_full, atol=1e-6, rtol=1e-6)


def test_option_a_distillation_joint_loss_and_frozen_operators():
    """Verifies Option A distillation with fast precomputation, joint GT+KD loss, frozen (A, B), and held-out RMSE."""
    torch.manual_seed(42)
    num_channels = 14
    order = 8
    seq_len = 20
    teacher = ToyLSTMTeacher(in_features=num_channels, hidden_dim=32, out_features=1)
    teacher.eval()

    X, y, _, _ = generate_synthetic_smoke_data(
        num_samples=160, seq_len=seq_len, num_channels=num_channels, seed=123
    )
    X_train, y_train = X[:128], y[:128]
    X_val, y_val = X[128:], y[128:]
    flat_train = X_train.reshape(-1, num_channels)
    mean = np.mean(flat_train, axis=0).astype(np.float32)
    std = np.where(np.std(flat_train, axis=0) < 1e-4, 1.0, np.std(flat_train, axis=0)).astype(np.float32)

    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)

    student = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=32,
        out_features=1,
        window_size=float(seq_len),
    )

    A_before = student.cell.A.clone()
    B_before = student.cell.B.clone()
    initial_val_rmse = evaluate_rmse(student, val_loader)

    start_time = time.time()
    history = distill(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        val_dataloader=val_loader,
        channel_mean=mean,
        channel_std=std,
        epochs=20,
        lr=3e-3,
        distill_lambda=0.5,
        verbose=False,
    )
    duration = time.time() - start_time

    # Fast precomputed training should complete well under 5 seconds
    assert duration < 10.0, f"Precomputed distillation took too long: {duration:.2f}s"

    # Loss and held-out validation RMSE must decrease
    assert history["total_loss"][-1] < history["total_loss"][0]
    assert history["gt_loss"][-1] > 0.0
    assert history["kd_loss"][-1] > 0.0
    assert all(l == pytest.approx(0.5) for l in history["lambda"])
    assert "val_rmse" in history and len(history["val_rmse"]) == 20

    final_val_rmse = evaluate_rmse(student, val_loader)
    assert final_val_rmse < initial_val_rmse, (
        f"Held-out validation RMSE did not improve: {initial_val_rmse:.5f} -> {final_val_rmse:.5f}"
    )

    # Frozen analytical operators (A_bar, B_bar) must remain bit-for-bit unchanged
    torch.testing.assert_close(student.cell.A, A_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(student.cell.B, B_before, atol=0.0, rtol=0.0)

    # Baked normalization must be updated
    torch.testing.assert_close(student.cell.mean, torch.from_numpy(mean).float())


def test_distillation_unlabeled_calibration_automatic_lambda_fallback():
    """Verifies automatic fallback to lambda=1.0 (pure teacher mimicry) when y_true is absent."""
    torch.manual_seed(99)
    num_channels = 8
    seq_len = 15
    teacher = ToyLSTMTeacher(in_features=num_channels, hidden_dim=16, out_features=1)
    teacher.eval()

    X, _, mean, std = generate_synthetic_smoke_data(
        num_samples=64, seq_len=seq_len, num_channels=num_channels, seed=99
    )
    # Create strictly unlabeled calibration dataloader (y=None)
    unlabeled_loader = create_dataloader(X, y=None, batch_size=16, shuffle=True)

    student = StudentModel(
        num_channels=num_channels,
        order=6,
        hidden_dim=16,
        out_features=1,
        window_size=float(seq_len),
    )

    history = distill(
        teacher=teacher,
        student=student,
        dataloader=unlabeled_loader,
        channel_mean=mean,
        channel_std=std,
        epochs=10,
        lr=3e-3,
        distill_lambda=0.5,  # Requested 0.5, must auto-fallback to 1.0
        verbose=False,
    )

    assert all(l == pytest.approx(1.0) for l in history["lambda"])
    assert all(g == pytest.approx(0.0) for g in history["gt_loss"])
    assert history["total_loss"][-1] == pytest.approx(history["kd_loss"][-1])
    assert history["kd_loss"][-1] < history["kd_loss"][0]
