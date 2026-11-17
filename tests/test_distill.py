import time
import numpy as np
import pytest
import torch
import torch.nn as nn

from chebyshev.datasets import create_dataloader, generate_synthetic_smoke_data
from chebyshev.distill import (
    TeacherStateHook,
    cosine_annealing_lambda,
    distill,
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
        # Average over sequence dimension
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
    out = teacher(x)
    trajectory = hook.get_trajectory()
    
    assert trajectory is not None
    assert trajectory.shape == (4, 20, 32)
    hook.remove()


def test_teacher_hook_non_recurrent_fallback():
    teacher = ToyNonRecurrentTeacher()
    hook = TeacherStateHook(teacher)
    
    # Should not throw and should report no recurrent layer
    assert not hook.has_recurrent_layer
    
    x = torch.randn(4, 20, 14)
    out = teacher(x)
    trajectory = hook.get_trajectory()
    assert trajectory is None
    hook.remove()


def test_cosine_annealing_schedule():
    epochs = 10
    l_start = 1.0
    l_end = 0.1
    
    val_0 = cosine_annealing_lambda(0, epochs, l_start, l_end)
    val_mid = cosine_annealing_lambda(5, epochs, l_start, l_end)
    val_end = cosine_annealing_lambda(10, epochs, l_start, l_end)
    
    assert pytest.approx(val_0, abs=1e-5) == 1.0
    assert pytest.approx(val_end, abs=1e-5) == 0.1
    assert 0.1 < val_mid < 1.0


def test_distillation_convergence_smoke_test():
    # 1. Setup toy teacher and synthetic dataset
    num_channels = 14
    order = 8
    seq_len = 20
    teacher = ToyLSTMTeacher(in_features=num_channels, hidden_dim=32, out_features=1)
    teacher.eval()
    
    X, y, mean, std = generate_synthetic_smoke_data(
        num_samples=128, seq_len=seq_len, num_channels=num_channels, seed=123
    )
    dataloader = create_dataloader(X, y, batch_size=32)
    
    student = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=32,
        out_features=1,
        window_size=seq_len,
    )
    
    start_time = time.time()
    history = distill(
        teacher=teacher,
        student=student,
        dataloader=dataloader,
        channel_mean=mean,
        channel_std=std,
        epochs=15,
        lr=3e-3,
        lambda_start=1.0,
        lambda_end=0.1,
        verbose=False,
    )
    duration = time.time() - start_time
    
    # Assert fast execution (< 15 seconds)
    assert duration < 15.0, f"Distillation took too long: {duration:.2f}s"
    
    # Assert loss decreased substantially
    initial_loss = history["total_loss"][0]
    final_loss = history["total_loss"][-1]
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    
    # Assert baked normalization was updated
    torch.testing.assert_close(student.cell.mean, torch.from_numpy(mean).float())
