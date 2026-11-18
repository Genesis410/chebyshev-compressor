import torch
import numpy as np
import pytest

from chebyshev.models import DepthwiseChebyshevCell, ReadoutMLP, StudentModel


def test_depthwise_cell_shapes_and_state_step():
    num_channels = 14
    order = 8
    batch_size = 4
    
    cell = DepthwiseChebyshevCell(num_channels=num_channels, order=order, window_size=30.0, dt=1.0)
    state = cell.init_state(batch_size)
    assert state.shape == (batch_size, num_channels, order)
    assert torch.all(state == 0.0)
    
    x_t = torch.randn(batch_size, num_channels)
    next_state = cell(x_t, state)
    
    assert next_state.shape == (batch_size, num_channels, order)
    assert not torch.all(next_state == 0.0)


def test_depthwise_cell_input_sanity_clamping_and_nan_quarantine():
    num_channels = 4
    order = 6
    batch_size = 2
    
    cell = DepthwiseChebyshevCell(num_channels=num_channels, order=order, clamp_value=5.0)
    state = cell.init_state(batch_size)
    
    # Test extreme outlier clamping
    x_extreme = torch.tensor([[1000.0, -999.0, 3.0, -2.0], [5.0, -5.0, 0.0, 10.0]])
    x_clamped = torch.tensor([[5.0, -5.0, 3.0, -2.0], [5.0, -5.0, 0.0, 5.0]])
    
    state_from_extreme = cell(x_extreme, state)
    state_from_clamped = cell(x_clamped, state)
    
    # Both should be identical due to clamping
    torch.testing.assert_close(state_from_extreme, state_from_clamped)
    
    # Test NaN, +Inf, -Inf quarantine -> all non-finite values map to 0.0 matching C99 0x7F800000U bit check
    x_nonfinite = torch.tensor([[float('nan'), float('inf'), float('-inf'), 3.0], [0.0, float('inf'), -1.0, float('nan')]])
    x_quarantined = torch.tensor([[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, -1.0, 0.0]])
    state_from_nonfinite = cell(x_nonfinite, state)
    state_from_quarantined = cell(x_quarantined, state)

    assert not torch.isnan(state_from_nonfinite).any(), "NaN leaked into recurrent state!"
    assert not torch.isinf(state_from_nonfinite).any(), "Inf leaked into recurrent state!"
    torch.testing.assert_close(state_from_nonfinite, state_from_quarantined)


def test_depthwise_cell_baked_normalization():
    num_channels = 2
    order = 4
    batch_size = 1
    
    cell = DepthwiseChebyshevCell(num_channels=num_channels, order=order)
    mean = np.array([100.0, 200.0], dtype=np.float32)
    std = np.array([10.0, 20.0], dtype=np.float32)
    cell.set_normalization(mean, std)
    
    state = cell.init_state(batch_size)
    
    # Passing the exact mean should result in normalized input = 0, so state change is 0
    x_mean = torch.tensor([[100.0, 200.0]])
    next_state = cell(x_mean, state)
    torch.testing.assert_close(next_state, state)


def test_student_model_forward_and_step_equivalence():
    num_channels = 14
    order = 8
    hidden_dim = 32
    out_features = 1
    batch_size = 3
    seq_len = 10
    
    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
        window_size=30.0,
    )
    
    x = torch.randn(batch_size, seq_len, num_channels)
    
    # 1. Forward full sequence
    seq_out, trajectory = model(x, return_sequence=True, return_trajectory=True)
    assert seq_out.shape == (batch_size, seq_len, out_features)
    assert trajectory.shape == (batch_size, seq_len, num_channels * order)
    
    # 2. Forward sequence-to-one (last step)
    last_out = model(x, return_sequence=False)
    assert last_out.shape == (batch_size, out_features)
    torch.testing.assert_close(last_out, seq_out[:, -1, :])
    
    # 3. Step-by-step equivalence
    state = model.cell.init_state(batch_size)
    step_outputs = []
    for t in range(seq_len):
        out_t, state = model.step(x[:, t, :], state)
        step_outputs.append(out_t)
        
    step_outputs = torch.stack(step_outputs, dim=1)
    torch.testing.assert_close(step_outputs, seq_out)


def test_student_model_gradient_flow():
    num_channels = 6
    order = 4
    hidden_dim = 16
    out_features = 1
    batch_size = 2
    seq_len = 5
    
    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    
    x = torch.randn(batch_size, seq_len, num_channels, requires_grad=True)
    y = model(x, return_sequence=False)
    loss = y.sum()
    loss.backward()
    
    # Check that Readout MLP parameters have gradients
    for name, param in model.readout.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.all(param.grad == 0.0), f"Parameter {name} has zero gradient"
        
    # Check that input x receives gradients (verifying gradient flows back through recurrent state)
    assert x.grad is not None
    assert not torch.all(x.grad == 0.0)


def test_student_model_state_dict_serialization():
    num_channels = 4
    order = 4
    model1 = StudentModel(num_channels=num_channels, order=order)
    model1.cell.set_normalization(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 0.5, 0.5, 0.5]))
    
    state_dict = model1.state_dict()
    assert "cell.A" in state_dict
    assert "cell.B" in state_dict
    assert "cell.mean" in state_dict
    assert "cell.inv_std" in state_dict
    assert "readout.mlp.0.weight" in state_dict
    
    model2 = StudentModel(num_channels=num_channels, order=order)
    model2.load_state_dict(state_dict)
    
    x = torch.randn(2, 5, num_channels)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)
    torch.testing.assert_close(out1, out2)


def test_long_horizon_numerical_stability_10000_steps():
    """Streams 10,000 sequential steps in float32 verifying zero NaN/Inf and bounded state norms."""
    torch.manual_seed(42)
    num_channels = 4
    order = 8
    cell = DepthwiseChebyshevCell(num_channels=num_channels, order=order, window_size=30.0, dt=1.0)
    state = cell.init_state(batch_size=1)

    t = torch.arange(10000, dtype=torch.float32)
    # Mix of sinusoidal drift + bounded stochastic noise
    inputs = torch.sin(2.0 * np.pi * t / 50.0).unsqueeze(-1).repeat(1, num_channels)
    inputs = inputs + 0.5 * torch.randn(10000, num_channels)

    checkpoints = {100, 500, 1000, 10000}
    norms = {}

    with torch.no_grad():
        for step in range(1, 10001):
            state = cell(inputs[step - 1 : step], state)
            if step in checkpoints:
                assert torch.isfinite(state).all(), f"Non-finite state detected at step {step}"
                norm_val = float(torch.linalg.norm(state).item())
                norms[step] = norm_val
                assert norm_val < 100.0, f"State norm exploded at step {step}: {norm_val}"

    assert set(norms.keys()) == checkpoints

