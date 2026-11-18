"""PyTorch Depthwise Chebyshev State-Space Cell and StudentModel."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from chebyshev.math_ops import get_chebyshev_operators


class DepthwiseChebyshevCell(nn.Module):
    """Depthwise Chebyshev State-Space Recurrent Cell.
    
    Maintains independent order-K Chebyshev polynomial state vectors for each
    input sensor channel while sharing transition dynamics (A_bar, B_bar).
    Executes continuous-time polynomial projection via linear time-invariant
    multiply-accumulate updates with baked-in sensor normalization and
    IEEE-754 input sanity clamping.
    """

    def __init__(
        self,
        num_channels: int = 14,
        order: int = 8,
        window_size: float = 30.0,
        dt: float = 1.0,
        clamp_value: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.order = order
        self.window_size = window_size
        self.dt = dt
        self.clamp_value = clamp_value

        # Analytical discretized operators
        A_np, B_np = get_chebyshev_operators(
            order=order, window_size=window_size, dt=dt, dtype=np.float32
        )
        self.register_buffer("A", torch.from_numpy(A_np))
        self.register_buffer("B", torch.from_numpy(B_np))

        # Channel normalization buffers (baked into model)
        self.register_buffer("mean", torch.zeros(num_channels, dtype=torch.float32))
        self.register_buffer("inv_std", torch.ones(num_channels, dtype=torch.float32))

    def set_normalization(
        self,
        mean: Union[np.ndarray, torch.Tensor],
        std: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Sets dataset channel mean and standard deviation for baked normalization.
        
        Args:
            mean: Array or tensor of shape (num_channels,).
            std: Array or tensor of shape (num_channels,).
        """
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()

        self.mean.copy_(mean)
        # Avoid division by zero
        inv_std = 1.0 / (std + 1e-8)
        self.inv_std.copy_(inv_std)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Allocates zero-initialized caller state tensor of shape (B, D_in, K)."""
        if device is None:
            device = self.A.device
        if dtype is None:
            dtype = self.A.dtype
        return torch.zeros(
            batch_size, self.num_channels, self.order, device=device, dtype=dtype
        )

    def forward(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Computes 1-step recurrent state transition for a batch of multi-channel inputs.
        
        Args:
            x_t: Input sensor readings of shape (batch_size, num_channels).
            state: Current state tensor of shape (batch_size, num_channels, order).
            
        Returns:
            next_state: Updated state tensor of shape (batch_size, num_channels, order).
        """
        # 1. Baked normalization
        x_norm = (x_t - self.mean) * self.inv_std

        # 2. Input sanity clamping and IEEE-754 non-finite (NaN / +/-Inf) quarantine (ADR-0007)
        # Quarantines NaN, +Inf, and -Inf to 0.0 before clamping finite spikes to [-clamp, +clamp],
        # matching the C99 exponent bit-mask check ((bits & 0x7F800000U) == 0x7F800000U -> 0.0f).
        x_finite = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        x_sanitized = torch.clamp(x_finite, -self.clamp_value, self.clamp_value)

        # 3. Linear state-space update: s_{t+1} = A s_t + B x_t
        # state: (B, D_in, K) @ A.T (K, K) -> (B, D_in, K)
        # B: (K,) -> view(1, 1, K)
        next_state = torch.matmul(state, self.A.T) + x_sanitized.unsqueeze(-1) * self.B.view(1, 1, -1)

        return next_state


class ReadoutMLP(nn.Module):
    """Lightweight 2-layer Feedforward Perceptron with ReLU activation."""

    def __init__(
        self, in_features: int, hidden_dim: int = 32, out_features: int = 1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class StudentModel(nn.Module):
    """Complete Chebyshev Recurrent Student Architecture.
    
    Combines Depthwise Chebyshev linear state memory with a non-linear Readout MLP.
    """

    def __init__(
        self,
        num_channels: int = 14,
        order: int = 8,
        hidden_dim: int = 32,
        out_features: int = 1,
        window_size: float = 30.0,
        dt: float = 1.0,
        clamp_value: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.order = order
        self.hidden_dim = hidden_dim
        self.out_features = out_features

        self.cell = DepthwiseChebyshevCell(
            num_channels=num_channels,
            order=order,
            window_size=window_size,
            dt=dt,
            clamp_value=clamp_value,
        )
        self.readout = ReadoutMLP(
            in_features=num_channels * order,
            hidden_dim=hidden_dim,
            out_features=out_features,
        )

    def step(
        self, x_t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a single inference step, returning (prediction, next_state)."""
        next_state = self.cell(x_t, state)
        flat_state = next_state.reshape(x_t.shape[0], -1)
        y_t = self.readout(flat_state)
        return y_t, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Processes sequential input tensor through the Chebyshev state-space model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_channels).
            state: Optional initial state of shape (batch_size, num_channels, order).
            return_sequence: If True, returns predictions at all time steps (B, T, D_out).
                             If False, returns prediction at the final step (B, D_out).
            return_trajectory: If True, returns concatenated state trajectory (B, T, D_in * K).
            
        Returns:
            output (Tensor) or (output, trajectory) if return_trajectory is True.
        """
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.cell.init_state(batch_size, device=x.device, dtype=x.dtype)

        step_outputs = []
        trajectories = []

        for t in range(seq_len):
            state = self.cell(x[:, t, :], state)
            flat_state = state.reshape(batch_size, -1)

            if return_trajectory:
                trajectories.append(flat_state)

            if return_sequence or t == seq_len - 1:
                y_t = self.readout(flat_state)
                step_outputs.append(y_t)

        if return_sequence:
            out = torch.stack(step_outputs, dim=1)
        else:
            out = step_outputs[-1]

        if return_trajectory:
            traj = torch.stack(trajectories, dim=1)
            return out, traj

        return out
