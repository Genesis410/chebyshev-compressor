"""Analytical continuous Chebyshev operators and double-precision Tustin discretization."""

from typing import Tuple, Union
import numpy as np


def continuous_chebyshev_operators(
    order: int, theta: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives continuous state-space operators (A, B) for Translated Chebyshev memory.
    
    The memory projects a continuous sliding window of length `theta` onto an
    orthonormal shifted Chebyshev polynomial basis.
    
    Args:
        order: Truncation order K of the polynomial expansion (number of states).
        theta: Length of the sliding temporal window.
        
    Returns:
        Tuple of (A, B) in float64 precision:
            A: (order, order) continuous transition matrix with stable non-positive real eigenvalues.
            B: (order, 1) continuous input projection vector.
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")
    if theta <= 0.0:
        raise ValueError(f"Window size theta must be > 0, got {theta}")

    # Orthonormal Chebyshev basis scaling: R_0 = 1.0, R_n = sqrt(2.0) for n >= 1
    R = np.full(order, np.sqrt(2.0), dtype=np.float64)
    R[0] = 1.0

    # Meshgrid of indices: i = row index, j = column index
    j, i = np.meshgrid(np.arange(order), np.arange(order))

    # Continuous transition matrix A and input projection vector B
    A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1.0) * R[None, :]
    A = -(1.0 / theta) * A
    B = (1.0 / theta) * R[:, None]

    return A, B


def discretize_tustin(
    A: np.ndarray, B: np.ndarray, dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts continuous-time operators (A, B) to discrete (A_bar, B_bar) via Bilinear Tustin Transform.
    
    Computed in double precision (float64) to prevent numerical cancellation:
        A_bar = (I - 0.5 * dt * A)^(-1) * (I + 0.5 * dt * A)
        B_bar = (I - 0.5 * dt * A)^(-1) * (dt * B)
        
    Args:
        A: (order, order) continuous state transition matrix.
        B: (order, 1) continuous input projection vector.
        dt: Sampling time step.
        
    Returns:
        Tuple of (A_bar, B_bar) in float64 precision.
    """
    if dt <= 0.0:
        raise ValueError(f"Sampling time step dt must be > 0, got {dt}")

    order = A.shape[0]
    I = np.eye(order, dtype=np.float64)

    # Invert (I - 0.5 * dt * A) in float64
    inv_term = np.linalg.inv(I - 0.5 * dt * A)
    A_bar = inv_term @ (I + 0.5 * dt * A)
    B_bar = inv_term @ (dt * B)

    return A_bar, B_bar


def get_chebyshev_operators(
    order: int = 8,
    window_size: float = 30.0,
    dt: float = 1.0,
    dtype: type = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-level deterministic builder for discrete Chebyshev State-Space operators.
    
    Calculates continuous operators in float64, discretizes via Tustin transform,
    and returns matrices cast to the requested precision (default float32).
    
    Args:
        order: Polynomial order K (number of coefficients per channel, default 8).
        window_size: Sliding temporal window length (default 30.0).
        dt: Discrete sampling interval (default 1.0).
        dtype: Output numpy datatype (default np.float32).
        
    Returns:
        Tuple of (A_bar, B_bar):
            A_bar: (order, order) transition matrix.
            B_bar: (order,) 1D input projection vector.
    """
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")
    if window_size <= 0.0:
        raise ValueError(f"Window size must be > 0, got {window_size}")
    if dt <= 0.0:
        raise ValueError(f"Sampling step dt must be > 0, got {dt}")

    # Compute in float64
    A_cont, B_cont = continuous_chebyshev_operators(order, theta=window_size)
    A_bar, B_bar = discretize_tustin(A_cont, B_cont, dt=dt)

    # Cast to target dtype and squeeze B to 1D
    return A_bar.astype(dtype), B_bar.squeeze(-1).astype(dtype)
