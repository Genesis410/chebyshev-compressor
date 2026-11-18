"""Analytical continuous Chebyshev-Gauss-Radau (CGR) Spectral-SAT operators and Tustin discretization.

Mathematical Formulation:
    Consider the sliding-window history function f_t(tau) = u(t - tau) for tau in [0, theta],
    where tau = 0 is the present (f_t(0) = u(t)) and tau = theta is the oldest remembered point.
    Mapping tau -> x = 1 - 2*tau/theta in [-1, 1] transforms the transport PDE
        df_t/dt = -df_t/dtau,  f_t(0) = u(t)
    into
        df_t/dt = (2/theta) * df_t/dx,  f_t(1) = u(t).

    Expanding f_t(x) in Shifted Chebyshev Polynomials of the First Kind:
        f_t(x) ~= sum_{k=0}^{K-1} c_k(t) * T_k(x),   T_k(x) = cos(k * arccos(x)),
    where the Chebyshev orthogonal projection weights under w(x) = (1 - x^2)^(-1/2) are
    w_norm_0 = 1/pi and w_norm_k = 2/pi (k >= 1), so gamma_k = pi * w_norm_k = {1 if k=0 else 2}.

    Enforcing the inflow boundary condition f_t(1) = u(t) weakly via the Chebyshev-Gauss-Radau
    (CGR) Simultaneous Approximation Term (SAT) at x_0 = 1 yields the exact linear ODE:
        dc/dt = A * c + B * u(t)
    with:
        D_{j,k} = k (if j == 0) or 2*k (if j >= 1) for k > j and (k - j) odd, else 0
        alpha   = (2*K - 1) / 2
        A       = (2 / theta) * (D - alpha * gamma @ 1^T)
        B       = (2 * alpha / theta) * gamma

    Properties:
        1. Exact Polynomial Transport: For any polynomial history of degree <= K - 1,
           dc/dt = A*c + B*u(t) satisfies the transport PDE identically (zero residual).
        2. Provable Lyapunov Stability: With the SPD CGR weighted energy matrix
           P_CGR = V^T diag(w_j * (1 + x_j)) V > 0, A satisfies A^T P_CGR + P_CGR A < 0
           for all K >= 1 and theta > 0, guaranteeing strict Hurwitz eigenvalues Re(lambda_i(A)) < 0.
        3. Conformal Tustin Stability: Bilinear discretization maps Re(lambda_i(A)) < 0 strictly
           into the open unit disk |lambda_i(A_bar)| < 1.0 for every dt > 0.
"""

from typing import Tuple
import numpy as np


def chebyshev_differentiation_matrix(order: int) -> np.ndarray:
    """Constructs the exact upper-triangular Chebyshev modal differentiation matrix D.

    Maps modal Chebyshev coefficients c of f(x) = sum_{k=0}^{K-1} c_k T_k(x)
    to modal coefficients of df/dx on [-1, 1]:
        D_{j, k} = k       if j == 0, k > 0, and k is odd
        D_{j, k} = 2 * k   if j >= 1, k > j, and (k - j) is odd
        D_{j, k} = 0       otherwise

    Args:
        order: Truncation order K >= 1.

    Returns:
        D: (order, order) float64 differentiation matrix.
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")

    D = np.zeros((order, order), dtype=np.float64)
    for k in range(1, order):
        for j in range(k):
            if (k - j) % 2 == 1:
                D[j, k] = float(k) if j == 0 else 2.0 * float(k)
    return D


def continuous_chebyshev_operators(
    order: int, theta: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives canonical continuous-time CGR Spectral-SAT operators (A, B).

    Args:
        order: Truncation order K of the Chebyshev expansion (number of states).
        theta: Length of the sliding temporal window (> 0).

    Returns:
        Tuple of (A, B) in float64 precision:
            A: (order, order) strictly Hurwitz continuous transition matrix.
            B: (order, 1) continuous input projection vector.
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")
    if theta <= 0.0:
        raise ValueError(f"Window size theta must be > 0, got {theta}")

    # Exact upper-triangular Chebyshev differentiation matrix D
    D = chebyshev_differentiation_matrix(order)

    # Chebyshev boundary projection weights: gamma_k = pi * c_k (1 for k=0, 2 for k>=1)
    # where c_0 = 1/pi, c_k = 2/pi are the orthogonal Chebyshev normalization constants.
    gamma = np.full((order, 1), 2.0, dtype=np.float64)
    gamma[0, 0] = 1.0

    # CGR Simultaneous Approximation Term (SAT) penalty factor: alpha = (2K - 1) / 2
    alpha = (2.0 * float(order) - 1.0) / 2.0
    ones_row = np.ones((1, order), dtype=np.float64)

    scale = 2.0 / float(theta)
    A = scale * (D - alpha * (gamma @ ones_row))
    B = scale * alpha * gamma

    return A, B


def chebyshev_lyapunov_matrix(order: int) -> np.ndarray:
    """Constructs the analytical SPD Lyapunov energy matrix P_CGR for order K.

    Uses the K-point Chebyshev-Gauss-Radau (CGR) quadrature nodes:
        x_j = cos(2 * pi * j / (2*K - 1)),  j = 0, ..., K-1
    with weights w_0 = pi / (2*K - 1) and w_j = 2*pi / (2*K - 1) for j >= 1,
    and Vandermonde matrix V_{j, k} = T_k(x_j):
        P_CGR = V^T @ diag(w_j * (1 + x_j)) @ V

    For the canonical continuous operator A, A^T P_CGR + P_CGR A is strictly
    negative definite for all order >= 1 and theta > 0.

    Args:
        order: Truncation order K >= 1.

    Returns:
        P_cgr: (order, order) symmetric positive-definite float64 matrix.
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")

    j_idx = np.arange(order, dtype=np.float64)
    k_idx = np.arange(order, dtype=np.float64)
    denom = 2.0 * float(order) - 1.0

    angles = 2.0 * np.pi * j_idx / denom
    x_nodes = np.cos(angles)
    weights = np.full(order, 2.0 * np.pi / denom, dtype=np.float64)
    weights[0] = np.pi / denom

    # Modal-to-nodal Chebyshev Vandermonde matrix: V[j, k] = cos(k * angles[j]) = T_k(x_j)
    V = np.cos(np.outer(angles, k_idx))
    P_cgr = V.T @ np.diag(weights * (1.0 + x_nodes)) @ V
    # Symmetrize to eliminate floating-point asymmetry
    return 0.5 * (P_cgr + P_cgr.T)


def evaluate_shifted_chebyshev_basis(
    order: int, tau_grid: np.ndarray, theta: float = 30.0
) -> np.ndarray:
    """Evaluates shifted Chebyshev basis T_k*(tau) = T_k(1 - 2*tau/theta) on tau_grid.

    Args:
        order: Polynomial order K >= 1.
        tau_grid: 1D array of lag times tau in [0, theta] (0 = present, theta = oldest).
        theta: Sliding temporal window length (> 0).

    Returns:
        T_mat: (order, len(tau_grid)) float64 basis matrix where row k is T_k*(tau).
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")
    if theta <= 0.0:
        raise ValueError(f"Window size theta must be > 0, got {theta}")

    tau = np.asarray(tau_grid, dtype=np.float64)
    x = np.clip(1.0 - 2.0 * tau / float(theta), -1.0, 1.0)
    arccos_x = np.arccos(x)
    k_idx = np.arange(order, dtype=np.float64)[:, None]
    return np.cos(k_idx * arccos_x[None, :])


def chebyshev_orthogonal_weights(order: int) -> np.ndarray:
    """Returns orthogonal Chebyshev inner-product normalization weights c_k.

    Under the Chebyshev weight measure w(x) = (1 - x^2)^(-1/2) on [-1, 1]:
        c_0 = 1 / pi,  c_k = 2 / pi  for k >= 1
    such that for any f(x) = sum_k a_k T_k(x), a_k = c_k * <f, T_k>_w.

    Args:
        order: Polynomial order K >= 1.

    Returns:
        weights: (order,) float64 array [1/pi, 2/pi, ..., 2/pi].
    """
    if order < 1:
        raise ValueError(f"Polynomial order must be >= 1, got {order}")
    weights = np.full(order, 2.0 / np.pi, dtype=np.float64)
    weights[0] = 1.0 / np.pi
    return weights


def reconstruct_chebyshev_history(
    state: np.ndarray, tau_grid: np.ndarray, theta: float = 30.0
) -> np.ndarray:
    """Reconstructs historical signal u(t - tau) from Chebyshev modal state vector.

    Because the canonical CGR operator stores modal expansion coefficients directly
    (with orthogonal normalization weights c_0 = 1/pi, c_k = 2/pi absorbed into
    gamma_k = pi * c_k during projection), reconstruction over tau in [0, theta] is:
        u_hat(t - tau) = sum_{k=0}^{K-1} state[..., k] * T_k(1 - 2*tau/theta).

    Args:
        state: State coefficient array of shape (..., order).
        tau_grid: 1D array of lag times tau in [0, theta].
        theta: Sliding temporal window length (> 0).

    Returns:
        Reconstructed signal array of shape (..., len(tau_grid)).
    """
    state_arr = np.asarray(state, dtype=np.float64)
    order = state_arr.shape[-1]
    T_mat = evaluate_shifted_chebyshev_basis(order, tau_grid, theta=theta)
    return state_arr @ T_mat


def discretize_tustin(
    A: np.ndarray, B: np.ndarray, dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts continuous-time operators (A, B) to discrete (A_bar, B_bar) via Bilinear Tustin Transform.

    Computed in double precision (float64) using linear solves to prevent numerical cancellation:
        A_bar = (I - 0.5 * dt * A)^(-1) * (I + 0.5 * dt * A)
        B_bar = (I - 0.5 * dt * A)^(-1) * (dt * B)

    Args:
        A: (order, order) continuous state transition matrix.
        B: (order, 1) continuous input projection vector.
        dt: Sampling time step (> 0).

    Returns:
        Tuple of (A_bar, B_bar) in float64 precision.
    """
    if dt <= 0.0:
        raise ValueError(f"Sampling time step dt must be > 0, got {dt}")

    order = A.shape[0]
    I = np.eye(order, dtype=np.float64)

    lhs = I - 0.5 * float(dt) * A
    A_bar = np.linalg.solve(lhs, I + 0.5 * float(dt) * A)
    B_bar = np.linalg.solve(lhs, float(dt) * B)

    return A_bar, B_bar


def get_chebyshev_operators(
    order: int = 8,
    window_size: float = 30.0,
    dt: float = 1.0,
    dtype: type = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-level deterministic builder for discrete Chebyshev State-Space operators.

    Calculates canonical continuous CGR Spectral-SAT operators in float64,
    discretizes via Bilinear Tustin transform in float64, and returns matrices
    cast to the requested precision (default float32).

    Args:
        order: Polynomial order K (number of coefficients per channel, default 8).
        window_size: Sliding temporal window length theta (default 30.0).
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

    A_cont, B_cont = continuous_chebyshev_operators(order, theta=window_size)
    A_bar, B_bar = discretize_tustin(A_cont, B_cont, dt=dt)

    return A_bar.astype(dtype), B_bar.squeeze(-1).astype(dtype)
