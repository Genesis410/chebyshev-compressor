import numpy as np
import pytest

from chebyshev.math_ops import (
    chebyshev_differentiation_matrix,
    chebyshev_lyapunov_matrix,
    chebyshev_orthogonal_weights,
    continuous_chebyshev_operators,
    discretize_tustin,
    get_chebyshev_operators,
    reconstruct_chebyshev_history,
)


def test_continuous_operators_shape_and_types():
    order = 8
    theta = 30.0
    A, B = continuous_chebyshev_operators(order, theta=theta)

    assert A.shape == (order, order)
    assert B.shape == (order, 1)
    assert A.dtype == np.float64
    assert B.dtype == np.float64


def test_continuous_operators_strict_hurwitz_stability():
    for order in [1, 2, 4, 8, 16, 32]:
        for theta in [10.0, 30.0, 100.0]:
            A, _ = continuous_chebyshev_operators(order, theta=theta)
            eigvals = np.linalg.eigvals(A)
            max_real = np.max(eigvals.real)
            assert max_real < 0.0, (
                f"Continuous A must be strictly Hurwitz for order={order}, theta={theta}: max_real={max_real}"
            )


def test_analytical_lyapunov_dissipation_certificate():
    """Verifies exact CGR weighted energy Lyapunov theorem: P_CGR > 0 and A^T P + P A < 0."""
    for order in [1, 2, 4, 8, 16, 32]:
        theta = 30.0
        A, _ = continuous_chebyshev_operators(order, theta=theta)
        P = chebyshev_lyapunov_matrix(order)

        # P must be symmetric positive definite
        np.testing.assert_allclose(P, P.T, atol=1e-14)
        p_eigs = np.linalg.eigvalsh(P)
        assert np.all(p_eigs > 0.0), f"P_CGR not SPD for K={order}: min_eig={p_eigs[0]}"

        # Lyapunov dissipation matrix A^T P + P A must be strictly negative definite
        lyap = A.T @ P + P @ A
        lyap = 0.5 * (lyap + lyap.T)
        lyap_eigs = np.linalg.eigvalsh(lyap)
        assert np.max(lyap_eigs) < 0.0, (
            f"Lyapunov dissipation failed for K={order}: max_eig={np.max(lyap_eigs)}"
        )

        # Constant state c = [1, 0, ..., 0]^T dissipation check: c^T M c < 0
        c_const = np.zeros(order, dtype=np.float64)
        c_const[0] = 1.0
        val_const = float(c_const @ lyap @ c_const)
        assert val_const < 0.0, f"Constant state energy must be strictly dissipative: {val_const}"


def test_exact_polynomial_transport_pde_residual():
    """Verifies dc/dt = A*c + B*u(t) satisfies transport PDE exactly for polynomials of degree <= K-1."""
    order = 8
    theta = 30.0
    A, B = continuous_chebyshev_operators(order, theta=theta)

    # 1. Constant DC steady-state gain: u(t) = 1 => -A^{-1} B = [1, 0, ..., 0]^T
    dc_state = (-np.linalg.solve(A, B)).ravel()
    expected_dc = np.zeros(order, dtype=np.float64)
    expected_dc[0] = 1.0
    np.testing.assert_allclose(dc_state, expected_dc, atol=1e-12)

    # 2. Degree-5 polynomial u(t) = (t / theta)^5 => exact PDE residual = 0
    t0 = 12.5
    p = (t0 - theta / 2.0) / theta
    q = 0.5
    # Project f_t(x) = (p + q*x)^5 and df_t/dt = (5/theta)*(p + q*x)^4 via Chebyshev-Gauss quadrature
    nodes = np.cos(np.pi * (np.arange(order) + 0.5) / order)
    weights = chebyshev_orthogonal_weights(order)
    # Gauss quadrature weight w_j = pi / K -> weights * (pi / K) = [1/K, 2/K, ..., 2/K]
    quad_scale = weights * (np.pi / float(order))

    f_vals = (p + q * nodes) ** 5
    df_vals = (5.0 / theta) * ((p + q * nodes) ** 4)
    basis_at_nodes = np.cos(np.outer(np.arange(order), np.pi * (np.arange(order) + 0.5) / order))

    c_exact = (quad_scale[:, None] * basis_at_nodes) @ f_vals
    dc_exact = (quad_scale[:, None] * basis_at_nodes) @ df_vals
    dc_ode = (A @ c_exact[:, None] + B * ((t0 / theta) ** 5)).ravel()

    np.testing.assert_allclose(dc_ode, dc_exact, rtol=1e-11, atol=1e-12)


def test_tustin_discretization_stability():
    for order in [2, 4, 8, 16, 32]:
        for theta in [10.0, 30.0, 100.0]:
            for dt in [0.1, 0.5, 1.0]:
                A, B = continuous_chebyshev_operators(order, theta=theta)
                A_bar, B_bar = discretize_tustin(A, B, dt=dt)

                assert A_bar.shape == (order, order)
                assert B_bar.shape == (order, 1)

                # All discrete eigenvalues must lie strictly inside the open unit disk
                max_eig_64 = np.max(np.abs(np.linalg.eigvals(A_bar)))
                assert max_eig_64 < 1.0, (
                    f"Discretized A_bar (float64) is unstable for order={order}, theta={theta}, dt={dt}: {max_eig_64}"
                )

                # Verify float32 quantization preserves strict unit-disk stability
                max_eig_32 = np.max(np.abs(np.linalg.eigvals(A_bar.astype(np.float32).astype(np.float64))))
                assert max_eig_32 < 1.0, (
                    f"Discretized A_bar (float32) is unstable for order={order}, theta={theta}, dt={dt}: {max_eig_32}"
                )


def test_signal_reconstruction_monotonic_spectral_convergence():
    """Verifies Chebyshev state reconstructs smooth low-frequency signals (omega << K/theta)
    with deterministic bounded L2 error that strictly decreases as K increases (4 -> 8 -> 16 -> 32).
    """
    theta = 30.0
    dt = 0.05  # Fine temporal resolution to isolate polynomial spatial approximation error
    t_steps = int(150.0 / dt)  # 5 full windows to reach steady-state history
    t_arr = np.arange(t_steps, dtype=np.float64) * dt

    # Smooth multi-tone signal with fundamental frequency below Nyquist bound even for small K
    # Periods 20s and 12s -> 1.5 and 2.5 cycles across window theta=30s
    def signal_fn(t: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * t / 20.0) + 0.5 * np.cos(2.0 * np.pi * t / 12.0) + 0.25

    u_arr = signal_fn(t_arr)
    tau_grid = np.linspace(0.0, theta, 200)
    t_now = t_arr[-1] + 0.5 * dt  # Tustin trapezoidal midpoint alignment
    u_true = signal_fn(t_now - tau_grid)

    # Verify orthogonal weight constants match canonical Chebyshev inner product
    w_ortho = chebyshev_orthogonal_weights(8)
    assert np.isclose(w_ortho[0], 1.0 / np.pi)
    assert np.allclose(w_ortho[1:], 2.0 / np.pi)

    errors = []
    for order in [4, 8, 16, 32]:
        A_bar, B_bar = get_chebyshev_operators(
            order=order, window_size=theta, dt=dt, dtype=np.float64
        )
        state = np.zeros(order, dtype=np.float64)
        for u_t in u_arr:
            state = A_bar @ state + B_bar * u_t

        u_rec = reconstruct_chebyshev_history(state, tau_grid, theta=theta)
        rel_l2_err = float(np.linalg.norm(u_true - u_rec) / np.linalg.norm(u_true))
        errors.append(rel_l2_err)

    # Strictly decreasing error with increasing polynomial order K
    assert errors[0] > errors[1] > errors[2] > errors[3], (
        f"Expected monotonic spectral convergence across K=[4,8,16,32], got {errors}"
    )
    # High-order reconstruction must be sub-0.1% relative L2 error
    assert errors[2] < 1e-3, f"K=16 relative L2 reconstruction error too high: {errors[2]}"
    assert errors[3] < 5e-4, f"K=32 relative L2 reconstruction error too high: {errors[3]}"


def test_get_chebyshev_operators_convenience_function():
    order = 8
    window_size = 30.0
    dt = 1.0

    # Check default float32 output
    A_bar, B_bar = get_chebyshev_operators(order=order, window_size=window_size, dt=dt, dtype=np.float32)
    assert A_bar.dtype == np.float32
    assert B_bar.dtype == np.float32
    assert A_bar.shape == (order, order)
    assert B_bar.shape == (order,)

    # Check float64 option
    A_bar_64, B_bar_64 = get_chebyshev_operators(order=order, window_size=window_size, dt=dt, dtype=np.float64)
    assert A_bar_64.dtype == np.float64
    assert B_bar_64.dtype == np.float64

    # Verify consistency between 32 and 64
    np.testing.assert_allclose(A_bar, A_bar_64, rtol=1e-5, atol=1e-5)


def test_invalid_parameters_raise_error():
    with pytest.raises(ValueError):
        get_chebyshev_operators(order=0)

    with pytest.raises(ValueError):
        chebyshev_differentiation_matrix(order=0)

    with pytest.raises(ValueError):
        chebyshev_lyapunov_matrix(order=0)

    with pytest.raises(ValueError):
        get_chebyshev_operators(order=8, window_size=0.0)

    with pytest.raises(ValueError):
        get_chebyshev_operators(order=8, window_size=-10.0)

    with pytest.raises(ValueError):
        get_chebyshev_operators(order=8, dt=0.0)
