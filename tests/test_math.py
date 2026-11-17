import numpy as np
import pytest

from chebyshev.math_ops import (
    continuous_chebyshev_operators,
    discretize_tustin,
    get_chebyshev_operators,
)


def test_continuous_operators_shape_and_types():
    order = 8
    theta = 30.0
    A, B = continuous_chebyshev_operators(order, theta=theta)
    
    assert A.shape == (order, order)
    assert B.shape == (order, 1)
    assert A.dtype == np.float64
    assert B.dtype == np.float64


def test_continuous_operators_stability():
    for order in [2, 4, 8, 16]:
        for theta in [10.0, 30.0, 100.0]:
            A, _ = continuous_chebyshev_operators(order, theta=theta)
            eigvals = np.linalg.eigvals(A)
            # Continuous system must have non-positive real parts for stability
            assert np.all(eigvals.real <= 1e-12), (
                f"Continuous A has unstable eigenvalues for order={order}, theta={theta}: {eigvals.real}"
            )


def test_tustin_discretization_stability():
    for order in [2, 4, 8, 16, 32]:
        for theta in [10.0, 30.0, 100.0]:
            for dt in [0.1, 0.5, 1.0]:
                A, B = continuous_chebyshev_operators(order, theta=theta)
                A_bar, B_bar = discretize_tustin(A, B, dt=dt)
                
                assert A_bar.shape == (order, order)
                assert B_bar.shape == (order, 1)
                
                # All discrete eigenvalues must lie strictly inside or on the unit disk
                eigs = np.abs(np.linalg.eigvals(A_bar))
                max_eig = np.max(eigs)
                assert max_eig <= 1.0 + 1e-9, (
                    f"Discretized A_bar is unstable for order={order}, theta={theta}, dt={dt}: max_eig={max_eig}"
                )


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
        get_chebyshev_operators(order=8, window_size=0.0)
        
    with pytest.raises(ValueError):
        get_chebyshev_operators(order=8, window_size=-10.0)
        
    with pytest.raises(ValueError):
        get_chebyshev_operators(order=8, dt=0.0)
