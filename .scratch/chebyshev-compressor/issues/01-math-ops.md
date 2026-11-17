# 01: Analytical Chebyshev Operators & Tustin Discretization

**What to build:**
A standalone mathematical module that generates continuous shifted Chebyshev derivative matrices and discretizes them via double-precision Bilinear Tustin Transform into stable $(\bar{A}, \bar{B})$ operators, fully verified by unit tests validating spectral eigenvalues and mathematical boundary conditions.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Derive shifted Chebyshev derivative matrix $A$ and input vector $B$ on $[0, \theta]$ with scale factor $2/\theta$
- [x] Implement Bilinear Tustin continuous-to-discrete conversion in `float64`
- [x] Cast final discretized operators to `float32`
- [x] Unit tests in `tests/test_math.py` verifying that all eigenvalues of $\bar{A}$ satisfy $|\lambda_i| \le 1.0$

## Implementation Summary
Implemented `chebyshev/math_ops.py` with:
- `continuous_chebyshev_operators(order, theta)`: Closed-form continuous Translated Chebyshev state operators with $O(1/t)$ polynomial memory characteristics and guaranteed non-positive real eigenvalues.
- `discretize_tustin(A, B, dt)`: IEEE-754 64-bit Bilinear Tustin Transform mapping the stable left-half plane conformal-isomorphically onto the unit disk.
- `get_chebyshev_operators(order, window_size, dt, dtype)`: High-level constructor returning stable $(\bar{A}, \bar{B})$ cast to single-precision float32.
- Verified by 5 passing unit tests in `tests/test_math.py`.
