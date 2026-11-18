# 01: Canonical Chebyshev Operators, Signal Reconstruction, and Long-Horizon Stability

Status: resolved
Priority: P0 (Blocker)
Testing Seam: Operator Seam (`chebyshev/math_ops.py`, `tests/test_math.py`, `tests/test_model.py`)
Blocked by: None (can start immediately)

## What to build
Lock in a single, mathematically rigorous continuous-time Shifted Chebyshev operator derivation $(A, B)$ from the sliding-window transport PDE $\partial f_t / \partial t = -\partial f_t / \partial \tau$ on $[0, \theta]$ in `chebyshev/math_ops.py`, expose a polynomial basis reconstruction helper for empirical validation, and expand automated mathematical tests to verify Lyapunov stability, signal reconstruction convergence, and 10,000-step long-horizon numerical boundedness.

## Acceptance Criteria

- [x] **Single Canonical Operator Definition**: Derived and implemented the exact continuous-time **Chebyshev-Gauss-Radau (CGR) Spectral-SAT** operator $(A, B)$ in `chebyshev/math_ops.py`:
  $$A = \frac{2}{\theta}\left(D - \alpha\,\boldsymbol{\gamma}\mathbf{1}^T\right), \quad B = \frac{2\alpha}{\theta}\,\boldsymbol{\gamma}, \quad \alpha = \frac{2K - 1}{2}$$
  where $D$ is the exact upper-triangular Chebyshev differentiation matrix and $\gamma_k = \pi c_k$ ($\gamma_0 = 1, \gamma_{k \ge 1} = 2$) reflects the exact Chebyshev orthogonal weights $c_0 = 1/\pi, c_k = 2/\pi$.
- [x] **Lyapunov & Spectral Stability Verification**: Verified via `tests/test_math.py`:
  - Strict Hurwitz eigenvalues $\max_i \mathrm{Re}(\lambda_i(A)) < 0$ across $K \in \{1, 2, 4, 8, 16, 32\}$ and $\theta \in \{10, 30, 100\}$.
  - Exact analytical CGR Lyapunov stability certificate $P_{\text{CGR}} = V^T \mathrm{diag}(w_j(1 + x_j)) V \succ 0$ and $A^T P_{\text{CGR}} + P_{\text{CGR}} A \prec 0$.
  - Exact polynomial transport PDE residual ($< 10^{-11}$) for polynomials up to degree $K - 1$ and exact DC steady-state gain $-A^{-1}B = [1, 0, \dots, 0]^T$.
  - Tustin discrete spectral radius $\rho(\bar{A}) < 1.0$ strictly in both `float64` and `float32`.
- [x] **Safeguard — Signal Reconstruction Test with Exact Orthogonal Normalization**:
  - Added `chebyshev_orthogonal_weights`, `evaluate_shifted_chebyshev_basis`, and `reconstruct_chebyshev_history` in `chebyshev/math_ops.py`.
  - Verified monotonic spectral convergence across $K \in \{4, 8, 16, 32\}$ on smooth low-frequency signals ($\omega \ll K/\theta$) with sub-0.1% relative $L_2$ error at $K = 16$ (`2.09e-4`) and $K = 32$ (`1.74e-4`).
- [x] **Long-Horizon Numerical Stability (10,000 Steps)**:
  - Added `test_long_horizon_numerical_stability_10000_steps` in `tests/test_model.py` verifying zero `NaN`/`Inf` and uniformly bounded state norms across $T \in \{100, 500, 1000, 10000\}$.
- [x] All 15 tests in `tests/test_math.py` and `tests/test_model.py` pass deterministically.

## Implementation Summary
- Replaced the mathematically inconsistent dense operator in `chebyshev/math_ops.py` (which previously produced $>100\%$ signal reconstruction error) with the exact Chebyshev-Gauss-Radau (CGR) Spectral-SAT operator derived from the transport PDE.
- Added `chebyshev_differentiation_matrix`, `chebyshev_lyapunov_matrix`, `chebyshev_orthogonal_weights`, `evaluate_shifted_chebyshev_basis`, and `reconstruct_chebyshev_history`.
- Replaced `np.linalg.inv` with `np.linalg.solve` in `discretize_tustin` for improved numerical conditioning.
- Verified by 15 passing tests in `tests/test_math.py` and `tests/test_model.py`.
