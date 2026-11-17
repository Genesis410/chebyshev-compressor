# Closed-Form Chebyshev Discretization via Double-Precision Tustin Transform

## Context
Projecting a continuous sliding window $[0, \theta]$ onto shifted orthogonal Chebyshev polynomials of the first kind $T_n^*(t)$ requires a continuous derivative transition matrix $A$ and input vector $B$. When converting this continuous state space into discrete time step $\Delta t$, naive numerical integration or single-precision matrix inversion risks catastrophic cancellation and numerical instability as polynomial order $K$ increases.

## Decision
1. We compute continuous-time operators $A$ and $B$ analytically incorporating the scaling factor $2/\theta$.
2. The continuous-to-discrete conversion $\bar{A} = (I - \frac{\Delta t}{2} A)^{-1} (I + \frac{\Delta t}{2} A)$ and $\bar{B} = (I - \frac{\Delta t}{2} A)^{-1} (\Delta t B)$ is computed strictly using IEEE-754 64-bit floating-point (`float64` / `double`) arithmetic before casting final coefficients to single-precision `float32`.

## Consequences
- Eliminates trial-and-error hyperparameter search for recurrent state matrices.
- Guarantees deterministic eigenvalues strictly within the stable unit disk ($|\lambda_i| \le 1$).
- Prevents roundoff error accumulation during matrix inversion.
