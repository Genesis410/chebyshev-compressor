# Depthwise Chebyshev Recurrence with Baked Input Normalization

## Context
Industrial multi-sensor workloads (such as NASA C-MAPSS with 14 active channels) present heterogeneous physical dimensions (temperatures, pressures, fan speeds) with vastly different dynamic ranges. Attempting to compress all channels into a single coupled Chebyshev state vector destroys per-channel spectral resolution, while feeding unnormalized raw sensor data to orthogonal Chebyshev polynomials on $[-1, 1]$ triggers numerical instability.

## Decision
We adopt a Depthwise Chebyshev State-Space architecture:
1. Each input channel $d \in \{1, \dots, D_{\text{in}}\}$ maintains its own independent order-$K$ Chebyshev state vector $s^{(d)}_t$, sharing the identical transition matrix $\bar{A} \in \mathbb{R}^{K \times K}$ and input vector $\bar{B} \in \mathbb{R}^K$.
2. Channel normalization parameters ($\mu_d$ and $\sigma_d^{-1}$) computed on the calibration dataset are baked directly into `chebyshev_engine.h` as `static const float` arrays, normalizing raw physical inputs on-the-fly before projection into the Chebyshev basis.
3. Cross-channel correlations are learned exclusively by the non-linear Readout MLP operating on the concatenated state vector $[s^{(1)}_t, \dots, s^{(D_{\text{in}})}_t]$.

## Consequences
- Keeps state memory microscopic ($D_{\text{in}} \cdot K \cdot 4$ bytes, e.g., 448 bytes for 14 sensors at order $K=8$), easily resident in MCU registers or L1 cache.
- Simplifies edge deployment by allowing embedded callers to feed raw physical sensor outputs directly into `chebyshev_step()`.
- Prevents cross-channel gradient interference during spectral polynomial representation.
