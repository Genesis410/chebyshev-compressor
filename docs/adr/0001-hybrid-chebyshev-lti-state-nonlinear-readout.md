# Hybrid Chebyshev LTI-State with Non-Linear Readout Head

## Context
Deploying multi-layer LSTM networks to microcontrollers (MCU) is bottle-necked by sequential transcendental gating operations ($\sigma$ and $\tanh$), costing dozens of CPU cycles per element. While an end-to-end linear recurrent model ($s_{t+1} = \bar{A} s_t + \bar{B} x_t$, $y_t = W s_t + b$) minimizes execution time, it acts purely as a linear time-invariant (LTI) filter, failing to capture complex non-linear anomaly thresholds and degrading accuracy well below 95%.

## Decision
We decouple the memory compression mechanism from the prediction mapping:
1. **Recurrence**: Strictly linear time-invariant orthogonal Chebyshev state updates ($s_{t+1} = \bar{A} s_t + \bar{B} x_t$) without non-linear activations inside the temporal loop, yielding $O(1)$ memory and pure multiply-accumulate (MAC) DSP execution.
2. **Readout**: A lightweight non-linear Multi-Layer Perceptron ($y_t = W_2 \, \text{ReLU}(W_1 s_t + b_1) + b_2$) applied once per step outside the recurrent state loop.

## Consequences
- Guarantees $O(K)$ state updates per step using diagonalized or block-diagonal Chebyshev matrices.
- Eliminates expensive transcendental function evaluations (`expf`) in the sequential loop.
- Retains over 95% of non-linear teacher predictive capability on complex edge benchmarks like NASA C-MAPSS.
