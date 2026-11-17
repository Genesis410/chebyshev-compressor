# 02: PyTorch Depthwise Chebyshev State-Space Model with Non-Linear Readout

**What to build:**
The complete PyTorch neural architecture featuring `DepthwiseChebyshevCell` and `StudentModel` with input sanity clamping ($[-5.0, 5.0]$) and a 2-layer ReLU Readout MLP, complete with forward-pass unit tests verifying recurrent gradient flow, state tensor dimensions, and causal sequence execution.

**Blocked by:** 01: Analytical Chebyshev Operators & Tustin Discretization

**Status:** resolved

- [x] Implement `DepthwiseChebyshevCell` maintaining per-channel $s_t \in \mathbb{R}^{D_{\text{in}} \times K}$
- [x] Implement input sanity clamping using `torch.clamp(x_norm, -5.0, 5.0)` and NaN quarantine
- [x] Implement `StudentModel` combining Depthwise recurrent cell and 2-layer Readout MLP ($D_{\text{in}} \cdot K \to \text{hidden} \to D_{\text{out}}$)
- [x] Unit tests in `tests/test_model.py` verifying batch forward shapes and backward gradient flow

## Implementation Summary
Implemented `chebyshev/models.py` containing:
- `DepthwiseChebyshevCell`: Depthwise linear recurrence ($s_{t+1} = \bar{A} s_t + \bar{B} x_t$) with baked normalization buffers (`mean`, `inv_std`), caller-allocated zero state initialization, and IEEE-754 input clamping + NaN quarantine.
- `ReadoutMLP`: Lightweight feedforward network ($D_{\text{in}} \cdot K \to \text{hidden} \to D_{\text{out}}$) with ReLU non-linearity.
- `StudentModel`: Unified architecture supporting sequence-to-sequence, sequence-to-one, single-step inference (`model.step`), and trajectory extraction for distillation.
- Verified by 6 comprehensive unit tests in `tests/test_model.py`.
