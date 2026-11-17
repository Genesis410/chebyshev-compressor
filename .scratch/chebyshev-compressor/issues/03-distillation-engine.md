# 03: Non-Invasive Trajectory Distillation Engine with Forward Hooks

**What to build:**
The distillation training loop and synthetic data generator, supporting non-invasive PyTorch forward hooks, linear trajectory adapter projection, cosine annealing $\lambda(t)$ scheduling, and normalized loss calculation, verifiable by training a student model on a synthetic multi-frequency benchmark in $<15$ seconds.

**Blocked by:** 02: PyTorch Depthwise Chebyshev State-Space Model with Non-Linear Readout

**Status:** resolved

- [x] Implement synthetic multi-frequency signal generator with anomaly spikes in `datasets.py`
- [x] Implement non-invasive PyTorch forward hooks with tuple output handling and `batch_first` transposition
- [x] Implement linear trajectory alignment adapter $P$ with cosine annealing weight schedule $\lambda(t) \in [1.0, 0.1]$
- [x] Implement normalized joint loss function ($\mathcal{L}_{\text{task}} + \lambda(t) \mathcal{L}_{\text{state}}$)
- [x] Verify convergence on synthetic smoke-test dataset in under 15 seconds

## Implementation Summary
- `chebyshev/datasets.py`: Implemented deterministic synthetic signal generator simulating machinery degradation trends with multi-frequency sinusoids, anomaly spikes, and channel normalization statistics.
- `chebyshev/distill.py`:
  - `TeacherStateHook`: Non-invasive PyTorch hook intercepting recurrent states (`nn.LSTM`, `nn.GRU`), standardizing batch dimensions, and providing graceful fallback for non-recurrent teachers.
  - `cosine_annealing_lambda`: Smoothly decays trajectory alignment weight $\lambda(t) \in [1.0, 0.1]$.
  - `distill`: Joint distillation training loop optimizing output matching and trajectory alignment with temporary linear adapter (discarded post-training) and baking sensor normalization statistics into the student model.
- Verified by 5 comprehensive unit tests in `tests/test_distill.py`. Total test suite: 16 tests passing.
