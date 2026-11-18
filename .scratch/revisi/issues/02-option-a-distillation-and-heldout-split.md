# 02: Option A Distillation Refactor, Fast State Precomputation, and Held-Out Engine Split

Status: resolved
Priority: P1 (Blocker)
Testing Seam: Primary Seam (`chebyshev/distill.py`, `chebyshev/datasets.py`, `chebyshev/cli.py`, `tests/test_distill.py`, `tests/test_cli.py`)
Blocked by: 01

## What to build
Refactor the distillation engine to implement **Option A (Fixed Analytical Orthogonal Memory with Supervised Multi-Objective Readout Distillation)**: eliminate the misleading discarded trajectory `adapter`, precompute frozen Chebyshev states before training for $>10\times$ CPU speedup, combine ground-truth supervision with teacher mimicry (with automatic $\lambda = 1.0$ fallback for unlabeled calibration datasets), and enforce strict engine-level train/validation splitting on NASA C-MAPSS FD001.

## Acceptance Criteria

- [ ] **Remove Discarded Trajectory Adapter**:
  - Remove the temporary `adapter` (`nn.Linear`) and unused `loss_state` trajectory alignment optimization from `chebyshev/distill.py` so that 100% of trained parameters (`student.readout`) persist into the exported C99 header.
  - Keep `TeacherStateHook` only if explicitly needed for optional diagnostics/inspection, or cleanly remove dead optimization hooks from the distillation loop.
- [ ] **Safeguard — Fast State Precomputation**:
  - In `chebyshev/distill.py`, because $(\bar{A}, \bar{B})$ in `student.cell` are frozen analytical buffers, run a single `torch.no_grad()` pass over the calibration/training dataset before the epoch loop to precompute the final Chebyshev state tensor $S \in \mathbb{R}^{N \times (D_{\text{in}} \cdot K)}$ (or $N \times T \times (D_{\text{in}} \cdot K)$ when sequence outputs are requested) along with `teacher_out` predictions.
  - Train only `student.readout` over mini-batches of precomputed $(S, y_{\text{teacher}}, y_{\text{true}})$, verifying zero modification to `student.cell.A` and `student.cell.B` and significantly faster CPU training.
- [ ] **Safeguard — Joint Loss & Automatic Unlabeled Fallback ($\lambda = 1.0$)**:
  - Implement the multi-objective distillation loss:
    $$\mathcal{L} = (1 - \lambda)\,\mathrm{MSE}(y_{\text{student}}, y_{\text{true}}) + \lambda\,\mathrm{MSE}(y_{\text{student}}, y_{\text{teacher}})$$
  - Default $\lambda \in (0, 1)$ (e.g., `alpha_kd=0.5` or `distill_lambda=0.5`) when ground-truth labels `y_true` are present in the dataloader.
  - Automatically fall back to $\lambda = 1.0$ (pure teacher mimicry $\mathrm{MSE}(y_{\text{student}}, y_{\text{teacher}})$) when `dataloader` yields unlabeled inputs (e.g., 1-tuple `(batch_x,)` or `batch_y is None`).
  - Record `total_loss`, `gt_loss`, `kd_loss`, and effective `lambda` in the returned training history dictionary.
- [ ] **Held-Out Engine Split for NASA C-MAPSS FD001**:
  - Update `chebyshev/datasets.py` to support deterministic engine-level (`unit_id`) train/validation splitting (e.g., 80% training engines, 20% held-out validation engines with zero engine overlap).
  - Compute channel normalization statistics (`mean`, `std`) strictly on the **training engines only** to prevent data leakage into the validation split.
- [ ] **CLI & Test Suite Updates**:
  - Update `chebyshev/cli.py` (`cheb distill`, `cheb --smoke-test`) to evaluate and log held-out validation RMSE against actual ground-truth RUL (in both normalized units and RUL cycles).
  - Update `tests/test_distill.py` and `tests/test_cli.py` to verify:
    - Fast state precomputation produces identical readout outputs and leaves $(\bar{A}, \bar{B})$ bit-for-bit frozen.
    - Joint loss reduces validation MSE on held-out batches.
    - Unlabeled dataloader automatically activates $\lambda = 1.0$ fallback without error.
    - Train/val engine IDs in C-MAPSS split are strictly disjoint.
