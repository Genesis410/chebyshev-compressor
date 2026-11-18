# Specification: Mathematical Rigor, Consistent Derivations, and Distillation Validation Revision

Status: ready-for-agent

## Problem Statement

An expert academic and engineering peer review of the repository identified critical discrepancies between theoretical claims, documentation, and the actual implementation:

1. **Mathematical Inconsistency in Operator Definitions**: The transition matrix definition in `README.md` specifies an upper triangular sparse recurrence ($A_{j,k} = -4k$ for $k > j$), whereas the continuous formulation in `THEORY.md` and the implementation in `math_ops.py` construct a dense matrix scaled by basis factors $R_n R_k$. This represents three conflicting formulations within the repository for the primary mathematical operator governing memory.
2. **Unsound Theoretical Claims**:
   - **NPLR Decomposition Violation**: The whitepaper claims an exact Normal Plus Low-Rank (NPLR) structure $A + A^T = -B B^T$, yet numerical verification of the implemented operator reveals $\|A + A^T + B B^T\| \approx 0.13111 \ne 0$. The energy conservation and Lyapunov dissipation proofs in the theory document do not hold for the implemented matrix.
   - **Invalid Asymptotic Algebraic Decay**: The documentation claims asymptotic algebraic decay $O(1/t)$ for the state transition operator $\|e^{At}\|$. For any finite-dimensional strictly stable linear time-invariant (LTI) matrix with $\mathrm{Re}(\lambda_i) < 0$, the operator norm decays exponentially as $O(e^{-\alpha t})$.
3. **Flawed Trajectory Distillation Mechanics**: In `distill.py`, the Chebyshev recurrence matrices $\bar{A}, \bar{B}$ are static non-trainable buffers. The trajectory alignment loss $\mathcal{L}_{\text{state}}$ only backpropagates into a temporary `adapter` projection layer that maps the student state to the teacher hidden dimensions. Because the adapter is discarded during C99 export and never modifies the student state space, the trajectory distillation does not actually teach or modify the student's recurrent memory dynamics. Furthermore, ground-truth targets are ignored during distillation, operating purely as unconstrained teacher imitation without held-out validation.
4. **Imprecise Performance and Embedded Claims**:
   - The claimed 448-byte footprint accounts strictly for the persistent state buffer ($D_{\text{in}} \times K \times 4$), omitting active stack frames, intermediate MLP activations, and input/output buffers.
   - The 185x latency benchmark compares an isolated single-step C call against a single-step PyTorch call rather than a fair, end-to-end streaming window evaluation with equivalent preprocessing and state management.
   - Compiler warning flags (`-Wall -Wextra -Werror -pedantic`) are conflated with certified MISRA-C compliance.

Without resolving these foundational discrepancies, the repository suffers from a credibility gap between its software engineering artifacts and its scientific claims.

---

## Solution

Revise the mathematical foundations, distillation architecture, documentation, and evaluation suite across four structured phases:

1. **Unify Mathematical Derivation**:
   - Standardize on a single, rigorous continuous Chebyshev State-Space formulation derived from first principles (the continuous sliding memory transport PDE projected onto the orthogonal shifted Chebyshev polynomial basis of the first kind).
   - Ensure the continuous operators $A, B$, the discrete operators $\bar{A}, \bar{B}$, the README equations, and the theory whitepaper represent identical mathematics.
   - Replace false NPLR and algebraic decay claims with correct Lyapunov stability theory and finite-window minimax polynomial approximation guarantees.
2. **Restructure Distillation Architecture**:
   - Lock in the architectural design as **Fixed Analytical Orthogonal Memory with Supervised Multi-Objective Readout Distillation**:
     - The Chebyshev recurrent cell dynamics $(\bar{A}, \bar{B})$ remain strictly fixed and analytically optimal from polynomial approximation theory.
     - Eliminate the misleading trajectory alignment adapter.
     - Implement a principled distillation objective combining soft teacher output mimicry with ground-truth target supervision:

       ```math
       \mathcal{L}_{\text{distill}} = (1 - \lambda)\mathcal{L}_{\text{task}}(y_{\text{student}}, y_{\text{true}}) + \lambda \tau^2 \mathcal{L}_{\text{KD}}(y_{\text{student}}, y_{\text{teacher}})
       ```
     - Evaluate the distilled student on a true held-out test partition of C-MAPSS turbofan engines.
3. **Clarify Embedded and Benchmark Claims**:
   - Distinguish **Persistent Recurrent State SRAM** (448 Bytes) from **Total Active SRAM Footprint** (documenting stack and scratch buffers).
   - Standardize benchmark protocols to report end-to-end streaming latency under identical conditions.
   - Scope compliance claims accurately to strict C99 pedantic compiler cleanliness, and validate `-ffast-math` behavior explicitly.
4. **Expand Scientific Validation Tests**:
   - Implement signal reconstruction tests verifying that the Chebyshev state vector accurately reconstructs the historical input signal over the temporal window.
   - Benchmark long-horizon numerical stability across 100, 500, 1,000, and 10,000 timesteps.
   - Run polynomial order ablations ($K \in \{4, 8, 16, 32\}$).

---

## User Stories

1. As a machine learning researcher, I want the continuous transition matrix $A$ in the README, `docs/THEORY.md`, and `math_ops.py` to be identical, so that I can reproduce and mathematically verify the memory operators without contradictions.
2. As a dynamical systems theorist, I want the theory whitepaper to provide a derivation of $A$ and $B$ from the transport PDE $\partial f_t / \partial t = -\partial f_t / \partial \tau$, so that the origin of every matrix coefficient is transparent.
3. As an academic reviewer, I want the whitepaper to accurately state the Lyapunov dissipative properties of the continuous matrix rather than claiming exact NPLR when $\|A + A^T + B B^T\| \ne 0$, so that the proofs remain mathematically sound.
4. As an academic reviewer, I want the documentation to accurately characterize finite-dimensional state transition decay as exponential with a polynomial-like effective window rather than claiming asymptotic $O(1/t)$ decay for a finite matrix, so that the asymptotic claims are mathematically honest.
5. As an embedded ML engineer, I want the distillation pipeline in `distill.py` to train parameters that actually persist into the exported model, so that computational effort spent during training directly improves edge model predictions.
6. As a developer, I want `distill.py` to remove the temporary trajectory adapter if it does not affect exported parameters, so that the codebase is free of unused or misleading optimization paths.
7. As an ML practitioner, I want distillation to incorporate ground-truth target supervision alongside teacher imitation, so that the student model optimizes for actual task accuracy (e.g. Remaining Useful Life) rather than compounding teacher errors.
8. As a predictive maintenance engineer, I want model performance to be evaluated on a separate set of test engine trajectories in NASA C-MAPSS FD001, so that published RMSE figures reflect real generalization.
9. As an embedded firmware engineer, I want documentation to explicitly distinguish between the 448-byte persistent recurrent state and the total runtime stack footprint, so that I can allocate microcontroller memory safely.
10. As a safety-critical software developer, I want the documentation to state "Zero-warning strict C99 compilation" rather than claiming full MISRA-C compliance without formal static analysis tool reports, so that regulatory expectations are respected.
11. As a TinyML developer, I want explicit test cases confirming that input sanity clamping prevents state poisoning and handles NaN/Inf inputs correctly even when compiled with `-ffast-math`, so that compiler optimizations do not break safety.
12. As a researcher, I want unit tests that reconstruct synthetic and real signals from the Chebyshev state coefficients $\sum_{k=0}^{K-1} s_k(t) \phi_k(\tau)$, so that empirical approximation fidelity is verified.
13. As a developer, I want test coverage verifying numerical stability and bounded state norms over 10,000 continuous streaming steps, so that long-running industrial sensors will not suffer from numerical drift or divergence.
14. As an experimentalist, I want an ablation test across polynomial orders $K \in \{4, 8, 16, 32\}$, so that the tradeoff between memory footprint, inference latency, and predictive accuracy is quantified.
15. As a contributor, I want the end-to-end benchmark to report fair comparisons between native C streaming and PyTorch streaming under matched preprocessing and state management, so that performance speedup figures are credible.

---

## Implementation Decisions

### 1. Mathematical Operator Standardization

- **Unified Basis Definition**: Standardize on shifted Chebyshev polynomials of the first kind $T_k^*(\tau) = T_k(2\tau/\theta - 1)$ on the domain $\tau \in [0, \theta]$ with the Chebyshev weight measure $\omega(\tau) = \frac{1}{\sqrt{\tau(\theta - \tau)}}$.
- **Orthonormal Scaling Factors**: Define basis normalization constants $R_0 = \sqrt{1/\pi}$ and $R_k = \sqrt{2/\pi}$ for $k \ge 1$ (or the integer-scaled convention with explicit normalization). The exact choice must be locked in one place and used consistently across documentation and code.
- **Continuous Operators Formulation**:
  - Write down the exact elements of $A \in \mathbb{R}^{K \times K}$ and $B \in \mathbb{R}^{K \times 1}$ in `math_ops.py` and mirror them in `docs/THEORY.md` and `README.md`.
  - Deprecate and remove the sparse upper-triangular formula currently shown in `README.md` if the dense scaled operator is the true discretization basis.
- **Correction of Stability and Decay Theory**:
  - Replace the strict NPLR equation $A + A^T = -BB^T$ with the generalized Lyapunov inequality $A^T P + P A \le 0$ for an appropriate positive-definite metric $P$.
  - State clearly that for any finite dimension $K$, the operator norm satisfies $\|e^{At}\| \le M e^{-\alpha t}$ with $\alpha = \min_i |\mathrm{Re}(\lambda_i(A))| > 0$. Explain that the Minimax polynomial property guarantees optimal approximation across the finite temporal window $[0, \theta]$ without exponential gradient explosion during backpropagation, without mischaracterizing finite-dimensional asymptotic behavior.

### 2. Distillation Architecture Revision

- **Architectural Paradigm**: Select **Fixed Analytical Orthogonal Memory (Option A)**:
  - The recurrent cell operators $(\bar{A}, \bar{B})$ remain fixed analytical constants calculated via double-precision Tustin transformation.
  - The Readout MLP head is the primary trainable component.
- **Fast State Precomputation**:
  - Because $(\bar{A}, \bar{B})$ are deterministic and frozen, precompute the Chebyshev state tensor $S$ (and teacher target predictions) once prior to the training epoch loop, then train only the Readout MLP directly over $S$. This accelerates CPU distillation by $>10\times$ and guarantees zero pole drift during optimization.
- **Distillation Loss Formulation & Unlabeled Fallback**:
  - Remove the unused `adapter` and `loss_state` trajectory alignment mechanism from `distill.py`.
  - Formulate the joint distillation loss as:
    $$\mathcal{L} = (1 - \lambda)\,\mathrm{MSE}(y_{\text{student}}, y_{\text{true}}) + \lambda\,\mathrm{MSE}(y_{\text{student}}, y_{\text{teacher}})$$
  - Default $\lambda \in (0, 1)$ (e.g., `0.5`) when ground-truth `y_true` is available; automatically fall back to $\lambda = 1.0$ (pure teacher mimicry) when distilling on unlabeled calibration datasets.
- **Validation Protocol**:
  - Split dataset by unit/engine ID (e.g., engines 1–80 for training, 81–100 for validation), fitting channel normalization statistics strictly on the training split.
  - Report RMSE against actual ground truth on the held-out validation split.

### 3. Documentation & Embedded Claims Alignment

- **IEEE-754 Bit-Level Sanitization (`-ffast-math` / `-ffinite-math-only` Safeguard)**:
  - Under `-ffast-math`, compilers enable `-ffinite-math-only` and may optimize away floating-point range checks for `NaN`/`Inf` as dead code.
  - Implement `chebyshev_sanitize(float x)` using strict-aliasing-safe C99 `memcpy` into `uint32_t` and checking the IEEE-754 exponent bits `(bits & 0x7F800000U) == 0x7F800000U`, returning `0.0f` deterministically for `NaN` and `±Inf` before clamping to `[-CHEB_INPUT_CLAMP, CHEB_INPUT_CLAMP]`.
  - Mirror identical non-finite -> `0.0` mapping in PyTorch `DepthwiseChebyshevCell` to maintain exact C/PyTorch parity.
- **Memory Terminology**:
  - Replace generic "448 Bytes RAM" badges and statements with:
    - **Persistent Recurrent State**: 448 Bytes (Static BSS for 14 channels $\times$ Order 8).
    - **Active Execution Stack**: ~256 Bytes (Local arrays for normalization, recurrence scratch buffer, and MLP hidden activations).
    - **Total Runtime SRAM Budget**: < 1 KB.
- **Standards Claims**:
  - Update "MISRA-C Compliance" references to "Strict C99 Cleanliness (`-Wall -Wextra -Werror -pedantic -std=c99` with zero warnings)".
- **Benchmark Claims**:
  - Clarify benchmark methodology: report both matched sequence-window streaming comparisons and single-step microbenchmarks, explicitly noting Python dispatch overhead vs native C compute.

---

## Testing Decisions

### What Makes a Good Test
- Tests must verify external mathematical invariants and observable system behavior, not private internal helper structures.
- All tests must be deterministic, portable across OS platforms, and execute without requiring external network access.

### Module Test Coverage
1. **Operator Correctness & Reconstruction (`test_math.py`)**:
   - Verify that the continuous operator $A$ has all eigenvalues in the open left-half complex plane: $\max \mathrm{Re}(\lambda_i(A)) < 0$.
   - Verify that the discrete operator $\bar{A}$ has spectral radius bounded strictly within the unit circle: $\rho(\bar{A}) < 1.0$.
   - **Signal Reconstruction Test**: Project a smooth, low-frequency test signal (with frequency strictly below the window Nyquist bound, $\omega \ll K/\theta$) into the Chebyshev state space and verify that polynomial reconstruction $\hat{u}(\tau) = \sum_{k=0}^{K-1} c_k s_k T_k^*(\tau)$ using the exact orthogonal Chebyshev normalization constants matching the canonical $B$ operator ($c_0, c_k$) achieves deterministic bounded $L_2$ reconstruction error that strictly decreases as order $K$ increases.
2. **Distillation Fidelity & Generalization (`test_distill.py`)**:
   - Verify that distillation training without adapter optimizes student readout parameters and decreases task loss on unseen validation batches.
   - Verify that joint loss correctly balances ground-truth supervision and teacher emulation.
3. **Long-Horizon Stability (`test_model.py`)**:
   - Stream 10,000 sequential random inputs through `DepthwiseChebyshevCell` and assert state norms remain finite and within bounded theoretical limits without overflow or NaN accumulation.
4. **NaN Quarantine & Fast-Math Safety (`test_export_c.py`)**:
   - Compile the generated C header with `-O3 -ffast-math` and verify that passing `NaN` or `Inf` sensor values is safely quarantined by `chebyshev_sanitize` without state poisoning.
5. **C99 Exporter & Parity (`test_export_c.py`)**:
   - Maintain the $\max |y_{\text{PyTorch}} - y_{\text{C}}| < 10^{-4}$ numerical parity test across 1,000 steps.

---

## Out of Scope

- Introducing fully learnable continuous recurrence matrices (Option B) that would require complex Riemannian manifold optimization or spectral regularization to guarantee stability.
- Replacing the C99 header-only design with a dynamic runtime library.
- Deploying to physical hardware in this sprint (remains documented as a hardware target; formal automated testing uses GCC/Clang on host runners).
- Formal MISRA-C commercial certification toolchains (e.g. Polyspace, PC-lint).

---

## Further Notes

- The reviewer's feedback represents a positive endorsement of the engineering architecture and TinyML vision, while providing a clear roadmap to elevate the project to publication-grade scientific rigor.
- Resolving the mathematical inconsistencies and distillation mechanics will establish a solid foundation for empirical papers and real-world industrial deployments.
