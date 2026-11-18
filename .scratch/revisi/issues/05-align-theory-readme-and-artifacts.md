# 05: Align Theory Whitepaper, README, ADRs, and Regenerate Verified Artifacts

Status: resolved
Priority: P0 / P2 (Documentation & Reproducibility Sync)
Testing Seam: End-to-End Pipeline & Documentation Integrity (`docs/THEORY.md`, `README.md`, `CONTRIBUTING.md`, `GLOSSARY.md`, `docs/adr/`, `cheb_out/`)
Blocked by: 01, 02, 03, 04

## What to build
Synchronize all mathematical equations, stability proofs, distillation descriptions, embedded memory tables, and benchmark numbers across `docs/THEORY.md`, `README.md`, `CONTRIBUTING.md`, and relevant ADRs so they match the revised code 100%, and regenerate all deterministic `cheb_out/` experimental artifacts and manifest hashes.

## Acceptance Criteria

- [ ] **Unify Core Equations Across `README.md`, `docs/THEORY.md`, and `chebyshev/math_ops.py`**:
  - Remove the conflicting sparse upper-triangular formula ($A_{j,k} = -4k$) from `README.md`.
  - Display the exact single canonical continuous $(A, B)$ derivation and Tustin discretization $(\bar{A}, \bar{B})$ consistently in both `README.md` and `docs/THEORY.md`, matching `chebyshev/math_ops.py` line for line.
- [ ] **Correct Theoretical Proofs in `docs/THEORY.md`**:
  - Remove the invalid strict NPLR equality claim ($A + A^T = -BB^T$ with $S^T = -S$) in Section 3.1; replace with a sound Lyapunov stability and strict Hurwitz eigenvalue analysis ($\mathrm{Re}(\lambda_i(A)) < 0 \implies |\lambda_i(\bar{A})| < 1$).
  - Replace the invalid finite-dimensional asymptotic $O(1/t)$ decay claim for $\|e^{At}\|$ with an accurate explanation: for finite order $K$, $\|e^{At}\| = O(e^{-\alpha t})$ guarantees BIBO stability, while the shifted Chebyshev Minimax property delivers near-optimal uniform approximation across the sliding window $[0, \theta]$ without non-linear recurrent gradient vanishing/explosion.
  - Ensure all display equations use GitHub KaTeX-compatible multi-line `$$` blocks with blank lines.
- [ ] **Honest Distillation, SRAM, and Standards Documentation (`README.md`, `CONTRIBUTING.md`, `GLOSSARY.md`, ADRs)**:
  - Document **Option A (Fixed Analytical Orthogonal Memory + Supervised Multi-Objective Readout Distillation with Fast State Precomputation)** accurately—removing any claim that trajectory alignment trains $(A, B)$.
  - Distinguish **Persistent Recurrent State (448 Bytes)** from **Active Stack Scratch** and **Total Runtime SRAM Budget (< 1 KB)** across badges, ASCII diagrams, and benchmark tables.
  - Replace "MISRA-C compliance" claims with "Strict C99 zero-warning compliance (`-std=c99 -Wall -Wextra -Werror -pedantic`)" and note `-ffast-math` IEEE-754 bit-level quarantine verification.
  - Update ADR-0007 (input sanity clamping) and ADR-0001/0005 if needed to reflect the `memcpy` exponent-bit check and Option A formulation.
- [ ] **Regenerate Verified Artifacts & Full Verification**:
  - Run `python -m chebyshev.cli --smoke-test --seed 42` to regenerate `cheb_out/teacher.pt`, `cheb_out/student.pt`, `cheb_out/calib.npy`, `cheb_out/chebyshev_engine.h`, `cheb_out/benchmark_results.json`, and `cheb_out/reproducibility_manifest.json`.
  - Populate `README.md` benchmark and ablation tables strictly with newly measured, reproducible numbers (including held-out validation RMSE and end-to-end sequence streaming latency).
  - Run full `pytest tests/ -v` and verify 100% pass rate with zero warnings.
