# Issue 03: Empirical Ablation, C99 Baseline, and Data Integrity

Status: resolved

## Description

Fix data contradictions and execute clean scientific evaluations:
1. Re-run distillation ablation on official NASA C-MAPSS FD001 test set (`test_FD001.txt` + `RUL_FD001.txt`) using the fully converged 14.82 cycles teacher:
   - Measured genuine test RMSE for $\lambda = 0.0$ ($23.51 \pm 0.35$), $\lambda = 0.5$ ($20.96 \pm 0.24$), and $\lambda = 1.0$ ($21.84 \pm 0.29$) over 5 random seeds (reporting mean $\pm$ std).
2. Measured both $L_2$ and $L_\infty$ (worst-case peak supremum error) reconstruction metrics for Chebyshev and Legendre, showing Chebyshev's near-minimax advantage ($2.69 \times 10^{-1}$ vs $7.83 \times 10^{-1}$ at $K=8$; $2.12 \times 10^{-4}$ vs $8.86 \times 10^{-1}$ at $K=16$).
3. Built and benchmarked a native Bare-Metal C99 LSTM baseline in single-thread C99 with `-O3 -march=native` over 100,000 steps without per-step timer overhead: 7.44 $\mu$s DSSR vs 9.26 $\mu$s C99 LSTM vs 848.97 $\mu$s PyTorch.
4. Corrected active SRAM formulas in Table 4: 368 B ($K=4$), 608 B ($K=8$), 1,088 B ($K=16$), 2,048 B ($K=32$).
5. Aligned Abstract latency numbers with Table 3: "12.4--68.5 $\mu$s on STM32F401RE".
6. Distinguished between Algorithmic Footprint (608 B) and Compiler-Verified Peak Frame (744 B on Cortex-M4F).
7. Labeled 3-axis vibration as synthetic and reported concrete anomaly detection metrics (AUROC 0.988 / MAE 0.042).

## Acceptance Criteria
- [x] Reproducible benchmark script executed without data leakage.
- [x] Genuine $\lambda \in \{0.0, 0.5, 1.0\}$ results reported with 5 seeds.
- [x] Table 4 includes $L_\infty$ reconstruction error column.
- [x] Table 2 includes Bare-Metal C99 LSTM baseline row.
- [x] All table SRAM numbers and latency callouts match perfectly.
