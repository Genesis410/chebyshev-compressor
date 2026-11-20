# Specification: Comprehensive Technical Report Audit, Mathematical Rigor, and Camera-Ready 8-Page Publication

Status: completed

## Problem Statement

The LaTeX manuscript (`report/main.tex`) for *Chebyshev State-Space Compression (DSSR)* underwent an exhaustive peer review audit. While the structural foundation is solid, critical defects remain across four major areas that would prevent publication or acceptance by high-tier conferences (EMSOFT, RTSS, MLSys, NeurIPS) and grant bodies:

1. **LaTeX Compilation Bugs & Code Flaws (P0)**:
   - Undefined control sequence `\Comment` in Algorithm 3 due to missing algorithmic packages.
   - Mismatched column definition in Table 3 (`{llrrrr}` for 5 data columns).
   - Internal brainstorming remnants (`\label{alg:option_a_distill}`).
   - Missing `\usepackage[T1]{fontenc}`, causing corrupted accented glyphs (e.g., Ré, Kajić) and hyphenation glitches.
   - Raw Unicode em-dashes (`—`) instead of LaTeX standard `---`.

2. **Mathematical Proof & Notation Flaws (P0)**:
   - **Incorrect Boundary Matrix $B_E$**: Written as $\frac{4(K-1)}{2K-1}$ instead of the exact derivative of the boundary Lagrange interpolant $\ell_0'(1) = \frac{K(K-1)}{3}$, which yields $[\tilde{D}^T W_E + W_E \tilde{D}]_{0,0} = \frac{4K(K-1)}{3(2K-1)}$ ($8/9$ for $K=2$).
   - **Algebraic Entry Error in $[\tilde{A}^T W_E + W_E \tilde{A}]_{0,0}$**: The formula $-\frac{2}{\theta}\frac{4K^2-8K+6}{2K-1}$ evaluates to $-4/\theta$ for $K=2$, contradicting the text's own $-5.111/\theta$. The exact analytical formula is $-\frac{4}{\theta}\frac{10K^2 - 10K + 3}{3(2K-1)}$, yielding true values after outer factor multiplication: $-4/\theta$ ($K=1$), $-10.222/\theta$ ($K=2$), $-16.800/\theta$ ($K=3$), and $-23.429/\theta$ ($K=4$).
   - **Missing Arrowhead Schur Complement Step**: Concluding $M = \tilde{A}^T W_E + W_E \tilde{A} \prec 0$ solely from negative diagonal entries and zero interior cross-coupling is invalid because boundary-to-interior coupling $\mathbf{b} \ne \mathbf{0}$. The proof requires the strict Schur complement condition $a + \mathbf{b}^T D_{\mathrm{int}}^{-1} \mathbf{b} < 0$.
   - **Notation Clashes & Dimension Errors**: Symbol collision of scalar input $c$ with modal vector $\mathbf{c}$ in Proposition 1; matrix dimension mismatch in Section 5.2 ($W_1 S$ on row-major batches instead of column vector $\mathbf{s} \in \mathbb{R}^{D_{\mathrm{in}}K}$); unproven claim $e_{\mathrm{SAT}} = O(1/K)$ asserted as "rigorous"; overnumbered intermediate equations.

3. **Experimental Contradictions & Data Integrity (P0)**:
   - **Catastrophic Distillation Contradiction ($\lambda=0.0$ at 3.81 cycles)**: Impossible performance on C-MAPSS FD001 (SOTA is ~11-12 cycles), destroying the motivation for distillation and revealing residual data leakage or latency copy-paste. Teacher mimicry ($\lambda=1.0$ at 41.81 cycles) reveals an unconverged teacher.
   - **STM32 Latency Contradiction**: Abstract states 12.4--15.8 $\mu$s, while Table 3 lists 12.40 $\mu$s (3-ch) and 68.50 $\mu$s (14-ch).
   - **Active SRAM Miscalculation in Table 4**: $K=4$ was reported as 304 B instead of $60(4)+128 = 368\text{ B}$; $K=32$ was reported as 2,176 B instead of $60(32)+128 = 2{,}048\text{ B}$.
   - **Transparent Stack Footprint Reporting**: Distinction required between Algorithmic SRAM (608 B) and Compiler-Verified Peak Frame (744 B on Cortex-M4F).
   - **Legendre Outperforming Chebyshev Without $L_\infty$ Metric**: Table 4 reports only $L_2$ error (where Legendre naturally excels), omitting $L_\infty$ supremum error which proves Chebyshev's near-minimax theoretical superiority.
   - **Unsubstantiated 3-Axis Vibration**: Lacks accuracy metrics and synthetic labeling.
   - **Apples-to-Oranges PyTorch vs C99**: Needs a Bare-Metal C99 LSTM baseline.

4. **Visual Layout, TikZ, & Micro-Polishing (P1 & P2)**:
   - TikZ architecture diagram mixes dataflow with memory metrics, lacks recurrent feedback loops ($z^{-1}$), has non-uniform node heights, and belongs at the top of Page 2.
   - Abused `\resizebox` across tables shrinks fonts to illegible ~5.5 pt.
   - Computer Modern font looks like a draft; needs `newtxtext,newtxmath`.
   - Abstract is overly verbose (~285 words); needs compression to 170--200 words.
   - Document must fit into exactly 8 pages with balanced columns.

---

## Solution

Revise the complete technical manuscript, mathematical proofs, experimental codebase, and visual assets:

1. **LaTeX Code Hardening**: Load `\usepackage[T1]{fontenc}`, `\usepackage{newtxtext,newtxmath}`, `\usepackage{cite}`, and `\usepackage{booktabs}`. Remove undefined macros (`\Comment`), rename all legacy labels to `dssr`, and replace Unicode characters with LaTeX conventions.
2. **Exact SBP Lyapunov Derivation & Schur Complement**: Derive the exact Gauss-Radau boundary flux $[\tilde{D}^T W_E + W_E \tilde{D}]_{0,0} = \frac{4K(K-1)}{3(2K-1)}$, the exact analytical formula $[\tilde{A}^T W_E + W_E \tilde{A}]_{0,0} = -\frac{4}{\theta}\frac{10K^2 - 10K + 3}{3(2K-1)}$, and the arrowhead Schur complement condition $a + \mathbf{b}^T D_{\mathrm{int}}^{-1} \mathbf{b} < 0$. Verify across all orders $K=1 \dots 32$ in automated unit tests.
3. **Rigorous Re-Benchmarking & Data Cleanliness**:
   - Re-run distillation ablations with the fully converged 14.82 cycles teacher on official test splits with 5 random seeds.
   - Measure $L_\infty$ and $L_2$ reconstruction errors for Chebyshev and Legendre, proving Chebyshev minimizes worst-case peak error.
   - Implement a native Bare-Metal C99 LSTM baseline to provide a fair C99-to-C99 architectural comparison.
   - Correct all SRAM formulas in Table 4 (368 B, 608 B, 1,088 B, 2,048 B) and align Abstract latency numbers with Table 3.
   - Label 3-axis vibration as synthetic and provide concrete anomaly AUROC/MAE metrics.
4. **Professional TikZ Vector Architecture**: Refactor Figure 1 using relative positioning, uniform node heights, recurrent feedback loops ($z^{-1}$), clean dashed memory enclosures for TFLM arena and DSSR SRAM, and anchor at top of Page 2.
5. **Camera-Ready 8-Page Layout**: Span wide tables across two columns (`table*`) at `\footnotesize` without font distortion. Format algorithms with clean rules. Balance columns on Page 8.

---

## User Stories

1. As an academic reviewer, I want mathematically exact SBP Lyapunov proofs with closed-form algebraic expressions and rigorous Schur complement conditions, so that I can verify continuous stability without encountering counter-examples.
2. As an embedded systems engineer, I want transparent SRAM reporting distinguishing algorithmic memory from compiler-verified stack frames, so that I can budget memory deterministically on bare-metal microcontrollers.
3. As a machine learning researcher, I want empirical ablations showing both $L_2$ and $L_\infty$ metrics across multiple seeds on official test sets, so that Chebyshev's minimax advantage over Legendre is experimentally substantiated.
4. As a TinyML practitioner, I want a direct C99-to-C99 comparison between Chebyshev DSSR and a native Bare-Metal LSTM, so that language-overhead distortions (Python vs C) are eliminated.
5. As an open-source contributor, I want clean, reproducible scripts that re-generate every single table and figure from scratch, so that research integrity is guaranteed.
6. As a reader of the PDF, I want crisp Times typography, properly aligned TikZ vector diagrams, and readable 8pt tables without resizing artifacts, so that the document meets premier IEEE/ACM publication standards.
7. As an archivist, I want clean bibliographic entries for ISO 26262 and IEEE 754 with complete publisher and standard metadata, so that citations are fully resolvable.

---

## Implementation Decisions

- **Preconditioning & Preamble**: Use `\usepackage[T1]{fontenc}`, `\usepackage{newtxtext,newtxmath}`, `\usepackage{cite}`. Set margins to `margin=0.72in, top=0.75in, bottom=0.75in` with `\setlength{\columnsep}{16pt}`.
- **Mathematical SBP Formulation**:
  - Boundary flux: $[\tilde{D}^T W_E + W_E \tilde{D}]_{0,0} = \frac{4K(K-1)}{3(2K-1)}$.
  - Boundary diagonal entry: $[\tilde{A}^T W_E + W_E \tilde{A}]_{0,0} = -\frac{4}{\theta}\frac{10K^2 - 10K + 3}{3(2K-1)}$.
  - Evaluated numerical examples: $-4/\theta$ ($K=1$), $-10.222/\theta$ ($K=2$), $-16.800/\theta$ ($K=3$), $-23.429/\theta$ ($K=4$).
  - Negative definiteness of arrowhead block matrix verified via strict Schur complement: $a + \mathbf{b}^T D_{\mathrm{int}}^{-1} \mathbf{b} < 0$.
- **Experimental Protocol**:
  - Official NASA C-MAPSS FD001 test split (`test_FD001.txt` + `RUL_FD001.txt`).
  - Reference Teacher: 2-layer LSTM ($H=64$, 14.82 cycles RMSE).
  - Student Model: Chebyshev DSSR ($K=8, H=32$).
  - Loss ablation: report genuine test RMSE across $\lambda \in \{0.0, 0.5, 1.0\}$ over 5 seeds.
  - Reconstruction error: report both $L_2$ (mean squared) and $L_\infty$ (worst-case supremum norm) for Chebyshev and Legendre.
  - Memory formulas: $M_{\mathrm{active}} = 60K + 128$ bytes ($D_{\mathrm{in}}=14, H=32$).
  - Baseline C99 LSTM: implement standalone C99 benchmark harness measuring latency of identical 2-layer LSTM.
- **TikZ Vector Graphic**:
  - Figure 1 placed at the start of Section 2.
  - Unit delay feedback loops ($z^{-1}$) over hidden states.
  - Background dashed boxes enclosing memory bounds.
  - Standardized node heights (`minimum height=1.1cm`).
- **Page Budget & Flow**:
  - Target: exactly 8 pages.
  - Use `table*` for Table 1, Table 3, Table 4. Single-column for Table 2.

---

## Testing Decisions

- **Mathematical Seam (`tests/test_math.py`)**:
  - Assert exact boundary entry values for $K \in \{1, 2, 3, 4, 8, 16, 32\}$.
  - Assert negative definiteness of the Schur complement $a + \mathbf{b}^T D_{\mathrm{int}}^{-1} \mathbf{b} < 0$.
  - Assert strictly Hurwitz eigenvalues and Lyapunov dissipation for all $K \in \{1, \dots, 32\}$.
- **Native C99 Benchmark Seam (`scripts/measure_c99_lstm_baseline.py`)**:
  - Measure execution latency of Chebyshev DSSR vs Bare-Metal C99 LSTM compiled with `-O3 -march=native` over 100,000 iterations.
- **Ablation & Test Set Seam (`scripts/run_scientific_benchmarks.py`)**:
  - Execute genuine 5-seed evaluation on official C-MAPSS test set.
  - Verify $L_\infty$ error containment for Chebyshev over Legendre.
- **Document Quality & Compilation Seam**:
  - Automated `pdflatex` compilation with zero undefined control sequence warnings.
  - Verification of exact 8-page count with `pypdf`.
  - Visual inspection of all 8 rendered page images.

---

## Out of Scope

- Implementing full Q7/Q15 integer fixed-point kernel (deferred to future work).
- Multi-rate asynchronous kalman filter state-space fusion (deferred to future work).
- Submitting the DOI to Zenodo (user will trigger once final camera-ready PDF is approved).

---

## Further Notes

- Author identity: Ihsan Akmal (`ihsanakmal151@gmail.com`), Independent Researcher.
- Repository: Genesis410/chebyshev-compressor.
