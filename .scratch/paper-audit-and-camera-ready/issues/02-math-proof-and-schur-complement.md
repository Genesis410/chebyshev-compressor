# Issue 02: Mathematical Proof Rigor, Boundary Flux, and Arrowhead Schur Complement

Status: resolved

## Description

Address four critical mathematical flaws in Theorem 2 and notation issues:
1. Exact boundary matrix flux: $[\tilde{D}^T W_E + W_E \tilde{D}]_{0,0} = \frac{4K(K-1)}{3(2K-1)}$, derived from Lagrange boundary interpolant derivative $\ell_0'(1) = \frac{K(K-1)}{3}$.
2. Exact analytical formula for boundary diagonal entry:
   $$[\tilde{A}^T W_E + W_E \tilde{A}]_{0,0} = -\frac{4}{\theta}\frac{10K^2 - 10K + 3}{3(2K-1)}$$
   Update concrete examples after multiplying by outer factor $2/\theta$: $-4/\theta$ ($K=1$), $-10.222/\theta$ ($K=2$), $-16.800/\theta$ ($K=3$), $-23.429/\theta$ ($K=4$).
3. Arrowhead block Schur complement step: Matrix $M = \begin{pmatrix} a & \mathbf{b}^T \\ \mathbf{b} & M_{\mathrm{int}} \end{pmatrix}$ with $M_{\mathrm{int}} \prec 0$ is strictly negative definite if and only if the Schur complement $S = a - \mathbf{b}^T M_{\mathrm{int}}^{-1} \mathbf{b} = -\frac{2}{\theta}(3K - 1) < 0$. Formulated and verified across orders in `tests/test_math.py`.
4. Qualitative error decomposition: rephrased $e_{\mathrm{SAT}} = O(1/K)$ as qualitative decomposition with appropriate citations (Carpenter et al. 1994, Hesthaven et al. 2007).
5. Disambiguated scalar input $u_0 \in \mathbb{R}$ from modal vector $\mathbf{c}$ in Proposition 1.
6. Corrected matrix dimensions in Section 5.2 to vector form $\hat{y} = \mathbf{w}_2^T \mathrm{ReLU}(W_1 \mathbf{s} + \mathbf{b}_1) + b_2$ for $\mathbf{s} \in \mathbb{R}^{D_{\mathrm{in}}K}$.
7. Removed equation numbering from non-referenced intermediate steps using `\begin{equation*}` or `\nonumber`.

## Acceptance Criteria
- [x] Analytical boundary entry verified symbolically and numerically in `tests/test_math.py`.
- [x] Schur complement condition proven and tested for $K=1 \dots 32$.
- [x] No symbol collisions or dimension mismatches.
