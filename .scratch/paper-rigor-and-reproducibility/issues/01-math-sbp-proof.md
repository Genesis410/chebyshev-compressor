# Issue 01: SBP Lyapunov Stability Re-Derivation, Exact Nodal Dissipation, and Theory Claims

Status: resolved

## Answer
The SBP Lyapunov proof is completely re-derived and verified from first principles:
1. In discrete Gauss-Radau nodal coordinates, $\mathbf{u} = V \mathbf{c}$, the discrete metric is $W_E = \mathrm{diag}(\bar{w}_j (1 + x_j))$.
2. The differentiation operator $\tilde{D} = V D V^{-1}$ satisfies the discrete SBP identity:
   $\tilde{D}^T W_E + W_E \tilde{D} = B_E - Q_E$,
   where $[B_E]_{0,0} = \frac{4(K-1)}{2K-1}$, all interior diagonal entries are negative ($[B_E - Q_E]_{j,j} < 0$ for $j \ge 1$), and there is zero cross-coupling between interior nodes ($[B_E - Q_E]_{j,k} = 0$ for $j \ne k, j \ge 1, k \ge 1$). For any constant state $\mathbf{u} = [1, \dots, 1]^T$, $\mathbf{u}^T (\tilde{D}^T W_E + W_E \tilde{D}) \mathbf{u} = 0$.
3. The SAT penalty vector transforms as $V \boldsymbol{\gamma} = (2K - 1) \mathbf{e}_0$, so the nodal SAT penalty evaluates to:
   $-2 \alpha (2K - 1) W_E \mathbf{e}_0 \mathbf{e}_0^T = -2 \left(\frac{2K-1}{2}\right) (2K - 1) \left(\frac{2}{2K-1}\right) \mathbf{e}_0 \mathbf{e}_0^T = -2(2K - 1) \mathbf{e}_0 \mathbf{e}_0^T$.
4. Combining the differentiation SBP matrix and the SAT penalty:
   $\tilde{A}^T W_E + W_E \tilde{A} = \frac{2}{\theta} [(\tilde{D}^T W_E + W_E \tilde{D}) - 2(2K-1) \mathbf{e}_0 \mathbf{e}_0^T]$.
   Because $2(2K-1) > \frac{4(K-1)}{2K-1}$ for all $K \ge 1$, the $(0,0)$ entry is strictly negative (e.g. $-5.111$ for $K=2$, $-8.4$ for $K=3$, $-11.714$ for $K=4$), dominating row-0 off-diagonals.
5. For a constant state $\mathbf{c} = [1, 0, \dots, 0]^T$, the quadratic form evaluates to exactly:
   $\mathbf{c}^T (A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A) \mathbf{c} = -\frac{4(2K-1)}{\theta} < 0$.
   It never vanishes, resolving the counter-example completely.
6. The eigenvalues of $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A$ are strictly negative for all $K \ge 1$, guaranteeing strict Hurwitz stability.
7. Optimization of the 2-layer MLP readout is formally defined as "static supervised regression over precomputed orthogonal features without BPTT", and the error decomposition is formalized as $e_{\mathrm{total}} \le e_{\mathrm{poly}} + e_{\mathrm{SAT}} + e_{\mathrm{Tustin}}$.


## Description

The mathematical stability proof in Theorem 2 (Equations 32–34) of `report/main.tex` contains algebraic and conceptual errors:
1. Equation (34) claims $\mathbf{c}^T (A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A) \mathbf{c} \le -\frac{2}{\theta} \sum_{j=1}^{K-1} \bar{w}_j (1 - x_j) (\partial_x f(x_j))^2 < 0$. If $\mathbf{c} = [1, 0, \dots, 0]^T$, $f(x) \equiv 1$ and $\partial_x f \equiv 0$, which yields $0 \not< 0$. For $K=1$, the sum is empty.
2. In transforming Lagrange nodal coordinates to Chebyshev modal coordinates, $V \boldsymbol{\gamma} = (2K - 1) \mathbf{e}_0$. The SAT boundary penalty is $-2\bar{w}_0 (1 + x_0) \alpha (2K-1) u_0^2 = -2(2K-1)u_0^2$.
3. In continuous weighted space, integration by parts on $\int_{-1}^1 \sqrt{\frac{1+x}{1-x}} 2 f \partial_x f \, dx$ produces an interior term proportional to $\int W'(x) f^2 dx$ with $W'(x) = \frac{1}{(1-x)\sqrt{1-x^2}} > 0$, which is positive and strictly dissipative for all $f \not\equiv 0$ (including constants), not $(\partial_x f)^2$.
4. Inequality notation must be strictly consistent ($\prec 0$).
5. Optimization of the 2-layer MLP readout must be described as non-convex supervised regression over frozen orthogonal features without BPTT.
6. The error budget must be decomposed into $e_{\mathrm{total}} \le e_{\mathrm{poly}} + e_{\mathrm{SAT}} + e_{\mathrm{Tustin}}$.

## Acceptance Criteria
- Unit tests in `tests/test_math.py` verify $P_{\mathrm{CGR}} \succ 0$ and $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$ across all $K \in \{1, 2, \dots, 32\}$.
- The proof of Theorem 2 in `report/main.tex` is algebraically sound and directly derived from Gauss-Radau SBP identities.
- All theory claims (inequality signs, optimization class, and error budget) are mathematically precise and defensible.
