# Specification: Academic Rigor, Mathematical Proof Revision, and Reproducible Re-Benchmarking

Status: completed

## Problem Statement

The LaTeX technical report (`report/main.tex` / `main_2.pdf`) documents the Chebyshev-Gauss-Radau (CGR) state-space architecture, Decoupled State-Space Readout (DSSR), and zero-allocation C99 inference engine. However, an academic and technical review identified foundational flaws across four areas that would result in immediate rejection by peer-reviewed conferences (EMSOFT, RTSS, MLSys, NeurIPS) or independent research grant committees (Emergent Ventures, NLnet, TinyML Foundation):

1. **Mathematical Invalidation in Stability and Error Bounds**:
   - **Algebraic Flaw in Theorem 2 (Equations 32–34)**: The proof claims that for all non-zero modal trajectories $\mathbf{c} \ne \mathbf{0}$, $\mathbf{c}^T (A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A) \mathbf{c} \le -\frac{2}{\theta}\sum_{j=1}^{K-1} \bar{w}_j (1 - x_j) (\partial_x f(x_j))^2 < 0$. For a constant modal state $\mathbf{c} = [1, 0, \dots, 0]^T$, the function is $f(x) \equiv 1$ and its derivative is identically zero ($\partial_x f \equiv 0$), collapsing the summation to $0$, which disproves the $< 0$ strict bound. For $K=1$, the summation index is empty ($= 0$).
   - **Missing Factor $(2K-1)$ in Boundary Penalty**: In transforming the Lagrange nodal basis to Chebyshev modal basis, $V \boldsymbol{\gamma} = (2K-1) \mathbf{e}_0$. Consequently, the boundary SAT penalty on $W_E \tilde{A} + \tilde{A}^T W_E$ evaluates to $-2\bar{w}_0(1+x_0)\alpha(2K-1)u_0^2 = -2(2K-1)u_0^2$, rather than $-4\alpha\bar{w}_0 u_0^2 = -2u_0^2$.
   - **Interior Integration by Parts Discrepancy**: Continuous integration by parts of $\int_{-1}^1 \sqrt{\frac{1+x}{1-x}} 2 f \partial_x f \, dx$ produces an interior term proportional to $\int W'(x) f(x)^2 dx$ where $W'(x) = \frac{1}{(1-x)\sqrt{1-x^2}} > 0$, rather than $(\partial_x f)^2$. The proof must be derived rigorously from exact Gauss-Radau Summation-By-Parts (SBP) discrete operators.
   - **Inequality Contradiction**: The Abstract states negative semi-definiteness ($A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \le 0$), while the body claims strict negative definiteness ($\prec 0$).
   - **Mischaracterization of Non-Convex Optimization**: Section 5.2 claims that training a two-layer MLP with ReLU activations converts distillation into a "simple convex regression task", which is mathematically false for multi-layer neural networks.
   - **Overclaim of Minimax Error Bounds ($L_\infty$)**: Section 2.3 equates the static Chebyshev projection Lebesgue constant bound ($\Lambda_8 \approx 1.84$) to the dynamic output of the CGR-SAT ODE after Tustin bilinear discretization, omitting boundary SAT penalty residuals and frequency warping errors.

2. **Experimental Integrity and Methodology Red Flags**:
   - **Suspicious Latency Numbers**: Table 2 lists single-step latency as $0.32\,\mu\text{s}$ and window latency ($T=50$) as exactly $16.00\,\mu\text{s}$ ($0.32 \times 50$), lacking realistic cache, loop, or timer variance.
   - **Ambiguous and Mismatched Hardware Measurements in Table 4**: Column "MCU Step Latency" omits the target MCU identity and mixes host CPU timings ($K=4$ @ 2.97 $\mu$s, $K=32$ @ 38.83 $\mu$s), ESP32 timings ($K=8$ @ 5.20 $\mu$s), and STM32 timings ($K=16$ @ 15.80 $\mu$s).
   - **ROM Size Calculation Inconsistencies**: Table 3 lists 3-channel vibration model parameters as $1.98\text{ KB}$ (matching $495 \times 4\text{ B}$), but lists 14-channel C-MAPSS parameters as $17.82\text{ KB}$ when $3,749 \times 4\text{ B} = 14,996\text{ B} = 15.00\text{ KB}$ ($14.64\text{ KiB}$). Furthermore, identical binary ROM sizes across Xtensa and ARM Thumb-2 ISAs are physically impossible without clarifying that the metric represents parameter `.rodata`.
   - **Unsubstantiated "3-Axis Vibration" Claims**: The Abstract and Table 3 cite a 3-axis vibration dataset, but Section 7 never explains the dataset provenance, task formulation, or accuracy metrics.
   - **Apples-to-Oranges Baselines**: Comparing hand-optimized C99 against Python PyTorch and generic TensorFlow Lite Micro arenas conflates language/framework overhead with architectural efficiency. A native bare-metal C99 recurrent baseline (or direct TFLM DSSR evaluation) is missing.
   - **Missing Chebyshev vs. Legendre Empirical Validation**: The theoretical superiority of Chebyshev ($L_\infty$) over Legendre ($L^2$) lacks empirical verification within the same C99 engine.
   - **Missing Distillation Ablation ($\lambda$)**: The report introduces loss weighting $\lambda \in [0, 1]$ in Algorithm 2, but never reports the value used or the ablation $\lambda = 0.0$ (direct supervised training without teacher).
   - **Non-Standard C-MAPSS Evaluation Split & False Citation**: C-MAPSS FD001 was evaluated on training units rather than the official 100 test trajectories (`test_FD001.txt` / `RUL_FD001.txt`), and footnote 3 in Table 4 falsely attaches citation [3] (Zheng et al., 2017) to a 14.82 cycles RMSE figure.

3. **Embedded C99 Technical and Safety Vulnerabilities**:
   - **Understated SRAM Footprint**: Claiming "608 Bytes total SRAM" accounts only for `chebyshev_state_t` (448 B) and local arrays (160 B), ignoring compiler stack frames, callee-saved registers, and caller input buffers.
   - **Overflow Vulnerability in `chebyshev_sanitize`**: The function checks for NaN/Inf before normalization. An extreme finite float input (e.g. $10^{38}$) can overflow during `(val - mean) * inv_std` into $\pm\infty$, bypassing the bit-mask gate and causing undefined behavior under `-ffast-math`.

4. **Visual Layout, Citations, and AI Stylometry**:
   - Duplicate code block of `chebyshev_sanitize` on page 6.
   - Inaccurate citations: citing Zheng et al. (2017) instead of Hochreiter & Schmidhuber (1997) for foundational LSTM; incomplete citations for ISO 26262 and IEEE Std 754-2019.
   - Typographical errors: $0.962E \to 0.9625$, $[0,0] \to [0, \theta]$, `-03` compiler flag.
   - Promotional LLM phrasing ("unprecedented 98.1% reduction", "comprehensive evaluations").

---

## Solution

Execute a rigorous, reproducible re-benchmarking and revision campaign across the codebase, experiment harness, and LaTeX manuscript:

1. **Rigorous SBP-SAT Stability and Error Proof**:
   - Re-derive the proof of Theorem 2 using discrete Gauss-Radau Summation-By-Parts operators. Formulate the boundary and internal dissipation terms accurately, proving that $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$ holds strictly for all $K \ge 1$ by showing that the discrete dissipation operator $Q_E \succeq 0$ and boundary term $(2 - 2) u_0^2 = 0$ combine with interior dissipation to strictly bound all non-zero states.
   - Verify negative definiteness symbolically and numerically for $K \in \{1, 2, \dots, 32\}$ in unit tests.
   - Align inequality notation to strict negative definiteness ($\prec 0$) throughout Abstract, Introduction, Theorem 2, and Conclusion.
   - Replace "convex regression task" with "static supervised feedforward regression over frozen orthogonal features without BPTT".
   - Formalize the error decomposition $e_{\mathrm{total}} \le e_{\mathrm{poly}} + e_{\mathrm{SAT}} + e_{\mathrm{Tustin}}$, detailing how each term contributes to the overall approximation.

2. **Genuine Hardware Re-Benchmarking and Empirical Integrity**:
   - Measure actual execution times using a standalone native C executable compiled with GCC `-O3` over $100,000$ iterations, recording authentic single-step latencies, sequence window latencies ($T=50$), throughput, and standard deviation.
   - Disambiguate Table 4 hardware profiles by reporting separate, dedicated columns or explicit benchmarks for Host x86 CPU, ESP32 (240 MHz Xtensa), and STM32 (84 MHz Cortex-M4F).
   - Clarify ROM footprint metrics: document Model Parameter ROM (`.rodata`) with exact byte counts ($1,980\text{ B} = 1.98\text{ KB}$ for 3-channel; $14,996\text{ B} \approx 15.00\text{ KB}$ for 14-channel) and report actual compiled binary `.text + .rodata` sections from compiler size utilities.
   - Formalize the 3-axis vibration case study in Section 7 as an industrial micro-edge anomaly detection application, reporting the dataset source, operational context, ROC-AUC/anomaly detection metrics, and ESP32 streaming profile.
   - Implement an apples-to-apples comparison by benchmarking a bare-metal C99 LSTM/GRU cell alongside DSSR under identical C99 compiler optimization flags.
   - Implement and evaluate a Legendre polynomial DSSR baseline (HiPPO-Leg / LMU) on the same dataset in the same C99 engine, verifying Chebyshev's minimax $L_\infty$ convergence advantage.
   - Execute an ablation study on the distillation parameter $\lambda \in \{0.0, 0.5, 1.0\}$, quantifying the exact empirical contribution of teacher mimicry versus direct supervised learning.
   - Integrate official NASA C-MAPSS FD001 test files (`test_FD001.txt`, `RUL_FD001.txt`), evaluate on the official 100 test engine trajectories, and attribute baseline performance figures transparently.

3. **Embedded C99 Hardening and Accurate Profiling**:
   - Profile the exact stack frame usage of `chebyshev_step` using `-fstack-usage` with GCC. Document persistent SRAM (`.bss`, 448 B), stack scratch frame, caller buffers, and total active SRAM footprint.
   - Harden `chebyshev_sanitize` against intermediate overflow by applying input clamping before normalization or inspecting the IEEE-754 bit-mask after multiplication, ensuring robust behavior under `-ffast-math`.

4. **Manuscript Polish, Citations, and Stylometry**:
   - Eliminate duplicate C code listings.
   - Correct foundational citations (Hochreiter & Schmidhuber 1997 for LSTM; complete standard bibliographic entries for ISO 26262:2018 and IEEE Std 754-2019).
   - Fix all typographical errors ($0.9625$, $[0, \theta]$, `-O3`, $P_k(t)$).
   - Replace hyperbolic AI language with restrained, precise scientific prose.

---

## User Stories

1. As a reviewer at EMSOFT/RTSS, I want the stability proof of Theorem 2 to derive from valid Gauss-Radau SBP identities without algebraic errors, so that the theoretical foundation is mathematically sound.
2. As a numerical analyst, I want the SBP quadratic form to correctly reflect positive interior dissipation $\int W'(x) f^2 dx$ and boundary cancellation $(2-2)u_0^2 = 0$, so that constant states $f \equiv 1$ do not invalidate the strict inequality.
3. As a control theorist, I want the notation for Lyapunov dissipation to be strictly consistent ($\prec 0$) between the Abstract, Theorems, and Conclusion, so that stability claims are unambiguous.
4. As a machine learning researcher, I want the paper to accurately characterize 2-layer MLP training as non-convex supervised regression rather than claiming convexity, so that optimization theory is respected.
5. As an applied mathematician, I want the paper to clarify the total error decomposition ($e_{\mathrm{poly}} + e_{\mathrm{SAT}} + e_{\mathrm{Tustin}}$) rather than attributing static Lebesgue constant bounds to dynamic discretized state updates, so that approximation guarantees are defensible.
6. As an experimentalist, I want all latency and throughput numbers in Table 2, Table 3, and Table 4 to derive from genuine, reproducible hardware executions, so that published benchmarks withstand scrutiny.
7. As a TinyML developer, I want Table 4 to explicitly specify the microcontroller target for every latency measurement, so that I can evaluate performance on relevant hardware.
8. As an embedded systems engineer, I want the paper to distinguish between Model Parameter ROM (`.rodata`) and compiled binary size (`.text + .rodata`), so that memory consumption is transparent.
9. As a predictive maintenance practitioner, I want Section 7 to provide full context on the 3-axis vibration dataset, including task formulation and evaluation metrics, so that empirical claims are substantiated.
10. As an embedded ML researcher, I want an apples-to-apples comparison between DSSR and a bare-metal C99 LSTM/GRU implementation, so that architectural efficiency is distinguished from programming language overhead.
11. As a state-space model researcher, I want an empirical comparison between Chebyshev and Legendre (HiPPO-Leg) bases within the same C99 engine, so that the practical advantage of Chebyshev polynomials is proven.
12. As a machine learning engineer, I want an ablation on the distillation loss weight $\lambda \in \{0.0, 0.5, 1.0\}$, so that the utility of teacher mimicry over direct supervision is quantified.
13. As a benchmark author, I want C-MAPSS FD001 evaluation to follow the official test protocol (`test_FD001.txt` with `RUL_FD001.txt`), so that published RMSE figures match established literature.
14. As a firmware engineer, I want stack usage to be verified using compiler instrumentation (`-fstack-usage`), so that peak SRAM consumption can be allocated safely on Cortex-M microcontrollers.
15. As a safety-critical software developer, I want `chebyshev_sanitize` to be protected against intermediate overflow from extreme finite floats under `-ffast-math`, so that state memory cannot be poisoned.
16. As a reader, I want the paper to contain a single, clean presentation of the C inference implementation without duplicate code listings.
17. As an academic scholar, I want citations to properly reference Hochreiter & Schmidhuber (1997) for LSTMs and full bibliographic records for ISO 26262 and IEEE 754.
18. As an academic reviewer, I want the manuscript to employ objective, measured scientific language free from promotional AI phrases.

---

## Implementation Decisions

### 1. Mathematical Proof and Theory Formulation
- **Summation-By-Parts Gauss-Radau Identity**:
  - Express the continuous energy balance from first principles using the Chebyshev-Gauss-Radau quadrature:
    $$E_d = \mathbf{c}^T P_{\mathrm{CGR}} \mathbf{c} = \mathbf{u}^T W_E \mathbf{u}$$
    where $W_E = \mathrm{diag}(\bar{w}_j (1 + x_j))$ with normalized weights $\bar{w}_j = w_j / \pi$.
  - Under nodal collocation, the discrete differentiation matrix $\tilde{D}$ satisfies the Gauss-Radau SBP property:
    $$\tilde{D}^T W_E + W_E \tilde{D} = B_E - Q_E$$
    where $B_E = \mathrm{diag}(2, 0, \dots, 0)$ is the boundary flux at $x_0 = 1$ and $Q_E \succeq 0$ is the symmetric internal dissipation matrix reflecting the continuous weight derivative $W'(x) = \frac{1}{(1-x)\sqrt{1-x^2}} > 0$.
  - The SAT penalty vector in nodal coordinates satisfies:
    $$V \boldsymbol{\gamma} = (2K - 1) \mathbf{e}_0$$
    The SAT contribution to $\tilde{A}^T W_E + W_E \tilde{A}$ evaluates to:
    $$-2 \bar{w}_0 (1 + x_0) \alpha \cdot (2K - 1) u_0^2 = -2 \left(\frac{1}{2K-1}\right) (2) \left(\frac{2K-1}{2}\right) u_0^2 = -2 u_0^2$$
    This exactly cancels the boundary flux:
    $$(2 - 2) u_0^2 = 0$$
  - The quadratic form becomes:
    $$\mathbf{c}^T (A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A) \mathbf{c} = -\mathbf{u}^T Q_E \mathbf{u} - 2\bar{\kappa} \|\mathbf{u}\|^2 \le -\kappa \|\mathbf{c}\|^2 < 0$$
    strictly for all $\mathbf{c} \ne \mathbf{0}$, ensuring all eigenvalues of $A$ have strictly negative real parts.
- **Strict Inequality Standardization**:
  - Replace all occurrences of $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \le 0$ with $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$.
- **Readout Optimization Characterization**:
  - Replace "convex regression task" with "static supervised parameter optimization over frozen orthogonal features without Backpropagation Through Time (BPTT)".
- **Error Budget Formalization**:
  - Decompose total system error into three distinct components:
    1. Polynomial truncation error $e_{\mathrm{poly}} = O(K^{-r})$ for $C^r$ functions, governed by Chebyshev minimax bounds.
    2. SAT boundary relaxation residual $e_{\mathrm{SAT}} = O(1/\alpha) = O(1/K)$.
    3. Bilinear Tustin discretization error $e_{\mathrm{Tustin}} = O(\Delta t^2)$.

### 2. Empirical Benchmarking and Reproducibility Suite
- **Native C99 Benchmark Harness**:
  - Execute `run_c_benchmark` with $100,000$ iterations using GCC `-O3` to capture authentic single-step latencies, streaming window latencies ($T=50$), and throughput.
  - Record the actual processor specifications (e.g., Intel Core i7-1165G7 @ 2.80GHz) and compiler version.
- **Microcontroller Benchmarking & Table Disambiguation**:
  - In Table 4, replace the generic "MCU Step Latency" column with explicit hardware columns:
    - Host x86 CPU (GCC `-O3`, 2.80 GHz)
    - ESP32 (Xtensa LX6, 240 MHz)
    - STM32 Nucleo-F401RE (ARM Cortex-M4F, 84 MHz)
  - Ensure all data points reflect real measurements or clearly cited hardware logs.
- **ROM Footprint Disaggregation**:
  - Define Model Parameter ROM (`.rodata`) explicitly:
    - 3-Channel ($K=8, H=16$): $495\text{ floats} \times 4\text{ B} = 1,980\text{ B} = 1.98\text{ KB}$.
    - 14-Channel ($K=8, H=32$): $3,749\text{ floats} \times 4\text{ B} = 14,996\text{ B} = 15.00\text{ KB}$ ($14.64\text{ KiB}$).
  - Document compiled application flash consumption (`.text + .rodata`) separately based on compiler map / size outputs.
- **3-Axis Bearing Vibration Application**:
  - Add a dedicated subsection in Section 7 documenting the 3-axis vibration dataset (bearing degradation monitoring), specifying the 3 input channels (radial, tangential, axial), sample rate, anomaly index regression/classification task, achieved accuracy/AUC, and on-device performance.
- **Apples-to-Apples Bare-Metal C99 Baseline**:
  - Benchmark a native C99 implementation of an LSTM/GRU cell compiled with `-O3` against DSSR, isolating architectural advantages from runtime framework differences.
- **Empirical Chebyshev vs. Legendre Comparison**:
  - Implement Legendre-basis DSSR (HiPPO-Leg recurrence) within the same C99 engine and evaluate under identical conditions, verifying the empirical benefit of Chebyshev $L_\infty$ representation.
- **Distillation Loss Ablation**:
  - Run full evaluation across $\lambda \in \{0.0, 0.5, 1.0\}$:
    - $\lambda = 0.0$: Direct supervised training of Readout MLP on frozen Chebyshev features.
    - $\lambda = 0.5$: Joint ground-truth and teacher supervision.
    - $\lambda = 1.0$: Pure teacher mimicry.
- **Official C-MAPSS FD001 Evaluation**:
  - Download and incorporate official `test_FD001.txt` and `RUL_FD001.txt`.
  - Evaluate both teacher LSTM and student DSSR on the 100 official test engines.
  - Remove misattributed citation `[3]` from the 14.82 cycles RMSE result, clarifying which results are baseline reproductions and which are proposed models.

### 3. Embedded C99 Engine Hardening
- **Stack Usage Measurement**:
  - Compile `chebyshev_engine.h` with `gcc -c -O3 -fstack-usage` to obtain the exact stack frame size.
  - Present the complete SRAM breakdown:
    - Persistent Recurrent State (`.bss`): $D_{\mathrm{in}} \times K \times 4\text{ bytes}$ (448 B for $K=8, D_{\mathrm{in}}=14$).
    - Active Stack Scratch Frame: measured bytes from compiler output.
    - Caller Input Buffer: $D_{\mathrm{in}} \times 4\text{ bytes}$ (56 B).
    - Total Active SRAM: sum of all three components.
- **Hardening `chebyshev_sanitize`**:
  - Prevent finite-to-infinite overflow by performing bounds checking prior to normalization or applying IEEE-754 exponent bit-mask checks post-normalization:
    ```c
    static inline float chebyshev_sanitize(float val, float mean, float inv_std, float clamp_val) {
        uint32_t raw_bits;
        memcpy(&raw_bits, &val, sizeof(val));
        if ((raw_bits & 0x7F800000U) == 0x7F800000U) {
            return 0.0f;
        }
        /* Bound raw input to prevent overflow during multiplication */
        if (val > 1.0e10f) val = 1.0e10f;
        if (val < -1.0e10f) val = -1.0e10f;
        float norm = (val - mean) * inv_std;
        if (norm > clamp_val) norm = clamp_val;
        if (norm < -clamp_val) norm = -clamp_val;
        return norm;
    }
    ```

### 4. Manuscript Layout, Typography, and Style
- Remove the redundant code listing of `chebyshev_sanitize` on page 6.
- Update references: add Hochreiter & Schmidhuber (1997) for LSTM; complete ISO 26262:2018 and IEEE Std 754-2019 entries.
- Correct typographical errors: replace `0.962E` with `0.9625`, verify rendering of $[0, \theta]$, ensure `-O3` uses capital 'O', and standardize polynomial notation ($P_k(t)$).
- Rephrase promotional text into neutral, rigorous academic discourse.

---

## Testing Decisions

### What Makes a Good Test
- **External Parity and Physical Validity**: Tests must measure observable behavior (eigenvalues, numerical stability, execution timings, output parity) rather than internal intermediate structures.
- **Strict Boundary Checks**: Test numerical operators against known analytical limits ($K=1$, constant signals, polynomial test functions).
- **Embedded Safety Invariance**: Safety filters must produce zero `NaN` or `Inf` escapes across millions of random and malformed inputs, even under aggressive compiler optimization flags (`-ffast-math`).

### Testing Seams
1. **Mathematical Stability Seam (`tests/test_math.py`)**:
   - Verify $P_{\mathrm{CGR}} \succ 0$ and $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$ for all $K \in \{1, 2, \dots, 32\}$.
   - Verify exact polynomial transport PDE residual for degree $\le K-1$.
   - Verify unconditioned unit-disk stability $\rho(\bar{A}) < 1.0$ under Tustin discretization across varying $\Delta t$ and $\theta$.
2. **Native C99 Execution and Hardware Seam (`tests/test_benchmark.py`)**:
   - Compile and execute the standalone native C99 harness using GCC `-O3`.
   - Measure genuine single-step latency, streaming window latency, throughput, and stack frame via `-fstack-usage`.
3. **Embedded Parity and Sanitization Seam (`tests/test_export_c.py`)**:
   - Test `chebyshev_sanitize` with extreme finite floats ($> 10^{30}$) under `-ffast-math` to verify that no `Inf` or `NaN` is produced.
   - Verify bit-level output parity between PyTorch, NumPy, and compiled C99 across 1,000 streaming steps.
4. **Empirical Evaluation and Ablation Seam (`chebyshev/distill.py`, `scripts/run_scientific_benchmarks.py`)**:
   - Evaluate teacher LSTM and student DSSR on official C-MAPSS FD001 test split (`test_FD001.txt` / `RUL_FD001.txt`).
   - Run distillation parameter ablation ($\lambda \in \{0.0, 0.5, 1.0\}$).
   - Run basis ablation comparing Chebyshev DSSR vs. Legendre DSSR (HiPPO-Leg).

### Prior Art
- Existing unit tests in `tests/test_math.py`, `tests/test_benchmark.py`, and `tests/test_export_c.py`.
- Hardware benchmark logs in `examples/esp32_vibration/BENCHMARK_LOG.md`.

---

## Out of Scope
- Porting the inference engine to C++ or non-C languages (the engine strictly adheres to ISO C99).
- Implementing dynamic runtime memory allocation or heap managers (the engine is strictly zero-malloc).
- Hardware PCB layout design or in-system debugging of physical sensors.
- Training foundation models or large-scale transformers.

---

## Further Notes
- The specification directly addresses every issue flagged in the academic review.
- Following confirmation of this specification, implementation tickets will be created to execute the mathematical re-derivation, empirical benchmark script, embedded C99 hardening, and LaTeX manuscript updates.
