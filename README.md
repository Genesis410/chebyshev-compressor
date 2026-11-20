# Chebyshev-Compressor: Analytical Chebyshev State-Space Compression to O(1) Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C99](https://img.shields.io/badge/Language-Strict%20C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Persistent State](https://img.shields.io/badge/Persistent%20State-448%20Bytes%20(608%20B%20Total%20SRAM)-success.svg)]()
[![Zero Allocation](https://img.shields.io/badge/Dynamic%20Alloc-0%20Bytes%20(malloc--free)-brightgreen.svg)]()
[![End-to-End Speedup](https://img.shields.io/badge/Speedup-5.7x%20Window%20%7C%20142.8x%20Step-orange.svg)]()

**Chebyshev-Compressor** compresses heavy recurrent neural networks (LSTMs, GRUs) into a **Fixed Analytical Orthogonal Chebyshev State-Space Model (Option A)** with a lightweight trainable Readout MLP. It exports directly to a standalone, zero-dependency, `malloc`-free strict C99 single-header library (`chebyshev_engine.h`) designed for real-time edge streaming on resource-constrained microcontrollers (ESP32, STM32, ARM Cortex-M0+/M4F).

---

## Key Architectural Advantages

```text
+-------------------------------------------------------------------------------+
| CONVENTIONAL RECURRENT NEURAL NETWORK (LSTM / GRU)                            |
|                                                                               |
|  Sensors [D] ---> [ 4x Non-Linear Gates: Sigmoid & Tanh ]                     |
|                   [ Coupled Hidden Matrices W_hh (H x H) ]                    |
|                   [ 2-Layer State: 2 x 2 x 32 x 4 = 512 B ] ---> Hidden [32]  |
|                   [ TFLM Tensor Arena Requires 16 - 32 KB]      (32,768 Bytes)|
+-------------------------------------------------------------------------------+
                                       |
         OPTION A: FROZEN ANALYTICAL CGR CHEBYSHEV + READOUT DISTILLATION
                                       v
+-------------------------------------------------------------------------------+
| CHEBYSHEV-GAUSS-RADAU (CGR) STATE-SPACE MODEL (O(1) CONSTANT STATE)           |
|                                                                               |
|  Sensors [D] ---> [ Depthwise Shifted Chebyshev Recurrence (CGR-SAT) ]        |
|                   [ Frozen Analytical Tustin Operators A_bar, B_bar  ]        |
|                   [ Minimax Optimal Window + Provable Lyapunov BIBO  ]        |
|                   [ Strict C99 Single Header, Zero malloc, Re-entrant]        |
|                                                                               |
|  14 Sensors x Order 8 = 448 B Persistent State + 160 B Stack = 608 B SRAM     |
+-------------------------------------------------------------------------------+
```

1. **Mathematically Grounded Orthogonal Window Memory (CGR Spectral-SAT)**: Continuous-time sliding-window transport ($\partial_t f_t = -\partial_\tau f_t$) is projected onto shifted Chebyshev polynomials of the first kind ($`T_k^*(\tau)`$) via a Chebyshev-Gauss-Radau Simultaneous Approximation Term (SAT). The resulting finite-dimensional operator satisfies an exact analytical Lyapunov dissipation theorem ($A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$), guaranteeing strictly Hurwitz eigenvalues, unconditional Tustin unit-disk stability ($\rho(\bar{A}) < 1.0$), and near-optimal Minimax uniform approximation across $[0, \theta]$.
2. **Explicit $O(1)$ SRAM Budget (448 B Persistent + 160 B Stack = 608 B Estimated Active Footprint)**: For 14 channels at order $K = 8$, the persistent recurrent state buffer occupies exactly $14 \times 8 \times 4 = 448$ bytes (caller-allocated `.bss`), plus 160 bytes of local call-stack scratch (`next_s[8]` + `h1[32]`) for an estimated active runtime SRAM footprint of **608 bytes** (excluding compiler register spills, struct alignment, and caller stack frames) with **zero heap allocation** (`malloc`-free).
3. **Option A Readout Distillation with Fast State Precomputation**: Because $(\bar{A}, \bar{B})$ are strictly frozen analytical operators, Chebyshev states $S \in \mathbb{R}^{N \times (D_{\mathrm{in}} \cdot K)}$ are precomputed once prior to training. Only the exported Readout MLP is optimized under a joint ground-truth + teacher mimicry objective (automatically falling back to $\lambda = 1.0$ on unlabeled calibration data), preventing pole drift and accelerating CPU training $>10\times$.
4. **Embedded-Safe IEEE-754 Quarantine (`-ffast-math` Immune)**: Raw physical sensor values are normalized in-place using baked $(\mu, \sigma^{-1})$ vectors and sanitized via strict-aliasing-safe `memcpy` IEEE-754 exponent bit-masking (`(bits & 0x7F800000U) == 0x7F800000U -> 0.0f`), verified directly under `-O3 -ffast-math` (`-ffinite-math-only`) to prevent `NaN`/`±Inf` state poisoning.
5. **Verified Numerical Parity & Strict C99 Cleanliness**: Automated tests verify $\max |y_{\mathrm{PyTorch}} - y_{\mathrm{C}}| < 10^{-4}$ over 1,000 steps (including spikes, `NaN`, and `±Inf`), 10,000-step `float32` boundedness, monotonic signal reconstruction convergence, and zero-warning compilation under `-std=c99 -Wall -Wextra -Werror -pedantic`.

---

## Empirical Benchmark & Ablation Results

All numbers below are directly measured and reproducible from `cheb --smoke-test --seed 42` and `python scripts/run_ablation.py --seed 42` on a single-threaded CPU (`torch.set_num_threads(1)`, GCC `-O3`, with `volatile float sink` preventing dead-code elimination).

### 1. Host CPU Streaming Benchmark (14 Channels, Order $K = 8$, Hidden Dim = 32)

| Metric | PyTorch LSTM (1-thread) | C99 Engine (`chebyshev_engine.h`) | Gain / Edge Benefit |
| :--- | ---: | ---: | :--- |
| **Single-Step Host Latency**\* | 848.97 µs | **5.95 µs** | **142.8x faster** (eliminates Python overhead) |
| **End-to-End Window Latency ($T = 50$)** | 1,489.40 µs | **260.02 µs** (5.20 µs/step) | **5.7x faster** (fair matched sequence ingestion) |
| **Streaming Throughput** | 1,178 steps/sec | **168,148 steps/sec** | **Real-time edge streaming** |
| **Recurrent State Vector (math)** | 512 Bytes ($2 \times 2 \times 32 \times 4$) | **448 Bytes** ($14 \times 8 \times 4$) | **12.5% state reduction** (linear memory) |
| **Active Call-Stack Scratch (local)** | Dynamic runtime stack | **160 Bytes** | `next_s[8]` + `h1[32]` |
| **Estimated Runtime Workspace vs TFLM Arena** | 32,768 Bytes (TFLM arena workspace) | **608 Bytes** (448 B state + 160 B stack) | **98.1% memory footprint reduction** (fits in 2 KB MCUs) |
| **Dynamic Heap Allocations (`malloc`)** | Enabled | **0 Bytes (`malloc`-free)** | Deterministic real-time execution |
| **C99 Compiler Strictness** | Requires Python / LibTorch | **Zero warnings (`-Werror -pedantic`)** | Verified on GCC & Clang |

\* *Note: Single-step PyTorch latency includes Python interpreter and tensor dispatch overhead. End-to-End Window Latency compares full $`T = 50`$ sequence ingestion including state initialization (`chebyshev_init`) and raw sensor normalization on identical single-thread CPU hardware. Baseline mathematical recurrent state: 2 layers x 2 state vectors (h, c) x 32 hidden units x 4 bytes = 512 bytes vs Chebyshev recurrent state $14 \times 8 \times 4 = 448$ bytes. Runtime memory footprint: ~32 KB workspace arena required by TensorFlow Lite Micro LSTM runtime vs 608 bytes estimated active C99 runtime footprint (448 B persistent state + 160 B stack scratch; excluding compiler spills and caller frames).*

### 2. On-Device Microcontroller Profiles (Measured Hardware Footprint)

| Microcontroller Target | Application Scenario | Model Config | Flash ROM (`.rodata`) | Persistent State (`.bss`) | Stack Scratch | Active SRAM Footprint\* | Measured Step Latency |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **ESP32 Xtensa @ 240 MHz** | 3-Axis Bearing Vibration ([Demo](examples/esp32_vibration/)) | $D_{\mathrm{in}}=3, K=8, H=16$ | **1.98 KB** | **96 Bytes** | 96 Bytes | **192 Bytes** | **3.8 µs** |
| **ESP32 Xtensa @ 240 MHz** | 14-Channel Turbofan RUL ([Demo](examples/arduino_cmapss/)) | $D_{\mathrm{in}}=14, K=8, H=32$ | **17.82 KB** | **448 Bytes** | 160 Bytes | **608 Bytes** | **5.2 µs** |
| **STM32F401RE Cortex-M4F @ 84 MHz** | 3-Axis Bearing Vibration | $D_{\mathrm{in}}=3, K=8, H=16$ | **1.98 KB** | **96 Bytes** | 96 Bytes | **192 Bytes** | **11.4 µs** |
| **STM32F401RE Cortex-M4F @ 84 MHz** | 14-Channel Turbofan RUL | $D_{\mathrm{in}}=14, K=8, H=32$ | **17.82 KB** | **448 Bytes** | 160 Bytes | **608 Bytes** | **15.8 µs** |

\* *Active SRAM Footprint is estimated as Persistent State (`.bss`) + local Call-Stack Scratch (`next_s` + `h1`), excluding compiler register spills and caller stack frames. Measurements taken using hardware cycle counters (`micros()` on ESP32, DWT `CYCCNT` on Cortex-M4F with hardware single-precision FPU). Zero dynamic heap allocations (`malloc`-free). See [examples/esp32_vibration/BENCHMARK_LOG.md](examples/esp32_vibration/BENCHMARK_LOG.md) for firmware memory map and profiling logs.*

### 3. Polynomial Order Ablation Study ($K \in \{4, 8, 16, 32\}$)

Reproducible via `python scripts/run_ablation.py --seed 42 --max-engines 50 --epochs 20 --iterations 50000` (evaluated on strictly disjoint held-out C-MAPSS FD001 engines with train-only normalization statistics):

| Order K | Signal Recon Rel L2 Error | Teacher Val RMSE (cycles) | Student Val RMSE (cycles) | Persistent State (B) | Stack Scratch (B) | Active SRAM (B) | C99 Step Latency (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4** | 9.4102e-01 | 41.75 | 21.50 | 224 | 144 | 368 | 2.973 |
| **8** | 2.5415e-01 | 41.75 | 21.00 | 448 | 160 | 608 | 5.750 |
| **16** | **2.0861e-04** | 41.75 | **20.56** | 896 | 192 | 1,088 | 16.351 |
| **32** | **1.7411e-04** | 41.75 | 23.24 | 1,792 | 256 | 2,048 | 38.827 |

*Key observations:*
- Signal reconstruction error drops strictly monotonically across $K = 4 \to 8 \to 16 \to 32$, reaching $< 0.021\%$ relative $L_2$ error at $K = 16$.
- $K = 8$ and $K = 16$ sit on the Pareto frontier: $K = 8$ fits in 608 B total SRAM at 5.75 µs/step, while $K = 16$ achieves the lowest held-out RMSE (20.56 cycles) before $K = 32$ begins overfitting the compact 32-unit readout MLP.

---

## 30-Second Quickstart

### 1. Minimal Installation (CPU-Optimized Wheel)

To avoid downloading multi-gigabyte CUDA packages on lightweight development machines, install PyTorch CPU first:

```bash
# 1. Install lightweight PyTorch CPU (~180 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Clone and install chebyshev-compressor in editable mode
git clone https://github.com/Genesis410/chebyshev-compressor.git
cd chebyshev-compressor
pip install -e .
```

### 2. End-to-End One-Liner Smoke Test

Verify the full pipeline—80/20 held-out dataset splitting, LSTM teacher training, Option A fast-precomputed distillation, C99 export, and native anti-DCE latency benchmarking:

```bash
cheb --smoke-test --seed 42
```

---

## CLI Command Suite Reference

The package provides a unified command-line tool `cheb`:

### 1. `cheb train-teacher`
Trains the baseline PyTorch LSTM teacher strictly on training engines and evaluates held-out validation RMSE:
```bash
cheb train-teacher \
  --data-dir ./data/cmapss \
  --hidden-dim 64 \
  --num-layers 2 \
  --epochs 20 \
  --lr 0.001 \
  --seed 42 \
  --out ./cheb_out/teacher.pt
```

### 2. `cheb distill`
Precomputes frozen CGR Chebyshev states once and optimizes the Readout MLP under Option A multi-objective distillation ($\lambda \in [0, 1]$, automatic fallback $\lambda = 1.0$ if unlabeled):
```bash
cheb distill \
  --teacher ./cheb_out/teacher.pt \
  --data-dir ./data/cmapss \
  --order 8 \
  --window-size 30 \
  --hidden-dim 32 \
  --distill-lambda 0.5 \
  --epochs 30 \
  --seed 42 \
  --out ./cheb_out/student.pt
```

### 3. `cheb export`
Generates the self-contained strict C99 single-header file with baked normalization, `memcpy` IEEE-754 sanitization, and CRC32 checksum:
```bash
cheb export \
  --model ./cheb_out/student.pt \
  --output ./cheb_out/chebyshev_engine.h
```

### 4. `cheb benchmark`
Compiles an isolated standalone C99 benchmark binary (`gcc`, `clang`, or `cl`) and audits both single-step and end-to-end window ($T = 50$) streaming latency against single-thread PyTorch:
```bash
cheb benchmark \
  --header ./cheb_out/chebyshev_engine.h \
  --iterations 100000
```

---

## Microcontroller & Embedded Deployment (C99)

The generated `chebyshev_engine.h` depends solely on ISO C99 standard headers (`<stdint.h>`, `<string.h>`).

### Standard C Usage Pattern

```c
#include <stdio.h>
#include "chebyshev_engine.h"

/* 1. Allocate persistent state in static BSS (448 bytes for 14 ch x order 8) */
static chebyshev_state_t engine_state;

int main(void) {
    /* 2. Initialize state vectors to zero */
    chebyshev_init(&engine_state);

    /* 3. Raw sensor input vector (14 channels) */
    float raw_sensors[CHEB_NUM_CHANNELS] = {
        642.35f, 1589.70f, 1400.60f, 554.36f, 2388.06f,
        9046.19f, 47.47f, 521.66f, 2388.02f, 8138.62f,
        8.4195f, 392.0f, 39.06f, 23.4190f
    };

    /* 4. Execute streaming inference step (normalizes, bit-sanitizes, steps, predicts) */
    float out[CHEB_NUM_OUTPUTS];
    chebyshev_step(&engine_state, raw_sensors, out);

    printf("Predicted Remaining Useful Life (RUL): %.2f\n", out[0]);
    return 0;
}
```

### Microcontroller Examples

- **3-Axis Bearing Vibration Monitoring**: [examples/esp32_vibration/esp32_vibration.ino](examples/esp32_vibration/esp32_vibration.ino) demonstrates rotating machinery anomaly tracking on 3 accelerometer channels with only **96 Bytes** persistent state RAM ($D_{\mathrm{in}} = 3, K = 8$).
- **Multi-Sensor Turbofan Degradation**: [examples/arduino_cmapss/arduino_cmapss.ino](examples/arduino_cmapss/arduino_cmapss.ino) demonstrates 14-channel C-MAPSS continuous RUL streaming with **448 Bytes** persistent state RAM ($D_{\mathrm{in}} = 14, K = 8$).

---

## Mathematical Foundations

The complete derivation from the sliding-window transport PDE, the Chebyshev-Gauss-Radau (CGR) Spectral-SAT boundary projection, the exact analytical Lyapunov stability proof, and double-precision Tustin discretization are documented in:

📄 **[docs/THEORY.md: Mathematical Foundations of Chebyshev State-Space Compression](docs/THEORY.md)**

### Summary of Canonical Core Equations

Every equation below matches [chebyshev/math_ops.py](chebyshev/math_ops.py) and [docs/THEORY.md](docs/THEORY.md) identically:

#### 1. Shifted Chebyshev-Gauss-Radau (CGR) Spectral-SAT Continuous Operator

Mapping sliding history $\tau \in [0, \theta]$ to $x = 1 - 2\tau/\theta \in [-1, 1]$ gives the modal ODE:

$$
\frac{d\mathbf{c}(t)}{dt} = A \mathbf{c}(t) + B u(t)
$$

with exact upper-triangular Chebyshev differentiation matrix $D \in \mathbb{R}^{K \times K}$, boundary projection weights $\gamma_k = \pi w_{\mathrm{norm}, k}$ ($\gamma_0 = 1$, $\gamma_{k \ge 1} = 2$), and CGR penalty $\alpha = \frac{2K - 1}{2}$:

$$
D_{j, k} = \begin{cases}
k, & \text{if } j = 0,\; k > 0,\; k \text{ is odd} \\
2k, & \text{if } j \ge 1,\; k > j,\; (k - j) \text{ is odd} \\
0, & \text{otherwise}
\end{cases}
$$

$$
A = \frac{2}{\theta}\left(D - \alpha\,\boldsymbol{\gamma}\mathbf{1}^T\right), \qquad B = \frac{2\alpha}{\theta}\,\boldsymbol{\gamma}
$$

With the analytical SPD CGR weighted energy matrix $P_{\mathrm{CGR}} = V^T \mathrm{diag}(w_j(1 + x_j)) V \succ 0$, $A$ satisfies the strict Lyapunov dissipation theorem $A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$, guaranteeing strictly Hurwitz eigenvalues ($\max_i \mathrm{Re}(\lambda_i(A)) < 0$) and exponential BIBO stability $\|e^{At}\| \le M e^{-\alpha_0 t}$.

#### 2. Double-Precision Tustin Bilinear Discretization (Δt = 1.0)

$$
\bar{A} = \left(I - \frac{\Delta t}{2} A\right)^{-1} \left(I + \frac{\Delta t}{2} A\right)
$$

$$
\bar{B} = \left(I - \frac{\Delta t}{2} A\right)^{-1} (\Delta t B)
$$

Conformal bilinear mapping guarantees strict unit-disk stability $\rho(\bar{A}) < 1.0$ for all $K \ge 1, \theta > 0, \Delta t > 0$.

#### 3. Depthwise Multi-Channel Step & Readout MLP

$$
\mathbf{s}_d[t] = \bar{A} \mathbf{s}_d[t-1] + \bar{B} \cdot \mathrm{sanitize}\!\left(\frac{u_d[t] - \mu_d}{\sigma_d}\right)
$$

$$
\hat{y}[t] = W_2 \cdot \mathrm{ReLU}\!\left(W_1\, \mathrm{vec}(\mathbf{S}[t]) + b_1\right) + b_2
$$

#### 4. Option A Multi-Objective Distillation Loss

$$
\mathcal{L} = (1 - \lambda)\,\mathrm{MSE}(\hat{y}_{\mathrm{student}}, y_{\mathrm{true}}) + \lambda\,\mathrm{MSE}(\hat{y}_{\mathrm{student}}, \hat{y}_{\mathrm{teacher}})
$$

with automatic fallback $\lambda = 1.0$ when distilling on unlabeled calibration datasets.

---

## Project Structure & Architecture

```text
.
├── chebyshev/
│   ├── __init__.py           # Package version and public API
│   ├── math_ops.py           # Canonical CGR Spectral-SAT (A, B), Lyapunov matrix, Tustin & reconstruction
│   ├── models.py             # DepthwiseChebyshevCell, ReadoutMLP, StudentModel
│   ├── distill.py            # Option A Fast State Precomputation & Multi-Objective Distillation Engine
│   ├── exporter.py           # Strict C99 code generator with memcpy IEEE-754 quarantine & CRC32
│   ├── benchmark.py          # Anti-DCE native C harness (single-step + window) & order ablation
│   ├── datasets.py           # NASA C-MAPSS FD001 engine-level splits & synthetic signal generator
│   ├── teacher.py            # Reference PyTorch LSTM / GRU teacher architectures
│   ├── utils.py              # Deterministic seeding, SHA-256/CRC32 hashing, and manifest generator
│   └── cli.py                # Unified 'cheb' CLI entrypoint
├── docs/
│   ├── THEORY.md             # Canonical CGR Spectral-SAT derivation & Lyapunov stability whitepaper
│   ├── COMMERCIAL.md         # Commercial services, Deployment Sprint, & Zero-Retention SLA
│   ├── adr/                  # Architectural Decision Records (ADRs 0001 - 0010)
│   └── agents/               # Domain guidelines, triage roles, and issue specs
├── examples/
│   ├── esp32_vibration/      # 3-axis bearing vibration anomaly monitoring (96 B state)
│   └── arduino_cmapss/       # 14-channel turbofan RUL streaming sketch (448 B state)
├── tools/
│   └── sanitize_client_data.py # Client-side zero-dependency privacy & unit stripping script
├── scripts/
│   ├── run_ablation.py       # Reproducible order K in {4, 8, 16, 32} ablation script
│   ├── setup.sh              # 1-click Linux/macOS contributor bootstrap script
│   └── setup.ps1             # 1-click Windows PowerShell bootstrap script
├── tests/
│   ├── test_math.py          # Hurwitz, Lyapunov P_CGR, PDE residual, Tustin, & reconstruction convergence
│   ├── test_model.py         # PyTorch forward, non-finite quarantine, & 10,000-step stability
│   ├── test_distill.py       # Fast precomputation, frozen (A, B), joint loss, & unlabeled fallback
│   ├── test_exporter.py      # C99 code generation, memcpy bit-mask, SRAM metadata, & CRC32
│   ├── test_export_c.py      # Strict C99, -O3 -ffast-math quarantine, & 1,000-step parity (< 1e-4)
│   ├── test_benchmark.py     # Native C single-step/window benchmarking & order ablation tests
│   ├── test_cli.py           # Held-out engine split verification & CLI end-to-end smoke-test
│   └── test_utils.py         # Seed determinism, manifest & calibration round-trip tests
├── .github/workflows/ci.yml  # Multi-OS CI (Ubuntu, Windows, macOS x Python 3.10-3.12)
├── CONTRIBUTING.md           # Contributor guidelines and strict C99 rules
├── Dockerfile                # Hermetic containerized dev environment
├── Makefile                  # Unix / macOS developer task runner
├── GLOSSARY.md               # Domain modeling vocabulary
└── pyproject.toml            # Minimal runtime dependency configuration
```

---

## Contributing & Developer Onboarding

We welcome researchers and embedded engineers! To contribute with full reproducibility:

- 📖 **Contributor Guidelines**: Read [CONTRIBUTING.md](CONTRIBUTING.md) for architectural commitments (zero-`malloc`, single-precision float rules, `-ffast-math`-safe IEEE-754 quarantine, and $10^{-4}$ numerical parity).
- 🚀 **1-Click Bootstrap**:
  - On Linux / macOS: run `./scripts/setup.sh`
  - On Windows: run `.\scripts\setup.ps1`
- 🐳 **Docker Dev Environment**:
  ```bash
  docker compose run --rm test
  ```
- 🤖 **Continuous Integration**: Every push and PR is verified across Ubuntu, Windows, and macOS on Python 3.10–3.12 via [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Running the Automated Test Suite

The test suite runs 40 automated unit, mathematical, and native C integration tests:

```bash
pytest tests/ -v
```

Key verified invariants:
- Analytical CGR Lyapunov dissipation ($A^T P_{\mathrm{CGR}} + P_{\mathrm{CGR}} A \prec 0$), strict Hurwitz eigenvalues, and exact polynomial PDE residual ($< 10^{-11}$).
- Monotonic signal reconstruction convergence across $K \in \{4, 8, 16, 32\}$ and Tustin unit-disk stability ($\rho(\bar{A}) < 1.0$) in `float64` and `float32`.
- 10,000-step long-horizon `float32` numerical boundedness.
- Option A fast state precomputation equivalence, frozen $(\bar{A}, \bar{B})$, automatic unlabeled $\lambda = 1.0$ fallback, and disjoint held-out C-MAPSS engine splitting.
- PyTorch vs. compiled C99 numerical parity ($\max |y_{\mathrm{PyTorch}} - y_{\mathrm{C}}| < 10^{-4}$) under both standard `-O2` and `-O3 -ffast-math` with injected `NaN`, `+Inf`, `-Inf`, and extreme spikes.
- Strict zero-warning C99 compilation under `-std=c99 -Wall -Wextra -Werror -pedantic`.

---

## Commercial Support & Deployment Services

Looking for customized model conversion, firmware profiling, or comparative baseline evaluation on your proprietary sensor streams? We offer standardized fixed-scope services under our [Commercial Services Guide](docs/COMMERCIAL.md):
- **Deployment Sprint (US$79)**: 1 recurrent model converted to verified C99 engine + numerical parity & memory audit on reference MCUs (5–7 calendar days).
- **Research Package (US$149)**: 4–6 page comparative technical audit report ($K \in \{4, 8, 16, 32\}$ vs baseline) + reproducibility bundle (7–10 calendar days).
- **Zero-Retention Guarantee**: Client-side data sanitization with permanent 14-day data deletion SLA.

To request a deployment sprint or benchmark package, reach out via [ihsanakmal151@gmail.com](mailto:ihsanakmal151@gmail.com?subject=%5BCommercial%20Inquiry%5D%20Chebyshev-Compressor%20Service) or [open a Commercial Inquiry on GitHub](https://github.com/Genesis410/chebyshev-compressor/issues/new?template=commercial_inquiry.md).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
