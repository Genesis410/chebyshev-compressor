# Contributing to Chebyshev-Compressor

Thank you for your interest in contributing to **Chebyshev-Compressor**!

This project bridges mathematical continuous-time state-space models and high-performance embedded systems. To ensure that every contribution maintains strict mathematical correctness, numerical parity, and microcontroller safety, please follow these guidelines.

---

## 1. Fast Development Environment Setup (< 3 Minutes)

We designed the development environment to be lightweight, reproducible, and runnable even on CPU-only laptops without needing multi-gigabyte CUDA downloads or specialized hardware.

### Prerequisites
- **Python**: Version 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.14).
- **C Compiler**: Any standard C99 compiler:
  - **Linux**: `gcc` or `clang` (`sudo apt-get install build-essential`)
  - **macOS**: Apple Clang (`xcode-select --install`)
  - **Windows**: MinGW-w64 (`gcc`), Clang, or MSVC (`cl.exe`).

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Genesis410/chebyshev-compressor.git
cd chebyshev-compressor

# 2. Create and activate a clean virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install lightweight PyTorch CPU wheel first (~180 MB instead of ~2.5 GB CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install chebyshev-compressor in editable development mode with dev tools
pip install -e ".[dev]" --no-build-isolation

# 5. Verify your setup by running the test suite
pytest tests/ -v

# 6. Verify end-to-end reproducible pipeline
cheb --smoke-test --seed 42
```

---

## 2. Architecture & Design Rules for Contributors

Any code submitted must adhere to the following non-negotiable architectural commitments:

### Rule 1: Zero Dynamic Allocation (`malloc`-free)
All generated C code in `chebyshev_engine.h` must be strictly `malloc`-free. State structures (`chebyshev_state_t`) must be allocated by the caller (either in static BSS or local stack memory).

### Rule 2: Header-Only `static inline` C99
All functions in `chebyshev_engine.h` must be declared `static inline`. This ensures the header can be included across multiple compilation units in large embedded firmware projects without causing `multiple definition` linker collisions.

### Rule 3: Single-Precision FPU & Literal Syntax
Embedded targets (e.g. ARM Cortex-M4F, ESP32) frequently feature single-precision hardware FPUs. 
- All floating-point operations must use `float` (32-bit).
- All literal values emitted by the C exporter must have an explicit decimal point or exponent followed by the `f` suffix (e.g., `1.0f`, `1.23e-4f`). Standard C99 disallows forms like `1f`.

### Rule 4: Compiler Optimization & `-ffast-math`-Safe Non-Finite Quarantine
Microcontroller build systems often enable `-ffast-math` (which implies `-ffinite-math-only`) or `/fp:fast`. Under these flags, compilers may optimize away floating-point `NaN`/`Inf` comparisons as dead code. Always use strict-aliasing-safe C99 `memcpy` IEEE-754 exponent bit-masking (`0x7F800000U`):
```c
static inline float chebyshev_sanitize(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) {
        return 0.0f; /* Catches NaN and +/-Inf deterministically under -ffast-math */
    }
    if (x > CHEB_INPUT_CLAMP) return CHEB_INPUT_CLAMP;
    if (x < -CHEB_INPUT_CLAMP) return -CHEB_INPUT_CLAMP;
    return x;
}
```

### Rule 5: Parity Invariance ($\max |y_{\text{PyTorch}} - y_{\text{C}}| < 10^{-4}$)
Any modification to `StudentModel` forward logic in [models.py](chebyshev/models.py) must be matched by an identical update to [exporter.py](chebyshev/exporter.py). The automated test [test_export_c.py](tests/test_export_c.py) verifies that after 1,000 steps of dynamic signals, impulses, and NaN injections, the discrepancy between PyTorch and native compiled C99 remains strictly below $10^{-4}$.

### Rule 6: Minimal Dependencies
The core package runtime must strictly depend only on:
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`

Do not add additional runtime dependencies.

---

## 3. Project Structure Tour

```text
.
├── chebyshev/
│   ├── math_ops.py       # Continuous A, B operators & float64 Tustin discretization
│   ├── models.py         # DepthwiseChebyshevCell, ReadoutMLP, StudentModel
│   ├── distill.py        # Non-invasive TeacherStateHook & Cosine Annealing Distillation
│   ├── exporter.py       # C99 code generator producing chebyshev_engine.h with CRC32
│   ├── benchmark.py      # Standalone C benchmark harness with monotonic clock
│   ├── datasets.py       # NASA C-MAPSS FD001 streaming loader & synthetic signals
│   ├── teacher.py        # Reference PyTorch LSTM / GRU teacher architectures
│   ├── utils.py          # Deterministic seeding, file hashing, and manifest generator
│   └── cli.py            # Unified 'cheb' CLI entrypoint
├── docs/
│   ├── THEORY.md         # Mathematical whitepaper (minimax, spectral decay, Tustin)
│   ├── adr/              # Architectural Decision Records (ADRs 0001 - 0008)
│   └── agents/           # Issue tracker specifications & triage vocabulary
├── examples/
│   └── arduino_cmapss/   # Ready-to-flash Arduino / ESP32 edge deployment sketch
└── tests/                # Comprehensive unit, parity, and benchmark test suite
```

---

## 4. Running Tests & Linters

Before creating a commit or opening a pull request, run the following commands:

### Run the Full Test Suite
```bash
pytest tests/ -v
```

### Run Strict C99 Compiler Check
Ensure that the exported C header compiles without any warnings under maximum strictness:
```bash
# Valid for Linux, macOS, and Windows MinGW:
gcc -x c -std=c99 -Wall -Wextra -Werror -pedantic -Wno-unused-function -fsyntax-only cheb_out/chebyshev_engine.h
clang -x c -std=c99 -Wall -Wextra -Werror -pedantic -Wno-unused-function -fsyntax-only cheb_out/chebyshev_engine.h
```

### Run Full Reproducible Pipeline
```bash
python -m chebyshev.cli --smoke-test --seed 42
```

---

## 5. Submitting a Pull Request (PR)

1. **Branch Naming**: Use descriptive prefixes:
   - `feat/feature-name` for new features or hardware targets
   - `fix/bug-description` for bug fixes
   - `docs/topic-name` for documentation improvements
2. **Issue Specification**: Reference existing issues in `.scratch/revisi/issues/` or create one following [issue-tracker.md](docs/agents/issue-tracker.md).
3. **PR Verification Checklist**:
   - [ ] All automated tests in `tests/` pass (including `-O3 -ffast-math` quarantine & signal reconstruction).
   - [ ] Parity between PyTorch and C99 is preserved ($\le 10^{-4}$).
   - [ ] C code passes `-Wall -Wextra -Werror -pedantic -std=c99`.
   - [ ] No heap allocations (`malloc`, `free`) added to C runtime.
   - [ ] Deterministic seed behavior preserved.
   - [ ] Domain terminology adheres to [GLOSSARY.md](GLOSSARY.md).

Thank you for helping push the boundaries of embedded neural compression!
