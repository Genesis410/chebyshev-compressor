# 05: Automated Compiler Discovery & Numerical Parity Verification

**What to build:**
The automated parity test suite that auto-detects host C compilers (`gcc`/`clang`/`cl`), compiles the generated C99 engine with strict flags (`-Wall -Wextra -Werror -pedantic -std=c99`), feeds 1,000 steps of normal, impulse, zero, and outlier inputs, and verifies that PyTorch and C99 predictions match with absolute error $< 10^{-4}$.

**Blocked by:** 03: Non-Invasive Trajectory Distillation Engine, 04: C99 Single-Header Code Generator & Exporter

**Status:** resolved

- [x] Implement compiler discovery using `shutil.which` for `gcc`, `clang`, or MSVC `cl.exe`
- [x] Auto-compile generated C engine with strict standard flags (`-Wall -Wextra -Werror -pedantic -std=c99`)
- [x] Feed 1,000 steps of identical sequential inputs through PyTorch and compiled C binary
- [x] Assert absolute difference $\max |y_{\text{PyTorch}} - y_{\text{C}}| < 10^{-4}$ under regular, impulse, zero, and NaN inputs

## Implementation Summary
- Implemented `tests/test_export_c.py`:
  - `find_c_compiler`: Auto-discovers host compiler (`gcc`, `clang`, MSVC `cl.exe`).
  - Strict compilation runner: Compiles generated `chebyshev_engine.h` with `-Wall -Wextra -Werror -pedantic -std=c99` (or `/W4 /WX /std:c11`), ensuring zero compiler warnings.
  - End-to-end numerical equivalence test: Feeds 1,000 continuous steps (including normal oscillations, extreme outlier spikes $\pm 5000.0$, injected NaNs and Infs, and zero decay) through PyTorch `StudentModel` and native C binary via binary file I/O.
  - Formally asserts $\max |y_{\text{PyTorch}} - y_{\text{C}}| < 10^{-4}$.
  - Verified multi-output parity (`out_features=3`).
- All 4 tests passing in `tests/test_export_c.py`. Total test suite: 22 tests passing.
