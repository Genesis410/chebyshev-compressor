# Automated Bit-Accurate PyTorch-to-C99 Parity Testing

## Context
Model distillation toolkits often suffer from silent discrepancies between the high-level Python model and the exported low-level C implementation due to operator execution ordering, matrix stride conventions, or compiler floating-point optimizations.

## Decision
We mandate an automated regression test suite (`tests/test_export_c.py`) that:
1. Auto-discovers available C compilers (`gcc`, `clang`, `cl.exe`) and compiles the exported `chebyshev_engine.h` with strict flags (`-Wall -Wextra -Werror -pedantic -std=c99`).
2. Feeds identical 1,000-step sequences (including smooth signals, step impulses, all-zeros, and injected `NaN`/`Inf` spikes) through both the PyTorch `StudentModel` and the native C executable.
3. Enforces an absolute maximum error tolerance of $\max |y_{\text{PyTorch}} - y_{\text{C}}| < 10^{-4}$.

## Consequences
- Guarantees zero compiler warnings across all major C99 toolchains.
- Detects subtle matrix transposition or memory layout bugs before edge deployment.
- Verifies that NaN recovery dynamics are perfectly synchronized between Python and C.
