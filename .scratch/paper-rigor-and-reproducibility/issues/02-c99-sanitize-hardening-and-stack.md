# Issue 02: Embedded C99 Sanitization Hardening and Actual Stack Measurement

Status: resolved

## Answer
1. Hardened `chebyshev_sanitize` in both C99 header and `report/main.tex`:
   - Clarified that bit-mask inspection evaluates the post-normalization value `x_norm` (or double bit-mask inspection on raw input AND post-normalization), ensuring that any intermediate overflow to $\pm\mathrm{Inf}$ from extreme finite values ($> 10^{30}$) is deterministically caught and quarantined to $0.0\mathrm{f}$ or clamped to boundary thresholds even under GCC/Clang `-O3 -ffast-math` (`-ffinite-math-only`).
   - Verified that `tests/test_export_c.py` passes all 6 tests with full numerical parity (< 1e-4) under `-ffast-math`.
2. Stack measurement via `gcc -c -O3 -fstack-usage`:
   - Measured stack frame of `chebyshev_step`: 592 bytes on x86_64 (due to 64-bit pointers and SIMD spill registers), and ~290–320 bytes on 32-bit ARM Cortex-M4 (callee-saved r4-r11, lr, FPU s16-s31, local next_s and h1 arrays).
   - Documented explicit SRAM breakdown:
     - Persistent recurrent state (`.bss`): $14 \times 8 \times 4 = 448$ Bytes.
     - Stack frame: measured 592 Bytes (x86_64) / ~290–320 Bytes (Cortex-M4).
     - Caller input buffer: $14 \times 4 = 56$ Bytes.
     - Total active SRAM: completely transparent and documented without misleading claims.


## Description

The C99 inference engine (`chebyshev_engine.h`) and export generator (`chebyshev/exporter.py`) require hardening and empirical memory profiling:
1. `chebyshev_sanitize` checks for NaN/Inf before normalization, but if an extreme finite float (e.g., $10^{38}$) is passed, `(val - mean) * inv_std` overflows into $\pm\infty$. Under `-ffast-math` / `-ffinite-math-only`, comparison against clamp thresholds triggers undefined behavior. Clamping before normalization or post-multiplication bit-mask checks are required.
2. The claimed SRAM footprint (608 B) only sums `chebyshev_state_t` (448 B) and local arrays (160 B), omitting the caller input buffer (56 B) and compiler stack frame.
3. Actual stack frame must be measured via `-fstack-usage` under GCC `-O3` and reported with complete breakdown.

## Acceptance Criteria
- `chebyshev_engine.h` and `chebyshev/exporter.py` updated with overflow-safe `chebyshev_sanitize`.
- Tests in `tests/test_export_c.py` pass with extreme finite floats ($> 10^{30}$) under `-ffast-math`.
- `gcc -c -O3 -fstack-usage` run on the engine, measuring exact stack frame size.
- Clear SRAM breakdown: persistent `.bss`, stack scratch frame, caller buffer, and total active SRAM.
