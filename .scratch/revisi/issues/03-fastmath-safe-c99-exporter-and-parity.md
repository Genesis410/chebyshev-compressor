# 03: Fast-Math-Safe IEEE-754 C99 Sanitization, Accurate SRAM Metadata, and Parity Tests

Status: resolved
Priority: P1 / P3 (Safety & Parity)
Testing Seam: Parity & Safety Seam (`chebyshev/exporter.py`, `chebyshev/models.py`, `tests/test_exporter.py`, `tests/test_export_c.py`)
Blocked by: 01

## What to build
Harden `chebyshev_sanitize` in the generated C99 engine using strict-aliasing-safe IEEE-754 integer bit inspection via `memcpy` so `NaN` and `±Inf` quarantine cannot be optimized away under `-ffast-math` / `-ffinite-math-only`. Align PyTorch non-finite sanitization for exact parity, report exact Persistent State vs Active Stack SRAM metadata in generated header comments, and replace uncertified MISRA-C claims with strict C99 compiler compliance.

## Acceptance Criteria

- [ ] **Safeguard — `-ffast-math` / `-ffinite-math-only` Immune `chebyshev_sanitize`**:
  - Include `<stdint.h>` and `<string.h>` in generated `chebyshev_engine.h`.
  - Replace the floating-point comparison NaN check in `chebyshev_sanitize(float x)` inside `chebyshev/exporter.py` with strict-aliasing-safe IEEE-754 exponent bit-mask checking via `memcpy`:
    ```c
    static inline float chebyshev_sanitize(float x) {
        uint32_t bits;
        memcpy(&bits, &x, sizeof(bits));
        if ((bits & 0x7F800000U) == 0x7F800000U) {
            return 0.0f; /* Catches NaN and +/-Inf deterministically even under -ffast-math */
        }
        if (x > CHEB_INPUT_CLAMP) return CHEB_INPUT_CLAMP;
        if (x < -CHEB_INPUT_CLAMP) return -CHEB_INPUT_CLAMP;
        return x;
    }
    ```
- [ ] **Exact PyTorch / C99 Non-Finite Parity**:
  - Update `DepthwiseChebyshevCell.forward` in `chebyshev/models.py` so `NaN`, `+Inf`, and `-Inf` all quarantine deterministically to `0.0` (`torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)` followed by `torch.clamp(..., -self.clamp_value, self.clamp_value)`), matching the C99 bit-mask semantics identically.
  - Update unit tests in `tests/test_model.py` verifying `NaN`, `+Inf`, `-Inf` -> `0.0` and finite extreme spikes (`±1e6`) -> `±5.0`.
- [ ] **Accurate Embedded Footprint Metadata & Standards Wording**:
  - Update generated header banner comments in `chebyshev/exporter.py` to explicitly distinguish:
    - **Persistent Recurrent State (BSS/caller-allocated)**: `CHEB_NUM_CHANNELS * CHEB_ORDER * 4` bytes.
    - **Active Call-Stack Scratch (local arrays in `chebyshev_step`)**: `(CHEB_ORDER + CHEB_HIDDEN_DIM) * 4` bytes.
    - **Total Estimated Runtime SRAM**: Persistent State + Active Stack Scratch.
  - Replace "MISRA-C compliant" docstrings/comments with "Strict C99 zero-warning (`-std=c99 -Wall -Wextra -Werror -pedantic`)".
- [ ] **Dedicated `-O3 -ffast-math` NaN/Inf Quarantine Test**:
  - Add a native compiler test in `tests/test_export_c.py` compiling a C runner against generated `chebyshev_engine.h` with `-std=c99 -O3 -ffast-math -Wall -Wextra -Werror -pedantic` (or MSVC `/fp:fast` equivalent).
  - Inject `NAN`, `INFINITY`, `-INFINITY`, and `±1e6f` spikes mid-stream and assert:
    - All outputs and state values remain strictly finite (zero state poisoning).
    - Outputs match PyTorch `StudentModel` within `< 1e-4` across normal, spiked, and non-finite steps.
- [ ] All strict C99 compliance tests (`-std=c99 -Wall -Wextra -Werror -pedantic`) and 1,000-step numerical parity tests (`max_err < 1e-4`) pass across GCC and Clang.
