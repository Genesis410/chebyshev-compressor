# Input Sanity Clamping and IEEE-754 Bit-Level Non-Finite Quarantine

## Context
In industrial sensor deployments, voltage spikes, electromagnetic interference, and faulty ADC channels frequently introduce extreme outliers or `NaN`/`Inf` floating-point values. Because the linear Chebyshev recurrence $s_{t+1} = \bar{A} s_t + \bar{B} x_t$ is strictly accumulative, a single corrupt input sample permanently poisons all state coefficients until the microcontroller is reset.

Furthermore, embedded builds commonly enable `-ffast-math` (which implies `-ffinite-math-only`) or MSVC `/fp:fast`. Under `-ffinite-math-only`, GCC and Clang assume `NaN` and `Inf` values never exist and may legally optimize away floating-point range checks such as `if (!(x >= -5.0f && x <= 5.0f))` as dead code.

## Decision
1. We implement strict-aliasing-safe C99 bit-level IEEE-754 exponent inspection via `memcpy` in `chebyshev_engine.h`, requiring only `<stdint.h>` and `<string.h>`:
   ```c
   #define CHEB_INPUT_CLAMP 5.0f

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
2. We mirror these exact semantics in PyTorch's `DepthwiseChebyshevCell.forward()` by calling `torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)` **prior to** `torch.clamp(..., -5.0, 5.0)`, guaranteeing bit-accurate parity across normal, spiked, `NaN`, `+Inf`, and `-Inf` inputs.
3. We verify this safeguard directly in CI via `test_fastmath_ieee754_nan_inf_quarantine_and_parity`, compiling with `-std=c99 -O3 -ffast-math -Wall -Wextra -Werror -pedantic` (or `/fp:fast`).

## Consequences
- Guarantees deterministic non-finite quarantine even when `-ffast-math` / `-ffinite-math-only` is active.
- Fully compliant with ISO C99 strict-aliasing rules via `memcpy`, compiling down to a single register bitmask instruction under `-O2`/`-O3`.
- Maintains exact `< 1e-4` numerical parity between PyTorch and compiled C99 under sensor faults.
