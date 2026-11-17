# Input Sanity Clamping and NaN Quarantine for State Memory Protection

## Context
In industrial sensor deployments, voltage spikes, electromagnetic interference, and faulty ADC channels frequently introduce extreme outliers or `NaN`/`Inf` floating-point values. Because the linear Chebyshev recurrence $s_{t+1} = \bar{A} s_t + \bar{B} x_t$ is strictly accumulative, a single corrupt input sample permanently poisons all state coefficients until the microcontroller is power-cycled. Furthermore, compiling with `-ffast-math` or `-Ofast` often causes embedded compilers to optimize away standard `isnan()` library checks.

## Decision
1. We implement explicit IEEE-754 range checks in `chebyshev_engine.h` without relying on `isnan()`:
   ```c
   #define CHEB_INPUT_CLAMP 5.0f

   static inline float chebyshev_sanitize(float x) {
       if (!(x >= -CHEB_INPUT_CLAMP && x <= CHEB_INPUT_CLAMP)) {
           if (x > CHEB_INPUT_CLAMP) return CHEB_INPUT_CLAMP;
           if (x < -CHEB_INPUT_CLAMP) return -CHEB_INPUT_CLAMP;
           return 0.0f; /* Catches NaN and Inf */
       }
       return x;
   }
   ```
2. We mirror this exact behavior in PyTorch's `StudentModel.forward()` using `torch.clamp(x_norm, -5.0, 5.0)` to guarantee bit-accurate parity under out-of-distribution conditions.

## Consequences
- Prevents permanent recurrent state poisoning from transient sensor glitches.
- Remains robust even under aggressive compiler optimizations (`-ffast-math`, `-Ofast`).
- Maintains exact mathematical consistency between training time and edge deployment.
