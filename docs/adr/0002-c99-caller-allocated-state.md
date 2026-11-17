# Pure C99 Header with Caller-Allocated State

## Context
Embedded deployment targets include resource-constrained microcontrollers (ESP32, ARM Cortex-M0/M4, STM32) and bare-metal environments governed by MISRA-C rules. Dynamic memory allocation (`malloc`, `free`) creates risks of heap exhaustion and memory fragmentation, while hidden global state variables in `.bss` prevent re-entrancy and concurrent multi-sensor monitoring.

## Decision
We implement `chebyshev_engine.h` as a zero-dependency, C99 single-header library where:
1. All state memory is encapsulated in a user-allocated struct (`chebyshev_state_t`) passed explicitly by pointer to every step function.
2. Model parameters are embedded as `static const float` arrays with single-precision `f` literal suffixes to prevent unintended 64-bit software floating-point emulation on single-precision FPUs.
3. No heap memory (`malloc`/`free`) or C++ standard libraries are used, with full C++ `extern "C"` compatibility for Arduino `.ino` integration.

## Consequences
- Enables deterministic memory footprints verified at compile time via `arm-none-eabi-size`.
- Ensures re-entrant, thread-safe inference for multiple concurrent sensor streams.
- Allows seamless compilation across GCC, Clang, MSVC, and Arduino toolchains without configuration.
