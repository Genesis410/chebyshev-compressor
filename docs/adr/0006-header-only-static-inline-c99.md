# Header-Only Static Inline C99 API Architecture

## Context
Embedded developers often include the single-header engine `chebyshev_engine.h` in multiple compilation translation units (e.g., `main.c`, `sensor_task.c`, or Arduino `.ino` files). If C functions are defined directly within a header file without proper link-time qualifiers, compilers will throw `multiple definition` linker errors (ODR violation in C++ / symbol collision in C).

## Decision
We mark all function implementations (`chebyshev_init`, `chebyshev_reset`, `chebyshev_step`) inside `chebyshev_engine.h` with the `static inline` function specifier and wrap them inside `#ifdef __cplusplus extern "C" { #endif` guards.

## Consequences
- Allows single-file drop-in integration without requiring separate C source files or complex build-system wiring.
- Allows embedded compilers to inline `chebyshev_step` directly into sensor acquisition interrupt service routines (ISRs) or high-frequency polling loops, eliminating function call overhead.
- Completely prevents duplicate symbol linker conflicts across multiple translation units.
