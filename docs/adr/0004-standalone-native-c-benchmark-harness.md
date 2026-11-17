# Standalone Native C Benchmark Harness for Latency Verification

## Context
A single inference step of `chebyshev_engine.h` executes in sub-microsecond time ($< 1\,\mu\text{s}$). Benchmarking this engine from Python using foreign-function interface bindings (`ctypes` or `cffi`) introduces an overhead of $1\text{–}2\,\mu\text{s}$ per call invocation, distorting reported latency by several hundred percent and invalidating audit claims of 50x speedups against PyTorch.

## Decision
The `cheb benchmark` command will generate, compile, and run a standalone native C harness binary:
1. Compiles a generated `benchmark_harness.c` using the host C compiler (`gcc -O3 -march=native` or `clang` / `cl.exe`).
2. Runs 100,000 continuous inference iterations within a native C loop using high-resolution monotonic clocks (`clock_gettime(CLOCK_MONOTONIC)` or Windows `QueryPerformanceCounter`).
3. Compares the resulting pure latency against a single-threaded PyTorch execution (`torch.set_num_threads(1)`).

## Consequences
- Eliminates Python runtime and FFI invocation noise entirely from published benchmarks.
- Produces verifiable, audit-proof latency figures in nanoseconds/microseconds.
- Requires a local C compiler in the benchmark environment (with graceful fallback to PyTorch-only metrics if no compiler is present).
