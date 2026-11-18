# Spec: Chebyshev-Compressor

Status: ready-for-agent

## Problem Statement

Engineers deploying time-series intelligence to edge devices and resource-constrained microcontrollers (such as ARM Cortex-M, ESP32, and low-power IoT chips) face severe computational and memory bottlenecks when attempting to run recurrent neural networks like LSTM and GRU. Although recurrent state memory at inference is ostensibly $O(1)$, standard recurrent architectures require evaluating four sequential gating functions per time step containing non-linear transcendental functions ($\sigma$ and $\tanh$). Microcontrollers without dedicated hardware acceleration or single-precision vector extensions expend dozens of CPU cycles per floating-point transcendental evaluation, driving inference latencies into milliseconds, draining battery life, and preventing high-frequency real-time anomaly detection. Furthermore, traditional recurrent architectures suffer from catastrophic forgetting and gradient instability over long temporal windows, while modern linear state-space alternatives lack accessible distillation toolkits capable of converting pre-trained PyTorch models directly into self-contained, zero-dependency C code.

## Solution

Chebyshev-Compressor is an open-source distillation toolkit and zero-dependency C99 inference engine that projects the continuous history of multi-sensor time-series signals onto an orthogonal Chebyshev polynomial basis. The compressed architecture combines a Depthwise Chebyshev State-Space Cell—which executes continuous-time polynomial projection via linear time-invariant recurrence using pure multiply-accumulate operations—with a lightweight non-linear Readout Multi-Layer Perceptron. Through a non-invasive trajectory matching distillation pipeline, the system extracts the representational capacity of heavy PyTorch Teacher Models without requiring changes to user source code. The resulting model is exported as a standalone, MISRA-C compliant C99 header featuring static constant parameters, caller-allocated state memory, baked-in physical normalization, and IEEE-754 input sanity clamping, enabling sub-microsecond inference with zero dynamic memory allocation.

## User Stories

1. As an embedded IoT developer, I want to compress a pre-trained PyTorch multi-layer LSTM model into an orthogonal state-space model using a single CLI command, so that I can deploy sensor analytics without designing custom neural architectures from scratch.
2. As a TinyML engineer, I want the compressed model to execute recurrent state updates using only linear multiply-accumulate operations, so that inference runs in sub-microsecond time on microcontrollers without hardware transcendental function accelerators.
3. As a firmware developer, I want the exported inference engine to be a single-header C99 file with zero external library dependencies, so that I can integrate it seamlessly into Arduino IDE, ESP-IDF, STM32Cube, or bare-metal embedded projects.
4. As an embedded systems architect, I want all state memory to be caller-allocated rather than stored in hidden global variables, so that the engine is fully re-entrant and can monitor multiple sensors concurrently across multiple threads or tasks.
5. As an embedded safety engineer, I want the inference engine to perform zero dynamic memory allocation (`malloc`/`free`), so that heap fragmentation and runtime memory exhaustion bugs are completely eliminated.
6. As a hardware hacker, I want physical sensor normalization parameters to be baked directly into the static C header as constants, so that my microcontroller firmware can pass raw physical ADC readings directly to the inference function without writing separate scaling logic.
7. As an industrial IoT engineer, I want explicit input sanity clamping and NaN quarantine logic inside the recurrent step function, so that sensor glitches, voltage spikes, and corrupted communication packets do not permanently poison the recurrent state vector.
8. As an embedded developer targeting power-efficient microcontrollers with single-precision FPUs, I want all weight constants and operations to use single-precision `float32` with literal `f` suffixes, so that the compiler does not inadvertently emit slow 64-bit software floating-point emulation instructions.
9. As a machine learning researcher, I want a non-invasive distillation pipeline that hooks into existing PyTorch models via forward hooks, so that I can distill arbitrary teacher architectures without modifying their class definitions or forward method signatures.
10. As a domain practitioner working on turbofan and machinery monitoring, I want out-of-the-box support for the NASA C-MAPSS dataset, so that I can immediately evaluate the toolkit on real-world multi-sensor degradation benchmarks.
11. As a performance auditor, I want a standalone native C benchmark harness that compiles and runs outside Python, so that reported inference latencies reflect pure bare-metal execution time without foreign-function interface overhead.
12. As a software developer testing integration, I want a built-in smoke-test flag that generates synthetic multi-frequency signals and executes the full distillation, export, and benchmark loop in under 15 seconds, so that I can verify my environment without downloading multi-megabyte datasets.
13. As a firmware engineer verifying model integrity during boot, I want a deterministic CRC32 model checksum baked into the generated C header, so that my bootloader can verify parameter integrity in Flash memory.
14. As an AI engineer benchmarking against baselines, I want an automated equivalence test suite that asserts the numerical maximum error between PyTorch and compiled C99 output is under $10^{-4}$ across 1,000 steps, so that I can trust the exporter's mathematical fidelity.
15. As an open-source contributor, I want a formal mathematical whitepaper in the repository explaining the minimax approximation property and polynomial decay of Chebyshev state space, so that I can understand the theoretical principles preventing vanishing gradients over long historical contexts.

## Implementation Decisions

1. **Decoupled Hybrid State-Space Architecture**: Memory accumulation is strictly separated from prediction mapping. The recurrent loop maintains a Depthwise Chebyshev State Vector updated via linear time-invariant operations, while an external Readout Multi-Layer Perceptron (default: input dimension $\to$ 32 hidden units $\to$ output dimension with ReLU activation) provides the non-linear capacity required to match or exceed 95% of teacher accuracy.
2. **Deterministic Closed-Form Discretization**: Transition matrices are computed analytically from continuous Chebyshev derivative operators on a sliding temporal window using the Bilinear Tustin Transform. Matrix inversions are calculated in 64-bit floating-point (`float64`) before being cast to 32-bit (`float32`) to eliminate numerical instability and roundoff drift.
3. **Depthwise Multi-Channel State Space**: Multi-sensor time-series inputs assign an independent order $K$ state vector to each physical sensor channel. All channels share identical transition dynamics, while channel coupling is handled downstream by the Readout MLP, restricting state RAM consumption to under 500 bytes for typical industrial setups.
4. **Trajectory Distillation with Annealed Adapter**: Distillation optimizes a joint objective combining output task matching and full-sequence state trajectory alignment. A linear adapter projection is trained end-to-end to align the concatenated Chebyshev state with the teacher's hidden representations, with the trajectory loss weight scheduled via cosine annealing and the adapter discarded prior to C export.
5. **Non-Invasive Teacher State Extraction**: PyTorch forward hooks attach automatically to the teacher's recurrent layers, normalizing output tuples and batch dimensions transparently, with graceful fallback to output matching if non-recurrent teachers are supplied.
6. **Self-Contained C99 Header Engine**: The exporter generates an `extern "C"` compatible header where all functions are qualified `static inline` to prevent multiple definition linker errors, weights are arranged in row-major order for optimal cache locality, and an anti in-place overwrite buffer preserves state orthogonality during recurrent updates.
7. **Robust Input Sanitization**: IEEE-754 range checks are applied directly after normalization in both Python and C99, clamping inputs to $[-5.0\text{f}, 5.0\text{f}]$ and resetting non-finite values to zero without relying on compiler-fragile library functions.
8. **Minimal Dependency Footprint**: Core library runtime dependencies are restricted to `torch`, `numpy`, and `tqdm`, utilizing built-in standard library utilities for CLI arguments, file streaming, and dataset parsing.

## Testing Decisions

1. **Testing Seam Hierarchy**:
   - **Primary High-Level Seam (CLI End-to-End)**: The unified command-line interface `cheb` is tested via programmatic execution of `distill`, `export`, and `benchmark` subcommands against synthetic and benchmark datasets.
   - **Mathematical Seam (Continuous & Discretized Operators)**: Unit tests verify that the discretized transition matrix eigenvalues lie strictly on or within the unit disk ($|\lambda_i| \le 1$) and satisfy orthogonality relations.
   - **Numerical Parity Seam (PyTorch vs Native C99)**: Automated test harnesses compile the exported C header with strict compiler flags (`-Wall -Wextra -Werror -pedantic -std=c99`) and assert that output sequences match PyTorch model predictions within an absolute tolerance of $10^{-4}$ under normal, impulse, zero, and outlier conditions.
2. **Behavioral Testing Focus**: Tests strictly validate external behavior (loss convergence, parity thresholds, state recovery from glitches, and compilation success) rather than private internal implementation details.

## Out of Scope

1. Dynamic memory allocation or heap-based state management in C.
2. Quantization to INT8 or fixed-point Q15 arithmetic (reserved for future post-v1.0 releases to avoid spectral pole shifting).
3. Large Language Model (LLM) attention or KV-cache compression.
4. Bidirectional sequence modeling for offline batch processing (the engine is strictly causal and forward-streaming).
5. Automatic hyperparameter neural architecture search for the Readout MLP.

## Further Notes

The codebase is developed with standard CPU-only constraints (AMD Ryzen 5 4650U, 16GB RAM) in mind, ensuring all distillation runs and benchmark validations execute swiftly without requiring GPU hardware. Artifacts from distillation runs are placed by convention into `./cheb_out/` to enable zero-configuration piping between commands.
