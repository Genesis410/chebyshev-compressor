# Chebyshev-Compressor

A lightweight distillation and conversion toolkit that transforms recurrent neural networks (LSTM/GRU) into orthogonal Chebyshev State-Space Models with O(1) state memory and zero-dependency C99 inference for TinyML and edge devices.

## Language

**Teacher Model**:
The pre-trained, heavy sequential network (e.g., PyTorch Multi-Layer LSTM or GRU) that serves as the behavioral ground truth during distillation.
_Avoid_: LLM, Large Foundation Model, GenAI

**Chebyshev State-Space Cell (CSSC)**:
A linear, time-invariant recurrent unit whose state vector represents the orthogonal Chebyshev polynomial expansion coefficients of the input signal over a historical time window.
_Avoid_: Gated Polynomial, Chebyshev LSTM, Polynomial RNN

**Chebyshev State Vector ($s_t$)**:
A fixed-size real vector $s_t \in \mathbb{R}^K$ (or $\mathbb{R}^{D_{\text{in}} \times K}$) containing the orthogonal spectral coefficients of the input history.
_Avoid_: KV Cache, Hidden Memory Matrix, Context Buffer

**Depthwise Chebyshev State**:
A structural decomposition where each input channel maintains its own independent order-$K$ Chebyshev state vector while sharing identical transition matrices ($\bar{A}, \bar{B}$).
_Avoid_: Coupled State, Multi-channel Flattened State

**Baked Normalization**:
The practice of embedding dataset channel statistics ($\mu_d$ and $\sigma_d^{-1}$) directly into the exported C header as static constants, allowing the inference engine to accept raw physical sensor readings without external pre-processing.
_Avoid_: External Pre-processing, Client-side Scaling, Raw Unscaled Input

**State Poisoning**:
The permanent, unrecoverable corruption of recurrent state memory caused by the ingress of non-finite (`NaN`/`Inf`) or extreme outlier values through an accumulative linear recurrence.
_Avoid_: Memory Leak, Buffer Overflow, Value Explosion

**Input Sanity Clamping**:
An explicit, strict-aliasing-safe IEEE-754 exponent bit-mask check (`0x7F800000U` via `memcpy`) followed by `[-5.0, 5.0]` range clamping applied immediately after normalization in both Python and C to quarantine `NaN`/`±Inf` and extreme spikes even under `-ffast-math` / `-ffinite-math-only`.
_Avoid_: Error Throwing, Dynamic Filtering, isnan-check

**Readout Distillation (Option A)**:
The optimization process of keeping the analytical Chebyshev-Gauss-Radau operators $(\bar{A}, \bar{B})$ frozen, precomputing the Chebyshev state tensor $S$ once, and training only the exported Readout Head under a joint ground-truth + teacher mimicry objective (automatically falling back to $\lambda = 1.0$ on unlabeled calibration data).
_Avoid_: Direct Weight Conversion, Weight Surgery, Discarded Alignment Adapter

**Readout Head**:
The lightweight non-linear feedforward projection layer (e.g., 2-layer MLP with ReLU) applied to the Chebyshev State Vector to produce final predictions.
_Avoid_: Non-linear recurrent gate, Activation function

**Static Inference Engine**:
A zero-allocation, header-only C99 library (`chebyshev_engine.h`) that executes model inference using caller-allocated state memory and pre-computed static constant weights.
_Avoid_: Dynamic C++ Runtime, Inference Framework, Heavy Engine

**Standalone Benchmark Harness**:
A self-contained native C executable compiled directly with high optimization flags (`-O3`) to benchmark raw inference latency without Python FFI (`ctypes`/`cffi`) call-overhead distortions.
_Avoid_: Python FFI benchmark, Ctypes loop
