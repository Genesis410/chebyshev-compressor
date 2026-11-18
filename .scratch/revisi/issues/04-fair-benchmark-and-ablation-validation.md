# 04: Fair End-to-End Streaming Benchmark, SRAM Breakdown, and Order Ablation Suite

Status: resolved
Priority: P2 (High — Scientific Validation)
Testing Seam: Primary Seam & Benchmark Harness (`chebyshev/benchmark.py`, `tests/test_benchmark.py`, ablation validation script/tests)
Blocked by: 01, 02, 03

## What to build
Upgrade the benchmark harness to measure fair end-to-end streaming sequence windows (e.g., `seq_len=50`, including state initialization and raw sensor normalization) alongside single-step microbenchmarks, report an honest Persistent State vs Active Stack SRAM breakdown, and add a reproducible polynomial order ablation study ($K \in \{4, 8, 16, 32\}$).

## Acceptance Criteria

- [ ] **Fair End-to-End Sequence Streaming Benchmark**:
  - Extend `chebyshev/benchmark.py` (`generate_c_benchmark_harness_source`, `run_c_benchmark`, `run_pytorch_benchmark`) to support a configurable `seq_len` (default `50` for end-to-end window evaluation, alongside single-step streaming latency).
  - Measure complete window execution in both native C (state init + `seq_len` steps of `chebyshev_step` on raw inputs) and single-threaded PyTorch (`torch.set_num_threads(1)` on a `(1, seq_len, D)` window), clearly reporting both **per-window latency** and **per-step amortized latency**.
  - Explicitly note in benchmark summaries that PyTorch CPU latency includes Python interpreter and tensor dispatch overhead.
- [ ] **Accurate SRAM Reporting**:
  - Emit both `CHEB_PERSISTENT_STATE_BYTES` (`sizeof(chebyshev_state_t)`) and `CHEB_STACK_SCRATCH_BYTES` (`(CHEB_ORDER + CHEB_HIDDEN_DIM) * sizeof(float)`) from the native C benchmark harness.
  - Update `compare_benchmarks` table formatting to display Persistent State RAM, Estimated Active Stack, and remove unverified `(MISRA-C)` labels (replace with `0 bytes (malloc-free C99)`).
- [ ] **Polynomial Order Ablation ($K \in \{4, 8, 16, 32\}$)**:
  - Add a reproducible ablation utility/script (e.g., `scripts/run_ablation.py` or module function + automated test in `tests/test_benchmark.py` / `tests/test_math.py`) evaluating $K \in \{4, 8, 16, 32\}$ on a fixed seed (`--seed 42`).
  - Record and output a structured JSON/Markdown table comparing:
    - Order $K$
    - Signal reconstruction $L_2$ error
    - Held-out validation RMSE
    - Persistent State Bytes ($D_{\text{in}} \times K \times 4$) & Active Stack Bytes
    - Native C99 latency ($\mu\text{s}$/step)
- [ ] All tests in `tests/test_benchmark.py` pass across Windows, macOS, and Linux CI runners.
