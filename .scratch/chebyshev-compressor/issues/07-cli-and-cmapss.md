# 07: Unified `cheb` CLI Suite & NASA C-MAPSS Turbofan Pipeline

**What to build:**
The complete command-line interface entrypoint (`cheb distill`, `cheb export`, `cheb benchmark`, `--smoke-test`) and NASA C-MAPSS FD001 dataset downloader/preprocessor, enabling end-to-end distillation, C export, and benchmark evaluation with default `./cheb_out/` artifact routing.

**Blocked by:** 03: Non-Invasive Trajectory Distillation Engine, 04: C99 Single-Header Code Generator & Exporter, 06: Standalone Native C Benchmark Harness

**Status:** resolved

- [x] Implement NASA C-MAPSS FD001 automatic downloader and windowed preprocessor in `datasets.py`
- [x] Implement baseline teacher trainer for C-MAPSS in `teacher.py`
- [x] Wire unified CLI commands (`distill`, `export`, `benchmark`, `--smoke-test`) in `chebyshev/cli.py`
- [x] Implement default `./cheb_out/` artifact convention for zero-configuration chaining

## Implementation Summary
- `chebyshev/datasets.py`: Implemented `load_cmapss_fd001` with the 14 informative sensor channels, piecewise-linear Remaining Useful Life calculation ($RUL = \min(\text{raw}, 125.0) / 125.0$), sliding temporal window extraction, and automatic offline replica generation fallback.
- `chebyshev/teacher.py`: Implemented `CMapssLSTMTeacher` (2-layer LSTM baseline) with training loop and checkpoint serialization.
- `chebyshev/cli.py`: Unified CLI entrypoint with `train-teacher`, `distill`, `export`, `benchmark`, and universal `--smoke-test` shortcut, routing artifacts into `./cheb_out/` by default.
- Verified by unit tests in `tests/test_cli.py` and end-to-end `python -m chebyshev.cli --smoke-test`. Total test suite: 29 tests passing.
