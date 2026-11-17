# 04: C99 Single-Header Code Generator & Exporter

**What to build:**
The exporter module that translates trained PyTorch student weights into a self-contained, MISRA-C compliant, `static inline` C99 header (`chebyshev_engine.h`) with row-major parameter matrices, caller-allocated state, baked-in channel normalization, anti in-place overwrite buffer, IEEE-754 input sanitization, and embedded CRC32 checksum.

**Blocked by:** 02: PyTorch Depthwise Chebyshev State-Space Model with Non-Linear Readout

**Status:** resolved

- [x] Implement parameter extraction and row-major array formatting with full `%.9g` float32 literal precision
- [x] Embed static constant normalization arrays (`CHEB_MEAN`, `CHEB_INV_STD`) into the header
- [x] Implement `chebyshev_sanitize` catching NaNs/Infs without compiler `isnan` dependencies
- [x] Implement `chebyshev_step` with stack temporary buffer `next_s[CHEB_ORDER]` to prevent in-place overwrites
- [x] Compute CRC32 model parameter checksum and format metadata comments

## Implementation Summary
- Implemented `chebyshev/exporter.py` with:
  - `format_float`: Guarantees exact IEEE-754 C99 single-precision literal syntax (`.0f` or `e...f`) avoiding double promotion.
  - Formats parameter matrices in spatial-locality row-major layout (`CHEB_A`, `CHEB_B`, `CHEB_MEAN`, `CHEB_INV_STD`, `CHEB_W1`, `CHEB_B1`, `CHEB_W2`, `CHEB_B2`).
  - Generates self-contained C99 header with re-entrant `chebyshev_state_t`, `chebyshev_sanitize`, `chebyshev_init`, `chebyshev_reset`, and `chebyshev_step`.
  - Computes CRC32 checksum over parameter byte array for firmware integrity validation.
  - Embeds RAM and Flash memory estimation metadata comments.
- Verified by unit tests in `tests/test_exporter.py`. Total test suite: 18 tests passing.
