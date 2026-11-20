# Issue 04: Manuscript Layout, Citations, Stylometry, and PDF Compilation

Status: resolved

## Description

The LaTeX manuscript (`report/main.tex` and `report/references.bib`) contains visual, typographic, and stylistic issues:
1. Duplicate `chebyshev_sanitize` code block on page 6.
2. Inaccurate citations: Hochreiter & Schmidhuber (1997) needed for LSTM; complete standard bibliographic entries for ISO 26262:2018 and IEEE Std 754-2019.
3. Typographical errors: replace `0.962E` with `0.9625`, verify $[0, \theta]$ notation, correct `-03` to `-O3`, and standardize polynomial symbols in Table 1 ($P_k(t)$).
4. Remove promotional AI language ("unprecedented", "comprehensive evaluations") and replace with objective academic phrasing.
5. Update all tables with the genuine re-benchmarked numbers from Issue 03 and re-derived proofs from Issue 01.
6. Recompile `report/main.pdf` and generate updated per-page artifact previews.

## Resolution Summary
- Removed duplicate C99 code block; single annotated listing now appears in Section 6.2.
- Updated `report/references.bib` with authoritative entries: Hochreiter & Schmidhuber (1997) for LSTM, full bibliographic standard entries for ISO 26262-6:2018, IEEE 754-2019, and MISRA C:2012.
- Fixed all typographical glitches: `0.9625`, proper $[0, \theta]$ intervals, `-O3` compiler flags, standardized polynomial symbols.
- Rewrote promotional LLM phrases into rigorous, objective academic language.
- Updated Table 1, Table 2, Table 3, and Table 4 with genuine empirical re-benchmarking data (5.83 µs CPU, 3.81 µs ESP32, 12.40 µs STM32, exact ROM byte counts).
- Compiled camera-ready `report/main.pdf` to exactly 8 pages with zero overflow and balanced columns.
- Rendered pages 1-8 to high-resolution PNG previews and updated `report_preview.md`.

## Acceptance Criteria
- [x] `report/main.tex` cleanly updated without duplicate listings or typos.
- [x] `report/references.bib` updated with authoritative entries.
- [x] `report/main.pdf` successfully compiled with zero errors (exact 8 pages).
- [x] Visual inspection artifacts updated in the artifact directory (`page_1.png` to `page_8.png`, `report_preview.md`).
