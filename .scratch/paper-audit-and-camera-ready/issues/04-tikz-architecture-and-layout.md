# Issue 04: TikZ Architectural Vector Diagram and Visual Presentation

Status: resolved

## Description

Refactor Figure 1 (TikZ architecture diagram) and layout presentation:
1. End both pathways (a) and (b) with an Output $\hat{y}_t$ box with matching margins.
2. Add background dashed enclosures using TikZ backgrounds/fit indicating memory bounds:
   - TFLM Tensor Arena (16--32 KB SRAM) enclosing the LSTM recurrent gates and hidden matrix.
   - Persistent State Buffer (448 B .bss) enclosing the Chebyshev state, and Call-Stack Scratch (160 B) enclosing the Readout MLP.
3. Add curved recurrent feedback arrows labeled $z^{-1}$ over the LSTM hidden state ($h_{t-1} \to h_t$) and Chebyshev state buffer ($s_{t-1} \to s_t$) with zero collision.
4. Standardize node heights (`minimum height=1.0cm`, `text depth=0.2ex`) and relative spacing (`right=0.5cm of ...`).
5. Anchors cleanly at the top of Page 3 within text width without clipping.

## Acceptance Criteria
- [x] Diagram displays clean parallel dataflow with matching terminal output blocks.
- [x] Memory footprint clearly demarcated via dashed background enclosures.
- [x] Recurrent unit delays ($z^{-1}$) explicitly indicated without label overlap.
- [x] Anchors cleanly at the top of Page 3 and fits within page margins.
