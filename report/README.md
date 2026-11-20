# Chebyshev-Compressor Technical Report

This directory contains the complete LaTeX source for the technical report:

> **Chebyshev State-Space Compression: Provably Stable Near-Minimax Continuous Memory in Sub-Kilobyte SRAM for Real-Time Edge Microcontrollers**  
> *Author:* Ihsan Akmal (Independent Researcher)

---

## 1. Quick Compilation

### Option A: Local Build (MiKTeX / TeX Live)
In your terminal (outside sandbox or normal PowerShell):
```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Or using `latexmk`:
```bash
latexmk -pdf main.tex
```

### Option B: Overleaf (Cloud)
1. Zip the `report/` folder (`main.tex` and `references.bib`).
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** > **Upload Project**.
3. Click **Recompile** to inspect and download the camera-ready PDF.

---

## 2. Zenodo DOI Minting Workflow

1. **Push to GitHub**:
   Ensure `main.tex`, `references.bib`, and `.zenodo.json` are committed to your GitHub repository (`Genesis410/chebyshev-compressor`).
2. **Link GitHub to Zenodo**:
   - Go to [Zenodo](https://zenodo.org/) and sign in with your GitHub account.
   - Navigate to **GitHub settings** on Zenodo (`https://zenodo.org/account/settings/github/`).
   - Toggle the switch to **ON** for repository `Genesis410/chebyshev-compressor`.
3. **Create a GitHub Release**:
   - On GitHub, go to **Releases** > **Draft a new release**.
   - Create a tag, e.g., `v1.0.0-report`.
   - Title: `v1.0.0: Technical Report & Analytical CGR Engine Release`.
   - Attach the compiled `main.pdf` as a release binary asset.
   - Click **Publish release**.
4. **DOI Generated**:
   - Zenodo automatically triggers an archive job, reads `.zenodo.json`, and mints a persistent DOI (e.g., `10.5281/zenodo.XXXXXXX`).
   - Copy the Zenodo DOI badge and paste it into `README.md`.

---

## 3. Positioning as an Independent Research Project

The combination of:
1. **Mathematical Rigor**: Continuous-to-discrete CGR SBP Lyapunov dissipation certificate ($A^T P + P A \prec 0$).
2. **Empirical Efficiency**: $608$ Bytes total active SRAM ($98.1\%$ memory reduction vs TFLM) with microsecond latency.
3. **Reproducible Artifact**: MIT License C99 single-header library and seed-fixed ablation scripts.
4. **Persistent Citation**: Zenodo citable DOI.

Provides the foundational evidence required to apply for competitive independent research grants, seed fellowships, and industry research collaborations.
