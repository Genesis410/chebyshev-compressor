# Awesome-Lists Submission Guide for Chebyshev-Compressor

Submitting **Chebyshev-Compressor** to popular curated GitHub "Awesome" lists drives high-intent, passive organic traffic from TinyML researchers, IoT engineers, and embedded developers worldwide.

Below are the 3 top target repositories, exact markdown snippets, and copy-paste Pull Request descriptions.

---

## Target 1: [AI-in-Transportation-Lab/awesome-tinyml](https://github.com/AI-in-Transportation-Lab/awesome-tinyml)

* **Repository**: `AI-in-Transportation-Lab/awesome-tinyml` (Largest active TinyML curated directory)
* **Target Section**: `### Frameworks and Tools` or `### Model Compression and Optimization`

### Markdown Snippet to Insert:
```markdown
- [Chebyshev-Compressor](https://github.com/Genesis410/chebyshev-compressor) - Constant-state recurrent neural network compression for resource-constrained embedded AI. Distills PyTorch/ONNX recurrent models into CGR Chebyshev state-space projections with an exact 96–448 byte persistent RAM C99 runtime for ESP32/STM32.
```

### Pull Request Submission Details:
* **PR Title**: `Add Chebyshev-Compressor to Model Compression / Tools`
* **PR Description**:
  ```markdown
  ### Description
  This PR adds **Chebyshev-Compressor** to the curated TinyML frameworks and tools list.

  ### Why it belongs here:
  - **Focus**: Constant-state recurrent model compression specifically designed for ultra-low-power microcontrollers (ESP32, STM32, Cortex-M).
  - **Solves a major TinyML problem**: Running LSTM or recurrent models on MCUs usually requires ~16–32 KB of SRAM arena (TFLite Micro). Chebyshev-Compressor projects continuous sliding windows onto orthogonal Chebyshev polynomials, reducing recurrent state to strictly 96 bytes (3 channels) or 448 bytes (14 channels).
  - **Open-source & Verified**: Strictly C99 compliant, `malloc`-free, with proven Lyapunov dissipation stability and automated C/PyTorch numerical parity tests.
  - **License**: MIT
  - **Link**: https://github.com/Genesis410/chebyshev-compressor
  ```

---

## Target 2: [umitkacar/awesome-tinyml](https://github.com/umitkacar/awesome-tinyml)

* **Repository**: `umitkacar/awesome-tinyml` (Edge AI & Microcontroller Machine Learning)
* **Target Section**: `## Frameworks` or `## Tools`

### Markdown Snippet to Insert:
```markdown
- [Chebyshev-Compressor](https://github.com/Genesis410/chebyshev-compressor) - Constant-memory recurrent model compression and C99 runtime for streaming sensor time-series on microcontrollers.
```

### Pull Request Submission Details:
* **PR Title**: `Add Chebyshev-Compressor (C99 Recurrent Model Compression)`
* **PR Description**:
  ```markdown
  Hello! I would like to propose adding **Chebyshev-Compressor** to this awesome TinyML list.

  - **Repository**: https://github.com/Genesis410/chebyshev-compressor
  - **Category**: Frameworks / Tools
  - **Summary**: An open-source compression pipeline that converts recurrent neural networks (LSTM/GRU) into continuous Chebyshev state-space representations. It exports to a single-header C99 library with zero dynamic memory allocation, running 3-axis vibration monitoring in 96 bytes of persistent RAM on ESP32.
  - **License**: MIT
  ```

---

## Target 3: [Premchand006/awesome-edge-ai](https://github.com/Premchand006/awesome-edge-ai)

* **Repository**: `Premchand006/awesome-edge-ai` (Edge AI and On-Device Machine Learning)
* **Target Section**: `### Edge Inference Frameworks` or `### Optimization & Compression`

### Markdown Snippet to Insert:
```markdown
- [Chebyshev-Compressor](https://github.com/Genesis410/chebyshev-compressor) - Constant-state recurrent model compression for resource-constrained microcontrollers (ESP32/STM32). Distills recurrent networks into continuous-time Chebyshev state-spaces with single-header C99 runtime.
```

### Pull Request Submission Details:
* **PR Title**: `Add Chebyshev-Compressor under Optimization & Compression`
* **PR Description**:
  ```markdown
  Hi! Proposing **Chebyshev-Compressor** for the Edge AI / TinyML section.

  - **URL**: https://github.com/Genesis410/chebyshev-compressor
  - **What it does**: Provides continuous-time polynomial state-space compression for streaming time-series sensor processing. It replaces heavy TFLM LSTM arenas with a caller-allocated C99 engine using strictly 96–448 bytes of SRAM and zero heap malloc.
  - **Target MCUs**: ESP32, STM32F4, ARM Cortex-M.
  ```

---

## 3-Minute Step-by-Step Submission Workflow:

You don't need any complex git setup to submit these PRs. You can do each one directly from your browser:

1. Open the target repository on GitHub (e.g., [`AI-in-Transportation-Lab/awesome-tinyml`](https://github.com/AI-in-Transportation-Lab/awesome-tinyml)).
2. Click on `README.md` and click the **Pencil icon (Edit this file)** in the top right.
3. GitHub will automatically create a fork under your account (`Genesis410/awesome-tinyml`).
4. Paste the markdown snippet into the appropriate alphabetical position in the list.
5. At the bottom, enter the PR title and description from this document.
6. Click **Propose changes** and then **Create pull request**.
