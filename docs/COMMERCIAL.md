# Commercial Deployment & Technical Benchmarking Services

**Chebyshev-Compressor** provides constant-state recurrent neural network compression for resource-constrained embedded microcontrollers. The core library, CLI, and C99 runtime are open-source and free under the [MIT License](../LICENSE).

For IoT startups, engineering teams, and academic research labs that require guaranteed numerical parity, firmware profiling, or comparative baseline evaluation on their proprietary sensor streams, we offer two standardized, fixed-scope **Productized Services**.

---

## 1. Service Packages

| Package | Deliverables | Target Timeline | Price |
| :--- | :--- | :--- | :--- |
| **Open-Source Core** | Python library, CLI (`cheb`), C99 exporter, comprehensive tests, docs | Instant | **Free (MIT)** |
| **Deployment Sprint** | Model conversion to `chebyshev_engine.h`, parity report, memory audit on reference MCU | 5–7 calendar days | **US$79** |
| **Research Package** | 4–6 page comparative audit report (order $K \in \{4, 8, 16, 32\}$ vs baseline) + scripts | 7–10 calendar days | **US$149** |

---

## 2. Package Details

### A. Deployment Sprint (US$79)

Designed for embedded developers who have a trained PyTorch/ONNX recurrent model and need a drop-in, zero-dependency C99 engine verified against their sensor domain.

- **Client Provides**:
  1. Trained PyTorch (`.pt`) or ONNX model checkpoint.
  2. Sample input sequence (`.npy` or `.csv`) processed through `tools/sanitize_client_data.py`.
- **Deliverables**:
  1. **Optimized `chebyshev_engine.h`**: Standalone, strict C99 single-header engine featuring baked normalization, IEEE-754 exponent bit-mask quarantine, and CRC32 model checksum.
  2. **Numerical Parity Certificate**: Automated verification proving $\max |y_{\mathrm{PyTorch}} - y_{\mathrm{C}}| < 10^{-4}$ across normal, zero, impulse, and extreme outlier signal phases.
  3. **Hardware Memory & Latency Audit**: Accurate breakdown of Persistent Recurrent State (`.bss`), active call-stack scratch, `.rodata` flash footprint, and step latency measured on our Reference Hardware Tier (ESP32 / STM32).
- **Revision Policy**: Includes 1 minor adjustment (e.g. adjusting polynomial order $K$ or normalization clipping threshold) requested within 3 calendar days of delivery.
- **Scope Boundary**: Deliverables are compiled and validated on the **Reference Hardware Tier** (ESP32 DevKit v1, STM32 Nucleo-F401RE). Custom PCB driver debugging, physical sensor wiring, and hardware-in-the-loop bring-up are explicitly out of scope (see [ADR-0009](adr/0009-reference-hardware-and-async-delivery-boundary.md)).

---

### B. Research & Benchmark Package (US$149)

Designed for academic researchers (MSc/PhD) and R&D engineering leads evaluating mathematical model compression against existing TinyML runtimes (TFLite Micro, Edge Impulse).

- **Client Provides**:
  1. Anonymized multi-channel dataset (`.npy` or `.csv`).
  2. Specification of baseline architecture (e.g. LSTM / GRU / MLP).
- **Deliverables**:
  1. **Automated Technical Audit Report (4–6 Pages, PDF)**:
     - Comparative validation table: Held-out validation RMSE/loss and inference throughput vs baseline.
     - Monotonic Order Ablation ($K \in \{4, 8, 16, 32\}$) identifying the Pareto frontier between accuracy, persistent SRAM, and C99 step latency.
     - Numerical stability analysis over long streaming horizons ($10^4$ steps).
     - Publication-ready high-resolution vector charts (SVG / 300 DPI PNG).
  2. **Hermetic Reproducibility Script Bundle**: Standalone Python and C scripts allowing full replication of all figures and tables with a single command.
- **Revision Policy**: Includes 1 minor adjustment (e.g. re-evaluating with an additional window parameter $\theta$) requested within 3 calendar days of delivery.

---

## 3. Communication SLA & Asynchronous Delivery

To maintain maximum technical focus and eliminate cross-timezone scheduling bottlenecks:
- **100% Asynchronous**: All project handoffs, specification clarifications, and artifact delivery occur via **GitHub Private Issue**, **Fiverr Order Messages**, or **Email**. We do not conduct live video/voice calls.
- **Response SLA**: Written responses within **24–48 business hours** (Monday through Friday, UTC+7).

---

## 4. Zero-Retention Data Privacy Guarantee

We adhere to a strict **Zero-Retention Policy** ([ADR-0010](adr/0010-client-data-sanitization-and-zero-retention.md)):
1. **Local Pre-Sanitization**: Clients run `tools/sanitize_client_data.py` locally before transmission. This utility strips physical units, channel labels, and proprietary sensor calibrations, scaling inputs to zero-mean unit-variance float32 arrays (`.npy`). *Note: Clients should retain their own local domain calibration notes (sampling rate $\Delta t$, target scale / threshold) for end-to-end firmware integration.*
2. **Permanent 14-Day Deletion**: All client-submitted model weights (`.pt`) and sanitized tensors (`.npy`) are permanently deleted from developer storage within 14 calendar days post-delivery acceptance.

---

## 5. How to Order

Commercial transactions are conducted with transparent buyer protection:

1. **Option 1: Fiverr Escrow (Recommended for new clients & corporate cards)**
   - Escrow buyer protection is handled securely through Fiverr custom offers.
   - Contact us to receive the direct Gig link: **[support@chebyshev-compressor.dev](mailto:support@chebyshev-compressor.dev)** (or via GitHub Discussions / private issue). Once your sanitized input shapes are reviewed, a custom Fiverr offer is issued for the agreed package ($79 or $149). Funds remain held in escrow until you inspect and accept the delivered C99 header and verification report.
2. **Option 2: Direct Wise International Invoice (For verified GitHub / Open-Source Partners)**
   - For direct corporate engagements or existing open-source contributors, invoices can be issued via Wise (direct international bank transfer / ACH / SEPA with minimal FX fees). Wise invoices provide direct wire settlement without third-party platform commissions.

To initiate an order, submit your sanitized data shape and target microcontroller via a private inquiry or email.
