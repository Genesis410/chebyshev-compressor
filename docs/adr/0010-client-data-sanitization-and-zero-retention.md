# Client Data Sanitization and Zero-Retention Privacy Policy

## Context
Industrial IoT startups and embedded engineering teams are hesitant to share proprietary sensor logs, physical unit calibrations, or trained neural weights with independent developers due to corporate trade secret protections and complex legal NDA overhead.

## Decision
We establish a zero-friction, privacy-preserving commercial workflow:
1. **Client-Side Data Sanitization**: Clients execute a standalone, zero-dependency Python script (`tools/sanitize_client_data.py`) locally before sharing. The script strips physical units, channel labels, and timestamp metadata, normalizing tensor streams to zero-mean unit-variance (`float32`).
2. **Zero-Retention Guarantee**: All client-submitted model checkpoints (`.pt`, `.onnx`) and sanitized evaluation tensors (`.npy`, `.csv`) are permanently erased from developer storage within 14 calendar days following deliverable acceptance.
3. **Reproducibility Isolation**: All benchmarking and distillation scripts emitted during commercial services run hermetically without phoning home or transmitting telemetry.

## Consequences
- Bypasses lengthy enterprise NDA legal friction, allowing rapid sprint commencement.
- Guarantees developer never possesses sensitive proprietary manufacturing or physical calibration data.
- Prevents accidental retention or leakage of third-party IP on local developer machines.
