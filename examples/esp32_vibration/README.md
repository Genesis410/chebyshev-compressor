# 3-Axis Bearing Vibration Anomaly Detection on ESP32

This example demonstrates deploying **Chebyshev-Compressor** to an ESP32 microcontroller for real-time streaming vibration anomaly monitoring (bearing defect detection).

## Key Metrics

- **Inputs**: 3 channels (Radial, Tangential, Axial vibration)
- **Polynomial Order**: $K = 8$
- **Persistent Recurrent State RAM**: **96 Bytes** ($3 \times 8 \times 4$ bytes)
- **Active Call-Stack Scratch**: 96 Bytes (`next_s[8]` + `h1[16]`)
- **Estimated Active SRAM Footprint**: **192 Bytes** (zero heap `malloc`)
- **Inference Latency**: ~3.5–5.0 µs per step on 240 MHz ESP32 Xtensa dual-core (see [BENCHMARK_LOG.md](BENCHMARK_LOG.md))

---

## 1. Quickstart (Data & C99 Header Generation)

Generate the trained student model, strict C99 header, and embedded test vectors:

```bash
python prepare_bearing_data.py
```

This generates:
- `chebyshev_engine.h`: Standalone strict C99 inference engine with baked 3-channel normalization.
- `vibration_sample_data.h`: 300 timesteps of continuous rotating machinery degradation (~3.6 KB `.rodata`, safe for default 1.2 MB app partitions).

---

## 2. Flashing to ESP32 (Arduino IDE / PlatformIO)

1. Open `esp32_vibration.ino` in Arduino IDE or PlatformIO.
2. Select your board (e.g. **ESP32 Dev Module** or **ESP32-S3**).
3. Connect your board via USB and click **Upload**.
4. Open the Serial Monitor at **115200 baud** to see real-time streaming anomaly index predictions and microsecond execution latency.

---

## 3. Mathematical Advantage Over FFT / LSTM

| Approach | Memory Required | Execution Overhead | Streaming Suitability |
| :--- | :--- | :--- | :--- |
| **FFT + Windowing** | 1,024–4,096 B buffer | Periodic blocking burst | Latency spikes during FFT calculation |
| **PyTorch / TFLM LSTM** | 16–32 KB SRAM arena | Sequential $\tanh/\sigma$ transcendentals | Exceeds low-end microcontroller RAM |
| **Chebyshev-Compressor** | **96 B persistent state** | Constant linear MAC updates | Seamless per-sample O(1) streaming |
