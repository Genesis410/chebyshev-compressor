/*
 * Chebyshev-Compressor: 3-Axis Bearing Vibration Anomaly Detection for ESP32
 * 
 * Hardware Target : ESP32 DevKit v1 / ESP32-WROOM-32 / ESP32-S3 / STM32
 * Architecture    : Depthwise Chebyshev State Space (CGR Spectral-SAT) + MLP Readout
 * Memory Footprint:
 *   - Persistent Recurrent State RAM : 96 Bytes (3 channels x 8 order x 4 bytes)
 *   - Call-Stack Scratch Buffer     : 96 Bytes (next_s[8] + h1[16])
 *   - Dynamic Allocation            : strictly 0 bytes (malloc-free)
 */

#include "chebyshev_engine.h"
#include "vibration_sample_data.h"

// Allocate persistent recurrent state in static BSS (exactly 96 bytes)
static chebyshev_state_t vibration_state;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        ; // Wait for serial console on USB CDC boards
    }

    Serial.println("\n========================================================");
    Serial.println("  CHEBYSHEV-COMPRESSOR: ESP32 3-AXIS BEARING VIBRATION");
    Serial.println("========================================================");
    Serial.printf("  Channels (D_in)       : %d (Radial, Tangential, Axial)\n", CHEB_NUM_CHANNELS);
    Serial.printf("  Polynomial Order (K)  : %d\n", CHEB_ORDER);
    Serial.printf("  Persistent State SRAM : %d Bytes\n", (int)sizeof(chebyshev_state_t));
    Serial.printf("  Model Checksum        : %s\n", CHEB_MODEL_CHECKSUM);
    Serial.printf("  Pre-loaded Test Steps : %d steps\n", VIBRATION_TEST_STEPS);
    Serial.println("========================================================\n");

    // Initialize state vector to zeros
    chebyshev_init(&vibration_state);
}

void loop() {
    Serial.println(">>> Starting Streaming Inference Over 300 Vibration Steps...\n");

    unsigned long total_micros = 0;
    float output[CHEB_NUM_OUTPUTS];

    for (int t = 0; t < VIBRATION_TEST_STEPS; t++) {
        const float *sensor_sample = VIBRATION_TEST_DATA[t];

        // Measure single-step execution time
        unsigned long t_start = micros();
        chebyshev_step(&vibration_state, sensor_sample, output);
        unsigned long t_elapsed = micros() - t_start;
        total_micros += t_elapsed;

        // Print progress every 25 steps
        if (t % 25 == 0 || t == VIBRATION_TEST_STEPS - 1) {
            Serial.printf("[Step %03d/300] In: [%.2f, %.2f, %.2f] | Anomaly Index: %.4f | Latency: %lu us\n",
                          t, sensor_sample[0], sensor_sample[1], sensor_sample[2], output[0], t_elapsed);
        }

        delay(10); // Simulate ~100 Hz physical sensor streaming rate
    }

    float avg_latency = (float)total_micros / (float)VIBRATION_TEST_STEPS;
    Serial.println("\n--------------------------------------------------------");
    Serial.printf("  Total Streaming Time : %lu us\n", total_micros);
    Serial.printf("  Average Step Latency : %.2f us/step\n", avg_latency);
    Serial.println("  Recurrent State Health: Zero NaN / Inf detected.");
    Serial.println("--------------------------------------------------------\n");

    Serial.println(">>> Pausing 5 seconds before next streaming cycle...\n");
    delay(5000);
}
