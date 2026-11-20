
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#define NUM_CHANNELS 14
#define CHEB_ORDER 8
#define HIDDEN_DIM 32
#define NUM_STEPS 100000

/* ================= Chebyshev DSSR Engine ================= */
typedef struct {
    float state[NUM_CHANNELS][CHEB_ORDER];
} chebyshev_state_t;

static float cheb_A[CHEB_ORDER][CHEB_ORDER];
static float cheb_B[CHEB_ORDER];
static float cheb_W1[HIDDEN_DIM][NUM_CHANNELS * CHEB_ORDER];
static float cheb_b1[HIDDEN_DIM];
static float cheb_W2[HIDDEN_DIM];
static float cheb_b2;

static inline float chebyshev_sanitize(float val, float mean, float inv_std, float clamp_val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) return 0.0f;
    float norm = (val - mean) * inv_std;
    memcpy(&bits, &norm, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) return (bits & 0x80000000U) ? -clamp_val : clamp_val;
    if (norm > clamp_val) return clamp_val;
    if (norm < -clamp_val) return -clamp_val;
    return norm;
}

float chebyshev_step(chebyshev_state_t *st, const float *raw_in) {
    float next_s[CHEB_ORDER];
    float h1[HIDDEN_DIM];

    for (int d = 0; d < NUM_CHANNELS; d++) {
        float u = chebyshev_sanitize(raw_in[d], 500.0f, 0.05f, 5.0f);
        for (int i = 0; i < CHEB_ORDER; i++) {
            float sum = cheb_B[i] * u;
            for (int j = 0; j < CHEB_ORDER; j++) {
                sum += cheb_A[i][j] * st->state[d][j];
            }
            next_s[i] = sum;
        }
        for (int i = 0; i < CHEB_ORDER; i++) {
            st->state[d][i] = next_s[i];
        }
    }

    for (int h = 0; h < HIDDEN_DIM; h++) {
        float acc = cheb_b1[h];
        for (int idx = 0; idx < NUM_CHANNELS * CHEB_ORDER; idx++) {
            acc += cheb_W1[h][idx] * st->state[idx / CHEB_ORDER][idx % CHEB_ORDER];
        }
        h1[h] = (acc > 0.0f) ? acc : 0.0f;
    }

    float out = cheb_b2;
    for (int h = 0; h < HIDDEN_DIM; h++) {
        out += cheb_W2[h] * h1[h];
    }
    return out;
}

/* ================= Bare-Metal C99 LSTM (Single Layer, H=32) ================= */
typedef struct {
    float h[HIDDEN_DIM];
    float c[HIDDEN_DIM];
} lstm_state_t;

/* LSTM gates: i, f, g, o. Each has W_ih (32x14), W_hh (32x32), b (32) */
static float lstm_W_ih[4 * HIDDEN_DIM][NUM_CHANNELS];
static float lstm_W_hh[4 * HIDDEN_DIM][HIDDEN_DIM];
static float lstm_b[4 * HIDDEN_DIM];
static float lstm_W_out[HIDDEN_DIM];
static float lstm_b_out;

static inline float sigmoidf_approx(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float lstm_step(lstm_state_t *st, const float *raw_in) {
    float gates[4 * HIDDEN_DIM];

    for (int g = 0; g < 4 * HIDDEN_DIM; g++) {
        float acc = lstm_b[g];
        for (int d = 0; d < NUM_CHANNELS; d++) {
            acc += lstm_W_ih[g][d] * raw_in[d];
        }
        for (int h = 0; h < HIDDEN_DIM; h++) {
            acc += lstm_W_hh[g][h] * st->h[h];
        }
        gates[g] = acc;
    }

    for (int h = 0; h < HIDDEN_DIM; h++) {
        float i_gate = sigmoidf_approx(gates[h]);
        float f_gate = sigmoidf_approx(gates[HIDDEN_DIM + h]);
        float g_gate = tanhf(gates[2 * HIDDEN_DIM + h]);
        float o_gate = sigmoidf_approx(gates[3 * HIDDEN_DIM + h]);

        st->c[h] = f_gate * st->c[h] + i_gate * g_gate;
        st->h[h] = o_gate * tanhf(st->c[h]);
    }

    float out = lstm_b_out;
    for (int h = 0; h < HIDDEN_DIM; h++) {
        out += lstm_W_out[h] * st->h[h];
    }
    return out;
}

int main() {
    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);

    /* Initialize dummy weights */
    for (int i = 0; i < CHEB_ORDER; i++) {
        cheb_B[i] = 0.01f;
        for (int j = 0; j < CHEB_ORDER; j++) cheb_A[i][j] = (i == j) ? 0.95f : 0.01f;
    }
    for (int h = 0; h < HIDDEN_DIM; h++) {
        cheb_b1[h] = 0.01f;
        cheb_W2[h] = 0.01f;
        for (int idx = 0; idx < NUM_CHANNELS * CHEB_ORDER; idx++) cheb_W1[h][idx] = 0.001f;
    }
    for (int g = 0; g < 4 * HIDDEN_DIM; g++) {
        lstm_b[g] = 0.01f;
        for (int d = 0; d < NUM_CHANNELS; d++) lstm_W_ih[g][d] = 0.001f;
        for (int h = 0; h < HIDDEN_DIM; h++) lstm_W_hh[g][h] = 0.001f;
    }
    for (int h = 0; h < HIDDEN_DIM; h++) lstm_W_out[h] = 0.01f;

    float raw_in[NUM_CHANNELS];
    for (int d = 0; d < NUM_CHANNELS; d++) raw_in[d] = 520.0f + d * 5.0f;

    /* Warm-up */
    chebyshev_state_t cheb_st;
    memset(&cheb_st, 0, sizeof(cheb_st));
    for (int step = 0; step < 1000; step++) {
        chebyshev_step(&cheb_st, raw_in);
    }

    lstm_state_t lstm_st;
    memset(&lstm_st, 0, sizeof(lstm_st));
    for (int step = 0; step < 1000; step++) {
        lstm_step(&lstm_st, raw_in);
    }

    /* Benchmark Chebyshev DSSR */
    volatile float cheb_sink = 0.0f;
    QueryPerformanceCounter(&t_start);
    for (int step = 0; step < NUM_STEPS; step++) {
        cheb_sink += chebyshev_step(&cheb_st, raw_in);
    }
    QueryPerformanceCounter(&t_end);
    double cheb_total_s = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;
    double cheb_lat_us = (cheb_total_s / NUM_STEPS) * 1e6;
    double cheb_fps = NUM_STEPS / cheb_total_s;

    /* Benchmark Bare-Metal C99 LSTM */
    volatile float lstm_sink = 0.0f;
    QueryPerformanceCounter(&t_start);
    for (int step = 0; step < NUM_STEPS; step++) {
        lstm_sink += lstm_step(&lstm_st, raw_in);
    }
    QueryPerformanceCounter(&t_end);
    double lstm_total_s = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;
    double lstm_lat_us = (lstm_total_s / NUM_STEPS) * 1e6;
    double lstm_fps = NUM_STEPS / lstm_total_s;

    printf("RESULTS:\n");
    printf("CHEB_LAT_US: %.3f\n", cheb_lat_us);
    printf("CHEB_THROUGHPUT: %.1f\n", cheb_fps);
    printf("LSTM_LAT_US: %.3f\n", lstm_lat_us);
    printf("LSTM_THROUGHPUT: %.1f\n", lstm_fps);
    printf("CHEB_SINK: %f, LSTM_SINK: %f\n", cheb_sink, lstm_sink);

    return 0;
}
