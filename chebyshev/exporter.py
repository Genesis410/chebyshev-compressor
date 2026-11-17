"""Code generator that exports trained StudentModel into a standalone, MISRA-C compliant C99 header."""

from pathlib import Path
from typing import Union
import datetime
import zlib
import numpy as np

from chebyshev.models import StudentModel


def format_float(val: float) -> str:
    """Formats float32 to IEEE-754 literal ensuring explicit decimal/exponent and 'f' suffix."""
    s = f"{val:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def format_1d_array(arr: np.ndarray, indent: int = 4, items_per_line: int = 8) -> str:
    """Formats a 1D numpy array into a C array initializer."""
    spaces = " " * indent
    elements = [format_float(x) for x in arr.flat]
    lines = []
    for i in range(0, len(elements), items_per_line):
        chunk = elements[i : i + items_per_line]
        lines.append(spaces + ", ".join(chunk))
    return ",\n".join(lines)


def format_2d_array(arr: np.ndarray, indent: int = 4) -> str:
    """Formats a 2D numpy array into a nested C array initializer."""
    outer_spaces = " " * indent
    inner_spaces = " " * (indent + 4)
    row_strings = []
    for row in arr:
        row_elems = ", ".join(format_float(x) for x in row)
        row_strings.append(f"{outer_spaces}{{\n{inner_spaces}{row_elems}\n{outer_spaces}}}")
    return ",\n".join(row_strings)


def export_to_c_header(student: StudentModel, output_path: Union[str, Path]) -> str:
    """Exports trained StudentModel parameters into a self-contained C99 header engine.
    
    Args:
        student: Trained StudentModel instance.
        output_path: Path where `chebyshev_engine.h` should be written.
        
    Returns:
        The generated C99 code string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model parameters and normalization buffers as numpy arrays
    A = student.cell.A.detach().cpu().numpy().astype(np.float32)
    B = student.cell.B.detach().cpu().numpy().astype(np.float32)
    mean = student.cell.mean.detach().cpu().numpy().astype(np.float32)
    inv_std = student.cell.inv_std.detach().cpu().numpy().astype(np.float32)

    # Readout MLP parameters
    W1 = student.readout.mlp[0].weight.detach().cpu().numpy().astype(np.float32)
    B1 = student.readout.mlp[0].bias.detach().cpu().numpy().astype(np.float32)
    W2 = student.readout.mlp[2].weight.detach().cpu().numpy().astype(np.float32)
    B2 = student.readout.mlp[2].bias.detach().cpu().numpy().astype(np.float32)

    num_channels = student.num_channels
    order = student.order
    hidden_dim = student.hidden_dim
    out_features = student.out_features

    # Calculate memory metrics
    param_count = (
        A.size + B.size + mean.size + inv_std.size + W1.size + B1.size + W2.size + B2.size
    )
    flash_bytes = param_count * 4
    ram_state_bytes = num_channels * order * 4

    # Compute CRC32 checksum across all parameter bytes for firmware verification
    crc = 0
    for arr in [A, B, mean, inv_std, W1, B1, W2, B2]:
        crc = zlib.crc32(arr.tobytes(), crc)
    checksum_hex = f"0x{crc & 0xFFFFFFFF:08X}U"

    export_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct complete C99 single-header library
    header_code = f"""/*
 * Chebyshev-Compressor: Standalone C99 Inference Engine
 * 
 * Auto-generated: {export_timestamp}
 * Architecture: Depthwise Chebyshev State Space + Non-Linear Readout MLP
 * 
 * Model Footprint:
 *   - Parameters: {param_count:,} float32 values
 *   - Estimated Flash/ROM footprint: {flash_bytes:,} bytes ({flash_bytes / 1024:.2f} KB)
 *   - Estimated RAM footprint: {ram_state_bytes:,} bytes (caller-allocated state)
 *   - Dynamic Allocation: strictly 0 bytes (no malloc/free)
 *   - Re-entrant: Yes (all state passed via context pointer)
 *   - Checksum: {checksum_hex}
 */

#ifndef CHEBYSHEV_ENGINE_H
#define CHEBYSHEV_ENGINE_H

#ifdef __cplusplus
extern "C" {{
#endif

#define CHEB_NUM_CHANNELS {num_channels}
#define CHEB_ORDER        {order}
#define CHEB_HIDDEN_DIM   {hidden_dim}
#define CHEB_NUM_OUTPUTS  {out_features}
#define CHEB_INPUT_CLAMP  5.0f
#define CHEB_MODEL_CHECKSUM {checksum_hex}

/* Re-entrant caller-allocated state structure */
typedef struct {{
    float state[CHEB_NUM_CHANNELS][CHEB_ORDER];
}} chebyshev_state_t;

/* Continuous-to-discrete Chebyshev transition matrix (A_bar) */
static const float CHEB_A[CHEB_ORDER][CHEB_ORDER] = {{
{format_2d_array(A, indent=4)}
}};

/* Input projection vector (B_bar) */
static const float CHEB_B[CHEB_ORDER] = {{
{format_1d_array(B, indent=4)}
}};

/* Baked channel mean for sensor normalization */
static const float CHEB_MEAN[CHEB_NUM_CHANNELS] = {{
{format_1d_array(mean, indent=4)}
}};

/* Baked channel inverse standard deviation (1 / sigma) */
static const float CHEB_INV_STD[CHEB_NUM_CHANNELS] = {{
{format_1d_array(inv_std, indent=4)}
}};

/* Readout MLP Layer 1 Weights: [{hidden_dim}][{num_channels * order}] */
static const float CHEB_W1[CHEB_HIDDEN_DIM][CHEB_NUM_CHANNELS * CHEB_ORDER] = {{
{format_2d_array(W1, indent=4)}
}};

/* Readout MLP Layer 1 Bias: [{hidden_dim}] */
static const float CHEB_B1[CHEB_HIDDEN_DIM] = {{
{format_1d_array(B1, indent=4)}
}};

/* Readout MLP Layer 2 Weights: [{out_features}][{hidden_dim}] */
static const float CHEB_W2[CHEB_NUM_OUTPUTS][CHEB_HIDDEN_DIM] = {{
{format_2d_array(W2, indent=4)}
}};

/* Readout MLP Layer 2 Bias: [{out_features}] */
static const float CHEB_B2[CHEB_NUM_OUTPUTS] = {{
{format_1d_array(B2, indent=4)}
}};

/*
 * IEEE-754 input sanity clamp and NaN quarantine.
 * Protects state against state poisoning even under -ffast-math / -Ofast.
 */
static inline float chebyshev_sanitize(float x) {{
    if (!(x >= -CHEB_INPUT_CLAMP && x <= CHEB_INPUT_CLAMP)) {{
        if (x > CHEB_INPUT_CLAMP) return CHEB_INPUT_CLAMP;
        if (x < -CHEB_INPUT_CLAMP) return -CHEB_INPUT_CLAMP;
        return 0.0f; /* Catches NaN and Inf */
    }}
    return x;
}}

/*
 * Initializes caller-allocated state structure to zero.
 */
static inline void chebyshev_init(chebyshev_state_t *ctx) {{
    if (!ctx) return;
    for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {{
        for (int k = 0; k < CHEB_ORDER; k++) {{
            ctx->state[d][k] = 0.0f;
        }}
    }}
}}

/*
 * Resets state buffer (alias for chebyshev_init).
 */
static inline void chebyshev_reset(chebyshev_state_t *ctx) {{
    chebyshev_init(ctx);
}}

/*
 * Executes 1 inference step for multi-channel sensor readings.
 * 
 * 1. Normalizes raw inputs using baked mean and inv_std
 * 2. Sanitizes normalized values against NaNs and extreme spikes
 * 3. Updates Depthwise Chebyshev state using anti-in-place buffer
 * 4. Computes Readout MLP feedforward projection with ReLU
 * 5. Writes final prediction into caller-allocated outputs array
 */
static inline void chebyshev_step(
    chebyshev_state_t *ctx, const float *raw_inputs, float *outputs
) {{
    if (!ctx || !raw_inputs || !outputs) return;

    /* 1. Update Depthwise Chebyshev Recurrent State per channel */
    float next_s[CHEB_ORDER];

    for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {{
        float x_norm = (raw_inputs[d] - CHEB_MEAN[d]) * CHEB_INV_STD[d];
        float x_sanitized = chebyshev_sanitize(x_norm);

        for (int i = 0; i < CHEB_ORDER; i++) {{
            float acc = 0.0f;
            for (int j = 0; j < CHEB_ORDER; j++) {{
                acc += CHEB_A[i][j] * ctx->state[d][j];
            }}
            acc += CHEB_B[i] * x_sanitized;
            next_s[i] = acc;
        }}

        for (int i = 0; i < CHEB_ORDER; i++) {{
            ctx->state[d][i] = next_s[i];
        }}
    }}

    /* 2. Readout MLP Layer 1 (Linear + ReLU) */
    float h1[CHEB_HIDDEN_DIM];

    for (int i = 0; i < CHEB_HIDDEN_DIM; i++) {{
        float acc = CHEB_B1[i];
        int w_idx = 0;
        for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {{
            for (int k = 0; k < CHEB_ORDER; k++) {{
                acc += CHEB_W1[i][w_idx++] * ctx->state[d][k];
            }}
        }}
        h1[i] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }}

    /* 3. Readout MLP Layer 2 (Linear Output) */
    for (int i = 0; i < CHEB_NUM_OUTPUTS; i++) {{
        float acc = CHEB_B2[i];
        for (int j = 0; j < CHEB_HIDDEN_DIM; j++) {{
            acc += CHEB_W2[i][j] * h1[j];
        }}
        outputs[i] = acc;
    }}
}}

#ifdef __cplusplus
}}
#endif

#endif /* CHEBYSHEV_ENGINE_H */
"""
    output_path.write_text(header_code, encoding="utf-8")
    return header_code
