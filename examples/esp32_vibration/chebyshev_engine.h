/*
 * Chebyshev-Compressor: Standalone Strict C99 Inference Engine
 * 
 * Auto-generated: 2026-11-20 09:05:13
 * Architecture: Depthwise Chebyshev State Space (CGR Spectral-SAT) + Readout MLP
 * Compliance: Strict C99 zero-warning (-std=c99 -Wall -Wextra -Werror -pedantic)
 * 
 * Model Footprint:
 *   - Parameters: 495 float32 values
 *   - Estimated Flash/ROM footprint (.rodata): 1,980 bytes (1.93 KB)
 *   - Persistent Recurrent State RAM (.bss/caller): 96 bytes
 *   - Active Call-Stack Scratch (next_s + h1): 96 bytes
 *   - Estimated Active Runtime SRAM Footprint: 192 bytes (State + Stack Scratch; excluding compiler spills/alignment/caller frames)
 *   - Dynamic Allocation: strictly 0 bytes (no malloc/free)
 *   - Re-entrant: Yes (all state passed via context pointer)
 *   - Checksum: 0xDA48E048U
 */

#ifndef CHEBYSHEV_ENGINE_H
#define CHEBYSHEV_ENGINE_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHEB_NUM_CHANNELS                3
#define CHEB_ORDER                       8
#define CHEB_HIDDEN_DIM                  16
#define CHEB_NUM_OUTPUTS                 1
#define CHEB_INPUT_CLAMP                 5.0f
#define CHEB_PERSISTENT_STATE_BYTES      96
#define CHEB_STACK_SCRATCH_BYTES         96
#define CHEB_ESTIMATED_ACTIVE_SRAM_BYTES 192
#define CHEB_TOTAL_SRAM_BYTES            192
#define CHEB_MODEL_CHECKSUM              0xDA48E048U

/* Re-entrant caller-allocated state structure */
typedef struct {
    float state[CHEB_NUM_CHANNELS][CHEB_ORDER];
} chebyshev_state_t;

/* Continuous-to-discrete Chebyshev transition matrix (A_bar) */
static const float CHEB_A[CHEB_ORDER][CHEB_ORDER] = {
    {
        0.841779113f, -0.0968282372f, -0.171131298f, -0.00826921966f, -0.186246857f, 0.0296162609f, -0.188413352f, 0.0168248862f
    },
    {
        -0.299200088f, 0.690826595f, -0.0737565309f, -0.343871385f, 0.0599879734f, -0.353656262f, 0.0981194898f, -0.329649657f
    },
    {
        -0.29649508f, -0.306378245f, 0.662654519f, 0.00638631964f, -0.376492918f, 0.0828096047f, -0.383367985f, 0.0556264147f
    },
    {
        -0.259667397f, -0.268322974f, -0.295443803f, 0.655277073f, 0.110187031f, -0.364697546f, 0.149235219f, -0.337066501f
    },
    {
        -0.244561598f, -0.25271365f, -0.278256744f, -0.324669093f, 0.601469696f, 0.155749112f, -0.413215041f, 0.123039715f
    },
    {
        -0.194450974f, -0.200932682f, -0.221241996f, -0.258144468f, -0.316871554f, 0.593769372f, 0.259425908f, -0.3698771f
    },
    {
        -0.179744601f, -0.18573609f, -0.204509407f, -0.238620937f, -0.292906463f, -0.375507325f, 0.500309646f, 0.246332079f
    },
    {
        -0.12255314f, -0.126638234f, -0.139438242f, -0.162696093f, -0.199708953f, -0.256027728f, -0.340697974f, 0.531590044f
    }
};

/* Input projection vector (B_bar) */
static const float CHEB_B[CHEB_ORDER] = {
    0.158220872f, 0.299200088f, 0.29649508f, 0.259667397f, 0.244561598f, 0.194450974f, 0.179744601f, 0.12255314f
};

/* Baked channel mean for sensor normalization */
static const float CHEB_MEAN[CHEB_NUM_CHANNELS] = {
    0.0356802791f, 0.0528048873f, 0.0165340099f
};

/* Baked channel inverse standard deviation (1 / sigma) */
static const float CHEB_INV_STD[CHEB_NUM_CHANNELS] = {
    1.39955008f, 1.22774553f, 2.60309863f
};

/* Readout MLP Layer 1 Weights: [16][24] */
static const float CHEB_W1[CHEB_HIDDEN_DIM][CHEB_NUM_CHANNELS * CHEB_ORDER] = {
    {
        0.257640153f, -0.0988398194f, -0.0977339819f, 0.213225514f, -0.0592695512f, 0.187898234f, -0.0481876954f, -0.0930406302f, -0.0393973961f, 0.228010163f, 0.0793053582f, 0.0730801374f, 0.067460008f, -0.0115682846f, -0.043187663f, -0.00800666958f, 0.123201713f, 0.025247233f, 0.187032312f, 0.0923612416f, 0.0641176775f, -0.0635857806f, 0.146770746f, -0.253671467f
    },
    {
        0.0986497775f, 0.123920046f, 0.0776082948f, 0.195287839f, 0.0985097587f, -0.0055488646f, -0.0900398716f, -0.244092867f, 0.142690986f, 0.120384902f, -0.222707778f, 0.100826226f, -0.0305153802f, 0.224321276f, -0.0199880246f, 0.212881789f, 0.11865817f, 0.142862022f, 0.106304042f, 0.110334642f, 0.108670667f, -0.124744616f, 0.0055956482f, -0.0753827542f
    },
    {
        0.166070089f, -0.00967341568f, 0.152548641f, 0.207738534f, 0.157865897f, 0.174209788f, 0.0419977419f, 0.0242721867f, -0.0101548154f, 0.233911037f, -0.0814943835f, 0.0544489212f, -0.113860734f, 0.258601129f, 0.0416514315f, 0.222564057f, -0.0596160702f, -0.0971607715f, 0.0350705646f, -0.0895081386f, 0.109592542f, 0.0741417259f, -0.0025634882f, 0.0618006438f
    },
    {
        0.150489315f, -0.203035176f, -0.0312575661f, 0.0709296018f, -0.163627386f, -0.00158865668f, -0.0968972519f, 0.364653826f, 0.0524055026f, 0.175140619f, 0.205954745f, -0.0343070328f, -0.0079039447f, -0.0489646941f, 0.0395993665f, -0.342909068f, 0.0340534449f, 0.146446273f, 0.220150799f, -0.0880793035f, -0.197612748f, 0.0113613009f, -0.0303217918f, 0.0557820126f
    },
    {
        -0.174849346f, -0.0160519592f, 0.361137778f, -0.247918636f, -0.155521482f, -0.127692699f, -0.239169255f, 0.195313677f, 0.357921332f, 0.3684946f, 0.315493673f, 0.0586991422f, -0.0682209283f, -0.0252446011f, -0.247681841f, -0.464643806f, -0.0619996265f, -0.0165013168f, 0.205521271f, -0.161604673f, -0.0055498709f, 0.109809399f, -0.123569489f, 0.304673105f
    },
    {
        0.0579447336f, 0.0507550538f, 0.061719235f, 0.162716463f, -0.0583024062f, 0.249840841f, 0.170819998f, -0.0895272717f, 0.1240739f, -0.0486468002f, 0.0752697438f, -0.116003618f, 0.22663334f, 0.248055458f, -0.0963912457f, -0.12317735f, -0.0701019093f, -0.0822113678f, -0.0161510669f, 0.127316609f, 0.00885760598f, 0.168800607f, 0.0517014563f, -0.114401802f
    },
    {
        -0.146259815f, 0.0544011071f, -0.0565896444f, -0.0369178951f, 0.00344858109f, -0.143163666f, 0.0312417708f, 0.347809732f, 0.315486193f, 0.314093292f, 0.119278461f, -0.0892522559f, 0.0186933763f, -0.228980184f, -0.0163910147f, -0.463485926f, -0.0844782293f, 0.0628723502f, 0.233554214f, -0.102567583f, -0.217178628f, 0.026712941f, 0.0513871275f, 0.414140016f
    },
    {
        0.0193667784f, 0.14170146f, -0.182506189f, 0.195272177f, -0.131804317f, -0.13932392f, 0.146487027f, -0.0453565866f, 0.125966191f, -0.142966047f, 0.154427588f, 0.193581432f, 0.0618049204f, 0.164658278f, -0.145762965f, -0.0450710207f, 0.062128067f, 0.115343302f, -0.10898862f, 0.129885286f, -0.0671784729f, -0.108947083f, 0.0399274528f, -0.162579015f
    },
    {
        -0.13723819f, 0.0723025724f, 0.253758281f, 0.0489089005f, -0.0444601327f, -0.16720444f, -0.186038047f, 0.55555439f, 0.0402875841f, 0.283239543f, 0.505329967f, 0.254822403f, 0.00656801416f, -0.155946761f, -0.249534845f, -0.255465001f, -0.138077259f, -0.200697586f, -0.0688301548f, 0.110314004f, 0.126515552f, 0.118484914f, 0.062185768f, 0.169946671f
    },
    {
        -0.113832228f, -0.125394672f, 0.0673494935f, 0.0793976784f, 0.0423551351f, -0.157029197f, 0.018131949f, -0.1083114f, -0.0401385427f, -0.0315987915f, 0.0101460628f, -0.178971022f, 0.174231172f, -0.086854741f, 0.063776046f, 0.140562892f, 0.082252115f, 0.142944694f, 0.115260243f, 0.149460733f, 0.0430798531f, 0.130367905f, -0.0887230188f, -0.109985977f
    },
    {
        0.112855107f, 0.0215393957f, 0.161060348f, -0.233742639f, 0.0155751789f, 0.0792323947f, -0.0782613531f, 0.379491448f, 0.294340104f, 0.0158512034f, 0.43336311f, -0.012783723f, 0.00218646484f, 0.0154154124f, 0.0949468836f, -0.339650363f, 0.166510224f, -0.193471983f, 0.110730074f, -0.0996551961f, -0.164514855f, -0.00675376691f, -0.0456482768f, 0.282557696f
    },
    {
        0.0888793692f, 0.0562534072f, 0.340444058f, -0.0562468953f, -0.0515382551f, -0.0634428337f, -0.0338167399f, 0.178052321f, 0.167368785f, 0.0353680179f, 0.322538853f, -0.12710847f, 0.241797537f, -0.227288499f, -0.124826007f, -0.328047305f, 0.206643611f, 0.0422530919f, -0.0820155367f, 0.121199146f, -0.240361065f, -0.0295287035f, 0.0751074776f, 0.451173931f
    },
    {
        0.0480613969f, -0.0402119495f, 0.182780266f, 0.252681196f, -0.0129194772f, 0.063355118f, -0.0283073094f, -0.0168568306f, 0.138019726f, 0.195505083f, -0.141779482f, -0.101620458f, 0.00997602567f, -0.06392961f, 0.0465542562f, 0.168997869f, -0.0648337752f, 0.0892731473f, -0.0912502557f, -0.0436231531f, -0.0424186252f, 0.0315699913f, -0.00309793395f, -0.0439015888f
    },
    {
        0.170482486f, 0.200652391f, 0.109814443f, 0.0449258722f, 0.0108992755f, 0.185430467f, -0.070073247f, 0.102943301f, -0.14799121f, -0.101216331f, 0.0304355603f, -0.0359542035f, 0.0868448168f, 0.224927574f, 0.14050132f, 0.246265307f, -0.0382479988f, -0.0478801318f, 0.0268536657f, 0.00813647825f, -0.0391421951f, 0.0366034321f, 0.233178809f, -0.142870188f
    },
    {
        0.114627421f, 0.0948205292f, 0.118205994f, 0.0550662428f, -0.0754028112f, 0.0842410624f, -0.0961371213f, -0.125288963f, 0.190805316f, 0.169188529f, -0.147513747f, 0.165793091f, 0.042325303f, 0.0523198694f, 0.196739495f, -0.0107176565f, -0.0649183542f, 0.128138214f, -0.0527254939f, -0.049000144f, -0.0823104084f, 0.200908184f, 0.0973955393f, -0.033089295f
    },
    {
        0.0631963164f, 0.0859795436f, -0.0579302423f, 0.129772738f, -0.0863604173f, 0.0625426248f, 0.0344486646f, -0.157181337f, 0.245214984f, 0.0986450687f, 0.0175506417f, -0.0642402992f, 0.100158922f, 0.126742199f, 0.178825557f, 0.0125228921f, -0.0336208679f, -0.110808574f, 0.0333474576f, 0.189041942f, 0.133265585f, 0.147532478f, 0.176518932f, 0.114880726f
    }
};

/* Readout MLP Layer 1 Bias: [16] */
static const float CHEB_B1[CHEB_HIDDEN_DIM] = {
    0.00202651555f, 0.0198259521f, -0.0069269021f, 0.0548310503f, 0.229659036f, 0.00690604886f, -0.0066297343f, -0.060408771f,
    0.0621475801f, -0.130421013f, 0.324881196f, 0.334537417f, -0.123696066f, 0.0168002285f, -0.0478045344f, -0.029581096f
};

/* Readout MLP Layer 2 Weights: [1][16] */
static const float CHEB_W2[CHEB_NUM_OUTPUTS][CHEB_HIDDEN_DIM] = {
    {
        0.0231259763f, -0.0728591383f, -0.0454848073f, 0.255621552f, 0.190384209f, -0.0843731239f, 0.0946527869f, -0.249401718f, 0.161955118f, 0.191904396f, 0.175453424f, 0.219666108f, -0.154007643f, -0.00675939862f, 0.147513509f, -0.114409313f
    }
};

/* Readout MLP Layer 2 Bias: [1] */
static const float CHEB_B2[CHEB_NUM_OUTPUTS] = {
    0.0270896908f
};

/*
 * Strict-aliasing-safe IEEE-754 exponent bit-mask check on raw memory.
 * Reads directly as uint32_t before any floating-point arithmetic so LLVM/GCC
 * -ffast-math (-ffinite-math-only) cannot attach nnan/ninf assumptions or
 * elide non-finite detection.
 */
static inline int chebyshev_is_nonfinite_ptr(const float *ptr) {
    uint32_t bits;
    memcpy(&bits, ptr, sizeof(bits));
    return (bits & 0x7F800000U) == 0x7F800000U;
}

/*
 * IEEE-754 bit-level non-finite quarantine and input sanity clamp.
 * Uses a volatile barrier + strict-aliasing-safe memcpy into uint32_t to inspect
 * the 8-bit exponent (0x7F800000U), guaranteeing NaN and +/-Inf are caught
 * deterministically even under GCC/Clang -ffast-math (-ffinite-math-only).
 */
static inline float chebyshev_sanitize(float x) {
    volatile float vx = x;
    float x_copy = vx;
    uint32_t bits;
    memcpy(&bits, &x_copy, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) {
        return 0.0f; /* Catches NaN and +/-Inf deterministically under -ffast-math */
    }
    if (x_copy > CHEB_INPUT_CLAMP) return CHEB_INPUT_CLAMP;
    if (x_copy < -CHEB_INPUT_CLAMP) return -CHEB_INPUT_CLAMP;
    return x_copy;
}

/*
 * Initializes caller-allocated state structure to zero.
 */
static inline void chebyshev_init(chebyshev_state_t *ctx) {
    if (!ctx) return;
    for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {
        for (int k = 0; k < CHEB_ORDER; k++) {
            ctx->state[d][k] = 0.0f;
        }
    }
}

/*
 * Resets state buffer (alias for chebyshev_init).
 */
static inline void chebyshev_reset(chebyshev_state_t *ctx) {
    chebyshev_init(ctx);
}

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
) {
    if (!ctx || !raw_inputs || !outputs) return;

    /* 1. Update Depthwise Chebyshev Recurrent State per channel */
    float next_s[CHEB_ORDER];

    for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {
        float x_sanitized;
        if (chebyshev_is_nonfinite_ptr(&raw_inputs[d])) {
            x_sanitized = 0.0f;
        } else {
            float x_norm = (raw_inputs[d] - CHEB_MEAN[d]) * CHEB_INV_STD[d];
            x_sanitized = chebyshev_sanitize(x_norm);
        }

        for (int i = 0; i < CHEB_ORDER; i++) {
            float acc = 0.0f;
            for (int j = 0; j < CHEB_ORDER; j++) {
                acc += CHEB_A[i][j] * ctx->state[d][j];
            }
            acc += CHEB_B[i] * x_sanitized;
            next_s[i] = acc;
        }

        for (int i = 0; i < CHEB_ORDER; i++) {
            ctx->state[d][i] = next_s[i];
        }
    }

    /* 2. Readout MLP Layer 1 (Linear + ReLU) */
    float h1[CHEB_HIDDEN_DIM];

    for (int i = 0; i < CHEB_HIDDEN_DIM; i++) {
        float acc = CHEB_B1[i];
        int w_idx = 0;
        for (int d = 0; d < CHEB_NUM_CHANNELS; d++) {
            for (int k = 0; k < CHEB_ORDER; k++) {
                acc += CHEB_W1[i][w_idx++] * ctx->state[d][k];
            }
        }
        h1[i] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }

    /* 3. Readout MLP Layer 2 (Linear Output) */
    for (int i = 0; i < CHEB_NUM_OUTPUTS; i++) {
        float acc = CHEB_B2[i];
        for (int j = 0; j < CHEB_HIDDEN_DIM; j++) {
            acc += CHEB_W2[i][j] * h1[j];
        }
        outputs[i] = acc;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CHEBYSHEV_ENGINE_H */
