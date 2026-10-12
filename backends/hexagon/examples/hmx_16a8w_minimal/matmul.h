#pragma once

#include <stdint.h>

enum { HMX_A16W8_M = 32, HMX_A16W8_K = 32, HMX_A16W8_N = 64 };
enum { HMX_A16W8_WORKSPACE_BYTES = 16384 };
enum { HMX_A16W8_TEST_CASES = 12 };

// Requant constants as QNN emitted them for the captured
// use_16a8w_m128_k128_n128 model: sa=1.4731697774550412e-05,
// sw=0.00452736672013998, sq=2.2494559743790887e-05, za=33401, zw=128,
// zq=29520.
#define HMX_A16W8_ZA 33401
#define HMX_A16W8_ZW 128
#define HMX_A16W8_ZQ 29520
// Decoded from the captured coefficient pair, including the half-LSB output
// bias.
#define HMX_A16W8_CONVERSION_MULT 3183617LL
#define HMX_A16W8_CONVERSION_BIAS 472328LL

// Row-major A/B/Q. Workspace must be in VTCM and 2048-byte aligned.
// B holds signed native weights: subtract 128 from QNN's uint8 weight
// container. The accumulator is scaled by 256 before conversion.
void hmx_matmul_a16w8_tile(
    const uint16_t* a,
    const int8_t* b,
    uint16_t* q,
    void* workspace);

// Per-tensor scales and variable activation/output zero points. Native weights
// are already centered at zero. Returns 1 before HMX for unsupported scales.
int hmx_matmul_a16w8_tile_quantized(
    const uint16_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    uint16_t zq,
    uint16_t* q,
    void* workspace);
