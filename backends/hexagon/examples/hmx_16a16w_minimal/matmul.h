#pragma once

#include <stdint.h>

enum { HMX_A16W16_M = 32, HMX_A16W16_K = 32, HMX_A16W16_N = 64 };
enum { HMX_A16W16_WORKSPACE_BYTES = 16384 };

// Requant constants as QNN emitted them for the captured
// use_16a16w_m128_k128_n128 model: sa=1.4731697774550412e-05,
// sw=1.7616483091842383e-05, sq=2.2494559743790887e-05, za=33401, zq=29520.
#define HMX_A16W16_ZA 33401
#define HMX_A16W16_ZQ 29520
#define HMX_A16W16_MULT 0.756091655

// Row-major A/B/Q. Workspace must be in VTCM and 2048-byte aligned.
// Q = clamp(zq + round(MULT * (floor(acc/65536) + corr[n])), 0, 65535)
void hmx_matmul_a16w16_tile(
    const uint16_t* a,
    const int16_t* b,
    uint16_t* q,
    void* workspace);

// Per-tensor scales and variable activation/output zero points. Native weights
// are already centered at zero. Returns 1 before HMX for unsupported scales.
int hmx_matmul_a16w16_tile_quantized(
    const uint16_t* a,
    const int16_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    uint16_t zq,
    uint16_t* q,
    void* workspace);
