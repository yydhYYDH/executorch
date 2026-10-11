#pragma once

#include <stdint.h>

enum { HMX_M = 64, HMX_K = 32, HMX_N = 64, HMX_WORKSPACE_BYTES = 12288 };

// Row-major A/B/Q/Y. Workspace must be in VTCM and 2048-byte aligned.
// A: scale=1/8, zero_point=128; B: scale=1/4, zero_point=0.
// Q: scale=1/32, zero_point=128; Y=(Q-128)/32.
void hmx_matmul_u8s8_tile(
    const uint8_t* a,
    const int8_t* b,
    uint8_t* q,
    float* y,
    void* workspace);
