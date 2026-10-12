#pragma once

#include <stdint.h>

enum { HMX_FP16A8W_M = 32, HMX_FP16A8W_K = 32, HMX_FP16A8W_N = 64 };
enum { HMX_FP16A8W_WORKSPACE_BYTES = 16384, HMX_FP16A8W_TEST_CASES = 10 };

// Row-major A/B/Q/Y. B is signed int8 with zero point zero; scale is per column.
// Q holds FP16 output bits; Y is their FP32 widening. Workspace is 2048-aligned VTCM.
// Call with HMX powered and locked. Scales and dequantized weights round to FP16.
void hmx_matmul_fp16a8w_tile(
    const __fp16* a,
    const int8_t* b,
    const float* scale,
    uint16_t* q,
    float* y,
    void* workspace);
