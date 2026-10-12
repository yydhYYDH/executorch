#pragma once

#include <stdint.h>

enum {
  HMX_LPBQ_M = 32,
  HMX_LPBQ_K = 128,
  HMX_LPBQ_N = 64,
  HMX_LPBQ_BLOCK = 32,
  HMX_LPBQ_BLOCKS = 4,
  HMX_LPBQ_WEIGHT_BYTES = 8192,
  HMX_LPBQ_TABLE_BYTES = 1024,
  HMX_LPBQ_WORKSPACE_BYTES = 16384,
  HMX_LPBQ_TEST_CASES = 12,
};

// Row-major A/W/Q/Y; W contains signed INT4 values, one per byte.
// scales[n,block] are nonnegative FP32 scales. Workspace is 2048-aligned VTCM.
// global_weight_scale is max(scales)/16 over the whole layer, or 0 for this
// tile. Returns 1 for unsupported parameters, before touching HMX.
// Q[r,n] = clamp(zq + round(gain[n] * (acc/256 + correction[n])), 0, 65535)
//   gain[n]       = (max_block scales[n,:] / 16) * sa / sq
//   correction[n] = -za * sum_k (w[k,n] * multiple[n,block]) / 256
// The 4-bit multiplier multiple[n,block] = round(16 * scales[n,block] /
// max_block scales[n,:]) is folded into the stored weight byte, so the hardware
// needs no separate block-scale operand.
int hmx_matmul_u16s4_lpbq_tile(
    // Quantized activation, row-major A[r*128 + k]. Integer, not float:
    // A_q = clamp(round(A_float / sa) + za, 0, 65535). Caller quantizes.
    const uint16_t* a,
    // Quantized weight, row-major W[(block*32 + k)*64 + n]. One signed INT4
    // value (-8..7) per byte, dequantized bounds sc[n,block]. Caller quantizes.
    const int8_t* w,
    // Per-(output channel, K block) FP32 weight scale, scales[n*4 + block].
    // Block size is 32 along K, so 4 blocks. The hardware has no scale operand;
    // these are consumed during preparation, not passed to HMX.
    const float* scales,
    // Anchor for the 4-bit multipliers: max(scales)/16 over the whole layer, or
    // 0 to derive it from this tile. It cancels algebraically in gain[n] and
    // only bounds normalized_scale to <= 1.
    float global_weight_scale,
    // Activation scale and zero point. za is an integer in the quantized domain.
    float sa,
    uint16_t za,
    // Output scale and zero point. zq is used twice: packed into the conversion
    // table bias (zq*16 + 8) and applied when dequantizing y.
    float sq,
    uint16_t zq,
    // Quantized output, row-major Q[r*64 + n].
    uint16_t* q,
    // Dequantized output, y[i] = (q[i] - zq) * sq. Produced by this operator,
    // not by HMX.
    float* y,
    // VTCM, 2048-byte aligned, HMX_LPBQ_WORKSPACE_BYTES. Layout below.
    void* workspace);

// Portable preparation for either host or DSP. No HMX or VTCM required.
int hmx_lpbq_prepare(
    const int8_t* w,
    const float* scales,
    float global_weight_scale,
    float sa,
    uint16_t za,
    float sq,
    uint16_t zq,
    int8_t* packed_weights,
    uint32_t* conversion_table);

// Prepared weights/table are copied to VTCM; their quantization must match
// sq/zq.
int hmx_matmul_u16s4_lpbq_prepared_tile(
    const uint16_t* a,
    const int8_t* packed_weights,
    const uint32_t* conversion_table,
    float sq,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace);
