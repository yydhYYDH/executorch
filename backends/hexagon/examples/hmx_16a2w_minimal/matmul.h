#pragma once

#include <stdint.h>

enum {
  HMX_A16W2_M = 32,
  HMX_A16W2_K = 32,
  HMX_A16W2_N = 64,
  HMX_A16W2_WEIGHT_BYTES = 512,
  HMX_A16W2_TABLE_BYTES = 1024,
  HMX_A16W2_WORKSPACE_BYTES = 16384
};

// Portable preparation; W contains logical int8 values, not packed 2-bit bytes.
// Returns 1 for unsupported parameters, leaving packed_weight/table unchanged.
int hmx_a16w2_prepare(
    const int8_t* w,
    const float weight_scales[64],
    float sa,
    float sq,
    uint16_t za,
    int zw,
    uint16_t zq,
    uint8_t packed_weight[512],
    uint32_t table[256]);

// Row-major A/Q/Y. Caller holds HMX power, resource lease and lock.
// Workspace must be VTCM, 2048-byte aligned and at least 16384 bytes.
void hmx_matmul_u16s2_prepared_tile(
    const uint16_t* a,
    const uint8_t packed_weight[512],
    const uint32_t table[256],
    float sq,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace);

int hmx_matmul_u16s2_tile(
    const uint16_t* a,
    const int8_t* w,
    const float weight_scales[64],
    float sa,
    float sq,
    uint16_t za,
    int zw,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace);
