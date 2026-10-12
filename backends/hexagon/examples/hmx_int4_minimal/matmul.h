#pragma once

#include <stdint.h>

enum { HMX_M = 64, HMX_K = 32, HMX_N = 128, HMX_WORKSPACE_BYTES = 16384 };
enum { HMX_A16_M = 32, HMX_A16_N = 64 };

// Row-major A/B/Q/Y. Workspace must be in VTCM and 2048-byte aligned.
// A: scale=1/8, zero_point=128; B: scale=1/4, zero_point=0.
// Q: scale=1/32, zero_point=128; Y=(Q-128)/32.
void hmx_matmul_u8s4_tile(
    const uint8_t* a,
    const int8_t* b,
    uint8_t* q,
    float* y,
    void* workspace);

// Same workspace and ownership requirements. A: scale=1/256, zero_point=32768.
// B: scale=1/4, zero_point=0. Q: scale=1/4, zero_point=32768.
// Q=clamp(floor(sum((A-32768)*B)/256)+32768,0,65535); Y=(Q-32768)/4.
void hmx_matmul_u16s4_tile(
    const uint16_t* a,
    const int8_t* b,
    uint16_t* q,
    float* y,
    void* workspace);

// Runtime quantization parameters; weights are centered before native packing.
// Returns 1 for unsupported conversion mode or centered weight range.
// Invalid parameters leave Q/Y/workspace unchanged. Y=(Q-zq)*sq.
int hmx_matmul_u8s4_tile_quantized(
    const uint8_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint8_t za,
    int weight_zero_point,
    uint8_t zq,
    uint8_t* q,
    float* y,
    void* workspace);

// Runtime quantization parameters; weights are centered before native packing.
// Returns 1 for unsupported conversion mode or centered weight range.
// Invalid parameters leave Q/Y/workspace unchanged. Y=(Q-zq)*sq.
int hmx_matmul_u16s4_tile_quantized(
    const uint16_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    int weight_zero_point,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace);
