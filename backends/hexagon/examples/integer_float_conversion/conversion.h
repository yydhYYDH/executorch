#pragma once

#include <stddef.h>
#include <stdint.h>

// Signed logical INT4 uses one int8_t per value; packing is a separate step.
// Invalid parameters/input return 1 without changing the destination.
int quantize_sint(
    const float* x,
    size_t count,
    unsigned bits,
    float scale,
    int zero_point,
    int8_t* q);
int dequantize_sint(
    const int8_t* q,
    size_t count,
    unsigned bits,
    float scale,
    int zero_point,
    float* x);
int requantize_sint(
    const int8_t* src,
    size_t count,
    unsigned src_bits,
    float src_scale,
    int src_zero_point,
    unsigned dst_bits,
    float dst_scale,
    int dst_zero_point,
    int8_t* dst);
int pack_int4(const int8_t* src, size_t count, uint8_t* dst);
void unpack_int4(const uint8_t* src, size_t count, int8_t* dst);
void fp32_to_fp16_bits(const float* src, size_t count, uint16_t* dst);
void fp16_bits_to_fp32(const uint16_t* src, size_t count, float* dst);
