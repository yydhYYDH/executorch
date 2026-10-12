#pragma once

#include <stdint.h>

// QNN 2.41's observed normal U16 conversion mode, accumulator scaled by 2^d.
// d is 8 for the byte-weight path and 16 for the split INT16 path.
// Returns 1 for unsupported scales/shifts, leaving words unchanged.
int hmx_u16_conversion_params(
    float activation_scale,
    float weight_scale,
    float output_scale,
    uint16_t output_zero_point,
    unsigned accumulator_fraction_bits,
    uint32_t words[2]);

// Normal U8 mode: d=0 for A8W8, d=4 for A8W4's x16 accumulator.
int hmx_u8_conversion_params(
    float activation_scale,
    float weight_scale,
    float output_scale,
    uint8_t output_zero_point,
    unsigned accumulator_fraction_bits,
    uint32_t* word);
