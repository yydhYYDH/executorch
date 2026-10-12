#include "conversion.h"

#include <math.h>
#include <string.h>

int hmx_u16_conversion_params(
    float activation_scale,
    float weight_scale,
    float output_scale,
    uint16_t output_zero_point,
    unsigned accumulator_fraction_bits,
    uint32_t words[2]) {
  if (!(activation_scale > 0 && weight_scale > 0 && output_scale > 0) ||
      !isfinite(activation_scale) || !isfinite(weight_scale) ||
      !isfinite(output_scale) ||
      (accumulator_fraction_bits != 8 && accumulator_fraction_bits != 16))
    return 1;
  // QNN's observed per-tensor path rounds the product before dividing in FP32.
  volatile float product = activation_scale * weight_scale;
  volatile float gain = product / output_scale;
  float value = gain;
  if (!isnormal(value))
    return 1;
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits += 2;
  int shift =
      (int)((bits >> 23) & 255) - 127 + 16 + (int)accumulator_fraction_bits;
  if (shift < 8 || shift > 22)
    return 1;
  uint32_t mantissa = ((bits >> 2) & 0x1fffffu) | 0x200000u;
  uint32_t encoded = mantissa ^ 0x200000u;
  uint32_t bias = (uint32_t)output_zero_point * 16 + 8;
  words[0] = ((encoded & 0x7ff) >> 1) | ((encoded & 1) << 31) |
      ((uint32_t)shift << 10) | ((bias & 0xfff) << 19);
  words[1] = ((encoded >> 11) & 0x3ff) | (((encoded >> 21) & 1) << 16) |
      ((bias & 0xff000) << 11);
  return 0;
}

int hmx_u8_conversion_params(
    float activation_scale,
    float weight_scale,
    float output_scale,
    uint8_t output_zero_point,
    unsigned accumulator_fraction_bits,
    uint32_t* word) {
  if (!(activation_scale > 0 && weight_scale > 0 && output_scale > 0) ||
      !isfinite(activation_scale) || !isfinite(weight_scale) ||
      !isfinite(output_scale) ||
      (accumulator_fraction_bits != 0 && accumulator_fraction_bits != 4))
    return 1;
  volatile float product = activation_scale * weight_scale;
  volatile float gain = product / output_scale;
  float value = gain;
  if (!isnormal(value))
    return 1;
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits += 0x800;
  int shift =
      (int)((bits >> 23) & 255) - 127 + 24 - (int)accumulator_fraction_bits;
  if (shift < 0 || shift > 31)
    return 1;
  uint32_t mantissa = ((bits >> 12) & 0x7ff) | 0x800;
  *word = ((mantissa >> 1) & 0x3ff) | ((mantissa & 1) << 31) |
      ((uint32_t)shift << 10) | ((uint32_t)output_zero_point << 23) | 0x400000;
  return 0;
}
