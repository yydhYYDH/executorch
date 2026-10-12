#include "../hmx_quant_params/conversion.h"
#include "matmul.h"

#include <math.h>

int hmx_a16w2_prepare(
    const int8_t* w,
    const float weight_scales[64],
    float sa,
    float sq,
    uint16_t za,
    int zw,
    uint16_t zq,
    uint8_t packed_weight[512],
    uint32_t table[256]) {
  if (!(sa > 0 && sq > 0) || !isfinite(sa) || !isfinite(sq) || zw < -128 ||
      zw > 127)
    return 1;
  float global_scale = 0;
  for (int n = 0; n < 64; ++n) {
    if (!(weight_scales[n] > 0) || !isfinite(weight_scales[n]))
      return 1;
    if (weight_scales[n] > global_scale)
      global_scale = weight_scales[n];
  }
  for (int i = 0; i < 32 * 64; ++i)
    if (w[i] - zw < -2 || w[i] - zw > 1)
      return 1;
  uint32_t coefficients[64][2];
  // QNN's per-channel preparation normalizes by the layer's largest scale.
  volatile float product = global_scale * sa;
  volatile float base_gain = product / sq;
  for (int n = 0; n < 64; ++n) {
    volatile float normalized = weight_scales[n] / global_scale;
    volatile float gain = normalized * base_gain;
    if (hmx_u16_conversion_params(gain, 1, 1, zq, 8, coefficients[n]))
      return 1;
  }
  for (int k = 0; k < 32; k += 16)
    for (int n = 0; n < 64; ++n)
      for (int i = 0; i < 4; ++i) {
        int index = (n / 32) * 256 + (k / 16) * 128 + (n % 32) * 4 + i;
        unsigned value = 0;
        for (int slot = 0; slot < 4; ++slot)
          value |= ((unsigned)(w[(k + i + 4 * slot) * 64 + n] - zw) & 3)
              << (2 * slot);
        packed_weight[index] = value;
      }
  for (int n = 0; n < 64; ++n) {
    int sum = 0;
    for (int k = 0; k < 32; ++k)
      sum += w[k * 64 + n] - zw;
    int num = (int)za * sum;
    int corr = num >= 0 ? (num + 128) / 256 : -((-num + 128) / 256);
    int index = ((n / 32) * 2 + n % 2) * 64 + ((n % 32) / 2) * 2;
    table[index] = coefficients[n][0];
    table[index + 1] = coefficients[n][1];
    table[index + 32] = (uint32_t)(-corr);
    table[index + 33] = 0;
  }
  return 0;
}
