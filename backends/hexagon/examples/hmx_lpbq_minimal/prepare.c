#include "matmul.h"

#include <math.h>
#include <string.h>

enum { M = HMX_LPBQ_M, K = HMX_LPBQ_K, N = HMX_LPBQ_N };

int hmx_lpbq_prepare(
    const int8_t* w,
    const float* scales,
    float global_weight_scale,
    float sa,
    uint16_t za,
    float sq,
    uint16_t zq,
    int8_t* wt,
    uint32_t* tab) {
  if (!(sa > 0 && sq > 0 && isfinite(sa) && isfinite(sq)))
    return 1;
  float tile_maximum = 0;
  for (int i = 0; i < N * HMX_LPBQ_BLOCKS; ++i) {
    if (!(scales[i] >= 0 && isfinite(scales[i])))
      return 1;
    if (scales[i] > tile_maximum)
      tile_maximum = scales[i];
  }
  if (global_weight_scale == 0)
    global_weight_scale = tile_maximum / 16;
  if (!isnormal(global_weight_scale) || global_weight_scale < tile_maximum / 16)
    return 1;
  volatile float product = global_weight_scale * sa;
  volatile float global_gain = product / sq;
  memset(tab, 0, 1024);
  for (int n = 0; n < N; ++n) {
    float maximum = 0;
    for (int block = 0; block < HMX_LPBQ_BLOCKS; ++block) {
      float s = scales[n * HMX_LPBQ_BLOCKS + block];
      if (!(s >= 0 && isfinite(s)))
        return 1;
      if (s > maximum)
        maximum = s;
    }
    if (!isnormal(maximum / 16))
      return 1;
    int sum = 0;
    for (int block = 0; block < HMX_LPBQ_BLOCKS; ++block) {
      float ratio = scales[n * HMX_LPBQ_BLOCKS + block] / (maximum / 16);
      int multiple = (int)ratio;
      float fraction = ratio - multiple;
      if (fraction > 0.5 || (fraction == 0.5 && (multiple & 1)))
        ++multiple;
      if (multiple < 1)
        multiple = 1;
      if (multiple > 16)
        multiple = 16;
      for (int k = 0; k < 32; ++k) {
        int v = w[(block * 32 + k) * N + n];
        if (v < -8 || v > 7)
          return 1;
        v *= multiple;
        wt[block * 2048 + (n / 32) * 1024 + (k / 4) * 128 + (n % 32) * 4 +
           k % 4] = (int8_t)v;
        sum += v;
      }
    }
    volatile float normalized_scale = (maximum / 16) / global_weight_scale;
    float gain = normalized_scale * global_gain;
    uint32_t bits;
    memcpy(&bits, &gain, sizeof(bits));
    bits += 2;
    int shift = (int)((bits >> 23) & 255) - 127 + 24;
    if (!isfinite(gain) || shift < 8 || shift > 22)
      return 1;
    uint32_t mult = ((bits >> 2) & 0x1fffffu) | 0x200000u;
    uint32_t encoded = mult ^ 0x200000u;
    uint32_t bias = (uint32_t)zq * 16 + 8;
    int index = ((n / 32) * 2 + n % 2) * 64 + ((n % 32) / 2) * 2;
    tab[index] = ((encoded & 0x7ff) >> 1) | ((encoded & 1) << 31) |
        ((uint32_t)shift << 10) | ((bias & 0xfff) << 19);
    tab[index + 1] = ((encoded >> 11) & 0x3ff) | (((encoded >> 21) & 1) << 16) |
        ((bias & 0xff000) << 11);
    long long numerator = (long long)za * sum;
    long long correction = (-numerator + 127) >> 8;
    tab[index + 32] = (uint32_t)(int32_t)correction;
  }

  return 0;
}
