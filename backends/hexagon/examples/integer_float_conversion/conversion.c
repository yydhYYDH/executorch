#include "conversion.h"

#include <math.h>
#include <string.h>

#ifdef __hexagon__
typedef __fp16 storage_half;
#else
typedef _Float16 storage_half;
#endif

static int valid_parameters(unsigned bits, float scale, int zero_point) {
  return (bits == 4 || bits == 8) && scale > 0 && isfinite(scale) &&
      zero_point >= -(1 << (bits - 1)) && zero_point < (1 << (bits - 1));
}

static int8_t
quantize_value(float x, unsigned bits, float scale, int zero_point) {
  volatile float divided = x / scale;
  float value = divided + zero_point;
  int lower = -(1 << (bits - 1));
  int upper = (1 << (bits - 1)) - 1;
  if (value <= lower)
    return lower;
  if (value >= upper)
    return upper;
  int truncated = (int)value;
  float fraction = value - truncated;
  if (fraction > 0.5f || (fraction == 0.5f && truncated % 2 != 0))
    ++truncated;
  else if (fraction < -0.5f || (fraction == -0.5f && truncated % 2 != 0))
    --truncated;
  return truncated;
}

int quantize_sint(
    const float* x,
    size_t count,
    unsigned bits,
    float scale,
    int zero_point,
    int8_t* q) {
  if (!valid_parameters(bits, scale, zero_point))
    return 1;
  for (size_t i = 0; i < count; ++i)
    if (isnan(x[i]))
      return 1;
  for (size_t i = 0; i < count; ++i)
    q[i] = quantize_value(x[i], bits, scale, zero_point);
  return 0;
}

int dequantize_sint(
    const int8_t* q,
    size_t count,
    unsigned bits,
    float scale,
    int zero_point,
    float* x) {
  if (!valid_parameters(bits, scale, zero_point))
    return 1;
  for (size_t i = 0; i < count; ++i)
    if (q[i] < -(1 << (bits - 1)) || q[i] >= (1 << (bits - 1)))
      return 1;
  for (size_t i = 0; i < count; ++i)
    x[i] = (q[i] - zero_point) * scale;
  return 0;
}

int requantize_sint(
    const int8_t* src,
    size_t count,
    unsigned src_bits,
    float src_scale,
    int src_zero_point,
    unsigned dst_bits,
    float dst_scale,
    int dst_zero_point,
    int8_t* dst) {
  if (!valid_parameters(src_bits, src_scale, src_zero_point) ||
      !valid_parameters(dst_bits, dst_scale, dst_zero_point))
    return 1;
  for (size_t i = 0; i < count; ++i)
    if (src[i] < -(1 << (src_bits - 1)) || src[i] >= (1 << (src_bits - 1)))
      return 1;
  for (size_t i = 0; i < count; ++i) {
    volatile float real = (src[i] - src_zero_point) * src_scale;
    dst[i] = quantize_value(real, dst_bits, dst_scale, dst_zero_point);
  }
  return 0;
}

int pack_int4(const int8_t* src, size_t count, uint8_t* dst) {
  for (size_t i = 0; i < count; ++i)
    if (src[i] < -8 || src[i] > 7)
      return 1;
  for (size_t i = 0; i < count; i += 2) {
    unsigned high = i + 1 < count ? (unsigned)src[i + 1] & 15 : 0;
    dst[i / 2] = ((unsigned)src[i] & 15) | (high << 4);
  }
  return 0;
}

void unpack_int4(const uint8_t* src, size_t count, int8_t* dst) {
  for (size_t i = 0; i < count; ++i) {
    int value = (src[i / 2] >> (4 * (i % 2))) & 15;
    dst[i] = value >= 8 ? value - 16 : value;
  }
}

void fp32_to_fp16_bits(const float* src, size_t count, uint16_t* dst) {
  for (size_t i = 0; i < count; ++i) {
    storage_half value = (storage_half)src[i];
    memcpy(dst + i, &value, sizeof(value));
  }
}

void fp16_bits_to_fp32(const uint16_t* src, size_t count, float* dst) {
  for (size_t i = 0; i < count; ++i) {
    storage_half value;
    memcpy(&value, src + i, sizeof(value));
    dst[i] = (float)value;
  }
}
