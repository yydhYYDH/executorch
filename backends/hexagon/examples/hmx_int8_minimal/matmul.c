#include "matmul.h"
#include "../hmx_quant_params/conversion.h"

__attribute__((noinline)) int hmx_matmul_u8s8_tile_quantized(
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
    void* workspace) {
  uint32_t coefficient;
  if (hmx_u8_conversion_params(sa, sw, sq, zq, 0, &coefficient))
    return 1;
  if (weight_zero_point < -128 || weight_zero_point > 127)
    return 1;
  for (int i = 0; i < 32 * 64; ++i) {
    int value = (int)b[i] - weight_zero_point;
    if (value < -128 || value > 127)
      return 1;
  }
  uint8_t* activation = workspace;
  int8_t* weight = (int8_t*)workspace + 2048;
  uint8_t* native_output = (uint8_t*)workspace + 4096;
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 8192);

  for (int r = 0; r < HMX_M; ++r) {
    for (int k = 0; k < HMX_K; ++k) {
      activation[(r / 4) * 128 + k * 4 + r % 4] = a[r * HMX_K + k];
    }
  }
  for (int k = 0; k < HMX_K; ++k) {
    for (int n = 0; n < HMX_N; ++n) {
      weight[(n / 32) * 1024 + (k / 4) * 128 + (n % 32) * 4 + k % 4] =
          (b[k * HMX_N + n] - weight_zero_point);
    }
  }
  for (int n = 0; n < HMX_N; ++n) {
    int sum = 0;
    for (int k = 0; k < HMX_K; ++k) {
      sum += (b[k * HMX_N + n] - weight_zero_point);
    }
    conversion[(n / 32) * 64 + n % 32] = coefficient;
    conversion[(n / 32) * 64 + 32 + n % 32] = (uint32_t)(-(int)za * sum);
  }

  asm volatile(
      "bias = mxmem2(%0)\n"
      "bias = mxmem2(%1)\n"
      "mxclracc\n"
      "{ activation.ub = mxmem(%2,%3):deep\n"
      "  weight.b = mxmem(%4,%3):deep }\n"
      "cvt.ub = acc(%5)\n"
      "mxmem(%6,%5) = cvt\n"
      "cvt.ub = acc(%7)\n"
      "mxmem(%8,%5) = cvt\n"
      :
      : "r"(conversion),
        "r"((uintptr_t)conversion + 257),
        "r"(activation),
        "r"(2047),
        "r"(weight),
        "r"(0),
        "r"(native_output),
        "r"(0x1001),
        "r"(native_output + 2048)
      : "memory");

  for (int r = 0; r < HMX_M; ++r) {
    for (int n = 0; n < HMX_N; ++n) {
      int native_index = (n / 32) * 2048 + (r / 4) * 128 + (n % 32) * 4 + r % 4;
      uint8_t value = native_output[native_index];
      q[r * HMX_N + n] = value;
      y[r * HMX_N + n] = ((int)value - zq) * sq;
    }
  }
  return 0;
}

void hmx_matmul_u8s8_tile(
    const uint8_t* a,
    const int8_t* b,
    uint8_t* q,
    float* y,
    void* workspace) {
  (void)hmx_matmul_u8s8_tile_quantized(
      a, b, 1.0f / 8, 1.0f / 4, 1.0f / 32, 128, 0, 128, q, y, workspace);
}
