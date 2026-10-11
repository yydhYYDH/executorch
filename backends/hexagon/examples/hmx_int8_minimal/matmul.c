#include "matmul.h"

__attribute__((noinline)) void hmx_matmul_u8s8_tile(
    const uint8_t* a,
    const int8_t* b,
    uint8_t* q,
    float* y,
    void* workspace) {
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
          b[k * HMX_N + n];
    }
  }
  for (int n = 0; n < HMX_N; ++n) {
    int sum = 0;
    for (int k = 0; k < HMX_K; ++k) {
      sum += b[k * HMX_N + n];
    }
    // Device-verified integer conversion: unit multiplier, output zero point
    // 128.
    conversion[(n / 32) * 64 + n % 32] = 0x40006000u;
    conversion[(n / 32) * 64 + 32 + n % 32] = (uint32_t)(-128 * sum);
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
      y[r * HMX_N + n] = ((int)value - 128) * (1.0f / 32.0f);
    }
  }
}
