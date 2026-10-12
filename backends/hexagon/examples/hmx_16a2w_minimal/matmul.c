#include "matmul.h"

#include <string.h>

__attribute__((noinline)) void hmx_matmul_u16s2_prepared_tile(
    const uint16_t* a,
    const uint8_t packed_weight[512],
    const uint32_t table[256],
    float sq,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace) {
  uint16_t* activation = workspace;
  uint8_t* weight = (uint8_t*)workspace + 2048;
  uint16_t* native_output = (uint16_t*)((uint8_t*)workspace + 4096);
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 12288);
  for (int r = 0; r < 32; ++r)
    for (int k = 0; k < 32; ++k)
      activation[(r / 2) * 64 + k * 2 + r % 2] = a[r * 32 + k];
  memcpy(weight, packed_weight, 512);
  memcpy(conversion, table, 1024);
  asm volatile(
      "mxclracc\n"
      "{ activation.ub = mxmem(%0,%1)\n"
      "weight.c = mxmem(%2,%3):deep }\n"
      :
      : "r"(activation), "r"(0x77c), "r"(weight), "r"(0x1ff)
      : "memory");
  for (int group = 0; group < 2; ++group) {
    uint32_t* table = conversion + group * 128;
    asm volatile(
        "bias = mxmem2(%0)\n"
        "cvt.uh = acc(%1):2x2\n"
        "bias = mxmem2(%2)\n"
        "cvt.uh = acc(%3):2x2\n"
        "mxmem(%4,%5) = cvt\n"
        :
        : "r"(table),
          "r"(0x401),
          "r"(table + 64),
          "r"(0x20c),
          "r"(native_output + group * 1024),
          "r"(0x700)
        : "memory");
  }
  for (int r = 0; r < HMX_A16W2_M; ++r) {
    for (int n = 0; n < HMX_A16W2_N; ++n) {
      int index = (n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2;
      uint16_t value = native_output[index];
      q[r * HMX_A16W2_N + n] = value;
      y[r * HMX_A16W2_N + n] = ((int)value - zq) * sq;
    }
  }
}

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
    void* workspace) {
  uint8_t packed_weight[512];
  uint32_t table[256];
  if (hmx_a16w2_prepare(
          w, weight_scales, sa, sq, za, zw, zq, packed_weight, table))
    return 1;
  hmx_matmul_u16s2_prepared_tile(
      a, packed_weight, table, sq, zq, q, y, workspace);
  return 0;
}
