#include "matmul.h"
#include "../hmx_quant_params/conversion.h"

#include <string.h>

enum { M = HMX_A16W8_M, K = HMX_A16W8_K, N = HMX_A16W8_N };

static void pack_a(const uint16_t* a, uint16_t* out) {
  memset(out, 0, 2048);
  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k)
      out[(r / 2) * 64 + k * 2 + (r % 2)] = a[r * K + k];
}

static void pack_w(const int8_t* b, uint8_t* out) {
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      out[(n / 32) * 1024 + (k / 4) * 128 + (n % 32) * 4 + k % 4] =
          (uint8_t)b[k * N + n];
}

// The two tables in each 32-column group select even and odd columns.

static void pack_table(
    const int8_t* b,
    uint16_t za,
    const uint32_t coefficient[2],
    uint32_t* out) {
  memset(out, 0, 2048);
  for (int n = 0; n < N; ++n) {
    int sum = 0;
    for (int k = 0; k < K; ++k)
      sum += b[k * N + n];
    long long num = (long long)za * sum;
    long long corr = num >= 0 ? (num + 128) / 256 : -((-num + 128) / 256);
    int index = ((n / 32) * 2 + n % 2) * 64 + ((n % 32) / 2) * 2;
    out[index] = coefficient[0];
    out[index + 1] = coefficient[1];
    out[index + 32] = (uint32_t)(int32_t)(-corr);
    out[index + 33] = 0;
  }
}

int hmx_matmul_a16w8_tile_quantized(
    const uint16_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    uint16_t zq,
    uint16_t* q,
    void* workspace) {
  uint32_t coefficient[2];
  if (hmx_u16_conversion_params(sa, sw, sq, zq, 8, coefficient))
    return 1;
  uint16_t* act = (uint16_t*)workspace; // 2048 B
  uint8_t* wt = (uint8_t*)workspace + 2048;
  uint16_t* native = (uint16_t*)((uint8_t*)workspace + 4096);
  uint32_t* tab = (uint32_t*)((uint8_t*)workspace + 12288);

  pack_a(a, act);
  pack_w(b, wt);
  pack_table(b, za, coefficient, tab);

  asm volatile("mxclracc\n" ::: "memory");
  // Low address bits select the four conversion banks; mxclracc does not reset
  // them.
  for (int bank = 0; bank < 4; ++bank) {
    uint8_t* table = (uint8_t*)tab + bank * 257;
    asm volatile("bias = mxmem2(%0)\n" : : "r"(table) : "memory");
  }
  asm volatile(
      "{ activation.ub = mxmem(%0,%1)\n"
      "weight.b = mxmem(%2,%3):deep }\n"
      :
      : "r"(act), "r"(0x77c), "r"(wt), "r"(0x7ff)
      : "memory");
  for (int group = 0; group < 2; ++group) {
    asm volatile(
        "cvt.uh = acc(%0):2x2\n"
        "cvt.uh = acc(%1):2x2\n"
        "mxmem(%2,%3) = cvt\n"
        :
        : "r"(group ? 0x2401 : 0x401),
          "r"(group ? 0x320c : 0x120c),
          "r"(native + 1024 * group),
          "r"(0x700)
        : "memory");
  }

  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n)
      q[r * N + n] =
          native[(n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2];
  return 0;
}

void hmx_matmul_a16w8_tile(
    const uint16_t* a,
    const int8_t* b,
    uint16_t* q,
    void* workspace) {
  (void)hmx_matmul_a16w8_tile_quantized(
      a,
      b,
      1.4731697774550412e-05f,
      0.00452736672013998f,
      2.2494559743790887e-05f,
      HMX_A16W8_ZA,
      HMX_A16W8_ZQ,
      q,
      workspace);
}
