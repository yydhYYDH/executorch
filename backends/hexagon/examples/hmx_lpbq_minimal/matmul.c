#include "matmul.h"

#include <math.h>
#include <string.h>

enum { M = HMX_LPBQ_M, K = HMX_LPBQ_K, N = HMX_LPBQ_N };

int hmx_matmul_u16s4_lpbq_prepared_tile(
    const uint16_t* a,
    const int8_t* packed_weights,
    const uint32_t* conversion_table,
    float sq,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace) {
  uint16_t* act = workspace;
  int8_t* wt = (int8_t*)workspace + 2048;
  uint16_t* native = (uint16_t*)((uint8_t*)workspace + 10240);
  uint32_t* tab = (uint32_t*)((uint8_t*)workspace + 14336);
  if (!(sq > 0 && isfinite(sq)))
    return 1;
  if (packed_weights != wt)
    memcpy(wt, packed_weights, HMX_LPBQ_WEIGHT_BYTES);
  if (conversion_table != tab)
    memcpy(tab, conversion_table, HMX_LPBQ_TABLE_BYTES);

  asm volatile("mxclracc\n" ::: "memory");
  for (int bank = 0; bank < 4; ++bank) {
    uint8_t* table = (uint8_t*)tab + bank * 257;
    asm volatile("bias = mxmem2(%0)\n" : : "r"(table) : "memory");
  }
  for (int block = 0; block < HMX_LPBQ_BLOCKS; ++block) {
    for (int r = 0; r < M; ++r)
      for (int k = 0; k < 32; ++k)
        act[(r / 2) * 64 + k * 2 + r % 2] = a[r * K + block * 32 + k];
    asm volatile(
        "{ activation.ub = mxmem(%0,%1)\n"
        "weight.b = mxmem(%2,%3):deep }\n"
        :
        : "r"(act), "r"(0x77c), "r"(wt + block * 2048), "r"(0x7ff)
        : "memory");
  }
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
    for (int n = 0; n < N; ++n) {
      int i = r * N + n;
      q[i] = native[(n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2];
      y[i] = ((int)q[i] - zq) * sq;
    }
  return 0;
}

int hmx_matmul_u16s4_lpbq_tile(
    const uint16_t* a,
    const int8_t* w,
    const float* scales,
    float global_weight_scale,
    float sa,
    uint16_t za,
    float sq,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace) {
  int8_t* wt = (int8_t*)workspace + 2048;
  uint32_t* tab = (uint32_t*)((uint8_t*)workspace + 14336);
  if (hmx_lpbq_prepare(w, scales, global_weight_scale, sa, za, sq, zq, wt, tab))
    return 1;
  return hmx_matmul_u16s4_lpbq_prepared_tile(
      a, wt, tab, sq, zq, q, y, workspace);
}
