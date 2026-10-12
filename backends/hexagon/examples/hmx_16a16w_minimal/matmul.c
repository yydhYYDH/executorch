#include "matmul.h"
#include "../hmx_quant_params/conversion.h"

#include <string.h>

enum { M = HMX_A16W16_M, K = HMX_A16W16_K, N = HMX_A16W16_N };

static void pack_a(const uint16_t* a, uint16_t* out) {
  memset(out, 0, 2048);
  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k)
      out[(r / 2) * 64 + k * 2 + (r % 2)] = a[r * K + k];
}

static void pack_w(const int16_t* b, uint8_t* out) {
  memset(out, 0, 4096);
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      int v = b[k * N + n];
      int low = ((v + 128) & 0xFF) - 128;
      int high = (v - low) / 256;
      int index = (n / 16) * 1024 + (k / 4) * 128 + (n % 16) * 8 + (k % 4);
      out[index] = (uint8_t)low;
      out[index + 4] = (uint8_t)high;
    }
  }
}

// The kernel pre-increments the table pointer by 0x100 before its first read,
// so column block b is written one 256-byte block past the base that is passed
// in.
static void pack_table(
    const int16_t* b,
    uint16_t za,
    const uint32_t coefficient[2],
    uint32_t* out) {
  memset(out, 0, 2048);
  for (int n = 0; n < N; ++n) {
    int sum = 0;
    for (int k = 0; k < K; ++k) {
      int v = b[k * N + n];
      sum += v > 32639 ? 32639 : v;
    }
    long long num = (long long)za * sum;
    long long corr =
        num >= 0 ? (num + 32768) / 65536 : -((-num + 32768) / 65536);
    int index = (n / 16 + 1) * 64 + (n % 16) * 2;
    out[index] = coefficient[0];
    out[index + 1] = coefficient[1];
    out[index + 32] = (uint32_t)(int32_t)(-corr);
    out[index + 33] = 0;
  }
}

int hmx_matmul_a16w16_tile_quantized(
    const uint16_t* a,
    const int16_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    uint16_t zq,
    uint16_t* q,
    void* workspace) {
  uint32_t coefficient[2];
  if (hmx_u16_conversion_params(sa, sw, sq, zq, 16, coefficient))
    return 1;
  uint16_t* act = (uint16_t*)workspace; // 2048 B
  uint8_t* wt = (uint8_t*)workspace + 2048; // 4096 B
  uint32_t* tab = (uint32_t*)((uint8_t*)workspace + 6144); // 2048 B
  uint16_t* native = (uint16_t*)((uint8_t*)workspace + 8192); // 8192 B

  pack_a(a, act);
  pack_w(b, wt);
  pack_table(b, za, coefficient, tab);

  asm volatile("mxclracc\n" ::: "memory");
  for (int block = 0; block < 4; ++block) {
    asm volatile(
        "{ activation.ub = mxmem(%0,%1)\n"
        "weight.b = mxmem(%2,%3):dilate }\n"
        "bias = mxmem2(%4)\n"
        "cvt.uh = acc(%5):2x2\n"
        "mxmem(%6,%7):2x2 = cvt\n"
        :
        : "r"(act),
          "r"(0x77c),
          "r"(wt + 1024 * block),
          "r"(0x3ff),
          "r"(tab + 64 * (block + 1)),
          "r"(0x600),
          "r"(native + 1024 * block),
          "r"(0x700)
        : "memory");
  }

  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n)
      q[r * N + n] =
          native[(n / 16) * 1024 + (r / 2) * 64 + (n % 16) * 2 + r % 2];
  return 0;
}

void hmx_matmul_a16w16_tile(
    const uint16_t* a,
    const int16_t* b,
    uint16_t* q,
    void* workspace) {
  (void)hmx_matmul_a16w16_tile_quantized(
      a,
      b,
      1.4731697774550412e-05f,
      1.7616483091842383e-05f,
      2.2494559743790887e-05f,
      HMX_A16W16_ZA,
      HMX_A16W16_ZQ,
      q,
      workspace);
}
