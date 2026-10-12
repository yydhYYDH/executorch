#include "matmul.h"

// Per-K-block accumulator: the device-verified u16s4 tile from
// examples/hmx_int4_minimal (SHA of its built .so:
// 9b6161d03d8972fdb92cbec5d48fe54406ff2ed687ff3730175681d4a95107dc for int8,
// int4 tile verified in the same harness). Fixed geometry M=32, K=32, N=64.
enum { TILE_M = 32, TILE_K = 32, TILE_N = 64 };

static void hmx_u16s4_accum(
    const uint16_t* a,
    const int8_t* b,
    uint16_t* q,
    void* workspace) {
  uint16_t* activation = workspace;
  uint8_t* weight = (uint8_t*)workspace + 2048;
  uint16_t* native_output = (uint16_t*)((uint8_t*)workspace + 4096);
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 12288);

  for (int r = 0; r < TILE_M; ++r) {
    for (int k = 0; k < TILE_K; ++k) {
      activation[(r / 2) * 64 + k * 2 + r % 2] = a[r * TILE_K + k];
    }
  }
  for (int k = 0; k < TILE_K; k += 8) {
    for (int n = 0; n < TILE_N; ++n) {
      for (int i = 0; i < 4; ++i) {
        int index = (n / 32) * 512 + (k / 8) * 128 + (n % 32) * 4 + i;
        weight[index] = ((uint8_t)b[(k + i) * TILE_N + n] & 15) |
            (((uint8_t)b[(k + i + 4) * TILE_N + n] & 15) << 4);
      }
    }
  }
  for (int n = 0; n < TILE_N; ++n) {
    int sum = 0;
    for (int k = 0; k < TILE_K; ++k) {
      sum += b[k * TILE_N + n];
    }
    int index = ((n / 32) * 2 + n % 2) * 64 + ((n % 32) / 2) * 2;
    conversion[index] = 0x00404000u;
    conversion[index + 1] = 0x40000000u;
    conversion[index + 32] = (uint32_t)(-128 * sum);
    conversion[index + 33] = 0;
  }

  asm volatile(
      "mxclracc\n"
      "{ activation.ub = mxmem(%0,%1)\n"
      "weight.n = mxmem(%2,%3):deep }\n"
      :
      : "r"(activation), "r"(0x77c), "r"(weight), "r"(0x3ff)
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
  for (int r = 0; r < TILE_M; ++r) {
    for (int n = 0; n < TILE_N; ++n) {
      int index = (n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2;
      q[r * TILE_N + n] = native_output[index];
    }
  }
}

__attribute__((noinline)) void hmx_matmul_u16s4_block_tile(
    const uint16_t* a,
    const int8_t* w,
    const int32_t* scale,
    uint16_t* q,
    float* y,
    void* workspace) {
  uint16_t partial[BHMX_M * BHMX_N];
  uint16_t abuf[BHMX_M * BHMX_BLOCK];
  int acc[BHMX_M * BHMX_N];

  for (int r = 0; r < BHMX_M; ++r) {
    for (int n = 0; n < BHMX_N; ++n) {
      acc[r * BHMX_N + n] = 0;
    }
  }
  for (int blk = 0; blk < BHMX_NBLOCK; ++blk) {
    for (int r = 0; r < BHMX_M; ++r) {
      for (int k = 0; k < BHMX_BLOCK; ++k) {
        abuf[r * BHMX_BLOCK + k] = a[r * BHMX_K + blk * BHMX_BLOCK + k];
      }
    }
    hmx_u16s4_accum(
        abuf,
        w + blk * BHMX_BLOCK * BHMX_N,
        partial,
        workspace);
    for (int r = 0; r < BHMX_M; ++r) {
      for (int n = 0; n < BHMX_N; ++n) {
        acc[r * BHMX_N + n] +=
            ((int)partial[r * BHMX_N + n] - 32768) * scale[n * BHMX_NBLOCK + blk];
      }
    }
  }
  for (int i = 0; i < BHMX_M * BHMX_N; ++i) {
    int v = acc[i] + 32768;
    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
    q[i] = (uint16_t)v;
    y[i] = (v - 32768) * (1.0f / 4.0f);
  }
}
