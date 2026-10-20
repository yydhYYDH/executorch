#include "matmul.h"

#include <string.h>

#include "../hmx_quant_params/conversion.h"

static void pack_activation(
    const uint8_t* a, int64_t m, int64_t k, int64_t m0, uint8_t* dst) {
  const int64_t chunks =
      (k + HMX_ANY_K_BLOCK - 1) / HMX_ANY_K_BLOCK;
  const int64_t rows = m - m0 < HMX_ANY_M_TILE ? m - m0 : HMX_ANY_M_TILE;
  memset(dst, 0, (size_t)chunks * HMX_ANY_K_BLOCK_BYTES);
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    uint8_t* tile = dst + chunk * HMX_ANY_K_BLOCK_BYTES;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t bk = 0; bk < HMX_ANY_K_BLOCK; ++bk) {
        const int64_t kk = chunk * HMX_ANY_K_BLOCK + bk;
        tile[(r / 4) * 128 + bk * 4 + r % 4] =
            kk < k ? a[(m0 + r) * k + kk] : 128;
      }
  }
}

static void pack_weight(
    const int8_t* b, int64_t n, int64_t k, int64_t n0, int8_t* dst) {
  const int64_t chunks =
      (k + HMX_ANY_K_BLOCK - 1) / HMX_ANY_K_BLOCK;
  const int64_t cols = n - n0 < HMX_ANY_N_TILE ? n - n0 : HMX_ANY_N_TILE;
  memset(dst, 0, (size_t)chunks * HMX_ANY_K_BLOCK_BYTES);
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    int8_t* tile = dst + chunk * HMX_ANY_K_BLOCK_BYTES;
    for (int64_t bk = 0; bk < HMX_ANY_K_BLOCK; ++bk) {
      const int64_t kk = chunk * HMX_ANY_K_BLOCK + bk;
      for (int64_t c = 0; c < cols; ++c)
        tile[(c / 32) * 1024 + (bk / 4) * 128 +
             (c % 32) * 4 + bk % 4] =
            kk < k ? b[kk * n + n0 + c] : 0;
    }
  }
}

static int prepare_table(
    const int8_t* b, int64_t n, int64_t k, int64_t n0,
    uint32_t* conversion) {
  uint32_t coefficient;
  if (hmx_u8_conversion_params(
          1.0f / 8, 1.0f / 4, 1.0f / 32, 128, 0, &coefficient))
    return 1;
  const int64_t cols = n - n0 < HMX_ANY_N_TILE ? n - n0 : HMX_ANY_N_TILE;
  for (int64_t c = 0; c < HMX_ANY_N_TILE; ++c) {
    int64_t sum = 0;
    for (int64_t kk = 0; kk < k && c < cols; ++kk)
      sum += b[kk * n + n0 + c];
    const int64_t correction = -128 * sum;
    if (correction < INT32_MIN || correction > INT32_MAX)
      return 1;
    conversion[(c / 32) * 64 + c % 32] = coefficient;
    conversion[(c / 32) * 64 + 32 + c % 32] = (uint32_t)correction;
  }
  return 0;
}

int hmx_matmul_u8s8_any_k(
    const uint8_t* a,
    const int8_t* b,
    int64_t m,
    int64_t n,
    int64_t k,
    uint8_t* native_tiles,
    void* workspace) {
  if (!a || !b || !native_tiles || !workspace || m <= 0 || n <= 0 || k <= 0)
    return 1;

  const int64_t chunks =
      (k + HMX_ANY_K_BLOCK - 1) / HMX_ANY_K_BLOCK;
  uint8_t* activation = (uint8_t*)workspace;
  int8_t* weight = (int8_t*)activation + chunks * HMX_ANY_K_BLOCK_BYTES;
  uint8_t* native_output =
      (uint8_t*)weight + chunks * HMX_ANY_K_BLOCK_BYTES;
  uint32_t* conversion = (uint32_t*)(native_output + HMX_ANY_TILE_BYTES);

  for (int64_t n0 = 0; n0 < n; n0 += HMX_ANY_N_TILE) {
    pack_weight(b, n, k, n0, weight);
    if (prepare_table(b, n, k, n0, conversion))
      return 1;
    for (int64_t m0 = 0; m0 < m; m0 += HMX_ANY_M_TILE) {
      pack_activation(a, m, k, m0, activation);
      asm volatile(
          "bias = mxmem2(%0)\n"
          "bias = mxmem2(%1)\n"
          "mxclracc\n"
          :
          : "r"(conversion), "r"((uintptr_t)conversion + 257)
          : "memory");
      for (int64_t chunk = 0; chunk < chunks; ++chunk)
        asm volatile(
            "{ activation.ub = mxmem(%0,%1):deep\n"
            "  weight.b = mxmem(%2,%1):deep }\n"
            :
            : "r"(activation + chunk * HMX_ANY_K_BLOCK_BYTES),
              "r"(HMX_ANY_K_BLOCK_BYTES - 1),
              "r"(weight + chunk * HMX_ANY_K_BLOCK_BYTES)
            : "memory");
      asm volatile(
          "cvt.ub = acc(%0)\n"
          "mxmem(%1,%0) = cvt\n"
          "cvt.ub = acc(%2)\n"
          "mxmem(%3,%0) = cvt\n"
          :
          : "r"(0), "r"(native_output), "r"(0x1001),
            "r"(native_output + 2048)
          : "memory");
      const int64_t tile_index = (n0 / HMX_ANY_N_TILE) +
          (n / HMX_ANY_N_TILE + (n % HMX_ANY_N_TILE != 0)) *
          (m0 / HMX_ANY_M_TILE);
      memcpy(native_tiles + tile_index * HMX_ANY_TILE_BYTES,
             native_output, HMX_ANY_TILE_BYTES);
    }
  }
  return 0;
}

int hmx_matmul_u8s8_any_k32(
    const uint8_t* a,
    const int8_t* b,
    int64_t m,
    int64_t n,
    uint8_t* native_tiles,
    void* workspace) {
  return hmx_matmul_u8s8_any_k(
      a, b, m, n, HMX_ANY_K_BLOCK, native_tiles, workspace);
}
