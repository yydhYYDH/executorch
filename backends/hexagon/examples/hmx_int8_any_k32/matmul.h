#pragma once

#include <stdint.h>

enum {
  HMX_ANY_M_TILE = 64,
  HMX_ANY_N_TILE = 64,
  HMX_ANY_K_BLOCK = 32,
  HMX_ANY_K_BLOCK_BYTES = 2048,
  HMX_ANY_TILE_BYTES = 4096,
  HMX_ANY_TABLE_BYTES = 512
};

#define HMX_ANY_WORKSPACE_BYTES(k) \
  (2 * (((k) + HMX_ANY_K_BLOCK - 1) / HMX_ANY_K_BLOCK) * \
       HMX_ANY_K_BLOCK_BYTES + \
   HMX_ANY_TILE_BYTES + HMX_ANY_TABLE_BYTES)

// A is M x K and B is K x N, both row-major. K may be any positive value.
// The last K block is zero-padded to 32: A uses 128 and B uses 0.
// Returns 1 for invalid K or when the conversion-table correction does not
// fit int32.
// Output is a sequence of native U8 tiles, one per 64x64 block:
// tile index = n_tile + n_tiles * m_tile, tile bytes = 4096.
// M and N may be arbitrary positive values.
int hmx_matmul_u8s8_any_k(
    const uint8_t* a,
    const int8_t* b,
    int64_t m,
    int64_t n,
    int64_t k,
    uint8_t* native_tiles,
    void* workspace);
