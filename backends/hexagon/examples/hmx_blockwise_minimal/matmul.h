#pragma once

#include <stdint.h>

// Block-wise int4 (LPBQ-style) standalone HMX tile.
//
// A uint16 activation and int4 weight use the device-verified u16s4 HMX tile
// from examples/hmx_int4_minimal as the per-K-block accumulator. The K axis is
// split into BHMX_BLOCK-wide blocks, each accumulated independently in HMX, and
// the block partials are combined with an integer per-(output, block) scale.
//
// Contract (exact integer, all inside this example):
//   acc_b[r,n] = sum_{k in block b} (A[r,k]-32768) * W[b*BLOCK+k,n]
//   T_b[r,n]   = clamp(floor(acc_b[r,n]/256) + 32768, 0, 65535)      // HMX
//   Q[r,n]     = clamp(sum_b (T_b[r,n]-32768) * scale[n,b] + 32768, 0, 65535)
//   Y[r,n]     = (Q[r,n]-32768) / 4
//
// A is uint16 (zero point 32768), W is int4 (-8..7) stored one per int8,
// scale is an integer [N][NBLOCK]. Workspace must be VTCM, 2048-byte aligned,
// and not alias other buffers.
enum {
  BHMX_M = 32,
  BHMX_K = 64,
  BHMX_N = 64,
  BHMX_BLOCK = 32,
  BHMX_NBLOCK = 2,
  BHMX_WORKSPACE_BYTES = 16384,
};

void hmx_matmul_u16s4_block_tile(
    const uint16_t* a,
    const int8_t* w,
    const int32_t* scale,
    uint16_t* q,
    float* y,
    void* workspace);
