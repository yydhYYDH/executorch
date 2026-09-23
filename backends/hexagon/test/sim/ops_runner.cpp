/* Runs the vendored htp-ops kernels inside hexagon-sim.
 *
 * Built as a plain Hexagon shared object with a main(), which the SDK's
 * run_main_on_hexagon_sim loads into the simulated QuRT. Undefined qurt_*
 * symbols are deliberate: the simulated RTOS resolves them at load time.
 *
 * Every descriptor here carries the numbers the emitters produce, so this
 * checks the emitted encoding against the kernel that consumes it. Results are
 * printed as raw fp16 bit patterns so the comparison against torch is on bits.
 */
#include <stdint.h>
#include <stdio.h>

#include "region_ops.h"

extern "C" int htp_ops_layer_norm(uint8_t *dst, uint8_t *src, uint8_t *gamma,
                                  uint8_t *beta, int32_t outterSize,
                                  int32_t innerSize, float epsilon,
                                  int32_t RMSNorm);
extern "C" int htp_ops_batch_matmul(uint8_t *dst, uint8_t *src0, uint8_t *src1,
                                    uint8_t *iter0, uint8_t *iter1,
                                    uint8_t *iter2, int32_t bytes,
                                    uint8_t *param);
extern "C" int htp_ops_raster_blit(uint8_t *dst, uint8_t **src, int src_number,
                                   uint8_t *region, int32_t regionCount,
                                   int32_t bytes);

#define ROWS 3
#define INNER 8
#define M 4
#define K 8
#define N 3

static void print_bits(const char *tag, const _Float16 *v, int n) {
  printf("%s", tag);
  for (int i = 0; i < n; ++i) printf(" %04x", ((const uint16_t *)v)[i]);
  printf("\n");
}

/* The buffers below are declared 128-byte aligned because the kernels read and
 * write them a vector at a time, and an HVX access wants that alignment. Without
 * it the array lands wherever the linker puts it and the result depends on the
 * size of the whole shared object rather than on the kernel -- a wrong answer
 * rather than a crash, from an edit that says nothing about memory. See
 * backends/hexagon/test/README.md. */

/* The fused norm. gamma and beta are fp32 even though src and dst are fp16,
 * which is only visible in layer_norm_ops.cc and reads garbage if missed. */
static void run_norm(void) {
  static _Float16 src[ROWS * INNER] __attribute__((aligned(128)));
  static _Float16 dst[ROWS * INNER] __attribute__((aligned(128)));
  static float gamma[INNER] __attribute__((aligned(128)));
  static float beta[INNER] __attribute__((aligned(128)));
  for (int i = 0; i < ROWS * INNER; ++i)
    src[i] = (_Float16)((float)((i * 7) % 13 - 6) * 0.5f);
  for (int i = 0; i < INNER; ++i) {
    gamma[i] = 1.0f + 0.125f * (float)i;
    beta[i] = 0.25f * (float)(i % 3);
  }
  print_bits("IN", src, ROWS * INNER);
  htp_ops_layer_norm((uint8_t *)dst, (uint8_t *)src, NULL, NULL, ROWS, INNER,
                     1e-5f, 1);
  print_bits("RMS", dst, ROWS * INNER);
  htp_ops_layer_norm((uint8_t *)dst, (uint8_t *)src, (uint8_t *)gamma,
                     (uint8_t *)beta, ROWS, INNER, 1e-5f, 0);
  print_bits("LN", dst, ROWS * INNER);
}

/* Operands are small integers, so products and sums are exact and agreement
 * does not depend on the order either side accumulates in. */
static void run_mm(void) {
  static _Float16 a[M * K] __attribute__((aligned(128)));
  static _Float16 b[K * N] __attribute__((aligned(128)));
  static _Float16 c[M * N] __attribute__((aligned(128)));
  for (int i = 0; i < M * K; ++i) a[i] = (_Float16)((float)((i * 5) % 7 - 3));
  for (int i = 0; i < K * N; ++i) b[i] = (_Float16)((float)((i * 3) % 5 - 2));
  print_bits("A", a, M * K);
  print_bits("B", b, K * N);

  HtpOpsLoopParam lp = {};
  lp.loopNumber = 1;
  lp.sizeXYZ[0] = M;
  lp.sizeXYZ[1] = K;
  lp.sizeXYZ[2] = N;
  lp.dstStrideXYZ[0] = N * 2;
  lp.dstStrideXYZ[1] = 0;
  lp.dstStrideXYZ[2] = 2;
  lp.src0StrideXYZ[0] = K * 2;
  lp.src0StrideXYZ[1] = 2;
  lp.src0StrideXYZ[2] = 0;
  lp.src1StrideXYZ[0] = 0;
  lp.src1StrideXYZ[1] = N * 2;
  lp.src1StrideXYZ[2] = 2;
  lp.outputElementSize = M * N;
  lp.input0Size = M * K;
  lp.input1Size = K * N;

  htp_ops_batch_matmul((uint8_t *)c, (uint8_t *)a, (uint8_t *)b, NULL, NULL,
                       NULL, 2, (uint8_t *)&lp);
  print_bits("MM", c, M * N);
}

/* A 2x3 transpose, which is the region _emit_permute produces. */
static void run_blit(void) {
  static _Float16 src[6] __attribute__((aligned(128)));
  static _Float16 dst[6] __attribute__((aligned(128)));
  for (int i = 0; i < 6; ++i) src[i] = (_Float16)((float)(i * 2 - 5));
  print_bits("SRC", src, 6);

  HtpOpsRasterRegion region = {};
  region.srcIndex = 0;
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size[0] = 1;
  region.size[1] = 2;
  region.size[2] = 3;
  region.srcStride[0] = 6;
  region.srcStride[1] = 3;
  region.srcStride[2] = 1;
  region.dstStride[0] = 6;
  region.dstStride[1] = 1;
  region.dstStride[2] = 2;

  uint8_t *sources[1] = {(uint8_t *)src};
  htp_ops_raster_blit((uint8_t *)dst, sources, 1, (uint8_t *)&region, 1, 2);
  print_bits("BLIT", dst, 6);
}

int main(void) {
  run_norm();
  run_mm();
  run_blit();
  return 0;
}
