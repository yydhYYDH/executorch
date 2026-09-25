/* The 1x1 direct entry point, on the simulator, against what torch would say.
 *
 * `htp_ops_conv1x1_direct_fp16` is the command 17 entry point and it forwards
 * to `hmx_im2col_convolution_fp16`, so this runner calls that name directly:
 * what is being checked is the function's own 1x1 activation fill (the direct
 * copy of the plane, and the strided direct gather) on the geometries the
 * emitter's `conv_1x1_direct_applies` accepts -- 1x1 window, unit dilation, no
 * padding, batch 1, a whole number of 32-channel reduction blocks, unit stride
 * with the output plane equal to the input plane or a stride above one.
 *
 * The operands are built the way the emitters build them: a 64-channel blocked
 * activation, a weight in the HMX 32x32 tiles, and a flat fp16 bias. The whole
 * output is printed as fp16 bit patterns, the weight is digested so the host can
 * check the bytes are the emitter's own packing, and the host compares torch's
 * fp64 convolution with the kernel's answer.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_queue.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "region_ops.h"

/* htp_ops.h is generated from the IDL, so the entry point is declared here. */
extern "C" int htp_ops_conv1x1_direct_fp16(uint8_t *output, uint8_t *input,
                                           uint8_t *weight, uint8_t *bias,
                                           const HmxIm2ColConvParam *params);

#define PACK 64
#define TILE 32

enum { kVectorBytes = 128 };
#define VECTOR_ALIGNED __attribute__((aligned(kVectorBytes)))
#define REQUIRE_VECTOR_ALIGNMENT(name)             \
  static_assert(__alignof__(name) == kVectorBytes, \
                "the kernel reads " #name " a vector at a time")

#define MAX_BLOCKS 2
#define MAX_AREA 81
#define MAX_OC 128

static __fp16 VECTOR_ALIGNED d1_src[MAX_BLOCKS * MAX_AREA * PACK];
static __fp16 VECTOR_ALIGNED d1_wgt[4 * 1 * 1 * 4 * 1024];
static __fp16 VECTOR_ALIGNED d1_bias[MAX_OC + PACK];
static __fp16 VECTOR_ALIGNED d1_dst[MAX_BLOCKS * MAX_AREA * PACK];

REQUIRE_VECTOR_ALIGNMENT(d1_src);
REQUIRE_VECTOR_ALIGNMENT(d1_wgt);
REQUIRE_VECTOR_ALIGNMENT(d1_bias);
REQUIRE_VECTOR_ALIGNMENT(d1_dst);

static void print_bits(const char *tag, const void *values, int count) {
  printf("%s", tag);
  for (int i = 0; i < count; ++i) printf(" %04x", ((const uint16_t *)values)[i]);
  printf("\n");
}

static void print_weight_digest(const char *tag, const __fp16 *values, int count) {
  uint32_t hash = 0;
  for (int i = 0; i < count; ++i) hash = hash * 31u + ((const uint16_t *)values)[i];
  printf("%s_WSUM %08x\n", tag, hash);
  char head[32];
  snprintf(head, sizeof(head), "%s_WHEAD", tag);
  print_bits(head, values, 16);
}

struct DirectCase {
  const char *tag;
  int ic, oc, stride, hw;
};

/* One case per fill the 1x1 dispatcher can take, all within the emitter's
 * restrictions: the direct copy of the plane (unit stride, D1_COPY), the strided
 * direct gather over a full row tile (D1_STRIDE), the same gather with a tile
 * that straddles a row (D1_SPLIT, an odd width so the 32-position tile crosses
 * the row boundary), and a ragged output channel count (D1_OC). */
static const DirectCase direct_cases[] = {
    {"D1_COPY", 64, 64, 1, 5},
    {"D1_STRIDE", 64, 64, 2, 8},
    {"D1_SPLIT", 64, 64, 3, 7},
    {"D1_OC", 64, 96, 1, 5},
};

static int out_extent(int in, int stride) { return (in - 1) / stride + 1; }

static void run_direct(const DirectCase &c) {
  const int blocks = (c.ic + PACK - 1) / PACK;
  const int groups = (c.ic + TILE - 1) / TILE;
  const int area = c.hw * c.hw;
  const int oh = out_extent(c.hw, c.stride);
  const int ow = oh;
  const int out_blocks = (c.oc + PACK - 1) / PACK;

  memset(d1_src, 0, sizeof(d1_src));
  memset(d1_wgt, 0, sizeof(d1_wgt));
  memset(d1_dst, 0, sizeof(d1_dst));
  for (int cb = 0; cb < blocks; ++cb)
    for (int y = 0; y < c.hw; ++y)
      for (int x = 0; x < c.hw; ++x)
        for (int lane = 0; lane < PACK; ++lane) {
          const int channel = cb * PACK + lane;
          if (channel >= c.ic) continue;
          d1_src[(y * c.hw + x) * PACK + lane] =
              (__fp16)(float)(((y * c.hw + x) * 7 + channel * 3) % 11 - 5);
        }
  for (int tile = 0; tile < (c.oc + TILE - 1) / TILE; ++tile)
    for (int ib = 0; ib < groups; ++ib) {
      const int tile_index = (tile * groups + ib) * 1024;
      for (int kin = 0; kin < TILE; ++kin) {
        const int channel = ib * TILE + kin;
        for (int cc = 0; cc < TILE; ++cc) {
          const int oc = tile * TILE + cc;
          if (oc >= c.oc || channel >= c.ic) continue;
          d1_wgt[tile_index + (kin / 2) * 64 + cc * 2 + (kin % 2)] =
              (__fp16)(float)(((oc * 3 + channel * 5) % 13) - 6);
        }
      }
    }
  for (int i = 0; i < c.oc + PACK; ++i)
    d1_bias[i] = (__fp16)(float)(i < c.oc ? (i % 5) - 2 : 0);

  HmxIm2ColConvParam p;
  memset(&p, 0, sizeof(p));
  p.im2col.padX = 0;
  p.im2col.padY = 0;
  p.im2col.dilateX = 1;
  p.im2col.dilateY = 1;
  p.im2col.strideX = c.stride;
  p.im2col.strideY = c.stride;
  p.im2col.kernelX = 1;
  p.im2col.kernelY = 1;
  p.im2col.icDiv4 = c.ic / 4;
  p.im2col.icup4 = (c.ic + 3) / 4 * 4;
  p.im2col.kernelCountUnit = groups;
  p.im2col.iw = c.hw;
  p.im2col.ih = c.hw;
  p.im2col.ow = ow;
  p.im2col.oh = oh;
  p.im2col.srcZStep = area * PACK;
  p.im2col.srcYStep = c.hw * PACK;
  p.im2col.packCUnit = PACK;
  p.im2col.destICStride = area * PACK;
  p.im2col.ic = c.ic;
  p.oc = c.oc;
  p.mp = 1;
  p.np = 2;
  p.batch = 1;

  int err = htp_ops_conv1x1_direct_fp16((uint8_t *)d1_dst, (uint8_t *)d1_src,
                                        (uint8_t *)d1_wgt, (uint8_t *)d1_bias, &p);
  printf("%s_RET %d\n", c.tag, err);
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_OUT", c.tag);
  print_bits(tag, d1_dst, out_blocks * oh * ow * PACK);
  print_weight_digest(c.tag, d1_wgt, ((c.oc + TILE - 1) / TILE) * groups * 1024);
}

int main(void) {
  power_setup();
  power_acquire();
  vtcm_manager_setup();
  vtcm_manager_acquire();
  hmx_manager_setup();
  hmx_queue_setup();
  worker_pool_global_init();

  for (unsigned i = 0; i < sizeof(direct_cases) / sizeof(direct_cases[0]); ++i)
    run_direct(direct_cases[i]);
  return 0;
}
