/* The convolution kernels, on the simulator, against what torch would say.
 *
 * Two entry points are exercised: `hvx_conv_depthwise2d_fp16` for a group per
 * channel, and `hmx_im2col_convolution_fp16` for the rest. Both take the DSP's
 * 64-channel blocked activation and a weight this backend rearranges at export
 * time, so the runner builds its operands the way the emitters do and prints the
 * whole output as fp16 bit patterns; the host side rebuilds the same tensors,
 * checks that the bytes it packs are the bytes the runner used, and compares
 * torch's convolution with the kernel's bit for bit.
 *
 * The HMX unit needs its operands in VTCM, so the managers have to be brought up
 * before either kernel runs, and the static buffers have to be 128-byte aligned:
 * the weight fill reaches VTCM through DMA, which copies a misaligned operand
 * wrong rather than failing.
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

/* region_ops.h has no prototype for either kernel and htp_ops.h is generated
 * from the IDL, so both are declared here. */
extern "C" int htp_ops_conv_depthwise2d_fp16(uint8_t *dst, uint8_t *src, uint8_t *weight,
                                             uint8_t *bias, int batch, int ih, int iw,
                                             int oh, int ow, int c4, int kernelY,
                                             int kernelX, int strideY, int strideX,
                                             int padY, int padX, int dilateY, int dilateX,
                                             int relu, int relu6);
extern "C" int htp_ops_im2col_convolution_fp16(uint8_t *output, uint8_t *input,
                                               uint8_t *weight, uint8_t *bias,
                                               const HmxIm2ColConvParam *params);
extern "C" int htp_ops_zero(uint8_t *dst, int32_t size);

#define PACK 64
#define TILE 32
#define MAT_ALIGN __attribute__((aligned(128)))

#define MAX_BLOCKS 2
#define MAX_BATCH 2
#define MAX_AREA 81
#define MAX_OC 128

static MAT_ALIGN __fp16 dw_src[MAX_BLOCKS * MAX_BATCH * MAX_AREA * PACK];
static MAT_ALIGN __fp16 dw_wgt[MAX_BLOCKS * 3 * 3 * PACK];
static MAT_ALIGN __fp16 dw_bias[MAX_BLOCKS * PACK];
static MAT_ALIGN __fp16 dw_dst[MAX_BLOCKS * MAX_BATCH * MAX_AREA * PACK];

static MAT_ALIGN __fp16 cv_src[MAX_BLOCKS * MAX_BATCH * MAX_AREA * PACK];
static MAT_ALIGN __fp16 cv_wgt[4 * 3 * 3 * 4 * 1024];
static MAT_ALIGN __fp16 cv_bias[MAX_OC + PACK];
static MAT_ALIGN __fp16 cv_dst[MAX_BLOCKS * MAX_BATCH * MAX_AREA * PACK];

static void print_bits(const char *tag, const void *values, int count) {
  printf("%s", tag);
  for (int i = 0; i < count; ++i) printf(" %04x", ((const uint16_t *)values)[i]);
  printf("\n");
}

/* One digest of a packed weight, so the host can check that the bytes this
 * runner gave the kernel are the bytes the emitter's packer produces. */
static void print_weight_digest(const char *tag, const __fp16 *values, int count) {
  uint32_t hash = 0;
  for (int i = 0; i < count; ++i) hash = hash * 31u + ((const uint16_t *)values)[i];
  printf("%s_WSUM %08x\n", tag, hash);
  char head[32];
  snprintf(head, sizeof(head), "%s_WHEAD", tag);
  print_bits(head, values, 16);
}

struct DepthCase {
  const char *tag;
  int channels, kernel, stride, pad, dilate, hw, relu, relu6, batch;
};

/* The geometry the depthwise emitter takes, one case per thing the walk does:
 * the origin, the stride, dilation, both activations, and a second channel
 * block over a batch of two. */
static const DepthCase depth_cases[] = {
    {"DW_1X1", 64, 3, 1, 1, 1, 8, 0, 0, 1},
    {"DW_2X2", 64, 3, 2, 1, 1, 9, 0, 0, 1},
    {"DW_DIL", 64, 3, 1, 0, 2, 7, 0, 0, 1},
    {"DW_RELU", 64, 3, 1, 1, 1, 8, 1, 0, 1},
    {"DW_RELU6", 64, 3, 1, 1, 1, 8, 0, 1, 1},
    {"DW_BLOCKS", 128, 3, 2, 1, 1, 9, 0, 0, 2},
};

static int out_extent(int in, int k, int stride, int pad, int dilate) {
  return (in + 2 * pad - dilate * (k - 1) - 1) / stride + 1;
}

static void run_depthwise(const DepthCase &c) {
  const int blocks = (c.channels + PACK - 1) / PACK;
  const int area = c.hw * c.hw;
  const int oh = out_extent(c.hw, c.kernel, c.stride, c.pad, c.dilate);
  const int ow = oh;

  memset(dw_src, 0, sizeof(dw_src));
  memset(dw_wgt, 0, sizeof(dw_wgt));
  memset(dw_dst, 0, sizeof(dw_dst));
  for (int b = 0; b < c.batch; ++b)
    for (int cb = 0; cb < blocks; ++cb)
      for (int y = 0; y < c.hw; ++y)
        for (int x = 0; x < c.hw; ++x)
          for (int lane = 0; lane < PACK; ++lane) {
            const int channel = cb * PACK + lane;
            if (channel >= c.channels) continue;
            dw_src[(cb * c.batch + b) * area * PACK + (y * c.hw + x) * PACK + lane] =
                (__fp16)(float)(((y * c.hw + x) * 7 + channel * 3) % 11 - 5);
          }
  for (int cb = 0; cb < blocks; ++cb)
    for (int ky = 0; ky < c.kernel; ++ky)
      for (int kx = 0; kx < c.kernel; ++kx)
        for (int lane = 0; lane < PACK; ++lane) {
          const int channel = cb * PACK + lane;
          if (channel >= c.channels) continue;
          /* The block index is outside the taps, which is the one thing the
           * emitter's own packer has to get right. */
          dw_wgt[((cb * c.kernel + ky) * c.kernel + kx) * PACK + lane] =
              (__fp16)(float)(((channel * 3 + ky * 7 + kx * 11) % 13) - 6);
        }
  for (int i = 0; i < blocks * PACK; ++i)
    dw_bias[i] = (__fp16)(float)(i < c.channels ? (i % 5) - 2 : 0);

  int err = htp_ops_conv_depthwise2d_fp16(
      (uint8_t *)dw_dst, (uint8_t *)dw_src, (uint8_t *)dw_wgt, (uint8_t *)dw_bias, c.batch,
      c.hw, c.hw, oh, ow, blocks, c.kernel, c.kernel, c.stride, c.stride, c.pad, c.pad,
      c.dilate, c.dilate, c.relu, c.relu6);
  printf("%s_RET %d\n", c.tag, err);
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_OUT", c.tag);
  print_bits(tag, dw_dst, blocks * c.batch * oh * ow * PACK);
  print_weight_digest(c.tag, dw_wgt, blocks * c.kernel * c.kernel * PACK);
}

struct ConvCase {
  const char *tag;
  int ic, oc, k, stride, pad, dilate, hw, batch;
  /* What to leave in the lanes past the last channel: 0 zero, 1 an infinity,
   * 2 a finite value with the weight's own padding lanes made nonzero too, so a
   * contribution from them would land in the answer. */
  int poison;
  /* Position tiles and channel tiles per pass, which the emitter pins. */
  int mp, np;
};

/* The geometry the im2col emitter takes: a pointwise layer, a strided window, no
 * padding, an input narrower than a 32-channel tile, a dilation, a batch, and an
 * output that is not a whole number of tiles. C8 and C9 are the same shape with
 * the padding lanes filled in. */
static const ConvCase conv_cases[] = {
    {"CV_3X3", 64, 64, 3, 1, 1, 1, 5, 1, 0, 1, 2},
    {"CV_STRIDE", 64, 64, 3, 2, 1, 1, 8, 1, 0, 1, 2},
    {"CV_NOPAD", 64, 64, 3, 1, 0, 1, 5, 1, 0, 1, 2},
    {"CV_1X1", 64, 96, 1, 1, 0, 1, 5, 1, 0, 1, 2},
    {"CV_IC96", 96, 64, 3, 1, 1, 1, 5, 1, 0, 1, 2},
    {"CV_DIL", 64, 64, 3, 1, 2, 2, 7, 1, 0, 1, 2},
    {"CV_BATCH", 64, 64, 3, 1, 1, 1, 5, 2, 0, 1, 2},
    {"CV_IC3", 3, 32, 3, 2, 1, 1, 8, 1, 0, 1, 2},
    /* The same convolution with the chunking the kernel defaults to, which is
     * what the emitter must not leave it to. */
    {"CV_ONE", 3, 32, 3, 2, 1, 1, 8, 1, 0, 1, 1},
    /* A ragged position tile over more than one channel tile, which is the
     * shape the single-tile store gets wrong. */
    {"CV_ODD", 64, 64, 3, 1, 1, 1, 5, 1, 0, 1, 1},
    {"CV_INF", 3, 32, 3, 2, 1, 1, 8, 1, 1, 1, 2},
    {"CV_READ", 3, 32, 3, 2, 1, 1, 8, 1, 2, 1, 2},
};

static void poison_padding(int channels, int batch, int area, int mode) {
  const int blocks = (channels + PACK - 1) / PACK;
  for (int cb = 0; cb < blocks; ++cb)
    for (int b = 0; b < batch; ++b)
      for (int m = 0; m < area; ++m)
        for (int lane = 0; lane < PACK; ++lane)
          if (cb * PACK + lane >= channels)
            cv_src[(cb * batch + b) * area * PACK + m * PACK + lane] =
                (__fp16)(float)(mode == 1 ? (1.0f / 0.0f) : 3.0f);
}

static void run_convolution(const ConvCase &c) {
  const int blocks = (c.ic + PACK - 1) / PACK;
  const int groups = (c.ic + TILE - 1) / TILE;
  const int area = c.hw * c.hw;
  const int oh = out_extent(c.hw, c.k, c.stride, c.pad, c.dilate);
  const int ow = oh;
  const int out_blocks = (c.oc + PACK - 1) / PACK;

  memset(cv_src, 0, sizeof(cv_src));
  memset(cv_wgt, 0, sizeof(cv_wgt));
  memset(cv_dst, 0, sizeof(cv_dst));
  for (int b = 0; b < c.batch; ++b)
    for (int cb = 0; cb < blocks; ++cb)
      for (int y = 0; y < c.hw; ++y)
        for (int x = 0; x < c.hw; ++x)
          for (int lane = 0; lane < PACK; ++lane) {
            const int channel = cb * PACK + lane;
            if (channel >= c.ic) continue;
            cv_src[(cb * c.batch + b) * area * PACK + (y * c.hw + x) * PACK + lane] =
                (__fp16)(float)(((y * c.hw + x) * 7 + channel * 3) % 11 - 5);
          }
  for (int tile = 0; tile < (c.oc + TILE - 1) / TILE; ++tile)
    for (int ky = 0; ky < c.k; ++ky)
      for (int kx = 0; kx < c.k; ++kx)
        for (int ib = 0; ib < groups; ++ib) {
          const int tile_index = (tile * c.k * c.k + ky * c.k + kx) * groups + ib;
          for (int kin = 0; kin < TILE; ++kin) {
            const int channel = ib * TILE + kin;
            for (int cc = 0; cc < TILE; ++cc) {
              const int oc = tile * TILE + cc;
              if (oc >= c.oc) continue;
              __fp16 value = 0;
              if (channel < c.ic)
                value =
                    (__fp16)(float)(((oc * 3 + channel * 5 + ky * 7 + kx * 11) % 13) - 6);
              else if (c.poison == 2)
                value = (__fp16)1.0f;
              cv_wgt[tile_index * 1024 + (kin / 2) * 64 + cc * 2 + (kin % 2)] = value;
            }
          }
        }
  if (c.poison) poison_padding(c.ic, c.batch, area, c.poison);
  for (int i = 0; i < c.oc + PACK; ++i)
    cv_bias[i] = (__fp16)(float)(i < c.oc ? (i % 5) - 2 : 0);

  HmxIm2ColConvParam p;
  memset(&p, 0, sizeof(p));
  p.im2col.padX = c.pad;
  p.im2col.padY = c.pad;
  p.im2col.dilateX = c.dilate;
  p.im2col.dilateY = c.dilate;
  p.im2col.strideX = c.stride;
  p.im2col.strideY = c.stride;
  p.im2col.kernelX = c.k;
  p.im2col.kernelY = c.k;
  p.im2col.icDiv4 = c.ic / 4;
  p.im2col.icup4 = (c.ic + 3) / 4 * 4;
  p.im2col.kernelCountUnit = c.k * c.k * groups;
  p.im2col.iw = c.hw;
  p.im2col.ih = c.hw;
  p.im2col.ow = ow;
  p.im2col.oh = oh;
  p.im2col.srcZStep = c.batch * area * PACK;
  p.im2col.srcYStep = c.hw * PACK;
  p.im2col.packCUnit = PACK;
  p.im2col.destICStride = area * PACK;
  p.im2col.ic = c.ic;
  p.oc = c.oc;
  p.mp = c.mp;
  p.np = c.np;
  p.batch = c.batch;

  int err = htp_ops_im2col_convolution_fp16((uint8_t *)cv_dst, (uint8_t *)cv_src,
                                            (uint8_t *)cv_wgt, (uint8_t *)cv_bias, &p);
  printf("%s_RET %d\n", c.tag, err);
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_OUT", c.tag);
  print_bits(tag, cv_dst, out_blocks * c.batch * oh * ow * PACK);
  if (!c.poison)
    print_weight_digest(c.tag, cv_wgt, ((c.oc + TILE - 1) / TILE) * c.k * c.k * groups * 1024);
}

/* Where the padding lanes actually are, so the host can see that the buffer the
 * kernel read was the one it meant: lane 3 of the first position, and the k = 3
 * lane of the first tile for four output channels. */
static void print_padding_probe(const ConvCase &c) {
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_A", c.tag);
  print_bits(tag, cv_src + 3, 4);
  snprintf(tag, sizeof(tag), "%s_W", c.tag);
  static __fp16 probe[4];
  for (int i = 0; i < 4; ++i) probe[i] = cv_wgt[(3 / 2) * 64 + i * 2 + (3 % 2)];
  print_bits(tag, probe, 4);
}

int main(void) {
  power_setup();
  power_acquire();
  vtcm_manager_setup();
  vtcm_manager_acquire();
  hmx_manager_setup();
  hmx_queue_setup();
  worker_pool_global_init();

  for (unsigned i = 0; i < sizeof(depth_cases) / sizeof(depth_cases[0]); ++i)
    run_depthwise(depth_cases[i]);
  for (unsigned i = 0; i < sizeof(conv_cases) / sizeof(conv_cases[0]); ++i) {
    run_convolution(conv_cases[i]);
    if (conv_cases[i].poison) print_padding_probe(conv_cases[i]);
  }

  /* htp_ops_zero: the memset a ragged channel count needs before the pack. */
  static uint8_t MAT_ALIGN scratch[256];
  memset(scratch, 0xAB, sizeof(scratch));
  int zero_ret = htp_ops_zero(scratch, 130);
  /* Printed as hex, which is how every other tag reads its words. */
  printf("ZERO %x %02x %02x %02x %02x\n", zero_ret, scratch[0], scratch[127], scratch[129],
         scratch[131]);
  return 0;
}
