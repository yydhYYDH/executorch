/* The quantized prefill matmul kernels, on the simulator, against arithmetic
 * that knows no layout.
 *
 * Two things are measured rather than read off the source. The first is the
 * weight: `htp_ops_weight_reorder_int4` is the vendored tree's own scalar-and-
 * HVX reorder into the int4 tile order the prefill kernels consume, so running
 * it on bytes the host built and printing the result is a measurement of that
 * order, which the host then has to reproduce byte for byte. The second is the
 * matmul itself: the operands arrive packed by the emitter's packer and by the
 * activation layout the kernel's own DMA descriptors describe, and the whole
 * output buffer comes back as fp16 bit patterns, so the host can compare it
 * against `sum_k a[m,k] * w[k,n] * scale[n]` -- an expectation that mentions
 * neither a tile, a nibble, a scale position nor the order rows come back in.
 *
 * The kernels stage their operands in VTCM and drive the HMX unit, so the
 * managers come up first, and every buffer is 128-byte aligned: a misaligned
 * operand reaches VTCM through DMA as wrong data rather than as an error.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_queue.h"
#include "dsp/hvx_utils.h"
#include "dsp/power.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "prefill_data.h"

extern "C" int htp_ops_matmul_q4a16_fp16(uint8_t *output, uint8_t *activation,
                                         uint8_t *weight, uint8_t *bias, int32_t m,
                                         int32_t k, int32_t n, int32_t weight_type,
                                         int32_t layout_type, int32_t mp, int32_t np,
                                         int32_t kp, int32_t scale_block_num,
                                         int32_t scale_asymmetric);
extern "C" int htp_ops_weight_reorder_int4(uint8_t *dst, uint8_t *src, int32_t ic,
                                           int32_t oc, int32_t alpha_size);
extern "C" void vtcm_manager_setup();
extern "C" int vtcm_manager_acquire();

enum { kVectorBytes = 128 };
#define VECTOR_ALIGNED __attribute__((aligned(kVectorBytes)))

static uint8_t VECTOR_ALIGNED g_out[kMaxOutputBytes];
static uint8_t VECTOR_ALIGNED g_reordered[kMaxReorderBytes];
static uint8_t VECTOR_ALIGNED g_act[kMaxActivationBytes];
static uint8_t VECTOR_ALIGNED g_weight[kMaxWeightBytes];
static uint8_t VECTOR_ALIGNED g_bias[kMaxBiasBytes];
static_assert(__alignof__(g_out) == kVectorBytes, "the output is written a vector at a time");
static_assert(__alignof__(g_act) == kVectorBytes, "the activation is read a vector at a time");
static_assert(__alignof__(g_weight) == kVectorBytes, "the weight is read a vector at a time");

static void print_bits(const char *tag, const uint8_t *data, int bytes) {
  printf("%s", tag);
  const uint16_t *bits = (const uint16_t *) data;
  for (int i = 0; i < bytes / 2; ++i) printf(" %04x", bits[i]);
  printf("\n");
}

/* The two HVX operations the reorder's scalar half is wrapped in, on an input
 * whose bytes say where they came from. Each returned byte identifies the input
 * byte it came from, which is what makes the host's model of these two a
 * measurement instead of a reading of the manual. */
static void measure_byte_ops(void) {
  static uint8_t in[128] VECTOR_ALIGNED;
  static uint8_t shuffled[128] VECTOR_ALIGNED;
  static uint8_t shifted[128] VECTOR_ALIGNED;
  static_assert(__alignof__(in) == kVectorBytes, "the probe reads a vector at a time");
  for (int i = 0; i < 128; ++i) in[i] = (uint8_t) i;
  HVX_Vector v = vmemu(in);
  vmemu(shuffled) = Q6_Vb_vshuff_Vb(v);
  for (int i = 0; i < 128; ++i) in[i] = (uint8_t) ((i * 2 + 1) & 0xff);
  vmemu(shifted) = Q6_Vh_vasl_VhR(vmemu(in), 4);
  printf("SHUFF");
  for (int i = 0; i < 128; ++i) printf(" %02x", shuffled[i]);
  printf("\n");
  printf("SHIFT");
  for (int i = 0; i < 128; ++i) printf(" %02x", shifted[i]);
  printf("\n");
}

static void run_reorder(const ReorderCase &c) {
  memset(g_reordered, 0x5a, sizeof(g_reordered));
  int ret = htp_ops_weight_reorder_int4(g_reordered, (uint8_t *) c.raw, c.ic, c.oc,
                                        c.alpha_bytes);
  printf("%s_RET %d\n", c.tag, ret);
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_BYTES", c.tag);
  print_bits(tag, g_reordered, c.reorder_bytes);
}

static void run_matmul(const PrefillCase &c) {
  memcpy(g_act, c.act, c.act_bytes);
  memcpy(g_weight, c.weight, c.weight_bytes);
  if (c.bias != 0) memcpy(g_bias, c.bias, c.bias_bytes);
  memset(g_out, 0xaa, sizeof(g_out));
  int ret = htp_ops_matmul_q4a16_fp16(g_out, g_act, g_weight, c.bias == 0 ? 0 : g_bias,
                                      c.m, c.k, c.n, 0, 1, c.mp, c.np, c.kp, c.scale_blocks, 0);
  printf("%s_RET %d\n", c.tag, ret);
  char tag[32];
  snprintf(tag, sizeof(tag), "%s_OUT", c.tag);
  print_bits(tag, g_out, c.out_bytes);
}

int main(void) {
  power_setup();
  power_acquire();
  vtcm_manager_setup();
  printf("VTCMACQUIRE %d\n", vtcm_manager_acquire());
  hmx_manager_setup();
  hmx_queue_setup();
  worker_pool_global_init();

  measure_byte_ops();
  for (unsigned i = 0; i < sizeof(kReorderCases) / sizeof(kReorderCases[0]); ++i)
    run_reorder(kReorderCases[i]);
  for (unsigned i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i)
    run_matmul(kCases[i]);
  return 0;
}
