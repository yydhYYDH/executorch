/* Walks the two quantized GEMV weight layouts on the DSP, one activation at a
 * time.
 *
 * The operands arrive packed by the emitter's own packers (the test writes them
 * into gemv_data.h) and the activations are built here, so the only thing this
 * file decides is which kernel gets called with what. Every row it prints is
 * compared on the host against arithmetic that knows nothing about tiles,
 * groups or nibbles.
 *
 * `unpack_vrmpy_weight_128` is included verbatim from the kernel: what the read
 * path does to a byte is a measurement here, not a transcription of the comment
 * that says what it should do.
 */
#include <stdint.h>
#include <stdio.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx_utils.h"
#include "gemv_data.h"

extern "C" int htp_ops_matmul_q4a16_gemv_i8(uint8_t *output, uint8_t *activation,
                                            uint8_t *weight, uint8_t *bias, int32_t k,
                                            int32_t n, int32_t scale_block_num,
                                            int32_t scale_asymmetric);
extern "C" int hmx_matmulw8a16block_gemv_i8(uint8_t *c, const uint8_t *a,
                                            const uint8_t *b_wt,
                                            const uint8_t *b_scale,
                                            const uint8_t *bias, int32_t K, int32_t N,
                                            int32_t scale_block_num);
extern "C" void vtcm_manager_setup();
extern "C" int vtcm_manager_acquire();

/* Verbatim from matmul_q4block_gemv_i8.c. */
static inline HVX_VectorPair unpack_vrmpy_weight_128(HVX_Vector v_int4) {
  const HVX_Vector v_mask_lo = Q6_Vb_vsplat_R(0x0f);
  const HVX_Vector v_eight = Q6_Vb_vsplat_R(0x08);
  HVX_Vector v_lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v_int4, v_mask_lo), v_eight);
  HVX_Vector v_hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v_int4, 4), v_eight);
  return Q6_W_vshuff_VVR(v_hi, v_lo, -1);
}

/* An HVX vector at `-mhvx-length=128b`, which is what the kernels here assume of
 * the buffers they are handed; the assertions below read it back out of the
 * declarations, since a byte array gives the linker no reason to align. kMaxK
 * is 8192 because the wide-K probes below need an activation that wide; the
 * shapes the per-position walk runs stay at 64 and 128. */
enum { kVectorBytes = 128, kMaxK = 8192, kMaxN = 64 };

static uint8_t g_act[kMaxK * 2] __attribute__((aligned(kVectorBytes)));
static uint16_t g_out[kMaxN] __attribute__((aligned(kVectorBytes)));
static uint16_t g_bias[kMaxN] __attribute__((aligned(kVectorBytes)));
static_assert(__alignof__(g_act) == kVectorBytes,
              "the activation is read a vector at a time");
static_assert(__alignof__(g_out) == kVectorBytes,
              "the output is written a vector at a time");
static_assert(__alignof__(g_bias) == kVectorBytes,
              "the bias is read a vector at a time");

/* One at `at`, and at every k when `at` is negative. */
static void activation(int k, int at) {
  for (int i = 0; i < 2 * k; ++i) g_act[i] = 0;
  if (at < 0) {
    for (int i = 0; i < k; ++i) g_act[2 * i + 1] = 0x3c;
  } else {
    g_act[2 * at + 1] = 0x3c;
  }
}

static void print_row(const char *tag, int at, int n) {
  printf("%s %d", tag, at);
  for (int i = 0; i < n; ++i) printf(" %04x", g_out[i]);
  printf("\n");
}

static void q4(const uint8_t *packed, const char *tag, int k, int n) {
  activation(k, -1);
  for (int i = 0; i < kMaxN; ++i) g_out[i] = 0xaaaa;
  int ret = htp_ops_matmul_q4a16_gemv_i8((uint8_t *) g_out, g_act,
                                         (uint8_t *) packed, NULL, k, n, 1, 0);
  if (ret != 0) printf("%s failed %d\n", tag, ret);
  print_row(tag, -1, n);
  for (int at = 0; at < k; ++at) {
    activation(k, at);
    for (int i = 0; i < kMaxN; ++i) g_out[i] = 0xaaaa;
    ret = htp_ops_matmul_q4a16_gemv_i8((uint8_t *) g_out, g_act, (uint8_t *) packed,
                                       NULL, k, n, 1, 0);
    if (ret != 0) printf("%s failed %d\n", tag, ret);
    print_row(tag, at, n);
  }
}

static void w8(const uint8_t *packed, const uint8_t *scales, const char *tag, int k,
               int n) {
  activation(k, -1);
  int ret = hmx_matmulw8a16block_gemv_i8((uint8_t *) g_out, g_act, packed, scales,
                                         (uint8_t *) g_bias, k, n, 1);
  if (ret != 0) printf("%s failed %d\n", tag, ret);
  print_row(tag, -1, n);
  for (int at = 0; at < k; ++at) {
    activation(k, at);
    ret = hmx_matmulw8a16block_gemv_i8((uint8_t *) g_out, g_act, packed, scales,
                                       (uint8_t *) g_bias, k, n, 1);
    if (ret != 0) printf("%s failed %d\n", tag, ret);
    print_row(tag, at, n);
  }
}

/* The all-ones answer alone: the per-position walk above prints one line per k,
 * and these shapes have thousands. `sf_from_w` turns the int32 accumulator into
 * fp32 with a magic number documented for |x| < 2^22, and with
 * `scale_block_num == 1` that accumulator spans the whole K, so the shapes here
 * are the ones where the bound can be reached. The weights are constant, so
 * nothing cancels and the accumulator is `|w| * 127 * k`: 127 * 127 * 512 for
 * W8W, 7 * 127 * 8192 for Q4W, against a bound of 4194304. */
static void q4_all_ones(const uint8_t *packed, const char *tag, int k, int n) {
  activation(k, -1);
  for (int i = 0; i < kMaxN; ++i) g_out[i] = 0xaaaa;
  int ret = htp_ops_matmul_q4a16_gemv_i8((uint8_t *) g_out, g_act,
                                         (uint8_t *) packed, NULL, k, n, 1, 0);
  if (ret != 0) printf("%s failed %d\n", tag, ret);
  print_row(tag, -1, n);
}

static void w8_all_ones(const uint8_t *packed, const uint8_t *scales,
                        const char *tag, int k, int n) {
  activation(k, -1);
  for (int i = 0; i < kMaxN; ++i) g_out[i] = 0xaaaa;
  int ret = hmx_matmulw8a16block_gemv_i8((uint8_t *) g_out, g_act, packed, scales,
                                         (uint8_t *) g_bias, k, n, 1);
  if (ret != 0) printf("%s failed %d\n", tag, ret);
  print_row(tag, -1, n);
}

int main(void) {
  vtcm_manager_setup();
  printf("vtcm acquired=%d\n", vtcm_manager_acquire());
  for (int i = 0; i < kMaxN; ++i) g_bias[i] = 0;

  /* The read path of the int4 layout, on an input whose bytes say where they
   * came from: input byte i is i, so a returned byte identifies both the input
   * byte and which of its two nibbles it was taken from. */
  {
    static uint8_t in[128] __attribute__((aligned(kVectorBytes)));
    static uint8_t pair[256] __attribute__((aligned(kVectorBytes)));
    static_assert(__alignof__(in) == kVectorBytes,
                  "the read path reads its input a vector at a time");
    static_assert(__alignof__(pair) == kVectorBytes,
                  "the read path writes its output a vector at a time");
    for (int i = 0; i < 128; ++i) in[i] = (uint8_t) i;
    HVX_VectorPair wv = unpack_vrmpy_weight_128(vmem(in));
    vmem((HVX_Vector *) pair) = Q6_V_lo_W(wv);
    vmem(((HVX_Vector *) pair) + 1) = Q6_V_hi_W(wv);
    printf("READPATH");
    for (int i = 0; i < 256; ++i) printf(" %02x", pair[i]);
    printf("\n");
  }

  q4(kQ4A, "Q4A", 64, 32);
  q4(kQ4B, "Q4B", 128, 64);
  w8(kW8A, kScalesA, "W8A", 64, 32);
  w8(kW8B, kScalesB, "W8B", 128, 64);

  /* The wide-K ladder: two rungs inside the accumulator bound, where the answer
   * has to be exact, and two past it. */
  w8_all_ones(kW8S, kScalesS, "W8S", 256, 32);
  q4_all_ones(kQ4S, "Q4S", 4224, 32);
  w8_all_ones(kW8W, kScalesW, "W8W", 512, 32);
  q4_all_ones(kQ4W, "Q4W", 8192, 32);
  /* And the control: the same K with weights of both signs, where the
   * per-channel sum cancels and the accumulator stays below the bound. */
  w8_all_ones(kW8M, kScalesM, "W8M", 8192, 32);
  q4_all_ones(kQ4M, "Q4M", 8192, 32);
  return 0;
}
