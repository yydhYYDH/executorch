/* The comparison routes, run against the vendored kernels inside hexagon-sim.
 *
 * Two questions this answers that the host cannot: whether a select at
 * bytes == 1 really produces a one-byte result from a two-byte condition, and
 * whether the DSP's own GREATER and LESS op types answer torch on the corner
 * cases a one-byte pack would inherit. The sixteen pairs are chosen for their
 * order relations rather than for their values: the two signed zeros, a NaN
 * against itself, each infinity against itself, and the fp16 extremes.
 */
#include <stdint.h>
#include <stdio.h>

extern "C" int htp_ops_binary_elementwise(uint8_t *dst, uint8_t *src0, uint8_t *src1,
                                           int32_t outSize, int32_t in0Size,
                                           int32_t in1Size, int32_t opType,
                                           int32_t bytes, int32_t inputBytes,
                                           int32_t inputIsFloat, int32_t outputIsFloat,
                                           const int32_t *broadcastParams,
                                           int32_t broadcastParamCount);
extern "C" int htp_ops_select(uint8_t *dst, uint8_t *cond_ptr, uint8_t *src1_ptr,
                              uint8_t *src2_ptr, int32_t outSize, int32_t condSize,
                              int32_t in1Size, int32_t in2Size, int32_t bytes,
                              int32_t condBytes, int32_t channelSize, int32_t innerSize);
extern "C" int htp_ops_reduction(uint8_t *dst, uint8_t *src, int32_t outside, int32_t reduce,
                                 int32_t inside, int32_t type, int32_t bytes);

#define N 16
#define ROWS 4
#define COLS 4
#define B 1

enum { kVectorBytes = 128 };

static void print_h(const char *tag, const _Float16 *v, int n) {
  printf("%s", tag);
  for (int i = 0; i < n; ++i) printf(" %04x", ((const uint16_t *)v)[i]);
  printf("\n");
}

static void print_b(const char *tag, const uint8_t *v, int n) {
  printf("%s", tag);
  for (int i = 0; i < n; ++i) printf(" %02x", v[i]);
  printf("\n");
}

static _Float16 A[N] __attribute__((aligned(kVectorBytes)));
static _Float16 Bv[N] __attribute__((aligned(kVectorBytes)));
static _Float16 D[N] __attribute__((aligned(kVectorBytes)));
static uint8_t OUT[N] __attribute__((aligned(kVectorBytes)));
static uint8_t ONE = 1;
static uint8_t ZERO = 0;
static_assert(__alignof__(A) == kVectorBytes, "A is read a vector at a time");
static_assert(__alignof__(Bv) == kVectorBytes, "Bv is read a vector at a time");
static_assert(__alignof__(D) == kVectorBytes, "D is written a vector at a time");
static_assert(__alignof__(OUT) == kVectorBytes, "OUT is written a vector at a time");

/* index  pair                                  what it is for
 *  0     0.0   vs  0.0        equal, both signs equal
 *  1    -0.0   vs  0.0        equal under IEEE, apart under a total order
 *  2     1.0   vs  1.0        equal, ordinary
 *  3    -1.0   vs  1.0        ordered, sign
 *  4     NaN   vs  NaN        unordered against itself
 *  5     inf   vs  inf        equal infinities
 *  6    -inf   vs -inf        equal infinities
 *  7     2.0   vs  1.0        ordered, positive
 *  8    -2.0   vs -1.0        ordered, negative
 *  9     0.5   vs  0.25       ordered, sub-unit
 * 10    -0.5   vs -0.25       ordered, negative sub-unit
 * 11     3.0   vs  3.0        equal, ordinary
 * 12     1e-4  vs  1e-4       equal, near the bottom of the normal range
 * 13    -1e-4  vs  1e-4       ordered, near the bottom
 * 14  65504   vs -65504       the fp16 extremes
 * 15 -65504   vs  65504       the fp16 extremes, reversed
 */
static const float A_IN[N] = {0.0f,  -0.0f,  1.0f,  -1.0f,  0.0f / 0.0f, 1.0f / 0.0f,
                              -1.0f / 0.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f, 1.0e-4f,
                              -1.0e-4f, 65504.0f, -65504.0f};
static const float B_IN[N] = {0.0f,  0.0f,  1.0f,  1.0f,  0.0f / 0.0f, 1.0f / 0.0f,
                              -1.0f / 0.0f, 1.0f, -1.0f, 0.25f, -0.25f, 3.0f, 1.0e-4f,
                              1.0e-4f, -65504.0f, 65504.0f};

static _Float16 RED_IN[ROWS * COLS] __attribute__((aligned(kVectorBytes)));
static _Float16 RED_OUT[ROWS] __attribute__((aligned(kVectorBytes)));
static_assert(__alignof__(RED_IN) == kVectorBytes, "RED_IN is read a vector at a time");
static_assert(__alignof__(RED_OUT) == kVectorBytes, "RED_OUT is written a vector at a time");

int main(void) {
  for (int i = 0; i < N; ++i) {
    A[i] = (_Float16)A_IN[i];
    Bv[i] = (_Float16)B_IN[i];
  }
  print_h("A", A, N);
  print_h("B", Bv, N);

  /* op 2 is SUB, op 9 is GREATER and op 10 is LESS, all three already in the
   * enum the DSP ships. outIsFloat=1 is what makes the two compare arms write
   * 1.0 and 0.0 rather than int32 1 and 0. */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)A, (uint8_t *)Bv, N, N, N, 2, 2,
                             2, 0, 1, NULL, 0);
  print_h("SUB", D, N);
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)A, (uint8_t *)Bv, N, N, N, 9, 2,
                             2, 0, 1, NULL, 0);
  print_h("GT", D, N);
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)A, (uint8_t *)Bv, N, N, N, 10, 2,
                             2, 0, 1, NULL, 0);
  print_h("LT", D, N);

  /* ne: the difference is the condition, one byte out. */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)A, (uint8_t *)Bv, N, N, N, 2, 2,
                             2, 0, 1, NULL, 0);
  htp_ops_select(OUT, (uint8_t *)D, &ONE, &ZERO, N, N, 1, 1, B, 2, 0, 0);
  print_b("NEBYTE", OUT, N);
  /* eq: the same two sources the other way round. */
  htp_ops_select(OUT, (uint8_t *)D, &ZERO, &ONE, N, N, 1, 1, B, 2, 0, 0);
  print_b("EQBYTE", OUT, N);
  /* gt packed: the DSP's own GREATER as a one-byte result. */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)A, (uint8_t *)Bv, N, N, N, 9, 2,
                             2, 0, 1, NULL, 0);
  htp_ops_select(OUT, (uint8_t *)D, &ONE, &ZERO, N, N, 1, 1, B, 2, 0, 0);
  print_b("GTBYTE", OUT, N);
  /* ge: the same comparison with the sources swapped, so !(b > a). */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)Bv, (uint8_t *)A, N, N, N, 9, 2,
                             2, 0, 1, NULL, 0);
  htp_ops_select(OUT, (uint8_t *)D, &ZERO, &ONE, N, N, 1, 1, B, 2, 0, 0);
  print_b("GEBYTE", OUT, N);

  /* The reduction's own guard, asked the question the route asks of it. */
  int red1 = htp_ops_reduction(OUT, (uint8_t *)A, 1, N, 1, 2, B);
  printf("REDONE %08x\n", (unsigned)red1);
  int red0 = htp_ops_reduction((uint8_t *)D, (uint8_t *)A, 1, N, 1, 2, 2);
  printf("REDTWO %08x\n", (unsigned)red0);

  /* the any == amax identity, on 0/1 at the width the reduction does take */
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      RED_IN[r * COLS + c] = (_Float16)(((r * 3 + c * 5) % 7) == 0 ? 1.0f : 0.0f);
    }
  }
  print_h("REDIN", RED_IN, ROWS * COLS);
  int red2 = htp_ops_reduction((uint8_t *)RED_OUT, (uint8_t *)RED_IN, ROWS, COLS, 1, 2, 2);
  printf("REDMAX %08x\n", (unsigned)red2);
  print_h("REDOUT", RED_OUT, ROWS);
  return 0;
}
