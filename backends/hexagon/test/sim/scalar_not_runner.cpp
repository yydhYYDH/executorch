/* A comparison against a literal, and a mask's negation, run in hexagon-sim.
 *
 * Four questions the host cannot answer about the commands its own emitters
 * write. Which arm a zero stride reaches: the emitter passes the binary
 * broadcast descriptor with the literal's stride at zero, not the kernel's
 * in1Size == 1 spelling, and those are separate branches with separate IEEE
 * behaviour. Whether a one-byte select reads a one-byte condition: the negation
 * is an all-1 select and the widths in the call are the only thing that could
 * get it wrong, so a two-byte condition over the same buffer is the control.
 * Whether a route measured at sixteen elements holds at 32, which is the length
 * whose lanes divide evenly. And what a conjunction of two masks costs, which
 * is the question a bool and/or asks.
 *
 * The sixteen values are the corner list the comparison-route file uses, read
 * here against a threshold rather than against each other: the two signed
 * zeros, a NaN, each infinity, and the fp16 extremes.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

#define N 16
#define ROWS 4
#define COLS 8
#define BIG (ROWS * COLS)
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
static _Float16 Abig[BIG] __attribute__((aligned(kVectorBytes)));
static _Float16 D[BIG] __attribute__((aligned(kVectorBytes)));
static _Float16 D2[BIG] __attribute__((aligned(kVectorBytes)));
static _Float16 D3[BIG] __attribute__((aligned(kVectorBytes)));
static uint8_t NOT[BIG] __attribute__((aligned(kVectorBytes)));
static uint8_t PACKED[BIG] __attribute__((aligned(kVectorBytes)));
static uint8_t ONE = 1;
static uint8_t ZERO = 0;
static_assert(__alignof__(A) == kVectorBytes, "A is read a vector at a time");
static_assert(__alignof__(Abig) == kVectorBytes, "Abig is read a vector at a time");
static_assert(__alignof__(D) == kVectorBytes, "D is written a vector at a time");
static_assert(__alignof__(NOT) == kVectorBytes, "NOT is written a vector at a time");
static_assert(__alignof__(PACKED) == kVectorBytes, "PACKED is written a vector at a time");

/* index  value        what it is for against a threshold
 *  0     0.0          equal to a zero threshold
 *  1    -0.0          equal to a zero threshold apart under a bit test
 *  2     1.0          equal to a one threshold
 *  3    -1.0          ordered, sign
 *  4     NaN          unordered against every threshold
 *  5    +inf          ordered above every threshold here
 *  6    -inf          ordered below every threshold here
 *  7     2.0          ordered above
 *  8    -2.0          ordered below
 *  9     0.5          between the two thresholds
 * 10    -0.5          between the two thresholds, negative
 * 11     3.0          ordered above
 * 12     1e-4         ordered above a zero threshold
 * 13    -1e-4         ordered below a zero threshold
 * 14  65504.0         the fp16 extreme
 * 15 -65504.0         the fp16 extreme, negative
 */
static const float A_IN[N] = {0.0f,  -0.0f,  1.0f,  -1.0f,  0.0f / 0.0f, 1.0f / 0.0f,
                              -1.0f / 0.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f, 1.0e-4f,
                              -1.0e-4f, 65504.0f, -65504.0f};

/* The descriptor _broadcast_tail writes for a (4, 8) output whose right-hand
 * operand is a one-element constant: the rank, the output extents, then each
 * stride table right-padded to the table's eight entries. The literal's table
 * is all zeroes, which is what makes the kernel read element zero at every
 * index. Written out literally rather than computed so that a change to the
 * emitter's tail has to change this file to go unnoticed. */
static int32_t kScalarTail[25] = {2, 4, 8, 0, 0, 0, 0, 0, 0,
                                  COLS, 1, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};

/* The one the emitter writes when both operands are the output. */
static int32_t kPairTail[25] = {2, 4, 8, 0, 0, 0, 0, 0, 0,
                                COLS, 1, 0, 0, 0, 0, 0, 0,
                                COLS, 1, 0, 0, 0, 0, 0, 0};

/* The commands run over 32 elements, so the sixteen values are repeated rather
 * than the array being read past: the first sixteen printed words are the
 * corner list and the rest is the same list again. */
static int run_scalar_compare(int op, float literal) {
  _Float16 k = (_Float16)literal;
  return htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)Abig, (uint8_t *)&k, BIG, BIG,
                                    1, op, 2, 2, 0, 0, kScalarTail, 25);
}

static int run_scalar_arm(int op, float literal) {
  _Float16 k = (_Float16)literal;
  return htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)Abig, (uint8_t *)&k, BIG, BIG,
                                    1, op, 2, 2, 0, 0, NULL, 0);
}

int main(void) {
  for (int i = 0; i < N; ++i) {
    A[i] = (_Float16)A_IN[i];
    Abig[i] = A[i];
    Abig[i + N] = A[i];
  }
  /* The input is printed first, so a route that disagreed with torch could not
   * be one the runner never received. */
  print_h("A", A, N);

  /* greater against zero, the spelling the emitter writes */
  printf("RC0 %d\n", run_scalar_compare(9, 0.0f));
  print_h("GT0", D, N);
  /* less against zero */
  run_scalar_compare(10, 0.0f);
  print_h("LT0", D, N);
  /* the same two at a threshold that splits the list differently */
  run_scalar_compare(9, 1.0f);
  print_h("GT1", D, N);
  run_scalar_compare(10, 1.0f);
  print_h("LT1", D, N);

  /* The kernel's own scalar arm, which is the branch a host that passed in1Size
   * and no descriptor would take. A different branch with its own IEEE
   * behaviour, so it is measured rather than assumed equivalent. */
  run_scalar_arm(9, 0.0f);
  print_h("ARM0", D, N);
  run_scalar_arm(10, 1.0f);
  print_h("ARM1", D, N);

  /* The tensor spelling, as the control that says this runner is comparing
   * rather than filling a buffer: the right operand is the output itself. */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)Abig, (uint8_t *)Abig, BIG, BIG, BIG,
                             9, 2, 2, 0, 0, kPairTail, 25);
  print_h("TIE0", D, N);
  /* The same comparison with no descriptor at all, which is the kernel's own
   * elementwise arm and the one that vectorises. */
  htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)Abig, (uint8_t *)Abig, BIG, BIG, BIG,
                             9, 2, 2, 0, 0, NULL, 0);
  print_h("TIEVEC", D, N);

  /* logical_not: the select between the two one-byte constants, condition and
   * result both at one byte, which is the whole emitter. The condition is the
   * *packed* mask _emit_compare leaves -- one byte an element -- and not the
   * fp16 flags behind it. */
  run_scalar_compare(9, 0.0f);
  htp_ops_select(PACKED, (uint8_t *)D, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
  print_b("PACK", PACKED, N);
  htp_ops_select(NOT, PACKED, &ZERO, &ONE, BIG, BIG, 1, 1, B, B, 0, 0);
  print_b("NOT", NOT, N);
  /* and the round trip: not(not(x)) is x, so the two selects in sequence have to
   * reproduce the first one. */
  {
    static uint8_t TWICE[BIG] __attribute__((aligned(kVectorBytes)));
    htp_ops_select(TWICE, NOT, &ZERO, &ONE, BIG, BIG, 1, 1, B, B, 0, 0);
    print_b("NOTAGAIN", TWICE, N);
  }
  /* The misreading this emitter must not do: the *flags* taken as a one-byte
   * condition, which is the same call with a two-byte buffer behind it. A byte
   * of an fp16 1.0 is 0x00, so the nonzero test sees zeroes and reads every
   * element as false. */
  htp_ops_select(NOT, (uint8_t *)D, &ZERO, &ONE, BIG, BIG, 1, 1, B, B, 0, 0);
  print_b("NOTTWO", NOT, N);
  /* The negative control: the packed mask declared two bytes wide, which pairs
   * the elements up. If this answered the same as NOT then the widths in the
   * call would not be what decides the answer and every other assertion in the
   * file would be about a parameter that does nothing. */
  htp_ops_select(NOT, PACKED, &ZERO, &ONE, BIG, BIG, 1, 1, B, 2, 0, 0);
  print_b("NOTWIDE", NOT, N);
  /* The other direction: a condition that is not 0 or 1, which is what a
   * nonzero test has to answer for and what a torch.bool slot never carries. */
  {
    static uint8_t MIXED[BIG] __attribute__((aligned(kVectorBytes)));
    for (int i = 0; i < N; ++i) {
      MIXED[i] = (uint8_t)(i == 0 ? 0x00 : (i == 1 ? 0xff : (i == 2 ? 0x01 : 0x00)));
    }
    print_b("MIXEDIN", MIXED, N);
    htp_ops_select(NOT, MIXED, &ZERO, &ONE, BIG, BIG, 1, 1, B, B, 0, 0);
    print_b("MIXEDOUT", NOT, N);
  }

  /* ---- what a conjunction of two masks costs, and what it costs once a mask
   * has been packed. The flags a comparison leaves are fp16 1.0 and 0.0, so
   * MIN(6) and MAX(5) combine them elementwise without anything that knows
   * about logic: three more commands over flags that already exist, and no
   * kernel. The same MIN over masks that have already been packed to one byte
   * an element is the negative control. A byte of 0x01 inside the two-byte
   * operand the binary command declares is the half-float 0x0001, a subnormal
   * rather than a one, and a subnormal is nonzero, so the nonzero test that
   * packs the result calls every element true. There is no command that unpacks
   * a bool, which is the reason a conjunction of two packed masks is a new op
   * type and not a composition. */
  {
    static uint8_t LEFT[BIG] __attribute__((aligned(kVectorBytes)));
    static uint8_t RIGHT[BIG] __attribute__((aligned(kVectorBytes)));
    /* Both masks are comparisons against a literal, and both flag vectors are
     * kept: the conjunction over the flags is the free route and the one over
     * the packed bytes is the control. The thresholds are a zero and the fp16
     * extreme, so the sixteen elements give six trues and seven. */
    run_scalar_compare(9, 0.0f);
    memcpy(D2, D, sizeof(_Float16) * BIG);
    htp_ops_select(LEFT, (uint8_t *)D, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
    run_scalar_compare(10, 65504.0f);
    memcpy(D3, D, sizeof(_Float16) * BIG);
    htp_ops_select(RIGHT, (uint8_t *)D, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
    print_b("GTPKG", LEFT, N);
    print_b("LTBIG", RIGHT, N);
    /* the conjunction of the two comparisons, over their fp16 flags */
    htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)D2, (uint8_t *)D3, BIG, BIG, BIG, 6, 2,
                               2, 0, 0, NULL, 0);
    htp_ops_select(NOT, (uint8_t *)D, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
    print_b("AND", NOT, N);
    /* the disjunction, over the same two flags */
    htp_ops_binary_elementwise((uint8_t *)D, (uint8_t *)D2, (uint8_t *)D3, BIG, BIG, BIG, 5, 2,
                               2, 0, 0, NULL, 0);
    htp_ops_select(NOT, (uint8_t *)D, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
    print_b("OR", NOT, N);
    /* the same MIN over the packed bytes, read at the two-byte width the binary
     * command declares */
    htp_ops_binary_elementwise((uint8_t *)D3, (uint8_t *)LEFT, (uint8_t *)RIGHT, BIG, BIG,
                               BIG, 6, 2, 2, 0, 0, NULL, 0);
    htp_ops_select(NOT, (uint8_t *)D3, &ONE, &ZERO, BIG, BIG, 1, 1, B, 2, 0, 0);
    print_b("ANDWIDE", NOT, N);
  }
  return 0;
}
