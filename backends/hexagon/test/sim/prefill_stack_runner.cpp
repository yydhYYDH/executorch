/* What the M <= 32 prefill kernel costs in stack, measured rather than read.
 *
 * `hmx_matmulq4fp16_mle32_part` builds one DMA descriptor per K/32 step, and at
 * 32 bytes each that array is exactly K bytes. On the stack it made this
 * branch's frame grow with K, which the phone cannot afford on the thread its
 * commands run on: K = 12736 aborts there with 0x8000040d while the M > 32
 * branch is fine at K = 25216. Nothing the host can compute sees that, and
 * neither does an ordinary run here -- on the simulator's main thread K = 12800
 * returns the right answer either way, because that stack is far larger than
 * the phone's.
 *
 * So the stack is chosen. Each case runs the kernel as a job on a one-worker
 * pool built with kStackBytes, while the kernel's own internal submits go to
 * the global pool and keep the normal worker stacks -- the phone's arrangement,
 * a small stack at the top of the call and ordinary stacks underneath. With
 * 8 KiB, K = 12800 on the M <= 32 branch overruns it and takes the process
 * down, while the same K on the M > 32 branch, which heap-allocates the same
 * array, does not.
 *
 * The data is generated rather than embedded, so this costs an ordinary
 * compile: every activation halfword is 1.0, every int4 digit is 1 after the +8
 * bias, and every scale is 1.0. Every output element is then exactly K, which
 * is what WRONG counts the violations of.
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

extern "C" int htp_ops_matmul_q4a16_fp16(uint8_t *output, uint8_t *activation,
                                         uint8_t *weight, uint8_t *bias, int32_t m,
                                         int32_t k, int32_t n, int32_t weight_type,
                                         int32_t layout_type, int32_t mp, int32_t np,
                                         int32_t kp, int32_t scale_block_num,
                                         int32_t scale_asymmetric);
extern "C" void vtcm_manager_setup();
extern "C" int vtcm_manager_acquire();

enum { kMaxK = 12800, kMaxM = 33, kMaxN = 32, kStackBytes = 8192 };

struct Case {
  const char *tag;
  int m, k, n;
};

/* A stack fault is fatal to the whole process, so the case that is expected to
 * overrun runs last: the tag before it is then the last thing in the log. */
static const Case kCases[] = {
    {"SMALLK64", 4, 64, 32},
    {"SMALLK512", 4, 512, 32},
    {"OTHERBRANCH", 33, 12800, 32},
    {"MLE32K12800", 4, 12800, 32},
};

#define ALIGNED __attribute__((aligned(128)))
static uint8_t ALIGNED g_out[(kMaxN + 63) / 64 * kMaxM * 128];
static uint8_t ALIGNED g_act[kMaxK * kMaxM * 2];
static uint8_t ALIGNED g_weight[(kMaxK / 32) * (kMaxN / 32) * 512 + kMaxN * 2];

struct ProbeJob {
  worker_synctoken_t token;
  int m, k, n;
  volatile int ret;
};

static void probe_job(void *dptr, int index) {
  ProbeJob *j = (ProbeJob *) dptr;
  (void) index;
  const int weight_bytes = (j->k / 32) * (j->n / 32) * 512;
  const int out_bytes = (j->n + 63) / 64 * j->m * 128;

  __fp16 *act = (__fp16 *) g_act;
  for (int i = 0; i < j->k * j->m; ++i) act[i] = (__fp16) 1.0f;
  memset(g_weight, 0x99, weight_bytes); /* digit 9 -> 9 - 8 == 1 in every nibble */
  {
    uint16_t *scales = (uint16_t *) (g_weight + weight_bytes);
    for (int i = 0; i < j->n; ++i) scales[i] = 0x3c00; /* fp16 1.0 */
  }
  memset(g_out, 0xaa, out_bytes);

  j->ret = htp_ops_matmul_q4a16_fp16(g_out, g_act, g_weight, 0, j->m, j->k, j->n, 0, 1,
                                     1, 2, j->k / 32, 1, 0);
  worker_pool_synctoken_jobdone(&j->token);
}

int main(void) {
  power_setup();
  power_acquire();
  vtcm_manager_setup();
  printf("VTCMACQ %x\n", vtcm_manager_acquire());
  hmx_manager_setup();
  hmx_queue_setup();
  worker_pool_global_init();

  for (unsigned i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
    const Case *c = &kCases[i];
    const int out_bytes = (c->n + 63) / 64 * c->m * 128;

    /* Printed and flushed before the kernel runs, so the last tag in the log
     * names the case that took the process down. */
    printf("%sSTART\n", c->tag);
    fflush(stdout);

    worker_pool_context_t pool = NULL;
    AEEResult created = worker_pool_init_ex(&pool, kStackBytes, 1);
    printf("%sSTACK %x\n", c->tag, kStackBytes);
    if (created != 0) {
      printf("%sCREATE %x\n", c->tag, (unsigned) created);
      fflush(stdout);
      continue;
    }

    ProbeJob job;
    memset(&job, 0, sizeof(job));
    job.m = c->m;
    job.k = c->k;
    job.n = c->n;
    job.ret = -1;
    worker_pool_synctoken_init(&job.token, 1);
    worker_pool_job_t entry;
    entry.fptr = probe_job;
    entry.dptr = &job;
    worker_pool_submit(pool, entry);
    worker_pool_synctoken_wait(&job.token);
    worker_pool_deinit(&pool);

    /* Every element should be exactly K, and K is exact in fp16 for these
     * shapes. The kernel writes the first N of each 64-channel pack per row, so
     * counting the K anywhere in the buffer is both the whole answer and free of
     * any assumption about where in the pack the N channels land: it catches a
     * missing element and a spurious one alike. Where they land is
     * test_prefill_on_sim.py's business, not this runner's. */
    union {
      __fp16 h;
      uint16_t u;
    } expected;
    expected.h = (__fp16) (float) c->k;
    const int real = c->m * c->n;
    const uint16_t *words = (const uint16_t *) g_out;
    int correct = 0;
    for (int e = 0; e < out_bytes / 2; ++e) {
      if (words[e] == expected.u) ++correct;
    }
    printf("%sRET %x\n", c->tag, (unsigned) job.ret);
    printf("%sM %x\n", c->tag, c->m);
    printf("%sK %x\n", c->tag, c->k);
    printf("%sVALUE %x\n", c->tag, words[0]);
    printf("%sEXPECT %x\n", c->tag, expected.u);
    printf("%sCORRECT %x\n", c->tag, (unsigned) correct);
    printf("%sREAL %x\n", c->tag, (unsigned) real);
    fflush(stdout);
  }
  return 0;
}
