#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

static uint16_t a[BHMX_M * BHMX_K];
static int8_t w[BHMX_K * BHMX_N];
static int32_t scale[BHMX_N * BHMX_NBLOCK];
static uint16_t q[BHMX_M * BHMX_N];
static float y[BHMX_M * BHMX_N];
static int64_t tests = 0, bad = 0;

static int floordiv(int value, int divisor) {
  int quotient = value / divisor;
  if (value < 0 && value % divisor) --quotient;
  return quotient;
}

// Exact scalar reference for the contract in matmul.h, including the HMX tile's
// own floor(acc/256) + 32768 requantization per block.
static void reference(uint16_t* out_q, float* out_y) {
  for (int r = 0; r < BHMX_M; ++r) {
    for (int n = 0; n < BHMX_N; ++n) {
      int acc = 0;
      for (int blk = 0; blk < BHMX_NBLOCK; ++blk) {
        int partial = 0;
        for (int k = 0; k < BHMX_BLOCK; ++k) {
          int kk = blk * BHMX_BLOCK + k;
          partial += ((int)a[r * BHMX_K + kk] - 32768) * w[kk * BHMX_N + n];
        }
        int t = floordiv(partial, 256) + 32768;
        t = t < 0 ? 0 : t > 65535 ? 65535 : t;
        acc += (t - 32768) * scale[n * BHMX_NBLOCK + blk];
      }
      int v = acc + 32768;
      v = v < 0 ? 0 : v > 65535 ? 65535 : v;
      out_q[r * BHMX_N + n] = (uint16_t)v;
      out_y[r * BHMX_N + n] = (v - 32768) * (1.0f / 4.0f);
    }
  }
}

static void fill(int test) {
  for (int n = 0; n < BHMX_N; ++n) {
    scale[n * BHMX_NBLOCK] = (n % 3) + 2;
    scale[n * BHMX_NBLOCK + 1] = (n % 5) + 1;
  }
  for (int r = 0; r < BHMX_M; ++r) {
    for (int k = 0; k < BHMX_K; ++k) {
      a[r * BHMX_K + k] = test == 0 ? (uint16_t)(32768 + (r * 3 + k * 7) % 9 - 4)
          : test == 1 ? (uint16_t)(32768 + (r * 17 + k * 29) % 257 - 128)
          : test == 3 ? (uint16_t)(32768 + 300 * ((r * 5 + k * 11) % 5 - 2))
                      : 32768;
    }
  }
  for (int k = 0; k < BHMX_K; ++k) {
    for (int n = 0; n < BHMX_N; ++n) {
      w[k * BHMX_N + n] = (int8_t)((k * 11 + n * 19 + k * n) % 16 - 8);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    FARF(ALWAYS, "BWEXAMPLE expected output directory argument");
    return 1;
  }
  int power_context = 0;
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  if (HAP_power_set(&power_context, &power)) {
    return 2;
  }
  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE* qfile = NULL;
  FILE* yfile = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, BHMX_WORKSPACE_BYTES, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1)) {
    goto cleanup;
  }
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context) {
    FARF(ALWAYS, "BWEXAMPLE resource acquire failed");
    goto cleanup;
  }
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context)) {
    FARF(ALWAYS, "BWEXAMPLE workspace or HMX lock failed");
    goto cleanup;
  }
  locked = 1;
  char path[1024];
  snprintf(path, sizeof(path), "%s/output.q", argv[1]);
  qfile = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.y", argv[1]);
  yfile = fopen(path, "wb");
  if (!qfile || !yfile) {
    goto cleanup;
  }

  for (int test = 0; test < 4; ++test) {
    fill(test);
    memset(q, 0xa5, sizeof(q));
    hmx_matmul_u16s4_block_tile(a, w, scale, q, y, workspace);

    uint16_t ref_q[BHMX_M * BHMX_N];
    float ref_y[BHMX_M * BHMX_N];
    reference(ref_q, ref_y);
    int q_bad = 0, y_bad = 0;
    for (int i = 0; i < BHMX_M * BHMX_N; ++i) {
      tests += 1;
      if (q[i] != ref_q[i]) ++q_bad;
      if (y[i] != ref_y[i]) ++y_bad;
      if (q[i] != ref_q[i] || y[i] != ref_y[i]) ++bad;
    }
    FARF(
        ALWAYS,
        "BWEXAMPLE case %d q_bad %d y_bad %d q %u %u %u %u",
        test,
        q_bad,
        y_bad,
        q[0],
        q[1],
        q[2],
        q[3]);
    if (fwrite(q, 1, sizeof(q), qfile) != sizeof(q) ||
        fwrite(y, 1, sizeof(y), yfile) != sizeof(y)) {
      goto cleanup;
    }
  }
  FARF(ALWAYS, "BWEXAMPLE total tests %lld bad %lld", (long long)tests, (long long)bad);
  result = bad ? 5 : 0;

cleanup:
  if (qfile) fclose(qfile);
  if (yfile) fclose(yfile);
  if (locked && context) HAP_compute_res_hmx_unlock(context);
  if (context) HAP_compute_res_release(context);
  FARF(ALWAYS, "BWEXAMPLE result %d", result);
  return result;
}
