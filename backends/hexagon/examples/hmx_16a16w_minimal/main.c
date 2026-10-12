#include <math.h>
#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

enum { M = HMX_A16W16_M, K = HMX_A16W16_K, N = HMX_A16W16_N };

static uint16_t a[M * K], q[M * N];
static int16_t b[K * N];
static uint16_t ref[M * N];

static long long round_half_away(long long num) {
  return num >= 0 ? (num + 32768) / 65536 : -((-num + 32768) / 65536);
}

static void reference(void) {
  for (int n = 0; n < N; ++n) {
    long long sum = 0;
    for (int k = 0; k < K; ++k) {
      int v = b[k * N + n];
      sum += v > 32639 ? 32639 : v;
    }
    long long corr = round_half_away((long long)HMX_A16W16_ZA * sum);
    for (int r = 0; r < M; ++r) {
      long long acc = 0;
      for (int k = 0; k < K; ++k) {
        int v = b[k * N + n];
        acc += (long long)a[r * K + k] * (v > 32639 ? 32639 : v);
      }
      long long t = (acc >> 16) - corr;
      double value = HMX_A16W16_ZQ + floor(HMX_A16W16_MULT * (double)t + 0.5);
      if (value < 0) value = 0;
      if (value > 65535) value = 65535;
      ref[r * N + n] = (uint16_t)value;
    }
  }
}

static unsigned check(void) {
  unsigned errors = 0;
  for (int i = 0; i < M * N; ++i)
    if (q[i] != ref[i]) ++errors;
  return errors;
}

int main(int argc, char** argv) {
  const char* out_dir = argc > 1 ? argv[1] : ".";
  char out_path[512];
  snprintf(out_path, sizeof(out_path), "%s/output.u16", out_dir);
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  int power_context = 0;
  if (HAP_power_set(&power_context, &power)) {
    FARF(ALWAYS, "A16W16EXAMPLE power set failed");
    return 1;
  }

  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE* qfile = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, HMX_A16W16_WORKSPACE_BYTES, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1))
    goto cleanup;
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context) goto cleanup;
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context))
    goto cleanup;
  locked = 1;

  for (int test = 0; test < 3; ++test) {
    uint32_t state = 1234567u + (uint32_t)test;
    for (int i = 0; i < M * K; ++i) {
      state = state * 1664525u + 1013904223u;
      int d = test == 0 ? 0
          : test == 1 ? (int)((state >> 20) % 65) - 32
                      : (int)((state >> 20) % 2049) - 1024;
      int v = HMX_A16W16_ZA + d;
      a[i] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : v);
    }
    for (int i = 0; i < K * N; ++i) {
      state = state * 1664525u + 1013904223u;
      b[i] = (int16_t)((int)((state >> 16) % 32768) - 16384);
    }
    memset(q, 0, sizeof(q));
    hmx_matmul_a16w16_tile(a, b, q, workspace);
    reference();
    unsigned errors = check();
    FARF(ALWAYS, "A16W16EXAMPLE test %d errors %u q0 %u %u %u %u", test, errors,
         q[0], q[1], q[2], q[3]);
    if (errors) {
      for (int i = 0; i < M * N; ++i) {
        if (q[i] != ref[i]) {
          FARF(ALWAYS, "A16W16EXAMPLE bad %d exp %u got %u", i, ref[i], q[i]);
          break;
        }
      }
    }
    if (errors) {
      result = 5;
      goto cleanup;
    }
  }

  qfile = fopen(out_path, "wb");
  if (!qfile) {
    FARF(ALWAYS, "A16W16EXAMPLE cannot open %s", out_path);
    result = 6;
    goto cleanup;
  }
  if (fwrite(q, 1, sizeof(q), qfile) != sizeof(q)) {
    result = 6;
    goto cleanup;
  }
  result = 0;

cleanup:
  if (qfile && fclose(qfile)) result = 4;
  if (locked && HAP_compute_res_hmx_unlock(context)) result = 4;
  if (context && HAP_compute_res_release(context)) result = 4;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power)) result = 4;
  FARF(ALWAYS, "A16W16EXAMPLE result %d", result);
  return result;
}
