#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

enum { M = HMX_A16W8_M, K = HMX_A16W8_K, N = HMX_A16W8_N };

static uint16_t a[M * K], q[M * N];
static int8_t b[K * N];
static uint16_t ref[M * N];

static void reference(void) {
  for (int n = 0; n < N; ++n) {
    long long sum = 0;
    for (int k = 0; k < K; ++k) sum += b[k * N + n];
    long long num = (long long)HMX_A16W8_ZA * sum;
    long long corr = num >= 0 ? (num + 128) / 256 : -((-num + 128) / 256);
    for (int r = 0; r < M; ++r) {
      long long acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (long long)a[r * K + k] * b[k * N + n];
      long long t = (acc >> 8) - corr;
      // The integer conversion preserves a sticky bit below the shifted accumulator.
      long long value =
          ((t * 256 + (t != 0)) * HMX_A16W8_CONVERSION_MULT +
           HMX_A16W8_CONVERSION_BIAS * (1LL << 26)) >> 30;
      if (value < 0) value = 0;
      if (value > 65535) value = 65535;
      ref[r * N + n] = (uint16_t)value;
    }
  }
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
    FARF(ALWAYS, "A16W8EXAMPLE power set failed");
    return 1;
  }

  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE* qfile = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, HMX_A16W8_WORKSPACE_BYTES, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1))
    goto cleanup;
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context) goto cleanup;
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context))
    goto cleanup;
  locked = 1;

  qfile = fopen(out_path, "wb");
  if (!qfile) {
    FARF(ALWAYS, "A16W8EXAMPLE cannot open %s", out_path);
    result = 6;
    goto cleanup;
  }
  for (int test = 0; test < HMX_A16W8_TEST_CASES; ++test) {
    uint32_t state = 1234567u + (uint32_t)test;
    for (int i = 0; i < M * K; ++i) {
      state = state * 1664525u + 1013904223u;
      int d = test == 0 ? 0
          : test == 1 ? (int)((state >> 20) % 65) - 32
                      : (int)((state >> 20) % 2049) - 1024;
      int v = HMX_A16W8_ZA + d;
      a[i] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : v);
      if (test == 3) a[i] = (uint16_t)(state >> 16);
      if (test == 4 || test == 6) a[i] = 0;
      if (test == 5 || test == 7) a[i] = 65535;
      if (test == 8) a[i] = HMX_A16W8_ZA;
      if (test == 9) a[i] = (uint16_t)(i * 63);
      if (test == 10) a[i] = HMX_A16W8_ZA - 4872;
      if (test == 11) a[i] = HMX_A16W8_ZA + 4872;
    }
    for (int i = 0; i < K * N; ++i) {
      state = state * 1664525u + 1013904223u;
      b[i] = (int8_t)((int)((state >> 16) % 256) - 128);
      if (test >= 4 && test <= 7) b[i] = test < 6 ? -128 : 127;
      if (test == 8 || test == 9) b[i] = 0;
      if (test == 10 || test == 11) b[i] = i / N == 0 ? 64 : 0;
    }
    memset(q, 0, sizeof(q));
    memset(workspace, test % 2 ? 0xa5 : 0x5a, HMX_A16W8_WORKSPACE_BYTES);
    hmx_matmul_a16w8_tile(a, b, q, workspace);
    reference();
    unsigned errors = 0;
    for (int i = 0; i < M * N; ++i)
      if (q[i] != ref[i]) ++errors;
    FARF(ALWAYS, "A16W8EXAMPLE test %d errors %u q0 %u %u %u %u", test, errors,
         q[0], q[1], q[2], q[3]);
    if (errors) {
      for (int i = 0; i < M * N; ++i) {
        if (q[i] != ref[i]) {
          FARF(ALWAYS, "A16W8EXAMPLE bad %d exp %u got %u", i, ref[i], q[i]);
          break;
        }
      }
      result = 5;
      goto cleanup;
    }
    if (fwrite(q, 1, sizeof(q), qfile) != sizeof(q)) {
      result = 6;
      goto cleanup;
    }
  }
  result = 0;

cleanup:
  if (qfile && fclose(qfile)) result = 4;
  if (locked && HAP_compute_res_hmx_unlock(context)) result = 4;
  if (context && HAP_compute_res_release(context)) result = 4;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power)) result = 4;
  FARF(ALWAYS, "A16W8EXAMPLE result %d", result);
  return result;
}
