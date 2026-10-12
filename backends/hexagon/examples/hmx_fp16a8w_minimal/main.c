#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

enum { M = HMX_FP16A8W_M, K = HMX_FP16A8W_K, N = HMX_FP16A8W_N };

static __fp16 a[M * K];
static int8_t b[K * N];
static float scales[N], y[M * N];
static uint16_t q[M * N];

static void inputs(int test) {
  for (int r = 0; r < M; ++r) {
    for (int k = 0; k < K; ++k) {
      float value = ((r * 3 + k * 7 + r * k) % 33 - 16) * (1.0f / 16.0f);
      if (test == 2) value = 0.0f;
      if (test == 3) value = r == k ? 1.0f : 0.0f;
      if (test == 4) value = r * K + k == 17 * K + 13 ? -0.5f : 0.0f;
      if (test == 5 || test == 6) value = test == 5 ? 1.0f : -1.0f;
      if (test == 9) value = ((r * 71 + k * 37) % 2049 - 1024) * (1.0f / 1024.0f);
      a[r * K + k] = (__fp16)value;
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      int value = (k * 11 + n * 19 + k * n) % 255 - 127;
      if (test == 1) value = 0;
      if (test == 4) value = k == 13 && n == 49 ? -127 : 0;
      if (test == 5 || test == 6) value = test == 5 ? -128 : 127;
      b[k * N + n] = (int8_t)value;
    }
  }
  for (int n = 0; n < N; ++n) {
    scales[n] = (1 + n % 7) * (1.0f / 256.0f);
    if (test == 7) scales[n] = (n % 13 + 1) * 0.000731f;
    if (test == 8) scales[n] = n % 2 ? 0.0078135006f : 0.00452736672f;
    if (test == 9) scales[n] = (1 + n % 7) * (1.0f / 64.0f);
  }
}

static unsigned check(void) {
  unsigned errors = 0;
  for (int r = 0; r < M; ++r) {
    for (int n = 0; n < N; ++n) {
      __fp16 scale_h = (__fp16)scales[n];
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        __fp16 weight = (__fp16)((float)b[k * N + n] * (float)scale_h);
        acc += (double)a[r * K + k] * (double)weight;
      }
      __fp16 expected = (__fp16)acc;
      uint16_t bits;
      memcpy(&bits, &expected, sizeof(bits));
      if (q[r * N + n] != bits || y[r * N + n] != (float)expected) {
        if (errors < 3)
          FARF(ALWAYS, "FP16A8WEXAMPLE bad r%d n%d exp%x got%x", r, n, bits, q[r * N + n]);
        ++errors;
      }
    }
  }
  return errors;
}

int main(int argc, char** argv) {
  const char* out_dir = argc > 1 ? argv[1] : ".";
  char path[512];
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  int power_context = 0;
  if (HAP_power_set(&power_context, &power)) return 1;

  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE *qfile = NULL, *yfile = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, HMX_FP16A8W_WORKSPACE_BYTES, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1)) goto cleanup;
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context) goto cleanup;
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) || HAP_compute_res_hmx_lock(context))
    goto cleanup;
  locked = 1;
  snprintf(path, sizeof(path), "%s/output.f16", out_dir);
  qfile = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.f32", out_dir);
  yfile = fopen(path, "wb");
  if (!qfile || !yfile) { result = 6; goto cleanup; }
  for (int test = 0; test < HMX_FP16A8W_TEST_CASES; ++test) {
    inputs(test);
    memset(workspace, test % 2 ? 0xa5 : 0x5a, HMX_FP16A8W_WORKSPACE_BYTES);
    hmx_matmul_fp16a8w_tile(a, b, scales, q, y, workspace);
    unsigned errors = check();
    FARF(ALWAYS, "FP16A8WEXAMPLE test%d errors%u q0:%x,%x,%x,%x", test, errors,
         q[0], q[1], q[2], q[3]);
    if (errors) { result = 5; goto cleanup; }
    if (fwrite(q, 1, sizeof(q), qfile) != sizeof(q) ||
        fwrite(y, 1, sizeof(y), yfile) != sizeof(y)) { result = 6; goto cleanup; }
  }
  result = 0;
cleanup:
  if (qfile && fclose(qfile)) result = 4;
  if (yfile && fclose(yfile)) result = 4;
  if (locked && HAP_compute_res_hmx_unlock(context)) result = 4;
  if (context && HAP_compute_res_release(context)) result = 4;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power)) result = 4;
  FARF(ALWAYS, "FP16A8WEXAMPLE result %d", result);
  return result;
}
