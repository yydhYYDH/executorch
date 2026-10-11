#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

static int power_context;
static uint8_t a[HMX_M * HMX_K], q[HMX_M * HMX_N];
static int8_t b[HMX_K * HMX_N];
static float y[HMX_M * HMX_N];

int main(int argc, char** argv) {
  if (argc != 2) {
    FARF(ALWAYS, "I8EXAMPLE expected output directory argument");
    return 1;
  }
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  if (HAP_power_set(&power_context, &power)) {
    return 2;
  }
  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE *qfile = NULL, *yfile = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, 16384, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1)) {
    goto cleanup;
  }
  context = HAP_compute_res_acquire(&attr, 100000);
  if (!context) {
    goto cleanup;
  }
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context)) {
    goto cleanup;
  }
  locked = 1;
  char path[1024];
  int path_length = snprintf(path, sizeof(path), "%s/output.u8", argv[1]);
  if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
    goto cleanup;
  }
  qfile = fopen(path, "wb");
  path_length = snprintf(path, sizeof(path), "%s/output.f32", argv[1]);
  if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
    goto cleanup;
  }
  yfile = fopen(path, "wb");
  if (!qfile || !yfile) {
    goto cleanup;
  }

  for (int test = 0; test < 3; ++test) {
    for (int r = 0; r < HMX_M; ++r) {
      for (int k = 0; k < HMX_K; ++k) {
        int value = (r * 3 + k * 7 + r * k) % 5 - 2;
        a[r * HMX_K + k] = test == 0 ? 128 + value
            : test == 1              ? (r * 17 + k * 29 + r * k * 3) % 256
                                     : 128;
      }
    }
    for (int k = 0; k < HMX_K; ++k) {
      for (int n = 0; n < HMX_N; ++n) {
        b[k * HMX_N + n] = test == 1 ? (k * 11 + n * 19 + k * n) % 17 - 8
                                     : (k * 3 + n * 7 + k * n) % 5 - 2;
      }
    }
    memset(q, 0xa5, sizeof(q));
    hmx_matmul_u8s8_tile(a, b, q, y, workspace);
    unsigned errors = 0;
    for (int r = 0; r < HMX_M; ++r) {
      for (int n = 0; n < HMX_N; ++n) {
        int acc = 0;
        for (int k = 0; k < HMX_K; ++k) {
          acc += ((int)a[r * HMX_K + k] - 128) * b[k * HMX_N + n];
        }
        int expected = acc + 128;
        expected = expected < 0 ? 0 : expected > 255 ? 255 : expected;
        int i = r * HMX_N + n;
        errors += q[i] != expected || y[i] != (expected - 128) / 32.0f;
      }
    }
    FARF(
        ALWAYS,
        "I8EXAMPLE case %d errors %u q %u %u %u %u",
        test,
        errors,
        q[0],
        q[1],
        q[2],
        q[3]);
    if (errors || fwrite(q, 1, sizeof(q), qfile) != sizeof(q) ||
        fwrite(y, 1, sizeof(y), yfile) != sizeof(y)) {
      goto cleanup;
    }
  }
  result = 0;

cleanup:
  if (qfile && fclose(qfile)) {
    result = 4;
  }
  if (yfile && fclose(yfile)) {
    result = 4;
  }
  if (locked && HAP_compute_res_hmx_unlock(context)) {
    result = 4;
  }
  if (context && HAP_compute_res_release(context)) {
    result = 4;
  }
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power)) {
    result = 4;
  }
  FARF(ALWAYS, "I8EXAMPLE result %d", result);
  return result;
}
