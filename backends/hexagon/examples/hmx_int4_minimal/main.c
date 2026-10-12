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
static uint16_t a16[HMX_A16_M * HMX_K], q16[HMX_A16_M * HMX_A16_N];

int main(int argc, char** argv) {
  if (argc != 2) {
    FARF(ALWAYS, "I4EXAMPLE expected output directory argument");
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
  FILE *qfile = NULL, *yfile = NULL, *q16file = NULL, *y16file = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, 16384, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1)) {
    goto cleanup;
  }
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context) {
    FARF(ALWAYS, "I4EXAMPLE resource acquire failed before compute");
    goto cleanup;
  }
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context)) {
    FARF(ALWAYS, "I4EXAMPLE workspace or HMX lock failed before compute");
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

  for (int test = 0; test < 5; ++test) {
    uint32_t state = 123456789;
    for (int r = 0; r < HMX_M; ++r) {
      for (int k = 0; k < HMX_K; ++k) {
        int value = (r * 3 + k * 7 + r * k) % 5 - 2;
        state = state * 1664525u + 1013904223u;
        a[r * HMX_K + k] = test == 0 ? 128 + value
            : test == 1              ? (r * 17 + k * 29 + r * k * 3) % 256
            : test == 3              ? 127 + (state >> 24) % 3
                                     : 128;
      }
    }
    for (int k = 0; k < HMX_K; ++k) {
      for (int n = 0; n < HMX_N; ++n) {
        state = state * 1664525u + 1013904223u;
        b[k * HMX_N + n] = test == 1 ? (k * 11 + n * 19 + k * n) % 16 - 8
            : test == 3              ? (int)((state >> 24) % 3) - 1
            : test == 4              ? 0
                                     : (k * 3 + n * 7 + k * n) % 5 - 2;
      }
    }
    if (test == 4) {
      a[17 * HMX_K + 13] = 129;
      a[63 * HMX_K + 31] = 130;
      a[0] = 127;
      b[13 * HMX_N + 97] = 1;
      b[31 * HMX_N] = -8;
      b[127] = 7;
      for (int i = 0; i < HMX_A16_M * HMX_K; ++i)
        a16[i] = 32768;
      hmx_matmul_u16s4_tile(a16, b, q16, y, workspace);
      unsigned errors = 0;
      for (int i = 0; i < HMX_A16_M * HMX_A16_N; ++i)
        errors += q16[i] != 32768 || y[i] != 0;
      FARF(ALWAYS, "A16EXAMPLE interleave errors %u", errors);
      if (errors)
        goto cleanup;
    }
    memset(q, 0xa5, sizeof(q));
    hmx_matmul_u8s4_tile(a, b, q, y, workspace);
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
        "I4EXAMPLE case %d errors %u q %u %u %u %u",
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
  path_length = snprintf(path, sizeof(path), "%s/output.u16", argv[1]);
  if (path_length < 0 || (size_t)path_length >= sizeof(path))
    goto cleanup;
  q16file = fopen(path, "wb");
  path_length = snprintf(path, sizeof(path), "%s/a16.f32", argv[1]);
  if (path_length < 0 || (size_t)path_length >= sizeof(path))
    goto cleanup;
  y16file = fopen(path, "wb");
  if (!q16file || !y16file)
    goto cleanup;

  for (int test = 0; test < 8; ++test) {
    uint32_t state = 987654321;
    for (int r = 0; r < HMX_A16_M; ++r) {
      for (int k = 0; k < HMX_K; ++k) {
        state = state * 1664525u + 1013904223u;
        a16[r * HMX_K + k] = test == 1
            ? 32768 + 256 * ((r * 3 + k * 7 + r * k) % 5 - 2)
            : test == 2 ? (r * 2017 + k * 2893 + r * k * 113) % 65536
            : test == 4 ? 0
            : test == 5 ? 65535
            : test == 6 ? state >> 16
            : test == 7 ? 32766 + (state >> 24) % 5
                        : 32768;
      }
    }
    for (int k = 0; k < HMX_K; ++k) {
      for (int n = 0; n < HMX_A16_N; ++n) {
        state = state * 1664525u + 1013904223u;
        b[k * HMX_A16_N + n] = test == 3 ? 0
            : test == 4                  ? -8
            : test == 5                  ? -8
            : test == 6                  ? (int)((state >> 24) % 16) - 8
            : test == 7                  ? (int)((state >> 24) % 5) - 2
                                         : (k * 11 + n * 19 + k * n) % 16 - 8;
      }
    }
    if (test == 3) {
      a16[17 * HMX_K + 13] = 33024;
      b[13 * HMX_A16_N + 49] = 1;
      a16[31 * HMX_K + 31] = 33280;
      b[31 * HMX_A16_N] = -8;
    }
    memset(q16, 0xa5, sizeof(q16));
    hmx_matmul_u16s4_tile(a16, b, q16, y, workspace);
    unsigned errors = 0;
    for (int r = 0; r < HMX_A16_M; ++r) {
      for (int n = 0; n < HMX_A16_N; ++n) {
        int acc = 0;
        for (int k = 0; k < HMX_K; ++k)
          acc += ((int)a16[r * HMX_K + k] - 32768) * b[k * HMX_A16_N + n];
        int quotient = acc / 256;
        if (acc < 0 && acc % 256)
          --quotient;
        int expected = quotient + 32768;
        expected = expected < 0 ? 0 : expected > 65535 ? 65535 : expected;
        int i = r * HMX_A16_N + n;
        errors += q16[i] != expected || y[i] != (expected - 32768) / 4.0f;
      }
    }
    FARF(
        ALWAYS,
        "A16EXAMPLE case %d errors %u q %u %u %u %u",
        test,
        errors,
        q16[0],
        q16[1],
        q16[2],
        q16[3]);
    if (errors || fwrite(q16, 1, sizeof(q16), q16file) != sizeof(q16) ||
        fwrite(y, sizeof(float), HMX_A16_M * HMX_A16_N, y16file) !=
            HMX_A16_M * HMX_A16_N)
      goto cleanup;
  }
  result = 0;

cleanup:
  if (q16file && fclose(q16file))
    result = 4;
  if (y16file && fclose(y16file))
    result = 4;
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
  FARF(ALWAYS, "I4EXAMPLE result %d", result);
  return result;
}
