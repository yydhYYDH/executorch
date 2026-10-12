#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

enum { M = HMX_LPBQ_M, K = HMX_LPBQ_K, N = HMX_LPBQ_N };
static uint16_t a[M * K], q[M * N];
static int8_t w[K * N], packed_weights[HMX_LPBQ_WEIGHT_BYTES];
static uint32_t conversion_table[HMX_LPBQ_TABLE_BYTES / 4];
static float scales[N * HMX_LPBQ_BLOCKS], y[M * N];

int main(int argc, char** argv) {
  const char* directory = argc > 1 ? argv[1] : ".";
  char path[512];
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  int power_context = 0;
  if (HAP_power_set(&power_context, &power))
    return 1;
  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0, result = 3;
  FILE *qfile = NULL, *yfile = NULL, *fixture = NULL, *prepared = NULL;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, HMX_LPBQ_WORKSPACE_BYTES, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1))
    goto cleanup;
  context = HAP_compute_res_acquire(&attr, 1000000);
  if (!context)
    goto cleanup;
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context))
    goto cleanup;
  locked = 1;
  snprintf(path, sizeof(path), "%s/output.q", directory);
  qfile = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.y", directory);
  yfile = fopen(path, "wb");
  if (!qfile || !yfile)
    goto cleanup;
  uint32_t fixture_count = 0;
  if (argc > 2) {
    fixture = fopen(argv[2], "rb");
    if (!fixture || fread(&fixture_count, 4, 1, fixture) != 1 ||
        fixture_count > 32)
      goto cleanup;
  }
  if (argc > 3) {
    uint32_t prepared_count;
    prepared = fopen(argv[3], "rb");
    if (!prepared || fread(&prepared_count, 4, 1, prepared) != 1 ||
        prepared_count != HMX_LPBQ_TEST_CASES + fixture_count)
      goto cleanup;
  }
  for (unsigned test = 0; test < HMX_LPBQ_TEST_CASES + fixture_count; ++test) {
    float sa = 1.0f / 4096, sq = 1.0f / 8192;
    float global_weight_scale = 0;
    uint16_t za = 32768, zq = 29520;
    if (test < HMX_LPBQ_TEST_CASES) {
      uint32_t state = 1234567u + test;
      for (int i = 0; i < M * K; ++i) {
        state = state * 1664525u + 1013904223u;
        int range = test == 1 ? 9 : test == 2 ? 257 : 1201;
        a[i] = (uint16_t)(za + (int)((state >> 16) % range) - range / 2);
        if (test == 0)
          a[i] = za;
        if (test == 4)
          a[i] = (uint16_t)(state >> 16);
        if (test == 6 || test == 7 || test == 11)
          a[i] = 65535;
        if (test == 8)
          a[i] = za + (i % K == 111 ? 1024 : 0);
      }
      for (int i = 0; i < K * N; ++i) {
        state = state * 1664525u + 1013904223u;
        w[i] = (int8_t)((state >> 16) % 16 - 8);
        if (test == 5)
          w[i] = 0;
        if (test == 6 || test == 7)
          w[i] = test == 6 ? -8 : 7;
        if (test == 8)
          w[i] = i / N == 111 && i % N == 49 ? -8 : 0;
        if (test == 11)
          w[i] = i / N < 32 ? 7 : i / N < 64 ? -7 : 0;
      }
      for (int n = 0; n < N; ++n) {
        for (int b = 0; b < HMX_LPBQ_BLOCKS; ++b) {
          float maximum = (16 + n % 16) * (1.0f / 1024);
          float ratio = b == 3 ? 16 : 1 + (n * 3 + b * 5) % 16;
          if (test == 10)
            ratio = b == 3 ? 16 : 0.5f + (n + b) % 15;
          scales[n * HMX_LPBQ_BLOCKS + b] = maximum * ratio * (1.0f / 16);
          if (test == 9 && n % 3 == 0 && b < 3)
            scales[n * HMX_LPBQ_BLOCKS + b] = 0;
          if (test == 11)
            scales[n * HMX_LPBQ_BLOCKS + b] = maximum;
        }
      }
      if (test == 11)
        za = 0;
      if (test == 4 || test == 9)
        zq = test == 4 ? 32768 : 65500;
    } else {
      if (fread(a, 1, sizeof(a), fixture) != sizeof(a) ||
          fread(w, 1, sizeof(w), fixture) != sizeof(w) ||
          fread(scales, 1, sizeof(scales), fixture) != sizeof(scales) ||
          fread(
              &global_weight_scale, sizeof(global_weight_scale), 1, fixture) !=
              1 ||
          fread(&sa, sizeof(sa), 1, fixture) != 1 ||
          fread(&sq, sizeof(sq), 1, fixture) != 1 ||
          fread(&za, sizeof(za), 1, fixture) != 1 ||
          fread(&zq, sizeof(zq), 1, fixture) != 1)
        goto cleanup;
    }
    memset(workspace, test % 2 ? 0xa5 : 0x5a, HMX_LPBQ_WORKSPACE_BYTES);
    if (prepared) {
      if (fread(packed_weights, 1, sizeof(packed_weights), prepared) !=
              sizeof(packed_weights) ||
          fread(conversion_table, 1, sizeof(conversion_table), prepared) !=
              sizeof(conversion_table) ||
          hmx_matmul_u16s4_lpbq_prepared_tile(
              a, packed_weights, conversion_table, sq, zq, q, y, workspace))
        goto cleanup;
    } else if (hmx_matmul_u16s4_lpbq_tile(
                   a,
                   w,
                   scales,
                   global_weight_scale,
                   sa,
                   za,
                   sq,
                   zq,
                   q,
                   y,
                   workspace)) {
      goto cleanup;
    }
    FARF(
        ALWAYS,
        "LPBQEXAMPLE test %u q0 %u %u %u %u",
        test,
        q[0],
        q[1],
        q[2],
        q[3]);
    if (fwrite(q, 1, sizeof(q), qfile) != sizeof(q) ||
        fwrite(y, 1, sizeof(y), yfile) != sizeof(y))
      goto cleanup;
  }
  result = 0;
cleanup:
  if (qfile && fclose(qfile))
    result = 4;
  if (yfile && fclose(yfile))
    result = 4;
  if (fixture && fclose(fixture))
    result = 4;
  if (prepared && fclose(prepared))
    result = 4;
  if (locked && HAP_compute_res_hmx_unlock(context))
    result = 4;
  if (context && HAP_compute_res_release(context))
    result = 4;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power))
    result = 4;
  FARF(ALWAYS, "LPBQEXAMPLE result %d host_prepared %d", result, argc > 3);
  return result;
}
