#include <stdio.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "matmul.h"

static int power_context;

enum {
  TEST_MAX_M = 256,
  TEST_MAX_N = 256,
  TEST_MAX_K = 512
};

static uint32_t native_index(int r, int n) {
  return (n / 32) * 2048 + (r / 4) * 128 + (n % 32) * 4 + r % 4;
}

static uint8_t expected_q(int64_t r, int64_t n, int64_t nn, int64_t kk,
                          const uint8_t* a, const int8_t* b) {
  int64_t acc = 0;
  for (int64_t k = 0; k < kk; ++k)
    acc += ((int32_t)a[r * kk + k] - 128) * b[k * nn + n];
  int64_t q = acc + 128;
  return (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
}

static void fill_inputs(uint8_t* a, int8_t* b, int64_t m, int64_t n,
                        int64_t k, int test) {
  for (int64_t r = 0; r < m; ++r)
    for (int64_t kk = 0; kk < k; ++kk) {
      int value = (int)(r * 17 + kk * 29 + r * kk * 3 + test * 11);
      a[r * k + kk] = test == 0 ? 128 : (uint8_t)value;
    }
  for (int64_t kk = 0; kk < k; ++kk)
    for (int64_t c = 0; c < n; ++c) {
      int value =
          (int)(kk * 11 + c * 19 + kk * c * 5 + test * 7) % 17 - 8;
      b[kk * n + c] = (int8_t)value;
    }
}

static int run_case(int64_t m, int64_t n, int64_t k, int test, FILE* file) {
  static uint8_t a[TEST_MAX_M * TEST_MAX_K];
  static int8_t b[TEST_MAX_K * TEST_MAX_N];
  static uint8_t tiles[4 * 4 * HMX_ANY_TILE_BYTES];
  if (m > TEST_MAX_M || n > TEST_MAX_N || k > TEST_MAX_K)
    return 1;

  fill_inputs(a, b, m, n, k, test);
  memset(tiles, 0xa5, sizeof(tiles));

  compute_res_attr_t attr;
  unsigned context = 0;
  int locked = 0;
  int result = 1;
  const uint32_t workspace_bytes =
      (HMX_ANY_WORKSPACE_BYTES(k) + 4095) & ~4095u;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, workspace_bytes, 1) ||
      HAP_compute_res_attr_set_hmx_param(&attr, 1))
    goto cleanup;
  context = HAP_compute_res_acquire(&attr, 100000);
  if (!context)
    goto cleanup;
  void* workspace = HAP_compute_res_attr_get_vtcm_ptr(&attr);
  if (!workspace || ((uintptr_t)workspace & 2047) ||
      HAP_compute_res_hmx_lock(context))
    goto cleanup;
  locked = 1;

  if (hmx_matmul_u8s8_any_k(a, b, m, n, k, tiles, workspace))
    goto cleanup;

  const int64_t mt = (m + HMX_ANY_M_TILE - 1) / HMX_ANY_M_TILE;
  const int64_t nt = (n + HMX_ANY_N_TILE - 1) / HMX_ANY_N_TILE;
  uint32_t errors = 0;
  for (int64_t mi = 0; mi < mt; ++mi) {
    for (int64_t ni = 0; ni < nt; ++ni) {
      const uint8_t* tile = tiles +
          (ni + nt * mi) * HMX_ANY_TILE_BYTES;
      for (int64_t r = 0; r < HMX_ANY_M_TILE; ++r) {
        for (int64_t c = 0; c < HMX_ANY_N_TILE; ++c) {
          if (mi * HMX_ANY_M_TILE + r >= m ||
              ni * HMX_ANY_N_TILE + c >= n)
            continue;
          uint8_t got = tile[native_index((int)r, (int)c)];
          uint8_t expected = expected_q(
              mi * HMX_ANY_M_TILE + r, ni * HMX_ANY_N_TILE + c, n, k, a, b);
          errors += got != expected;
        }
      }
    }
  }
  FARF(ALWAYS,
       "ANYK case test %d m %lld n %lld k %lld tiles %lldx%lld errors %u",
       test, (long long)m, (long long)n, (long long)k,
       (long long)mt, (long long)nt, errors);
  if (errors ||
      fwrite(tiles, 1, (size_t)(mt * nt) * HMX_ANY_TILE_BYTES, file) !=
          (size_t)(mt * nt) * HMX_ANY_TILE_BYTES)
    goto cleanup;
  result = 0;

cleanup:
  if (locked && HAP_compute_res_hmx_unlock(context))
    result = 1;
  if (context && HAP_compute_res_release(context))
    result = 1;
  return result;
}

static int reject_k(int64_t k) {
  uint8_t a[1];
  int8_t b[1];
  uint8_t tiles[1];
  uint8_t workspace[1];
  return hmx_matmul_u8s8_any_k(a, b, 1, 1, k, tiles, workspace) == 1;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    FARF(ALWAYS, "ANYK expected output directory");
    return 1;
  }
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  if (HAP_power_set(&power_context, &power))
    return 2;

  char path[1024];
  int length = snprintf(path, sizeof(path), "%s/tiles.u8", argv[1]);
  if (length < 0 || (size_t)length >= sizeof(path))
    return 3;
  FILE* file = fopen(path, "wb");
  if (!file)
    return 3;

  const int64_t shapes[][3] = {
      {1, 1, 1}, {1, 1, 31}, {1, 1, 33}, {1, 63, 64},
      {63, 1, 95}, {63, 63, 96}, {64, 64, 127}, {65, 65, 129},
      {127, 129, 256}, {129, 127, 320}, {200, 200, 512},
      {200, 200, 31}, {1, 1, 512}, {65, 65, 33},
  };
  int result = 0;
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
    if (run_case(
            shapes[i][0], shapes[i][1], shapes[i][2], (int)(i % 3), file)) {
      FARF(ALWAYS, "ANYK failed m %lld n %lld k %lld",
           (long long)shapes[i][0], (long long)shapes[i][1],
           (long long)shapes[i][2]);
      result = 4;
      break;
    }
  }
  if (result == 0 && (!reject_k(0) || !reject_k(-1))) {
    FARF(ALWAYS, "ANYK failed invalid K rejection");
    result = 4;
  }
  if (fclose(file))
    result = 4;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power))
    result = 4;
  FARF(ALWAYS, "ANYK result %d", result);
  return result;
}
