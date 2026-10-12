#include <stdio.h>

#include <string.h>
#include "../hmx_int4_minimal/matmul.h"

int hmx_matmul_u8s8_tile_quantized(
    const uint8_t*,
    const int8_t*,
    float,
    float,
    float,
    uint8_t,
    int,
    uint8_t,
    uint8_t*,
    float*,
    void*);
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "conversion.h"

static uint16_t a[64 * 32], q[64 * 128];
static uint8_t a8[64 * 32], q8[64 * 128];
static int8_t w[32 * 128];
static float y[64 * 128];
static uint8_t saved_workspace[16384];

int main(int argc, char** argv) {
  if (argc != 3)
    return 1;
  FILE *input = NULL, *coefficients = NULL, *output = NULL, *dequantized = NULL;
  char path[512];
  unsigned context = 0;
  int locked = 0, result = 1, power_context = 0;
  HAP_power_request_t power = {0};
  power.type = HAP_power_set_HMX;
  power.hmx.power_up = 1;
  if (HAP_power_set(&power_context, &power))
    return 1;
  compute_res_attr_t attr;
  if (HAP_compute_res_attr_init(&attr) ||
      HAP_compute_res_attr_set_vtcm_param(&attr, 16384, 1) ||
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
  memset(workspace, 0x5a, 16384);
  memcpy(saved_workspace, workspace, 16384);
  memset(q, 0x5a, sizeof(q));
  memset(q8, 0x5a, sizeof(q8));
  memset(y, 0x5a, sizeof(y));
  for (int kind = 0; kind < 3; ++kind) {
    for (int bad = 0; bad < 2; ++bad) {
      w[0] = bad ? 127 : 0;
      float sa = bad ? 0.125f : 0;
      int zw = bad ? -1 : 0;
      int rc = kind == 0
          ? hmx_matmul_u8s8_tile_quantized(
                a8, w, sa, 0.25f, 0.03125f, 0, zw, 0, q8, y, workspace)
          : kind == 1
          ? hmx_matmul_u8s4_tile_quantized(
                a8, w, sa, 0.25f, 0.03125f, 0, zw, 0, q8, y, workspace)
          : hmx_matmul_u16s4_tile_quantized(
                a, w, sa, 0.25f, 8.0f, 0, zw, 0, q, y, workspace);
      if (rc != 1 || memcmp(saved_workspace, workspace, 16384))
        goto cleanup;
      const uint8_t* outputs[] = {(const uint8_t*)q, q8, (const uint8_t*)y};
      const unsigned sizes[] = {sizeof(q), sizeof(q8), sizeof(y)};
      for (int buffer = 0; buffer < 3; ++buffer)
        for (unsigned i = 0; i < sizes[buffer]; ++i)
          if (outputs[buffer][i] != 0x5a)
            goto cleanup;
    }
  }
  input = fopen(argv[2], "rb");
  snprintf(path, sizeof(path), "%s/coefficients.u32", argv[1]);
  coefficients = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.u16", argv[1]);
  output = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.f32", argv[1]);
  dequantized = fopen(path, "wb");
  if (!input || !coefficients || !output || !dequantized)
    goto cleanup;
  uint32_t count;
  if (fread(&count, 4, 1, input) != 1 || count > 512)
    goto cleanup;
  for (unsigned test = 0; test < count; ++test) {
    float sa, sw, sq;
    int32_t za, zw, zq, kind;
    uint32_t pair[2] = {0};
    if (fread(&sa, 4, 1, input) != 1 || fread(&sw, 4, 1, input) != 1 ||
        fread(&sq, 4, 1, input) != 1 || fread(&za, 4, 1, input) != 1 ||
        fread(&zw, 4, 1, input) != 1 || fread(&zq, 4, 1, input) != 1 ||
        fread(&kind, 4, 1, input) != 1 || kind < 0 || kind > 2)
      goto cleanup;
    int m = kind == 2 ? 32 : 64, n = kind == 1 ? 128 : 64;
    if (za < 0 || zq < 0 || za > (kind == 2 ? 65535 : 255) ||
        zq > (kind == 2 ? 65535 : 255) ||
        fread(a, 2, m * 32, input) != (unsigned)m * 32 ||
        fread(w, 1, 32 * n, input) != (unsigned)32 * n)
      goto cleanup;
    int rc;
    if (kind == 2) {
      rc = hmx_u16_conversion_params(sa, sw, sq, zq, 8, pair) ||
          hmx_matmul_u16s4_tile_quantized(
               a, w, sa, sw, sq, za, zw, zq, q, y, workspace);
    } else {
      for (int i = 0; i < m * 32; ++i) {
        if (a[i] > 255)
          goto cleanup;
        a8[i] = a[i];
      }
      rc = hmx_u8_conversion_params(sa, sw, sq, zq, kind == 1 ? 4 : 0, pair);
      if (!rc)
        rc = kind == 0 ? hmx_matmul_u8s8_tile_quantized(
                             a8, w, sa, sw, sq, za, zw, zq, q8, y, workspace)
                       : hmx_matmul_u8s4_tile_quantized(
                             a8, w, sa, sw, sq, za, zw, zq, q8, y, workspace);
      for (int i = 0; i < m * n; ++i)
        q[i] = q8[i];
    }
    if (rc || fwrite(pair, 4, 2, coefficients) != 2 ||
        fwrite(q, 2, m * n, output) != (unsigned)m * n ||
        fwrite(y, 4, m * n, dequantized) != (unsigned)m * n)
      goto cleanup;
  }
  result = 0;
cleanup:
  if (input && fclose(input))
    result = 1;
  if (coefficients && fclose(coefficients))
    result = 1;
  if (output && fclose(output))
    result = 1;
  if (dequantized && fclose(dequantized))
    result = 1;
  if (locked && HAP_compute_res_hmx_unlock(context))
    result = 1;
  if (context && HAP_compute_res_release(context))
    result = 1;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power))
    result = 1;
  return result;
}
