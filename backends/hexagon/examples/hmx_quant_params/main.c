#include <stdio.h>

#include "../hmx_16a16w_minimal/matmul.h"
#include "../hmx_16a8w_minimal/matmul.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "conversion.h"

static uint16_t a[32 * 32], q[32 * 64];
static int8_t w8[32 * 64];
static int16_t w16[32 * 64];

int main(int argc, char** argv) {
  if (argc != 3)
    return 1;
  FILE *input = NULL, *coefficients = NULL, *output = NULL;
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
  input = fopen(argv[2], "rb");
  snprintf(path, sizeof(path), "%s/coefficients.u32", argv[1]);
  coefficients = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.u16", argv[1]);
  output = fopen(path, "wb");
  if (!input || !coefficients || !output)
    goto cleanup;
  uint32_t count;
  if (fread(&count, 4, 1, input) != 1 || count > 256)
    goto cleanup;
  for (unsigned test = 0; test < count; ++test) {
    float sa, sw, sq;
    uint16_t za, zq;
    uint32_t d, pair[2];
    if (fread(&sa, 4, 1, input) != 1 || fread(&sw, 4, 1, input) != 1 ||
        fread(&sq, 4, 1, input) != 1 || fread(&za, 2, 1, input) != 1 ||
        fread(&zq, 2, 1, input) != 1 || fread(&d, 4, 1, input) != 1 ||
        hmx_u16_conversion_params(sa, sw, sq, zq, d, pair))
      goto cleanup;
    for (int r = 0; r < 32; ++r)
      for (int k = 0; k < 32; ++k) {
        int value = za + (r * 71 + k * 113 + r * k) % 4097 - 2048;
        a[r * 32 + k] = value < 0 ? 0 : value > 65535 ? 65535 : value;
      }
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < 64; ++n) {
        int low = (k * 11 + n * 19 + k * n) % 256 - 128;
        int high = (k * 17 + n * 7 + k * n) % 63 - 31;
        w8[k * 64 + n] = low;
        w16[k * 64 + n] = high * 256 + low;
      }
    int rc = d == 8 ? hmx_matmul_a16w8_tile_quantized(
                          a, w8, sa, sw, sq, za, zq, q, workspace)
                    : hmx_matmul_a16w16_tile_quantized(
                          a, w16, sa, sw, sq, za, zq, q, workspace);
    if (rc || fwrite(pair, 4, 2, coefficients) != 2 ||
        fwrite(q, 2, 32 * 64, output) != 32 * 64)
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
  if (locked && HAP_compute_res_hmx_unlock(context))
    result = 1;
  if (context && HAP_compute_res_release(context))
    result = 1;
  power.hmx.power_up = 0;
  if (HAP_power_set(&power_context, &power))
    result = 1;
  return result;
}
