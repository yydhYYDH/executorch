#include <stdio.h>

#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "matmul.h"

static uint16_t a[32 * 32], q[32 * 64];
static int8_t w[32 * 64];
static float scales[64], y[32 * 64];
static uint8_t packed[512];
static uint32_t table[256];

int main(int argc, char** argv) {
  if (argc != 3)
    return 1;
  FILE *input = NULL, *coefficients = NULL, *output = NULL, *yf = NULL,
       *wf = NULL;
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
  snprintf(path, sizeof(path), "%s/table.u32", argv[1]);
  coefficients = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.u16", argv[1]);
  output = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/output.f32", argv[1]);
  yf = fopen(path, "wb");
  snprintf(path, sizeof(path), "%s/weights.u8", argv[1]);
  wf = fopen(path, "wb");
  if (!input || !coefficients || !output || !yf || !wf)
    goto cleanup;
  uint32_t count;
  if (fread(&count, 4, 1, input) != 1 || count > 256)
    goto cleanup;
  for (unsigned test = 0; test < count; ++test) {
    float sa, sq;
    int32_t za, zw, zq;
    if (fread(&sa, 4, 1, input) != 1 || fread(&sq, 4, 1, input) != 1 ||
        fread(&za, 4, 1, input) != 1 || fread(&zw, 4, 1, input) != 1 ||
        fread(&zq, 4, 1, input) != 1 || za < 0 || za > 65535 || zq < 0 ||
        zq > 65535 || fread(scales, 4, 64, input) != 64 ||
        fread(a, 2, 1024, input) != 1024 || fread(w, 1, 2048, input) != 2048)
      goto cleanup;
    if (hmx_a16w2_prepare(w, scales, sa, sq, za, zw, zq, packed, table))
      goto cleanup;
    if (test % 2 == 0) {
      if (hmx_matmul_u16s2_tile(
              a, w, scales, sa, sq, za, zw, zq, q, y, workspace))
        goto cleanup;
    } else {
      hmx_matmul_u16s2_prepared_tile(a, packed, table, sq, zq, q, y, workspace);
    }
    if (fwrite(table, 4, 256, coefficients) != 256 ||
        fwrite(packed, 1, 512, wf) != 512 ||
        fwrite(q, 2, 2048, output) != 2048 || fwrite(y, 4, 2048, yf) != 2048)
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
  if (yf && fclose(yf))
    result = 1;
  if (wf && fclose(wf))
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
