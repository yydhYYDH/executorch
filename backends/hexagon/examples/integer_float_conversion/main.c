#include "conversion.h"

#include <stdio.h>

static float x[4096], f8[4096], f4[4096], widened[4096];
static int8_t q8[4096], q4[4096], unpacked[4096], r4[4096], r8[4096];
static uint8_t packed[2048];
static uint16_t half[4096];

int main(int argc, char** argv) {
  if (argc != 3)
    return 1;
  FILE* input = fopen(argv[1], "rb");
  FILE* output = fopen(argv[2], "wb");
  int status = 1;
  if (!input || !output)
    goto cleanup;
  uint32_t cases;
  if (fread(&cases, 4, 1, input) != 1 || cases > 64)
    goto cleanup;
  for (uint32_t test = 0; test < cases; ++test) {
    uint32_t count;
    float s8, s4;
    int32_t z8, z4;
    if (fread(&count, 4, 1, input) != 1 || count > 4096 ||
        fread(&s8, 4, 1, input) != 1 || fread(&s4, 4, 1, input) != 1 ||
        fread(&z8, 4, 1, input) != 1 || fread(&z4, 4, 1, input) != 1 ||
        fread(x, 4, count, input) != count)
      goto cleanup;
    if (quantize_sint(x, count, 8, s8, z8, q8) ||
        quantize_sint(x, count, 4, s4, z4, q4) ||
        pack_int4(q4, count, packed) ||
        dequantize_sint(q8, count, 8, s8, z8, f8) ||
        dequantize_sint(q4, count, 4, s4, z4, f4) ||
        requantize_sint(q8, count, 8, s8, z8, 4, s4, z4, r4) ||
        requantize_sint(q4, count, 4, s4, z4, 8, s8, z8, r8))
      goto cleanup;
    unpack_int4(packed, count, unpacked);
    fp32_to_fp16_bits(x, count, half);
    fp16_bits_to_fp32(half, count, widened);
    if (fwrite(q8, 1, count, output) != count ||
        fwrite(q4, 1, count, output) != count ||
        fwrite(packed, 1, (count + 1) / 2, output) != (count + 1) / 2 ||
        fwrite(unpacked, 1, count, output) != count ||
        fwrite(f8, 4, count, output) != count ||
        fwrite(f4, 4, count, output) != count ||
        fwrite(r4, 1, count, output) != count ||
        fwrite(r8, 1, count, output) != count ||
        fwrite(half, 2, count, output) != count ||
        fwrite(widened, 4, count, output) != count)
      goto cleanup;
    fp32_to_fp16_bits(f8, count, half);
    if (fwrite(half, 2, count, output) != count)
      goto cleanup;
    fp32_to_fp16_bits(f4, count, half);
    if (fwrite(half, 2, count, output) != count ||
        quantize_sint(widened, count, 8, s8, z8, q8) ||
        quantize_sint(widened, count, 4, s4, z4, q4) ||
        fwrite(q8, 1, count, output) != count ||
        fwrite(q4, 1, count, output) != count)
      goto cleanup;
  }
  for (uint32_t bits = 0; bits < 65536; bits += 4096) {
    for (uint32_t i = 0; i < 4096; ++i)
      half[i] = bits + i;
    fp16_bits_to_fp32(half, 4096, widened);
    fp32_to_fp16_bits(widened, 4096, half);
    if (fwrite(widened, 4, 4096, output) != 4096 ||
        fwrite(half, 2, 4096, output) != 4096)
      goto cleanup;
  }
  status = 0;
cleanup:
  if (input && fclose(input))
    status = 1;
  if (output && fclose(output))
    status = 1;
  return status;
}
