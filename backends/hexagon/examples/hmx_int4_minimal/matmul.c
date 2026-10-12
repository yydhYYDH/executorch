#include "matmul.h"
#include "../hmx_quant_params/conversion.h"

__attribute__((noinline)) int hmx_matmul_u8s4_tile_quantized(
    const uint8_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint8_t za,
    int weight_zero_point,
    uint8_t zq,
    uint8_t* q,
    float* y,
    void* workspace) {
  uint32_t coefficient;
  if (hmx_u8_conversion_params(sa, sw, sq, zq, 4, &coefficient))
    return 1;
  if (weight_zero_point < -128 || weight_zero_point > 127)
    return 1;
  for (int i = 0; i < 32 * 128; ++i) {
    int value = (int)b[i] - weight_zero_point;
    if (value < -8 || value > 7)
      return 1;
  }
  uint8_t* activation = workspace;
  uint8_t* weight = (uint8_t*)workspace + 2048;
  uint8_t* native_output = (uint8_t*)workspace + 4096;
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 12288);

  for (int r = 0; r < HMX_M; ++r) {
    for (int k = 0; k < HMX_K; ++k)
      activation[r * 32 + k] = a[r * HMX_K + k];
  }
  for (int k = 0; k < HMX_K; ++k) {
    for (int n = 0; n < 64; ++n) {
      int index = (n / 32) * 1024 + (k / 4) * 128 + (n % 32) * 4 + k % 4;
      weight[index] = ((uint8_t)(b[k * HMX_N + (n / 32) * 64 + n % 32] -
                                 weight_zero_point) &
                       15) |
          (((uint8_t)(b[k * HMX_N + (n / 32) * 64 + n % 32 + 32] -
                      weight_zero_point) &
            15)
           << 4);
    }
  }
  for (int n = 0; n < HMX_N; ++n) {
    int sum = 0;
    for (int k = 0; k < HMX_K; ++k) {
      sum += (b[k * HMX_N + n] - weight_zero_point);
    }
    // The :2x:deep nibble path scales the byte accumulator by 16.
    conversion[(n / 32) * 64 + n % 32] = coefficient;
    conversion[(n / 32) * 64 + 32 + n % 32] = (uint32_t)(-(int)za * 16 * sum);
  }

  asm volatile(
      "mxclracc\n"
      "{ activation.ub = mxmem(%0,%1):cm\n"
      "weight.n = mxmem(%2,%3):2x:deep }\n"
      :
      : "r"(activation), "r"(0x71f), "r"(weight), "r"(2047)
      : "memory");
  // Converting group 0 first destroys state needed by group 1.
  for (int index = 0; index < 4; ++index) {
    int group = index < 2 ? 1 - index : index;
    uint32_t* table = conversion + group * 64;
    int selector = (group == 1 || group == 2);
    if (group % 2 == 0) {
      asm volatile(
          "bias = mxmem2(%0)\ncvt.ub = acc(%1):sc0\nmxmem(%2,%3):cm = cvt\n"
          :
          : "r"(table),
            "r"(selector),
            "r"(native_output + group * 2048),
            "r"(0x700)
          : "memory");
    } else {
      asm volatile(
          "bias = mxmem2(%0)\ncvt.ub = acc(%1):sc1\nmxmem(%2,%3):cm = cvt\n"
          :
          : "r"(table),
            "r"(selector),
            "r"(native_output + group * 2048),
            "r"(0x700)
          : "memory");
    }
  }

  for (int r = 0; r < HMX_M; ++r) {
    for (int n = 0; n < HMX_N; ++n) {
      int native_index = (n / 32) * 2048 + r * 32 + n % 32;
      uint8_t value = native_output[native_index];
      q[r * HMX_N + n] = value;
      y[r * HMX_N + n] = ((int)value - zq) * sq;
    }
  }
  return 0;
}

__attribute__((noinline)) int hmx_matmul_u16s4_tile_quantized(
    const uint16_t* a,
    const int8_t* b,
    float sa,
    float sw,
    float sq,
    uint16_t za,
    int weight_zero_point,
    uint16_t zq,
    uint16_t* q,
    float* y,
    void* workspace) {
  uint32_t coefficient[2];
  if (hmx_u16_conversion_params(sa, sw, sq, zq, 8, coefficient))
    return 1;
  if (weight_zero_point < -128 || weight_zero_point > 127)
    return 1;
  for (int i = 0; i < 32 * 64; ++i) {
    int value = (int)b[i] - weight_zero_point;
    if (value < -8 || value > 7)
      return 1;
  }
  uint16_t* activation = workspace;
  uint8_t* weight = (uint8_t*)workspace + 2048;
  uint16_t* native_output = (uint16_t*)((uint8_t*)workspace + 4096);
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 12288);

  for (int r = 0; r < HMX_A16_M; ++r) {
    for (int k = 0; k < HMX_K; ++k) {
      activation[(r / 2) * 64 + k * 2 + r % 2] = a[r * HMX_K + k];
    }
  }
  for (int k = 0; k < HMX_K; k += 8) {
    for (int n = 0; n < HMX_A16_N; ++n) {
      for (int i = 0; i < 4; ++i) {
        int index = (n / 32) * 512 + (k / 8) * 128 + (n % 32) * 4 + i;
        weight[index] =
            ((uint8_t)(b[(k + i) * HMX_A16_N + n] - weight_zero_point) & 15) |
            (((uint8_t)(b[(k + i + 4) * HMX_A16_N + n] - weight_zero_point) &
              15)
             << 4);
      }
    }
  }
  for (int n = 0; n < HMX_A16_N; ++n) {
    int sum = 0;
    for (int k = 0; k < HMX_K; ++k) {
      sum += (b[k * HMX_A16_N + n] - weight_zero_point);
    }
    // The two tables in each N32 group select even and odd columns.
    int index = ((n / 32) * 2 + n % 2) * 64 + ((n % 32) / 2) * 2;
    conversion[index] = coefficient[0];
    conversion[index + 1] = coefficient[1];
    int num = (int)za * sum;
    int corr = num >= 0 ? (num + 128) / 256 : -((-num + 128) / 256);
    conversion[index + 32] = (uint32_t)(-corr);
    conversion[index + 33] = 0;
  }

  asm volatile(
      "mxclracc\n"
      "{ activation.ub = mxmem(%0,%1)\n"
      "weight.n = mxmem(%2,%3):deep }\n"
      :
      : "r"(activation), "r"(0x77c), "r"(weight), "r"(0x3ff)
      : "memory");
  for (int group = 0; group < 2; ++group) {
    uint32_t* table = conversion + group * 128;
    asm volatile(
        "bias = mxmem2(%0)\n"
        "cvt.uh = acc(%1):2x2\n"
        "bias = mxmem2(%2)\n"
        "cvt.uh = acc(%3):2x2\n"
        "mxmem(%4,%5) = cvt\n"
        :
        : "r"(table),
          "r"(0x401),
          "r"(table + 64),
          "r"(0x20c),
          "r"(native_output + group * 1024),
          "r"(0x700)
        : "memory");
  }
  for (int r = 0; r < HMX_A16_M; ++r) {
    for (int n = 0; n < HMX_A16_N; ++n) {
      int index = (n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2;
      uint16_t value = native_output[index];
      q[r * HMX_A16_N + n] = value;
      y[r * HMX_A16_N + n] = ((int)value - zq) * sq;
    }
  }
  return 0;
}

void hmx_matmul_u8s4_tile(
    const uint8_t* a,
    const int8_t* b,
    uint8_t* q,
    float* y,
    void* workspace) {
  (void)hmx_matmul_u8s4_tile_quantized(
      a, b, 1.0f / 8, 1.0f / 4, 1.0f / 32, 128, 0, 128, q, y, workspace);
}

void hmx_matmul_u16s4_tile(
    const uint16_t* a,
    const int8_t* b,
    uint16_t* q,
    float* y,
    void* workspace) {
  (void)hmx_matmul_u16s4_tile_quantized(
      a, b, 1.0f / 256, 1.0f / 4, 1.0f / 4, 32768, 0, 32768, q, y, workspace);
}
