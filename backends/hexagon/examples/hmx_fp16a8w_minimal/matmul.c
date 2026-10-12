#include "matmul.h"

#include <string.h>

enum { M = HMX_FP16A8W_M, K = HMX_FP16A8W_K, N = HMX_FP16A8W_N };

void hmx_matmul_fp16a8w_tile(
    const __fp16* a,
    const int8_t* b,
    const float* scale,
    uint16_t* q,
    float* y,
    void* workspace) {
  __fp16* activation = workspace;
  __fp16* weight = (__fp16*)((uint8_t*)workspace + 2048);
  __fp16* output = (__fp16*)((uint8_t*)workspace + 6144);
  uint32_t* conversion = (uint32_t*)((uint8_t*)workspace + 12288);

  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k)
      activation[(r / 2) * 64 + k * 2 + r % 2] = a[r * K + k];
  for (int n = 0; n < N; ++n) {
    __fp16 scale_h = (__fp16)scale[n];
    for (int k = 0; k < K; ++k) {
      int index = (n / 32) * 1024 + (k / 2) * 64 + (n % 32) * 2 + k % 2;
      weight[index] = (__fp16)((float)b[k * N + n] * (float)scale_h);
    }
  }
  memset(conversion, 0, 256);
  for (int n = 0; n < 32; ++n) conversion[n] = 0x00003c00;

  for (int group = 0; group < 2; ++group) {
    asm volatile(
        "mxclracc.hf\n"
        "{ activation.hf = mxmem(%0,%1)\n"
        "weight.hf = mxmem(%2,%3) }\n"
        "bias = mxmem2(%4)\n"
        "cvt.hf = acc(%5)\n"
        "mxmem(%6,%7) = cvt\n"
        :
        : "r"(activation), "r"(0x77c), "r"(weight + group * 1024), "r"(0x7ff),
          "r"(conversion), "r"(2), "r"(output + group * 1024), "r"(0x700)
        : "memory");
  }
  for (int r = 0; r < M; ++r) {
    for (int n = 0; n < N; ++n) {
      __fp16 value = output[(n / 32) * 1024 + (r / 2) * 64 + (n % 32) * 2 + r % 2];
      memcpy(&q[r * N + n], &value, sizeof(value));
      y[r * N + n] = (float)value;
    }
  }
}
