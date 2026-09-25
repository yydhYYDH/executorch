#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stddef.h>
#include <stdint.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx_utils.h"

extern "C" {

static inline HVX_Vector htp_ops_arg_reduce_vec(HVX_Vector best_v,
                                                HVX_Vector value_v,
                                                int32_t is_min) {
  return is_min ? Q6_Vhf_vmin_VhfVhf(best_v, value_v)
                : Q6_Vhf_vmax_VhfVhf(best_v, value_v);
}

static inline int htp_ops_arg_reduce_is_nan(uint16_t bits) {
  return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0;
}

static inline int htp_ops_arg_reduce_is_zero(uint16_t bits) {
  return (bits & 0x7FFFu) == 0;
}

/* This is the vector walk used by both topk and the position command. The
 * vector part covers size & -64 elements; the scalar tail is deliberately
 * strict, so the first occurrence wins ordinary ties. */
static inline uint16_t htp_ops_arg_reduce_bits(const __fp16* src,
                                               int32_t size,
                                               int32_t is_min) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16* ptr = src;
  int i = 0;

  __fp16 best_scalar = src[0];
  HVX_Vector best_v = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)ptr);
    best_v = htp_ops_arg_reduce_vec(best_v, v, is_min);
    ptr += vec_len;
  }

  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 64), is_min);
  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 32), is_min);
  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 16), is_min);
  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 8), is_min);
  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 4), is_min);
  best_v = htp_ops_arg_reduce_vec(
      best_v, Q6_V_vror_VR(best_v, 2), is_min);

  __attribute__((aligned(128))) uint16_t tmp[vec_len];
  vmemu((HVX_Vector*)tmp) = best_v;
  uint16_t best_bits = tmp[0];
  best_scalar = *(__fp16*)&best_bits;

  for (; i < size; ++i) {
    const __fp16 value = src[i];
    const int wins = is_min ? value < best_scalar : value > best_scalar;
    if (wins) {
      best_scalar = value;
      best_bits = ((const uint16_t*)src)[i];
    }
  }
  return best_bits;
}

static inline uint16_t htp_ops_topk_fp16_max_bits(const __fp16* src,
                                                   int32_t size) {
  return htp_ops_arg_reduce_bits(src, size, 0);
}

AEEResult htp_ops_topkv2_k1_fp16(uint8_t* values,
                                  uint8_t* indices,
                                  uint8_t* input,
                                  int32_t rowSize,
                                  int32_t rows) {
  if (values == nullptr || indices == nullptr || input == nullptr ||
      rowSize <= 0 || rows <= 0) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  __fp16* valueOut = (__fp16*)values;
  int32_t* indexOut = (int32_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    const uint16_t bestBits = htp_ops_topk_fp16_max_bits(row, rowSize);
    const uint16_t* rowBits = (const uint16_t*)row;
    int32_t bestIndex = 0;
    for (int i = 0; i < rowSize; ++i) {
      if (rowBits[i] == bestBits) {
        bestIndex = i;
        break;
      }
    }
    ((uint16_t*)valueOut)[r] = bestBits;
    indexOut[r] = bestIndex;
  }
  return 0;
}

/* One row produces one Torch-compatible position. NaN is handled before the
 * numeric reduction: CPU argmax and argmin both choose the first NaN. The
 * zero special case makes -0 and +0 one tie, rather than making the sign chosen
 * by the HVX max/min instruction decide the index. */
AEEResult htp_ops_argmax_fp16(uint8_t* indices,
                               const uint8_t* input,
                               int32_t rowSize,
                               int32_t rows,
                               int32_t is_min) {
  if (indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0 ||
      (is_min != 0 && is_min != 1)) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  int64_t* indexOut = (int64_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    const uint16_t* rowBits = (const uint16_t*)row;
    int32_t bestIndex = -1;
    for (int32_t i = 0; i < rowSize; ++i) {
      if (htp_ops_arg_reduce_is_nan(rowBits[i])) {
        bestIndex = i;
        break;
      }
    }
    if (bestIndex < 0) {
      const uint16_t bestBits = htp_ops_arg_reduce_bits(row, rowSize, is_min);
      if (htp_ops_arg_reduce_is_zero(bestBits)) {
        for (int32_t i = 0; i < rowSize; ++i) {
          if (htp_ops_arg_reduce_is_zero(rowBits[i])) {
            bestIndex = i;
            break;
          }
        }
      }
      if (bestIndex < 0) {
        for (int32_t i = 0; i < rowSize; ++i) {
          if (rowBits[i] == bestBits) {
            bestIndex = i;
            break;
          }
        }
      }
    }
    indexOut[r] = bestIndex;
  }
  return 0;
}

} // extern "C"
