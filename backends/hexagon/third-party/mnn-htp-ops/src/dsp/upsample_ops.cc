/* Bilinear upsampling of an NCHW fp16 plane set, by an integer factor per axis.
 *
 * ATen spells this op as aten::upsample_bilinear2d.vec, and the arithmetic the
 * portable kernel does is four taps with two pairs of weights. Reproducing it
 * bit for bit means reproducing three things exactly, and each is a line of ATen
 * rather than a choice:
 *
 *   - the scale. compute_scales_value returns (float)(1.0 / scale_factor) when
 *     the caller gave a factor and (float)in / out when it gave a size. For an
 *     integer factor the two are the same correctly-rounded float, so the
 *     command carries the integer and this computes (float)(1.0 / (double)s).
 *   - the source coordinate. area_pixel_compute_source_index is
 *     scale * (dst + 0.5) - 0.5, clamped at zero -- one expression over the
 *     *output* index. Splitting it into k + (scale * (t + 0.5) - 0.5) for the
 *     phase t = dst % s and the block k = dst / s changes the last bit, because
 *     the product is no longer rounded on its own. The per-index form is what is
 *     written here, which costs one multiply per output element and is the
 *     reason the inner loop is scalar.
 *   - the accumulation order. h0l * (w0l * a + w1l * b) + h1l * (w0l * c +
 *     w1l * d) in fp32, with the fp16 loads widened, and one rounding to fp16
 *     at the store. The separable form top + h1l * (bot - top) is equal over
 *     the reals and is not the same function in floating point.
 */
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stddef.h>
#include <stdint.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

extern "C" {

/* The widest output row the two axis tables cover, and so the size of the
 * stack frame: 2 axes * 512 entries * 8 bytes = 8192 bytes. The bound is set by
 * the DSP RPC thread's stack, not by VTCM: a 32768-byte frame overflows it and
 * the command group comes back 0x8000040d, which is the same signature the skel
 * records for an -O0 branch chain at 14 KB. A vocoder row is a few hundred
 * samples wide and a super-resolution row a couple of hundred, so 512 is above
 * every geometry either family produces; hexagon_ops.py refuses anything wider,
 * and the two bounds are meant to move together. */
#define HTP_OPS_BILINEAR_AXIS_MAX 512

/* One output index: the source position below the tap and the weight above it.
 * low is the floor of the source coordinate and w the fraction, with the
 * src < 0 clamp ATen applies already folded in. The upper tap is low + 1
 * clamped to the input extent, which is why it is not stored. */
struct htp_bilinear_axis {
  int32_t low;
  float w;
};

/* The frame is the number the op-support table and the report both quote, so it
 * is asserted rather than described: a layout change that widened the entry
 * would otherwise silently double the VTCM the kernel asks for. */
static_assert(sizeof(struct htp_bilinear_axis) == 8, "the axis entry must stay 8 bytes");
static_assert(sizeof(struct htp_bilinear_axis) * 2 * HTP_OPS_BILINEAR_AXIS_MAX == 8192,
              "the two axis tables must stay 8192 bytes");

static inline float htp_bilinear_source(float scale, int32_t dst) {
  float src = scale * ((float)dst + 0.5f) - 0.5f;
  return src < 0.0f ? 0.0f : src;
}

static int htp_bilinear_build_axis(struct htp_bilinear_axis* axis,
                                   int32_t extent,
                                   int32_t out_extent) {
  if (extent < 1 || extent > out_extent || out_extent > HTP_OPS_BILINEAR_AXIS_MAX) {
    return -1;
  }
  const int32_t scale = out_extent / extent;
  if (scale * extent != out_extent) {
    return -1;
  }
  const float ratio = (float)(1.0 / (double)scale);
  for (int32_t dst = 0; dst < out_extent; ++dst) {
    const float src = htp_bilinear_source(ratio, dst);
    const int32_t low = (int32_t)src;
    if (low < 0 || low >= extent) {
      return -1;
    }
    axis[dst].low = low;
    axis[dst].w = src - (float)low;
  }
  return 0;
}

AEEResult htp_ops_upsample_bilinear2d_fp16(uint8_t* dst,
                                           const uint8_t* src,
                                           int32_t planes,
                                           int32_t in_h,
                                           int32_t in_w,
                                           int32_t out_h,
                                           int32_t out_w) {
  if (dst == nullptr || src == nullptr || planes <= 0 || in_h <= 0 || in_w <= 0 ||
      out_h <= 0 || out_w <= 0) {
    return -1;
  }
  if (out_h % in_h != 0 || out_w % in_w != 0) {
    return -1;
  }

  __attribute__((aligned(128))) struct htp_bilinear_axis rows[HTP_OPS_BILINEAR_AXIS_MAX];
  __attribute__((aligned(128))) struct htp_bilinear_axis cols[HTP_OPS_BILINEAR_AXIS_MAX];
  if (htp_bilinear_build_axis(rows, in_h, out_h) != 0 ||
      htp_bilinear_build_axis(cols, in_w, out_w) != 0) {
    return -1;
  }

  const __fp16* in = (const __fp16*)src;
  __fp16* out = (__fp16*)dst;
  const int32_t in_plane = in_h * in_w;
  const int32_t out_plane = out_h * out_w;

  for (int32_t plane = 0; plane < planes; ++plane) {
    const __fp16* planeIn = in + (int32_t)plane * in_plane;
    __fp16* planeOut = out + (int32_t)plane * out_plane;
    for (int32_t oy = 0; oy < out_h; ++oy) {
      const int32_t y0 = rows[oy].low;
      int32_t y1 = y0 + 1;
      if (y1 > in_h - 1) {
        y1 = in_h - 1;
      }
      const float h1l = rows[oy].w;
      const float h0l = 1.0f - h1l;
      const __fp16* topRow = planeIn + y0 * in_w;
      const __fp16* botRow = planeIn + y1 * in_w;
      __fp16* outRow = planeOut + oy * out_w;
      for (int32_t ox = 0; ox < out_w; ++ox) {
        const int32_t x0 = cols[ox].low;
        int32_t x1 = x0 + 1;
        if (x1 > in_w - 1) {
          x1 = in_w - 1;
        }
        const float w1l = cols[ox].w;
        const float w0l = 1.0f - w1l;
        const float top = w0l * (float)topRow[x0] + w1l * (float)topRow[x1];
        const float bot = w0l * (float)botRow[x0] + w1l * (float)botRow[x1];
        outRow[ox] = (__fp16)(h0l * top + h1l * bot);
      }
    }
  }
  return 0;
}

}  // extern "C"
