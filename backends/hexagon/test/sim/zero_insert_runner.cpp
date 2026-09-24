/* Does one raster region zero-insert, or does a fast path claim it?

 * A transposed convolution is lowered to a plain convolution over a
 * zero-interleaved input, and the interleaving is two DSP commands: a
 * `DSP_OP_ZERO` that clears the destination plane buffer and one
 * `DSP_OP_RASTER_BLIT` that scatters the input into it. This runner hands
 * `htp_ops_raster_blit` exactly that one region and asks whether the
 * destination came out as `dst[p, y*S + a, x*S + b] = src[p, y, x]` when
 * a == b == 0 and zero everywhere else -- and, when it did not, what it
 * produced instead.
 *
 * The region is the emitter's: `regionCount=1`, `bytes=2`, `src_number=1`,
 * then the twelve int32 fields of `HtpOpsRasterRegion`
 *
 *   size      = [P, H, W]
 *   srcStride = [H*W, W, 1]
 *   dstStride = [uh*uw, S*uw, S]
 *
 * with P the plane count, H and W the *source* plane's extents, S the
 * interleave factor and the destination plane (uh, uw) = ((H-1)*S+1,
 * (W-1)*S+1). The destination element order is therefore
 *
 *   dst[p*uh*uw + y*S*uw + x*S] = src[p*H*W + y*W + x],
 *
 * which is what the runner rebuilds as its own intent and what the host side
 * independently reproduces in numpy. The kernel's fast paths are checked
 * against precisely this: `htp_ops_try_interleave_c64_single_blit` claims any
 * region with size[2] == 16, srcStride[2] == 1 and dstStride[2] == 4, which is
 * the ZI_W16_S4 case below.
 *
 * ZI_W16_S4_CUT is the same case with the width cut into two regions of 15 and
 * 1 columns, which that fast path cannot claim. Both encode the same
 * destination, so the two have to agree -- and because the first reaches that
 * path and the second does not, the agreement is what says the two write the
 * same bytes rather than that one path agrees with itself.
 *
 * Every source element carries a distinct small integer, so a permuted,
 * interleaved or shifted result is a different value and not a rounding
 * difference; the destination is poisoned before the clear so a region that
 * never ran is a mismatch rather than a row of plausible zeros.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsp/power.h"
#include "dsp/worker_pool.h"
#include "region_ops.h"

/* region_ops.h has no prototype for either kernel and htp_ops.h is generated
 * from the IDL, so both are declared here, as conv_runner.cpp does. */
extern "C" int htp_ops_zero(uint8_t *dst, int32_t size);
extern "C" int htp_ops_raster_blit(uint8_t *dst, uint8_t **src, int src_number,
                                   uint8_t *region, int32_t regionCount,
                                   int32_t bytes);

/* Source and destination element counts over the whole case table are 384 and
 * 1270; the buffers are declared for 4096 so a case added later fails loudly
 * (see run_case) rather than overflowing a static array. */
enum { kMaxElements = 4096 };
enum { kVectorBytes = 128 };
#define VECTOR_ALIGNED __attribute__((aligned(kVectorBytes)))
#define REQUIRE_VECTOR_ALIGNMENT(name)                 \
  static_assert(__alignof__(name) == kVectorBytes,     \
                "the kernels read " #name " a vector at a time")

static __fp16 VECTOR_ALIGNED source[kMaxElements];
static __fp16 VECTOR_ALIGNED destination[kMaxElements];
static __fp16 VECTOR_ALIGNED intent[kMaxElements];
REQUIRE_VECTOR_ALIGNMENT(source);
REQUIRE_VECTOR_ALIGNMENT(destination);
REQUIRE_VECTOR_ALIGNMENT(intent);

struct ZeroCase {
  const char *tag;
  int planes, h, w, s;
  /* 0: one region across the whole width. 1: the same scatter with the width
   * cut into 15 + 1 columns, which htp_ops_try_interleave_c64_single_blit
   * cannot claim. Both describe the same destination, so the two forms have to
   * agree -- and only the first reaches that path, which is what makes their
   * agreeing evidence about it. */
  int cut;
};

/* Sizes are kept small on purpose: several of the kernel's fast paths require
 * size[2] >= 4096 or a 64-lane geometry, and a case that large would be testing
 * a different question. */
static const ZeroCase zero_cases[] = {
    {"ZI_SMALL", 1, 3, 5, 2, 0},
    {"ZI_OX", 1, 4, 7, 3, 0},
    {"ZI_PLANES", 3, 3, 5, 2, 0},
    {"ZI_W16_S4", 1, 2, 16, 4, 0},
    {"ZI_W16_S4_CUT", 1, 2, 16, 4, 1},
    {"ZI_W64", 2, 3, 64, 2, 0},
    {"ZI_S1H", 2, 9, 4, 2, 0},
    {"ZI_H64", 1, 64, 4, 2, 0},
};

/* One region: the source reads at stride 1 along a row and the destination
 * writes at the interleave factor, with the row and plane strides of each
 * layout. The destination's own anchor is the first column's interleaved
 * position, which is why the cut's second region starts at 15 * s. */
static void set_region(const ZeroCase &c, int plane, int uw, int offset, int width,
                       HtpOpsRasterRegion *r) {
  memset(r, 0, sizeof(*r));
  r->srcIndex = 0;
  r->srcOffset = offset;
  r->dstOffset = offset * c.s;
  r->size[0] = c.planes;
  r->size[1] = c.h;
  r->size[2] = width;
  r->srcStride[0] = c.h * c.w;
  r->srcStride[1] = c.w;
  r->srcStride[2] = 1;
  r->dstStride[0] = plane;
  r->dstStride[1] = c.s * uw;
  r->dstStride[2] = c.s;
}

static int build_regions(const ZeroCase &c, int plane, int uw,
                         HtpOpsRasterRegion *out) {
  if (!c.cut) {
    set_region(c, plane, uw, 0, c.w, out);
    return 1;
  }
  if (c.w != 16) return 0;
  set_region(c, plane, uw, 0, 15, out + 0);
  set_region(c, plane, uw, 15, 1, out + 1);
  return 2;
}

/* FNV-1a over the raw bytes, so the host can hash the same buffer it built. */
static uint32_t fnv1a(const void *bytes, size_t count) {
  const uint8_t *data = (const uint8_t *)bytes;
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < count; ++i) {
    hash ^= data[i];
    hash *= 16777619u;
  }
  return hash;
}

static void run_case(const ZeroCase &c) {
  const int uh = (c.h - 1) * c.s + 1;
  const int uw = (c.w - 1) * c.s + 1;
  const long source_plane = (long)c.h * c.w;
  const long dest_plane = (long)uh * uw;
  const long source_count = (long)c.planes * source_plane;
  const long dest_count = (long)c.planes * dest_plane;
  if (source_count > kMaxElements || dest_count > kMaxElements) {
    printf("%s HASH=00000000 MISMATCH=-1\n", c.tag);
    return;
  }

  /* Every source element a different nonzero integer: 1 .. P*H*W, which is at
   * most 384 here and so exact in fp16. A scatter that lands an element one
   * plane, one row or one column away therefore reads a different number. */
  memset(source, 0, sizeof(source));
  for (int p = 0; p < c.planes; ++p)
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const long index = p * source_plane + y * (long)c.w + x;
        source[index] = (__fp16)(float)(index + 1);
      }

  /* Poisoned, not cleared: the `DSP_OP_ZERO` in front of the blit is half of
   * what is under test, and a buffer that was already zero would hide it. */
  memset(destination, 0xAB, (size_t)dest_count * sizeof(__fp16));

  HtpOpsRasterRegion regions[2];
  const int region_count = build_regions(c, uh * uw, uw, regions);
  if (region_count == 0) {
    printf("%s HASH=00000000 MISMATCH=-2\n", c.tag);
    return;
  }

  uint8_t *sources[1] = {(uint8_t *)source};
  const int zero_ret =
      htp_ops_zero((uint8_t *)destination, (int32_t)(dest_count * sizeof(__fp16)));
  const int blit_ret = htp_ops_raster_blit((uint8_t *)destination, sources, 1,
                                           (uint8_t *)regions, region_count,
                                           (int32_t)sizeof(__fp16));

  /* The intent, built here the way the emitter's strides describe it. */
  memset(intent, 0, sizeof(intent));
  for (int p = 0; p < c.planes; ++p)
    for (int y = 0; y < c.h; ++y)
      for (int x = 0; x < c.w; ++x) {
        const long index = p * dest_plane + y * (long)c.s * uw + x * (long)c.s;
        intent[index] = source[p * source_plane + y * (long)c.w + x];
      }

  const uint16_t *got = (const uint16_t *)destination;
  const uint16_t *want = (const uint16_t *)intent;
  long mismatches = 0;
  for (long i = 0; i < dest_count; ++i)
    if (got[i] != want[i]) ++mismatches;

  printf("%s HASH=%08x MISMATCH=%ld\n", c.tag,
         fnv1a(destination, (size_t)dest_count * sizeof(__fp16)), mismatches);
  printf("%s_SRC HASH=%08x\n", c.tag,
         fnv1a(source, (size_t)source_count * sizeof(__fp16)));
  printf("%s_RET %d %d\n", c.tag, zero_ret, blit_ret);
  printf("%s_DEST", c.tag);
  const long shown = dest_count < 256 ? dest_count : 256;
  for (long i = 0; i < shown; ++i) printf(" %04x", got[i]);
  printf("\n");
  if (mismatches) {
    /* What the kernel wrote where it wrote something else, so a wrong mapping
     * can be read off the line rather than guessed at from a hash. */
    printf("%s_BAD", c.tag);
    int printed = 0;
    for (long i = 0; i < dest_count && printed < 16; ++i) {
      if (got[i] == want[i]) continue;
      printf(" %lx %04x %04x", i, got[i], want[i]);
      ++printed;
    }
    printf("\n");
  }
}

int main(void) {
  power_setup();
  power_acquire();
  /* The blit hands its fast paths to the DSP's worker pool, and several of
   * them are gated on there being more than one worker, so the pool has to come
   * up the way the skel brings it up. */
  worker_pool_global_init();

  for (unsigned i = 0; i < sizeof(zero_cases) / sizeof(zero_cases[0]); ++i)
    run_case(zero_cases[i]);
  return 0;
}
