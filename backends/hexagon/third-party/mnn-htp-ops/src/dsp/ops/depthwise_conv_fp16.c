#include <stdbool.h>
#include <stdint.h>

#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"



int hvx_conv_depthwise2d_fp16(__fp16* restrict dst, const __fp16* restrict src, const __fp16* restrict weight,
                              const __fp16* restrict bias, int batch, int ih, int iw, int oh, int ow, int c4,
                              int kernelY, int kernelX, int strideY, int strideX, int padY, int padX, int dilateY,
                              int dilateX, int relu, int relu6) {
    const int pack = __HVX_LENGTH__ / (int)sizeof(__fp16);
    if (batch <= 0 || ih <= 0 || iw <= 0 || oh <= 0 || ow <= 0 || c4 <= 0) {
        return -1;
    }

    const HVX_Vector vZero = Q6_V_vzero();
    __fp16 relu6_val = (__fp16)6.0f;
    const HVX_Vector vRelu6 = Q6_Vh_vsplat_R(fp16_to_bits(&relu6_val));

    /* The kernel reads each input vector once per tap, so reading it from the
     * (uncached) arena is a kernelY * kernelX-fold read amplification at DDR
     * bandwidth. One channel block fits VTCM, so it is copied in once and the
     * taps read the on-chip copy; the block is left on the arena only when it
     * does not fit. */
    uint8_t* vtcm_base = (uint8_t*)vtcm_manager_get_vtcm_base();
    const size_t block_elems = (size_t)ih * iw * pack;
    const bool stage = vtcm_base != NULL &&
        block_elems * sizeof(__fp16) <= (size_t)vtcm_manager_get_vtcm_size();

    for (int b = 0; b < batch; ++b) {
        for (int cb = 0; cb < c4; ++cb) {
            const __fp16* srcBase = src + ((size_t)(cb * batch + b) * ih * iw * pack);
            __fp16* dstBase = dst + ((size_t)(cb * batch + b) * oh * ow * pack);

            const __fp16* srcBlock = srcBase;
            if (stage) {
                __fp16* on_chip = (__fp16*)vtcm_base;
                for (size_t i = 0; i < block_elems; i += pack) {
                    vmemu((HVX_Vector*)(on_chip + i)) =
                        vmemu((const HVX_Vector*)(srcBase + i));
                }
                srcBlock = on_chip;
            }

            const __fp16* wBase = weight + (cb * kernelY * kernelX * pack);
            const __fp16* biasPtr = bias + (cb * pack);
            const HVX_Vector vBias = vmemu((const HVX_Vector*)biasPtr);

            for (int oy = 0; oy < oh; ++oy) {
                int inYOrigin = oy * strideY - padY;
                for (int ox = 0; ox < ow; ++ox) {
                    int inXOrigin = ox * strideX - padX;

                    HVX_Vector acc = vBias;

                    for (int ky = 0; ky < kernelY; ++ky) {
                        int inY = inYOrigin + ky * dilateY;
                        if (inY < 0 || inY >= ih) {
                            continue;
                        }
                        for (int kx = 0; kx < kernelX; ++kx) {
                            int inX = inXOrigin + kx * dilateX;
                            if (inX < 0 || inX >= iw) {
                                continue;
                            }
                            const __fp16* inPtr = srcBlock + (inY * iw + inX) * pack;
                            const __fp16* wPtr = wBase + ((ky * kernelX + kx) * pack);
                            HVX_Vector vIn = vmemu((const HVX_Vector*)inPtr);
                            HVX_Vector vW = vmemu((const HVX_Vector*)wPtr);
                            acc = Q6_Vhf_vmpyacc_VhfVhfVhf(acc, vIn, vW);
                        }
                    }

                    if (relu || relu6) {
                        acc = Q6_Vhf_vfmax_VhfVhf(acc, vZero);
                        if (relu6) {
                            acc = Q6_Vhf_vfmin_VhfVhf(acc, vRelu6);
                        }
                    }

                    __fp16* outPtr = dstBase + (oy * ow + ox) * pack;
                    vmemu((HVX_Vector*)outPtr) = acc;
                }
            }
        }
    }

    return 0;
}
