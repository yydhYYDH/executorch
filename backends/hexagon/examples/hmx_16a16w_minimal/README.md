# hmx_16a16w_minimal

**Variable-parameter follow-up (2026-10-12):** `matmul.c` now generates its
coefficient pair through the shared encoder, instead of embedding captured
words. `hmx_matmul_a16w16_tile_quantized` accepts FP32 `sa/sw/sq` and uint16
`za/zq`; the original entry wraps it with the old model's quantization values.
See [parameter generation](../hmx_quant_params/README.md) for the 71 device
cases, historical QNN coefficient checks and supported range. The fixed-model
record below is retained as the original numerical evidence.

Standalone HMX tile for QNN's `use_16a16w` config (uint16 activation, int16
weight, uint16 output), reverse-engineered from the QNN 2.41.0 HTP v79 skel and
verified bit-exact against QNN's own reference on a OnePlus 13 (PJZ110, SM8750).

## Shape

`M=32, K=32, N=64`, one 16 KB VTCM workspace.

```bash
./build.sh
python3 run_device.py
```

The example runs three tests with random weights and activations, computes the
requant reference in C on the DSP, and compares. It exits 0 only if all 2048
outputs match exactly:

```text
A16W16EXAMPLE test 0 errors 0 q0 29520 29519 29519 29520
A16W16EXAMPLE test 1 errors 0 q0 29528 29536 29524 29496
A16W16EXAMPLE test 2 errors 0 q0 29522 30134 29760 28716
A16W16EXAMPLE result 0
```

## Tile

```asm
mxclracc
for block in 0..3:
  { activation.ub = mxmem(A, 0x77c)
    weight.b     = mxmem(W + 1024*block, 0x3ff):dilate }
  bias = mxmem2(T + 64*(block+1))
  cvt.uh = acc(0x600):2x2
  mxmem(out + 1024*block, 0x700):2x2 = cvt
```

`:2x2` belongs on the store. The table base is offset by one 256-byte block
because the kernel pre-increments the table pointer before its first read.

## Layouts

```text
A[(r/2)*64 + k*2 + (r%2)]                        = A_q[r][k]        // uint16
W[(n/16)*1024 + (k/4)*128 + (n%16)*8 + (k%4)]    = low              // int16 as 2 bytes
W[.. + 4]                                        = high
low = ((v+128)&0xFF)-128 ; high = (v-low)/256
T[(n/16 + 1)*64 + (n%16)*2]                      = coefficient
T[(n/16 + 1)*64 + (n%16)*2 + 32]                 = -corr[n]
out[(n/16)*1024 + (r/2)*64 + (n%16)*2 + (r%2)]   = Q[r][n]
```

The output row-pair stride is **64 words**; a 16-column block spans 1024 words, so
a 32x64 output needs 4096 words of VTCM.

## Requant

```text
t = floor(acc_raw / 65536) + stored[n]
Q = clamp( zq + round(K * t), 0, 65535 )
stored[n] = -round_half_away( za * sum_k min(W[k][n], 32639) / 65536 )
K = 0.756091655, za = 33401, zq = 29520
```

`K`, `za` and `zq` are the values QNN emitted for the captured
`use_16a16w_m128_k128_n128` model (`sa=1.4731697774542e-05`,
`sw=1.7616483091842e-05`, `sq=2.2494559743791e-05`); they also live inside the
conversion table's coefficient half, which carries the multiplier and the output
zero point. Originally the coefficient was copied verbatim (`0x28403de7 0x3980020c`).
The new shared encoder reproduces these words from the scale/zero-point values;
its normal-mode bitfields are decoded and checked against 39 captured models.

## Limits

The 32-term accumulator is 32 bits, so weights near `32767` overflow and the
hardware saturates. `W_eff` is also capped at `32639` (`0x7F7F`), matching QNN.
