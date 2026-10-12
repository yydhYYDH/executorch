# Standalone HMX FP16A8W tile

**Device verified.** On PJZ110 / SM8750 / HTP v79, 2026-10-12: 10 cases,
20,480 FP16 output elements and their 20,480 FP32 widenings, zero bit mismatches
against both a DSP scalar reference and an independent NumPy host reference.
The DSP library calls HMX directly through the Hexagon SDK launcher; it does not
load QNN, ExecuTorch, or HexKL.

This reproduces the operand preparation, tile layout and instruction family
captured for QNN 2.41 `use_fp16a8w` with
`enable_native_mixed_precision_ops=false`. That configuration runs
`hmx_v73_convf16_1x1_stride1` at `0x2c7a00`: both HMX operands are **FP16**.
INT8 describes the stored weights, which are dequantized before the matmul.
This is not a direct FP16-by-INT8 HMX instruction.

## Interface and arithmetic

Fixed shape `A[32,32] @ B[32,64]`, row-major interface:

| Argument | Type | Meaning |
|---|---|---|
| A | `__fp16` | FP16 activation, no activation quantizer |
| B | `int8_t` | signed integer weights, zero point zero |
| scale | `float[64]` | one FP32 weight scale per output column |
| Q | `uint16_t[32,64]` | FP16 output bits |
| Y | `float[32,64]` | FP32 widening of the FP16 output |
| workspace | VTCM | 16 KiB, aligned to 2048 bytes |

The caller powers and locks HMX and acquires VTCM; `main.c` demonstrates the
complete acquire/call/release sequence. `matmul.c` owns packing, dequantization,
the inline HMX instructions and output unpacking.

```text
scale_h[n] = fp16(scale[n])
W_h[k,n]   = fp16(float(B[k,n]) * float(scale_h[n]))
Q[r,n]     = fp16(HMX_dot(A[r,:], W_h[:,n]))
Y[r,n]     = fp32(Q[r,n])
```

The scale rounding is deliberate: the earlier QNN capture matched
`fp16(B * fp16(scale))` for **512/512** captured weights. Multiplying by the
original FP32 scale before rounding the product does not reproduce that buffer.
Tests 7 and 8 exercise this distinction; 410 and 185 prepared weights differ
when scale rounding is omitted, respectively, and final outputs differ too.

There is no output quantizer or output dequantization. HMX converts its accumulator
to FP16, then the example widens those FP16 results to FP32. Returning FP32 does
not remove the FP16 output rounding.

## Native layout and instructions

All indices below are FP16 elements. For the fixed K32 tile:

```text
A_native[(r/2)*64 + 2*k + r%2] = A[r,k]
W_native[(n/32)*1024 + (k/2)*64 + (n%32)*2 + k%2] = W_h[k,n]
Q[r,n] = out_native[(n/32)*1024 + (r/2)*64 + (n%32)*2 + r%2]
```

One activation tile is 2048 bytes. The two N32 weight tiles occupy 4096 bytes.
Each N32 group computes with the sequence:

```text
mxclracc.hf
{ activation.hf = mxmem(A_native,0x77c)
  weight.hf = mxmem(W_native_group,0x7ff) }
bias = mxmem2(identity_table)
cvt.hf = acc(2)
mxmem(output_group,0x700) = cvt
```

The identity table is 256 bytes: first 32 uint32 words `0x00003c00` (FP16 scale
1.0 and bias 0), then 128 zero bytes. Neither operand load, conversion nor store
uses the integer A16W8 `:deep` / `:2x2` modifiers. The table is initialized on
every call. All output locations are checked after alternating workspace fills
`0x5a` and `0xa5`.

## Reproduce

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_fp16a8w_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python backends/hexagon/examples/hmx_fp16a8w_minimal/run_device.py
```

Default SDK `/home/yydh/hexagon-tutorial/tools/hexagon-sdk`, tools 19.0.04.
Build overrides: `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`, `HMX_EXAMPLE_BUILD_DIR`.
Device options: `--sdk`, `--build-dir`, `--host`, `--remote-dir`.

The device script waits for llama-server/llama-bench, verifies uploaded hashes,
checks the DSP main return value, retrieves all output arrays and compares their
raw bits with a NumPy reference. The host reference dequantizes each weight to
FP16, computes a float64 dot, rounds once to FP16 and widens to FP32. All selected
cases agree exactly; this does not assert that every FP16 input and every reduction
length matches an ideal float64 dot bit-for-bit.

| Case | Coverage | FP16 / FP32 mismatches |
|---|---|---:|
| 0 | varying A, signed weights, per-column scales | 0 / 0 |
| 1 | zero weights | 0 / 0 |
| 2 | zero activation | 0 / 0 |
| 3 | identity A, exposes each K/column mapping | 0 / 0 |
| 4 | single nonzero output at row 17, column 49 | 0 / 0 |
| 5..6 | signed-byte endpoints -128 / 127 | 0 / 0 |
| 7..8 | non-dyadic FP32 scales and scale-rounding controls | 0 / 0 |
| 9 | finer activation fractions and varied column scales | 0 / 0 |

Artifacts: `.tmp/scratch/mnv2/hmx_fp16a8w_minimal/example/` contains
`output.f16`, `output.f32`, `checks.json`, `device_hashes.txt`, `device.log`,
`example.farf.log` and decoded `example.asm`. Each output file contains ten
case-major `[32,64]` arrays. The script records output and deployed-library hashes.

Scope: finite inputs and scales in the demonstrated ranges, fixed tile, no bias,
no performance tuning, no whole-QNN-model output comparison, and no investigation
of QNN's alternative native mixed-precision option. The example extends the
captured INT8 recipe's weight range to include -128 as a standalone boundary test.
