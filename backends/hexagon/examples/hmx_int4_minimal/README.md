# Standalone HMX A8W4 and A16W4 tiles

## Runtime quantization parameters (2026-10-12)

`hmx_matmul_u8s4_tile_quantized` and `hmx_matmul_u16s4_tile_quantized` now accept new FP32 `sa/sw/sq` and `za/zw/zq`, generate conversion
parameters and weight-sum corrections, and return Q plus `Y=(Q-zq)*sq`.
No captured numeric constants or QNN runtime are required. Weights are logical
int8 containers; `Wq-zw` must fit the native signed INT8/INT4 range. Unsupported
scales/shifts or centered weights return 1 before modifying outputs/workspace.
The fixed-parameter entry points below remain compatibility wrappers.

See [the parameter API and verification report](../hmx_quant_params/README.md)
for signatures, supported modes, reproduction commands and device evidence:
92 parameterized cases per family, plus one newly exported unmodified QNN
model per family, passed with zero Q/Y mismatches. The tile geometry and
VTCM/resource ownership requirements below still apply.


Device-verified on PJZ110 / SM8750 / HTP v79, 2026-10-12. These two callable
examples directly execute HMX through the Hexagon SDK launcher and do not load
QNN. Their native instruction families and weight grouping came from the
independently confirmed QNN 2.41 entries; see [dispatch evidence](../../QNN_INT4_MATMUL_DISPATCH.md).

## Interfaces and exact arithmetic

Both functions are in [matmul.c](matmul.c), declared in [matmul.h](matmul.h).
All exposed A/B/Q/Y arrays have ordinary row-major layout. B uses an int8 API
container with values **-8..7**, packed to signed four-bit two's-complement
nibbles before HMX loading.

| Function | M×K×N | A | Q | Y |
|---|---|---|---|---|
| `hmx_matmul_u8s4_tile` | 64×32×128 | uint8 | uint8 | FP32 |
| `hmx_matmul_u16s4_tile` | 32×32×64 | uint16 | uint16 | FP32 |

A8W4 uses A scale=1/8, zero_point=128; B scale=1/4, zero_point=0;
Q scale=1/32, zero_point=128:

```text
S[r,n] = sum_k (A[r,k]-128)*B[k,n]
Q[r,n] = clamp(S[r,n]+128, 0, 255)
Y[r,n] = (Q[r,n]-128)/32
```

A16W4 uses A scale=1/256, zero_point=32768; B scale=1/4, zero_point=0;
Q scale=1/4, zero_point=32768:

```text
S[r,n] = sum_k (A[r,k]-32768)*B[k,n]
Q[r,n] = clamp(floor(S[r,n]/256)+32768, 0, 65535)
Y[r,n] = (Q[r,n]-32768)/4
```

The A16 formula includes **floor**, including for negative values: S=-1 yields
Q=32767. This verified fixed conversion is not PyTorch's default nearest rounding.
The full uint16 activation is processed as byte components; it is neither FP16
nor an activation truncated to uint8. The output is requantized, then explicitly
dequantized to FP32, rather than an unquantized integer dot product.

The caller provides at least 16,384 bytes of VTCM, 2048-byte aligned, and holds
the HMX lease/power vote. Buffers must not alias. [main.c](main.c) supplies setup
and releases resources on all paths. The two variants can share the workspace
sequentially. Device tests also interleave A8 → A16 → A8 calls.

These constants implement the stated fixed scales/zero points. There is no
arbitrary-shape, K accumulation, tail, per-channel, blockwise or production
backend integration. Scalar packing and FP32 dequantization make these semantics
examples, not tuned performance kernels. Arbitrary QNN A16 conversion parameters
and bit-identical replication of the original QNN graph remain unverified.

## A8 native layout and conversion

Workspace offsets: activation 0, packed weights 2048, native output 4096,
conversion tables 12288. Activation is row-major for this `:cm` load.
For g=0..1, j=0..31, k=0..31:

```text
W byte index = g*1024 + (k/4)*128 + j*4 + k%4
low nibble  = B[k,64*g+j]
high nibble = B[k,64*g+32+j]
Q byte index = (n/32)*2048 + r*32 + n%32
```

```asm
mxclracc
{ activation.ub = mxmem(A,0x71f):cm
  weight.n = mxmem(W,0x7ff):2x:deep }
bias = mxmem2(table)
cvt.ub = acc(selector):sc0  // even output group
cvt.ub = acc(selector):sc1  // odd output group
mxmem(output,0x700):cm = cvt
```

Conversion order is groups **1,0,2,3**, with selectors **0,1,1,0** indexed by
group. Moving group 0 before group 1 failed the complete output comparison.
The effective byte accumulator on this nibble path has a factor of 16: each
group's first 32 table words are `0x40005000`, and its next 32 words are
`-128*16*sum_k B[k,n]`. These implement scale 1/16 and output zero point 128.

## A16 native layout and conversion

The same workspace offsets hold 2048 activation bytes, 1024 weight bytes,
4096 output bytes and 1024 conversion bytes. All indices below are integer:

```text
A uint16 index = (r/2)*64 + k*2 + r%2
W byte index = (n/32)*512 + (k/8)*128 + (n%32)*4 + k%4
  for k=8*t+i, i=0..3:
    low nibble  = B[8*t+i,n]
    high nibble = B[8*t+i+4,n]
Q uint16 index = (n/32)*1024 + (r/2)*64 + (n%32)*2 + r%2
```

Thus A interleaves pairs of rows; W's high nibble crosses K, whereas the A8
high nibble crosses N. The complete native tile formulas passed varied-value
and isolated-position device checks. They do not describe all QNN graph tensors.

The four conversion tables select columns **0,2,...30**; **1,3,...31**;
**32,34,...62**; **33,35,...63**. Treating them as four consecutive 16-column
ranges produced wrong zero-point compensation. For output column n:

```text
group = (n/32)*2 + n%2
index = group*64 + 2*((n%32)/2)       // uint32 word index
conversion[index]    = 0x00404000
conversion[index+1]  = 0x40000000
conversion[index+32] = -128*sum_k B[k,n]
conversion[index+33] = 0
```

These coefficient pairs and correction words are verified together for the
stated arithmetic; their individual bitfield meanings are not all decoded.

```asm
mxclracc
{ activation.ub = mxmem(A,0x77c)
  weight.n = mxmem(W,0x3ff):deep }
// Repeat for the two N32 output groups, advancing table by 512 bytes:
bias = mxmem2(table)
cvt.uh = acc(0x401):2x2
bias = mxmem2(table+256)
cvt.uh = acc(0x20c):2x2
mxmem(output,0x700) = cvt
```

`cvt.uh` generates uint16 Q; scalar unpack/dequantization subsequently generates
Y. No scalar unpacking of W into an int8 matrix, FP16 conversion of A, or byte
plane matmuls substitute for this native `weight.n:deep` computation.

## Reproduce

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_int4_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_int4_minimal/run_device.py
```

Override `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`, `HMX_EXAMPLE_BUILD_DIR`, or the
runner's `--host/--sdk/--build-dir/--remote-dir` options as needed. The build uses
`hexagon-clang -mv79 -mhmx -O2` with warnings as errors, retaining decoded assembly
with `--mcpu=hexagonv79 --mattr=+hmx`. The phone script checks uploaded hashes,
waits for llama-server/llama-bench (llama-cli is permitted by the user), uses
private libraries, records root logcat and requires explicit DSP main return 0.
An ambiguous DSP failure is not automatically retried.

## Device evidence

A8's five cases cover small signed values, activation byte range, nibble -8,
saturation, zero-point input, random non-saturated values, and three isolated
row/K/column positions. A16's eight cases add full uint16 values with both bytes
varying, isolated positions, positive saturation, the lower output boundary,
non-periodic random values and negative fractional values that distinguish floor
from nearest rounding. Both DSP scalar references and independent host NumPy
matrix products compare complete outputs:

| Variant | Cases | Integer Q elements | FP32 Y elements | Mismatches |
|---|---:|---:|---:|---:|
| A8W4 | 5 | 40,960 | 40,960 | 0 |
| A16W4 | 8 | 16,384 | 16,384 | 0 |

The additional interleaved zero-point A16 call is checked on the DSP and does
not add another output record. Final hashes and artifacts are listed below;
no simulator or timing claim is involved.

| Artifact | SHA256 |
|---|---|
| DSP example SO | `2aaa2b384adf12de02979d7aefaed434f6cc855ab00af959597acc7bba281f9f` |
| A8 Q | `0723284da11a76822ac701c7045080f820efba16fcc39f929dd10792ec8d9de8` |
| A8 FP32 Y | `a90944f51a76bf00c909b42a13cedf6f696c9bc5937ea46bec36c190c6422e92` |
| A16 Q | `f237e270774769e9c1e24cb6507d4775f9e06d2e53fcd2b6e5b08065796935b7` |
| A16 FP32 Y | `7f3cf42d4922477a5c3d94cba9e233228a8dd6832b3781352b312f9c49cadbc8` |

Evidence directory: `.tmp/scratch/mnv2/hmx_int4_minimal/example/` contains
`checks.json`, `device_hashes.txt`, `device.log`, `example.farf.log`, `example.asm`,
`output.u8`, `output.f32`, `output.u16`, and `a16.f32`. The SDK launcher and its
skel identities are retained in `device_hashes.txt`.
