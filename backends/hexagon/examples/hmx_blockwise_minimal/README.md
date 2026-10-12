# Standalone HMX per-block INT4 sum (independent arithmetic)

**This is not the QNN LPBQ path.** For the QNN-compatible byte-weight route,
use [hmx_lpbq_minimal](../hmx_lpbq_minimal/README.md). This older example is kept
as a contrast: it converts/saturates each block and combines the outputs on the
CPU, whereas QNN accumulates effective byte weights and converts once.

**Device tier, bit-exact.** Verified on PJZ110 / SM8750 / HTP v79 on 2026-10-12:
8192 checked elements across 4 cases, **0 quantized mismatches, 0 FP32
mismatches**. The DSP `.so` runs HMX directly through the Hexagon SDK launcher and
does not load QNN, ExecuTorch, or HexKL. No production backend change, no
simulator.

This is the int4 **block-wise** (LPBQ-style) counterpart of the per-tensor
[`hmx_int4_minimal`](../hmx_int4_minimal/README.md) and the byte-weight
[`hmx_int8_minimal`](../hmx_int8_minimal/README.md) examples.

## Contract (exact integer, all inside this example)

Fixed shape `A[32,64] @ W[64,64]`, row-major interface. The K axis is split into
two 32-wide blocks; each block is accumulated independently in HMX and the
partials are combined with an integer per-(output, block) scale.

| tensor | type | meaning |
|---|---|---|
| A | uint16 | activation, zero point 32768 |
| W | int8 | int4 values `-8..7`, one per byte, `W[k,n]` |
| scale | int32 | `scale[n,b]`, per output column and K-block |
| Q | uint16 | requantized output |
| Y | FP32 | dequantized output |

```text
acc_b[r,n] = sum_{k in block b} (A[r,k]-32768) * W[b*32+k,n]
T_b[r,n]   = clamp(floor(acc_b[r,n]/256) + 32768, 0, 65535)      // HMX tile
Q[r,n]     = clamp(sum_b (T_b[r,n]-32768) * scale[n,b] + 32768, 0, 65535)
Y[r,n]     = (Q[r,n]-32768) / 4
```

`T_b` is produced by the **device-verified u16s4 HMX tile** (same instruction
sequence and byte layout as `hmx_int4_minimal`): `activation.ub = mxmem(A,0x77c)`
/ `weight.n = mxmem(W,0x3ff):deep` / `cvt.uh = acc(0x401|0x20c):2x2`. The
block-wise layer only adds the per-block split and integer combine; it does not
claim to be QNN's exact runtime byte kernel.

## Implementation

- `matmul.c`: byte pack per K-block → verified u16s4 HMX tile per block → scalar
  integer combine + FP32 dequantization.
- `main.c`: HMX power-up, 16 KiB VTCM lease, lock, four full numerical cases,
  output files, release.
- `build.sh`: compiles these two files (`hexagon-clang -mv79 -mhmx -O2
  -Wall -Wextra -Werror`) and keeps the disassembly.
- `run_device.py`: waits for phone workloads, deploys over SSH with hash checks,
  runs on the DSP, retrieves full outputs, and re-checks every element on the
  host with NumPy.

## Reproduce

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_blockwise_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python backends/hexagon/examples/hmx_blockwise_minimal/run_device.py
```

Default SDK `/home/yydh/hexagon-tutorial/tools/hexagon-sdk`, tools 19.0.04.
`HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`, `HMX_EXAMPLE_BUILD_DIR` override the
build; the run script exposes `--sdk`, `--build-dir`, `--host`, `--remote-dir`.
The runner's process exit code only reports RPC status; the script additionally
requires `main() ... return value of 0` and re-checks the full output.

## Evidence

Products in `.tmp/scratch/mnv2/hmx_blockwise_minimal/example/`:

- `output.q`: four `[32,64]` uint16 blocks, case-major.
- `output.y`: the same order as little-endian FP32.
- `checks.json`: host verification, full-output hashes, and the device `.so`
  hash, plus raw FARF/`device.log`/`device_hashes.txt`/`example.asm`.

| case | A | Q mismatches | FP32 mismatches |
|---|---|---:|---:|
| 0 | varies ±4 around zero point | 0/2048 | 0/2048 |
| 1 | varies ±128 around zero point | 0/2048 | 0/2048 |
| 2 | all zero point | 0/2048 | 0/2048 |
| 3 | varies ±600 around zero point | 0/2048 | 0/2048 |

`output.q` SHA256
`f0c9affeb71113d41c2fbbf72be7f33a28e44f1292fd45bc756d9a9144fb412b` and
`output.y` SHA256
`38136762039e9e51a502423e97e8dfa383fbeb2ed78290f75061564133077495` are
reproducible across runs. The device `.so` SHA256 (`checks.json`, e.g.
`77b99d7c83cb8d36c8f8bc17d477916a9d758dd4cbca2b5df1d8de9776e636cf`) is not
stable across rebuilds; `run_device.py` records and verifies whichever `.so` it
deploys.

## Scope and honest limits

- The contract fixes the block scales as **integers** and uses the verified tile's
  own `floor(acc/256)` requantization per block. It is not QNN's floating-point
  LPBQ requantization, so a device-vs-QNN byte diff is not expected here.
- `block_size` is along K only (32); per-output-channel block scales are expressed
  through `scale[n,b]`, but the demonstration does not vary `block_size`, M, or N.
- On HTP v79 / QNN 2.41 the `use_16a4w_block` `Linear` actually dispatches to the
  byte-weight `hmx_v75_convhbh1x1_stride1` (`0x2b25c0`); that kernel's param struct
  exposes conversion selectors (`0x401`, `0x20c`) but **no** block-scale pointer,
  so QNN's block scale is not observable at the kernel entry. This example is the
  independent int4 block-wise arithmetic, not a byte-exact copy of that kernel.
  See [QNN_INT4_MATMUL_DISPATCH.md](../../QNN_INT4_MATMUL_DISPATCH.md) and
  `.tmp/scratch/mnv2/qnn_int4_block/REPORT.md`.
- No HVX pack, DMA pipelining, multi-tile K accumulation, or performance tuning.
  No speed comparison with QNN.
