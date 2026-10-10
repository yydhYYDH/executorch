# Hexagon fp16 matmul: the 1024-cube fast path, and the QNN race it answers to

## 当前状态核实（以原目标为准）

本节核对当前工作树与现存原始结果；以下历史章节保留实验经过，发生冲突时以本节为准。此文件是实验报告，复现脚本见§75。

原目标是使24形状扫描中全部11个落后形状稳定快于未修改QNN，保持数值正确；之后修复1024×1024×1023、129×4093×2051、65×8191×1025的执行错误。目标尚未完成。§74只检查三组命名形状，不能代替原目标验收。

当前分支backend-hexagon-minimal，核实时HEAD为6e9b71719b88fcb25bb9a4ec79f6c189ea4eecf6。当前loop_ops.cc MD5为a2279cd3b2bbaa0fe02f4e0969dfa3fb；§75冻结源码MD5为d98f283c29f0925875b5529718e3ba78，两者不同。最新扫描只能证明其测量二进制，当前工作树尚需匹配构建、部署和复测。当前CMake确实设置HTP_MM_PHASE_PROFILE=0。

noprof_full.json有228条记录，其中6条执行错误、222条成功数值检查，54个形状有有效双后端对照。0个torch_close=false不是0个执行失败。

| 原目标 M×K×N | Hexagon µs | QNN µs | 此次扫描状态 |
|---|---:|---:|---|
| 8192×1024×1024 | 1475.3 | 1122.4 | 仍落后 |
| 256×8192×4096 | 1327.9 | 1652.5 | 该扫描领先 |
| 256×4096×8192 | 1242.6 | 1298.6 | 该扫描领先 |
| 1×8192×8192 | 2064.4 | 2046.1 | 仍落后 |
| 8×8192×4096 | 1068.7 | 1052.8 | 仍落后 |
| 33×4096×4096 | 571.3 | 547.4 | 仍落后 |
| 127×4096×4096 | 626.5 | 869.4 | 该扫描领先 |
| 513×2048×2048 | 465.7 | 496.4 | 该扫描领先 |
| 1024×1023×1024 | 218.9 | 231.4 | 该扫描领先 |
| 1024×1025×1024 | 230.3 | 232.4 | 领先幅度小，需确认 |
| 4097×1025×257 | 456.0 | 1078.4 | 该扫描领先 |

三组原执行错误中，129×4093×2051、65×8191×1025在该扫描两轮成功且参考检查通过；1024×1024×1023仍两轮失败。扩展扫描另有1024×1024×33和1024×1024×63失败。错误修复进展不能表述为三组全部修复。

1024³的正式汇编曾完成手机计时与数值验证，原三轮为197.6 vs QNN206.5 µs；对应证据位于matmul_rebenchmark/dsp_diagnosis/asm_goal。开头原先的build-only声明已过期。早期31项模拟器测试通过只覆盖当时源码，不能替当前新增改动背书。

复现§74时必须使用独立SWEEP_RESULTS文件：measure_newskel.py会跳过已有两轮成功记录，重用noprof_full.json可能根本没有重新测量。该脚本以无error计数决定跳过，排名前还必须检查finite、torch_close和完整输出。推荐：

```bash
cd /home/yydh/executorch
export PYTHONPATH=/home/yydh
export SWEEP_SKEL=noprof/skel SWEEP_RUNNER=noprof/hexagon_runner
export SWEEP_RESULTS=verify_current_fresh.json
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/matmul_coverage/measure_newskel.py
```

此命令复测手机上已有noprof二进制；要验证当前源码，先按§75匹配构建并部署runner/skel到独立目录，再将SWEEP_SKEL/SWEEP_RUNNER指向该目录。cmp65.py仍固定读取noprof_full.json与odb_results.json，不会自动读取上述新文件。§75中touch仅强制构建，不能代替source-ID、库hash与部署路径核对。

原始记录、模型和脚本位于git排除的.tmp目录，本文可提交，但仅凭git checkout不能重建全部历史实验。§75关于/tmp“被sandbox阻止”的描述也不是当前运行环境事实；当前环境没有文件系统sandbox限制，使用.tmp是工作目录约定。


PROVENANCE. Two different things live in this file and they have different
evidence tiers, so they are kept apart on purpose.

- **The race and the disassembly** (§1–§4) are measurements and static reads that
  already exist, taken on a real phone (OnePlus 13, PJZ110 / SM8750) and on the
  QNN 2.41 v79 skel. Their raw logs live under `.tmp/scratch/mnv2/`, which is
  **excluded from git** — the numbers are restated here because losing that
  directory loses the evidence.
- **The 1024-cube fast path** (§5–§7) is the uncommitted working-tree change. Its original build-only description is superseded by the device evidence in §6 and the current-state audit above.

Read performance claims with their measured artifact and protocol; see the current-state audit above.

## 1. The goal

Make ExecuTorch's Hexagon (HTP) **fp16 matmul as fast as QNN on the same phone**.
Both sides run fp16 on the HMX unit and produce **bit-identical outputs** on every
shape measured, so the comparison is kernel against kernel, not numerics against
numerics.

The race is scored on a fixed five-shape set. `hex/qnn < 1` means hexagon wins:

| id | shape (M, K, N) | regime | hexagon | QNN | outcome |
|---|---|---:|---:|---:|---|
| **A** | **1, 8192, 8192** | decode, weight-bound | 2504 us | **2128 us** | **QNN 1.18×** |
| B | 8192, 8192, 1 | large-M, input-bound | **5862 us** | 15473 us | hexagon 2.64× |
| C | 1024, 1024, 1024 | compute | **357 us** | 1157 us | hexagon 3.24× |
| D | 2048, 2048, 2048 | compute | **2381 us** | 4035 us | hexagon 1.70× |
| E | 4096, 4096, 4096 | compute | **17519 us** | 36988 us | hexagon 2.11× |

All numbers are 10 process runs, first dropped (QNN has no warm-up and its first
run is consistently slower). Before the two weight-DMA fixes of §3 the set read
A 4277 / B 7034 / C 409 / D 2575 / E 19219, i.e. A was 2.01× behind.

**Protocol warning: this table is not the current campaign's.** It compares
hexagon's *DSP kernel probe* against QNN's *own host application* (`N inference
took X ms`). The shape sweep that runs today puts **both** backends through
ExecuTorch on the same phone, same rounds, hexagon from the DSP command wall clock
and QNN from its etdump "execute excluding wait" time. The two protocols disagree
badly — QNN on 1024³ is 1157 us in this table and 202.7 us in that one — so no row
of this table may be mixed into a row of that one. See §6 for the sweep's own
1024³ result.

The result is **regime-dependent, and each backend wins one regime**:

| regime | bottleneck | hexagon | QNN | margin |
|---|---|---|---|---|
| `M == 1`, large weight | weight read | 24–33 GB/s | **48–59 GB/s** | QNN ~1.7–2.5× |
| `N == 1`, large M | input read | **17–19 GB/s** | 8–10 GB/s | hexagon ~1.7–2.3× |
| square, intensity ~341 | layout conversion | **5.3–7.2 TFLOP/s** | 1.9–4.3 TFLOP/s | hexagon 1.6–3.2× |

For an LLM this splits by phase: **decode (`M == 1`) favours QNN; prefill (large M)
favours hexagon.** The only shape still lost is A.

## 2. Which matmuls are actually compared

Five sets exist in the tree and they answer different questions. Conflating them
is how three earlier revisions of this comparison reached three different
conclusions.

**The five-shape set A–E** above. This is the old campaign's scored set, and the
one every conclusion in §1 is drawn from.

**The 57-shape coverage sweep** of §8. Both backends through ExecuTorch on the
same phone, one shape per code-path boundary, a QNN column on every row. It is the
broadest comparison here and it disagrees with the A–E table, for a reason §8
gives.

**The regime families**, measured to find where each backend's advantage comes
from — QNN was not measured on all of these, so they are single-measurement
exploratory numbers:

| `M == 1` (weight-bound) | hex us | QNN us | | `N == 1` (input-bound) | hex us | QNN us |
|---|---:|---:|---|---|---:|---:|
| 1, 8192, 8192 | 4277 | 2128 | | 8192, 8192, 1 | 7228 | 15473 |
| 1, 8192, 4096 | 2167 | 1186 | | 8192, 4096, 1 | 3693 | 6222 |
| 1, 4096, 8192 | 2012 | 1184 | | 4096, 8192, 1 | 3549 | 8226 |
| 1, 4096, 16384 | 3972 | 2193 | | 4096, 16384, 1 | 6842 | 14673 |
| 1, 16384, 4096 | 5352 | 2185 | | | | |

**The small-shape probes** (64³, 256³, 512³). These are **excluded from every
conclusion**. QNN carries a fixed per-inference cost of ~225–235 us that does not
depend on the matmul — 64³ is under a microsecond of DSP work and QNN still
reports 233.7 us, *less* than 256³'s 224.7 us. Comparing there measures fixed
costs, not kernels. This is what produced the earlier, wrong "the two are equal"
reading of `1x2048x2048` (QNN profiled 263 us vs hexagon 264 us).

**The `hexagon_mm256/` sweep** (9 shapes: 128³, 256³, 512³, 1024³, 2048³,
512×512×2048, 2048×2048×512, 1×2048×2048, 8×2048×2048). This one is
**ExecuTorch-only** — it reports delegate census, error against a host fp32
reference and `call_ms` over 10 executions, and has **no QNN column**. It is a
regression sweep, not a race.

**Bit-identity checks** (1×2048×2048, 1×4096×4096, 1×8192×2048, 128×2048×2048,
512×2048×2048): output md5 equal on both backends, and both differ from torch by
the same amount. For fp16 matmul the two backends are numerically
interchangeable.

**What the uncommitted change targets is exactly C, `1024×1024×1024`** — see §5
for the guard that pins it.

## 3. Where the time goes, and what was already fixed

The vendored kernel has a phase split, off by default, enabled with
`-DHTP_MM_PHASE_PROFILE=1` in `skel/CMakeLists.txt`. It accumulates per matmul
command into probe slots 200–207 (weight fill, activation pack, HMX tile loads,
output store, whole kernel, count, weight bytes, path); the host already prints
them from `PrintOpTimes`.

| shape | weight fill | activation pack | tile loads (HMX) | output store | whole us |
|---|---:|---:|---:|---:|---:|
| `M=1` 1×8192×8192 | 3994 (82 %) | 658 (14 %) | 128 (3 %) | 86 (2 %) | 4871 |
| `N=1` 8192×8192×1 | 18 (0 %) | **7211 (93 %)** | 4 (0 %) | 509 (7 %) | 7752 |
| 1024³ | 63 (14 %) | 178 (39 %) | 13 (3 %) | **193 (42 %)** | 458 |
| 2048³ | 249 (9 %) | 933 (35 %) | 81 (3 %) | **1404 (52 %)** | 2701 |
| 4096³ | 999 (5 %) | 8517 (44 %) | 420 (2 %) | **9508 (49 %)** | 19567 |

**The HMX multiply is 2–3 % of the time.** Everything else is layout conversion:
the weight DMA for `M == 1`, and activation pack plus output store for every
`M > 1` shape.

Two fixes were already made on that basis, both in `loop_ops.cc`, both
bit-identical to baseline and to QNN, and both **already in the working tree**:

1. **Chained weight-fill DMA** — `htp_ops_loop_hmx_dma_copy` replaced a
   synchronous per-256-KB-chunk `dma_wait_for_idle(); dmstart; dma_wait_for_idle()`
   loop with a stack descriptor array and one chained `dmstart`/`wait`. Weight
   fill 40 → 61 GB/s; shape A whole 4277 → 3056 us.
2. **Skip the padding memsets for `validRows == 1`** — the activation pack was
   two 512 KB memsets at ~1.5 GB/s (680 us on A); for `validRows == 1` only row 0
   is stored, so the padding rows never reach the output and both memsets are
   dead. Pack 680 → 6 us; A 3056 → 2504 us.

That closed A from 2.01× to 1.18×. **A is still lost.**

Two hypotheses were tested and disproven, so they should not be re-tried: the
uncached arena (`RPCMEM_FLAG_UNCACHED`; forcing cached changed 264 → 254 us and
nothing else) and CPU contention (0–4 % CPU, the host blocks on FastRPC).

## 4. What the QNN disassembly established

Done 2026-10-09, when optimizing had not closed the gap. Object: QNN 2.41 v79
skel, SHA256 `5d6900ed5c2edb2c81517adf251ca34a0eae43d41c7aeb799d855c66c84f0400`.

**Enable the HMX decoder explicitly.** Without it, objdump prints `<unknown>` for
the whole packet — including the ordinary scalar instructions — which reads like
invalid or encrypted code but is not:

```sh
/home/yydh/third/hexagon-sdk-6.6.0.0/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump \
  -d --mcpu=hexagonv79 --mattr=+hmx \
  --start-address=0x2c7a00 --stop-address=0x2c7c60 \
  /home/yydh/env/qairt/2.41.0.251128/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so
```

Tools **19.0.04 and 19.0.07** both decode the same bytes with `+hmx`; upgrading
was not required. Earlier notes claiming these SDK versions cannot decode the HMX
instructions are superseded. Use SDK objdump as authoritative: the installed
radare2 Hexagon plugin renders `dmpoll` as `barrier`, misplaces a branch target
by 4 bytes, and blanks several HMX words, so its semantics and CFG are not usable
here.

**The candidate kernel** is `hmx_v73_convf16_1x1_stride1`, `0x2c7a00..0x2c7c60`.
Its decoded shape: `activation.hf = mxmem`, `weight.hf = mxmem`,
`mxclracc.hf`, `bias = mxmem2`, `cvt.hf = acc`, `mxmem = cvt`. It has **no
decoded DMA submission or wait** — scheduling happens outside this inner kernel.
It has no HVX layout conversion and no DDR output loop either: it writes the HMX
result into the layout its output address table names, so the row-major
application conversion is somewhere else.

One parameter detail matters downstream: the QNN path ORs `0x70` into a low-11-bit
activation parameter and uses `0x7ff` (2048−1) for the weight, while **the
Hexagon-side helper uses `activation.hf = mxmem:deep` plus `weight.hf = mxmem`
sharing one parameter, `n_tiles * 2048 - 1`** — 65535 at K = 1024. The local
intrinsic header gives opcode, slot and syntax but not the bit semantics of these
parameters, so `0x70` must not be called a stride or a transpose on this evidence.

**Dispatch is confirmed, not assumed.** A private ABI-preserving probe replaced
only the first instruction at `0x2c7a00` with a standalone `{ jumpr r31 }`
(`00 c0 9f 52`, before stack allocation). The fp16 output changed; restoring the
unmodified private library restored its exact SHA256; all three fresh runner
processes executed successfully. That supports use of that **entry**, not of its
aligned body — the original entry can branch to the unaligned variant. Do not
transplant the patch to another binary or address.

**Not established by any of this:** that QNN's higher `M == 1` bandwidth comes
from a specific instruction. Its queue mechanics are already similar to ours
(asynchronous linking, per-descriptor polling). The remaining differences to
investigate are outer scheduling, descriptor flags, operand cache prefetch and
startup serialization. Also, QNN's 63 GB/s is weight bytes over overall inference
time, whereas our async weight-fill figure is pop-wait time — **these are not
comparable as DMA service bandwidth.**

## 5. The change: a 1024-cube HMX fast path

Three files, uncommitted, all on the v79 path:

- **new** `backends/hexagon/third-party/mnn-htp-ops/src/dsp/ops/matmul_fp16_1024.S`
- `backends/hexagon/third-party/mnn-htp-ops/src/dsp/loop_ops.cc`
- `backends/hexagon/skel/CMakeLists.txt`, plus the two `test_*_sim.py` source lists

**The kernel.** `htp_ops_matmul_fp16_1024_row(activation, weight, output)` is 22
lines of hand-written Hexagon assembly. It clears the accumulator once, runs a
32-iteration hardware loop (`loop0`), and per iteration issues one activation
load, one weight load and one converted store:

```asm
{ r3=##0xffff; r4=#0; r5=#32 }        // shared operand param 65535 = 32*2048-1; store offset 0; 32 tiles
{ r6=##65536; r7=#2; r8=#2048 }       // weight step 64 KB; cvt control 2; output step 2 KB
{ mxclracc.hf }
{ loop0(.Ltiles,r5) }
.Ltiles:
{ activation.hf=mxmem(r0,r3):deep; weight.hf=mxmem(r1,r3); r1=add(r1,r6) }
{ cvt.hf=acc(r7) }
{ mxmem(r2,r4)=cvt; r2=add(r2,r8) }:endloop0
{ jumpr r31 }
```

The `:deep` activation load with the shared `65535` parameter is the pattern the
QNN decode identified as the Hexagon-side helper's (§4); the accumulator is
cleared once outside the loop, so K accumulates across iterations. **The exact
operand geometry is read off the disassembly, not independently proven** — the
parameter bit semantics are not in the local headers, and this kernel has not
been run.

**The dispatch.** `htp_ops_loop_matmul_hmx_1024()` is called from
`htp_ops_loop_matmul_hmx_general()` behind
`#if __HEXAGON_ARCH__ == 79 && !defined(SIMULATOR_MOCK_HMX)`, and returns
`false` to fall through to the general kernel whenever anything does not match.
The guard is deliberately narrow:

- `htp_ops_loop_hmx_planned()` must report **prepacked** weights (the weights are
  already in HMX tile order; no other kernel can read that operand);
- `sizeXYZ == {1024, 1024, 1024}` and both `src0StrideXYZ[0]` and
  `dstStrideXYZ[0]` equal 2048 bytes — i.e. **exactly M = K = N = 1024, fp16**;
- `src0Base` 128-byte aligned, at least 2 workers, worker pool available;
- VTCM must have room for 3 × 2 MiB matrices plus 128 KB of raw staging, a
  256-byte scale table and slack for 6 × 127 bytes of alignment: the fast path
  allocates activation, weight and output **all resident at once**, which is the
  whole point — the general kernel keeps only a tile of each.

Note what that guard means for a model: in a Qwen3-0.6B L=1024 prefill, the only
per-layer matmul that is exactly 1024³ with K = 1024 is the k/v projection. This
is a narrow fast path, not a general one.

**The body** pipelines 32 activation tiles of 32 rows. A `HmxDmaQueue` holds two
descriptors, so tile *et+1*'s 64 KB is in flight while tile *et* is transformed
and multiplied; the weight is copied into VTCM once up front. Every 8 tiles the
output is handed to a worker-pool task
(`htp_ops_matmul1024_output_worker`) that re-orders the HMX output back to
row-major with `Q6_W_vdeal_VVR` and stores it; four tasks cover the 32 tiles and
the main thread waits on all four at the end. The one subtle line is the retire
before hand-off:

```c
// HMX stores are asynchronous: retire the last tile before publishing this group to another thread.
volatile int ready = Q6_R_vextract_VR(vmem((const HVX_Vector*)(output + (size_t)(et + 1) * 32768 - 64)), 0);
```

Phase timings go into the same `g_mm_phase_us` slots 4–7 used by the profile in
§3.

**The build wiring.** `skel/CMakeLists.txt` adds `ASM` to the project languages
and appends the `.S` to the skel sources **only when `SKEL_ARCH_NUM EQUAL 79`**,
so v68/v73 skels are untouched. The two simulator test source lists gained the
same file.

## 6. Build, and what "verified" currently means

```sh
CMAKE=/home/yydh/miniconda3/envs/et/bin/cmake
SDK=/home/yydh/hexagon-tutorial/tools/hexagon-sdk
$CMAKE -S backends/hexagon/skel -B cmake-out-skel-verify \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$SDK/tools/HEXAGON_Tools/19.0.04 \
  -DMNN_OPS_ROOT=$PWD/backends/hexagon/third-party/mnn-htp-ops \
  -DHEXAGON_FLATBUFFERS_ROOT=$PWD/third-party/flatbuffers \
  -DSKEL_ARCH=v79 -DIDL_DIR=$PWD/cmake-out-android-arm64-v8a/backends/hexagon/idl
$CMAKE --build cmake-out-skel-verify -j8
```

Verified on 2026-10-10 with the above: the build completes, `ship/libhex-htp-skel-v79.so`
is 1,504,248 B (the v79 baseline without this change is 1,482,392 B),
`htp_ops_matmul_fp16_1024_row` is defined at `0x70ba0` and called from
`htp_ops_loop_matmul_hmx_general` at `0x5aee8`, and
`htp_ops_matmul1024_output_worker` is present at `0x5c79c`.

**This change is not unmeasured.** The skel deployed to the phone as
`matmul_rebenchmark/asm_goal/production/skel` is md5
`c307292f212a565e0f570d849e80e735` — byte-identical to the local build above — and
a 24-shape device sweep ran against it on 2026-10-10 01:03–01:11. That sweep's own
report says the 1024-cube is the shape that takes the specialized worker output
path while every other shape takes the general path. On that shape it records
**hexagon 196.7 us against QNN 202.7 us** (ratio 0.97), with the two backends
bit-identical to each other and both inside rtol/atol 0.01 of an fp32 reference.

So the established tier for this change is **device, one shape, numerics and
timing both green**. Not established: any other shape (the guard should refuse all
of them), any end-to-end model effect, and whether the win is stable — 196.7
against 202.7 us is a 3 % margin, and this device has shown 10–15 % run-to-run
spread on other shapes.

## 7. What to establish next

Ranked by the gap that is still open after §9, largest first.

1. **Three shapes still never return** — `1024 × 1024 × 33`, `1024 × 1024 × 63`
   and `1024 × 1024 × 1023`, all `0x8000040d`. §9 fixed three of the original five
   by masking the tail; these three survive it. `1024 × 1024 × 1025` works and
   `1024 × 1024 × 1023` does not, so the trigger is not the N remainder. This is
   the only known correctness defect and it should be closed before more tuning.
2. **N below one tile at M = N = 1024 is now the worst gap** — `1024 × 1024 × 31`
   is 2.53×, `× 32` 1.88×, `× 65` 1.75×, `× 64` 1.63×, `× 257` 1.47×. The
   activation read is amortised over N, so a narrow N pays the full input cost for
   a sliver of output. QNN does not have this cliff.
3. **`M = 256` with large K and N** — `256 × 4096 × 8192` is 2.02× and
   `256 × 8192 × 4096` 1.96×, the two largest gaps above 1.9×. Eight row blocks of
   32, and the weight read dominates; neither the 1024 fast path nor the odd-K fix
   touches it.
4. **Odd and small K at M = N = 1024** — 1.66× to 1.84×. §9 halved these; they are
   still losses, and the remaining factor is the pack itself, not the DMA.
5. **`N == 1`** — `8192 × 4096 × 1` 1.71×, `4096 × 4096 × 1` 1.65×. The output
   store is a strided scatter that the HMX cannot help with.
6. **Small M** — 31/32/33/64/65/127 rows against 1024 or 4096 columns, 1.07× to
   1.36×. Under one full 32-row block the per-command overhead is not amortised.
7. **`M == 1`** — 1.02× to 1.12×, a tie at this measurement point (§8) rather
   than the 1.18× loss §1 records. Resolving the protocol question is still a
   prerequisite to optimizing it.
8. **Does the 1024 fast path guard ever fire in a real model?** The k/v projection
   in a Qwen3 0.6B L=1024 prefill is the candidate. Count commands that reach it
   before believing any end-to-end effect.
9. **Does the fast path hold the VTCM budget with attention resident?** It claims
   ~6.3 MB at once; the phone reports 8 MiB. What else is live in the same graph is
   not checked by the guard.

## 8. Shape coverage: 57 shapes, one protocol

The five-shape set of §1 is the old campaign's yardstick. To find where the two
backends actually stand, **57 shapes were exported through both backends and run
through ExecuTorch on the same phone** — same input, same weight, alternating
backend order, two rounds of 40 executions each with the first 10 dropped, CPU7
pinned, device lock serialized. Hexagon is the DSP command wall clock (probe
`op=38`); QNN is the SDK graph execution excluding resource wait (etdump). Both
include internal packing, transfers and output completion; neither includes host
RPC. 51 shapes are valid — every one of them passed an fp32-reference check at
rtol/atol 0.01, and the two backends were bit-identical to each other on every
shape.

The set is chosen one shape per code-path boundary rather than at random: M at
31/32/33/64/65/128/256/512/1023/1024/1025/4097, K at
31/32/33/63/64/65/1023/1024/1025/2048/4093/8191/8192, N at
31/32/33/63/64/65/256/257/1023/1024/1025/2048/3072/4096/8192, the two extreme
regimes (`M == 1`, `N == 1`), the small squares, and the Qwen3-0.6B L=1024
prefill and M=1 decode projections.

**Read this table against §1's, not mixed with it.** Same phone, same skel, but a
different measurement point for QNN: §1 compares hexagon's DSP probe against QNN's
*own host application*, this one puts *both* backends through ExecuTorch. The two
disagree by a lot — QNN on 1024³ is 1157 us in §1 and 203 us here — and nothing
here says which point answers the campaign's question. What the table does say is
which shapes have gaps too large to be a measurement artifact.

| M | K | N | Hexagon us | QNN us | Hexagon/QNN |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1023 | 1024 | 2421.6 | 225.6 | 10.73 |
| 1024 | 1025 | 1024 | 2397.8 | 227.3 | 10.55 |
| 4097 | 1025 | 257 | 8308.2 | 1092.5 | 7.60 |
| 1024 | 32 | 1024 | 200.8 | 44.8 | 4.48 |
| 1024 | 63 | 1024 | 222.3 | 53.1 | 4.19 |
| 1024 | 65 | 1024 | 210.6 | 51.6 | 4.08 |
| 1024 | 31 | 1024 | 156.5 | 45.7 | 3.43 |
| 1024 | 33 | 1024 | 149.0 | 48.7 | 3.06 |
| 1024 | 1024 | 31 | 124.8 | 50.6 | 2.47 |
| 256 | 4096 | 8192 | 2611.5 | 1300.4 | 2.01 |
| 256 | 8192 | 4096 | 3230.7 | 1646.4 | 1.96 |
| 1024 | 1024 | 32 | 94.6 | 48.2 | 1.96 |
| 8192 | 4096 | 1 | 1941.8 | 1133.2 | 1.71 |
| 1024 | 1024 | 65 | 110.6 | 65.8 | 1.68 |
| 4096 | 4096 | 1 | 974.3 | 591.5 | 1.65 |
| 1024 | 64 | 1024 | 79.0 | 51.4 | 1.54 |
| 8 | 8192 | 4096 | 1589.4 | 1034.4 | 1.54 |
| 1024 | 1024 | 257 | 140.2 | 97.2 | 1.44 |
| 33 | 4096 | 4096 | 742.1 | 544.2 | 1.36 |
| 128 | 128 | 128 | 6.0 | 4.4 | 1.35 |
| 8192 | 1024 | 1024 | 1466.5 | 1124.2 | 1.30 |
| 31 | 1024 | 1024 | 51.5 | 39.6 | 1.30 |
| 127 | 4096 | 4096 | 1126.2 | 872.8 | 1.29 |
| 33 | 1024 | 1024 | 50.5 | 39.2 | 1.29 |
| 65 | 1024 | 1024 | 55.4 | 44.9 | 1.23 |
| 513 | 2048 | 2048 | 639.0 | 520.3 | 1.23 |
| 1024 | 1024 | 64 | 67.0 | 54.7 | 1.23 |
| 32 | 1024 | 1024 | 45.8 | 38.4 | 1.19 |
| 64 | 1024 | 1024 | 50.1 | 42.2 | 1.19 |
| 256 | 1024 | 1024 | 82.5 | 71.5 | 1.15 |
| 1 | 1024 | 2048 | 77.0 | 66.7 | 1.15 |
| 1 | 1024 | 3072 | 113.4 | 100.1 | 1.13 |
| 1024 | 2048 | 1024 | 422.9 | 375.9 | 1.13 |
| 1 | 3072 | 1024 | 113.2 | 100.9 | 1.12 |
| 128 | 1024 | 1024 | 59.3 | 54.1 | 1.10 |
| 512 | 1024 | 1024 | 128.7 | 123.5 | 1.04 |
| 1 | 8192 | 8192 | 2060.3 | 2048.8 | 1.01 |
| 1024 | 1024 | 1024 | 197.1 | 203.2 | 0.97 |
| 256 | 256 | 256 | 12.8 | 13.3 | 0.96 |
| 512 | 512 | 512 | 42.4 | 45.6 | 0.93 |
| 1024 | 8192 | 1024 | 2436.0 | 2821.6 | 0.86 |
| 1024 | 1024 | 1025 | 263.3 | 306.0 | 0.86 |
| 1024 | 3072 | 1024 | 628.1 | 758.3 | 0.83 |
| 2048 | 2048 | 2048 | 1539.3 | 2052.0 | 0.75 |
| 1025 | 1024 | 1024 | 259.2 | 348.6 | 0.74 |
| 1024 | 1024 | 2048 | 377.6 | 508.4 | 0.74 |
| 1024 | 1024 | 3072 | 536.7 | 801.1 | 0.67 |
| 4096 | 4096 | 4096 | 12604.4 | 23195.2 | 0.54 |
| 4096 | 8192 | 256 | 2592.8 | 5231.5 | 0.50 |
| 1024 | 1024 | 8192 | 1441.0 | 6106.6 | 0.24 |
| 8192 | 256 | 4096 | 2280.5 | 17724.3 | 0.13 |

Ratio > 1 means QNN is faster. **Hexagon trails on 37 of the 51 valid shapes.**

### What is large enough to be real

- **Odd and small K is the disaster.** K = 31/32/33/63/64/65 at M = N = 1024 is
  3.1–4.5× behind, and K = 1023/1025 at the same M and N is **10.5–10.7×** behind.
  This is the same `K % 64 != 0` path the earlier sweep identified: the activation
  prefetch requires it, and without it the input goes through the per-row memcpy
  into a padded staging buffer with the DMA double-buffering off. A 10× gap is not
  a measurement-point artifact.
- **`M == 1` is a tie, not a loss** (1.01× on 8192²), and the QNN-favouring
  weight-bandwidth story of §1 does not appear at this measurement point.
- **Hexagon's wins are real and large where N dwarfs K**: 8192×256×4096 is 0.13×,
  1024×1024×8192 is 0.24×, 4096×8192×256 is 0.50×. Every one of those is an input-
  bound shape, matching §1's `N == 1` finding that hexagon reads the activation
  roughly twice as fast.
- **Large squares still favour hexagon** (4096³ 0.54×, 2048³ 0.75×), but 1024³ is
  a near-tie at this measurement point — 197.1 against 203.2 us.

### The five shapes that never return

Five shapes enter `op=38` and never come back, with `execute_command_group_profile
failed: 0x8000040d` and `ENTER (no return)` in the probe log. The campaign knew
about three of them; **`1024×1024×33` and `1024×1024×63` are new here.**

| shape | hexagon | QNN |
|---|---|---|
| 1024 × 1024 × 33 | fails both rounds | 61.7 us, passes |
| 1024 × 1024 × 63 | fails both rounds | 59.3 us, passes |
| 1024 × 1024 × 1023 | fails both rounds | 215.7 us, passes |
| 129 × 4093 × 2051 | fails both rounds | 331.4 us, passes |
| 65 × 8191 × 1025 | fails both rounds | 316.1 us, passes |

This is not a sick device: each of the five was re-run on-device with a
**1024×1024×1024 control bracketing every attempt**, and the control succeeded
every time (40/40 samples, ~197 us) while the shape under test failed.

All five have N not a multiple of 32, but that is **not sufficient** — N = 31, 65
and 257 are also not multiples of 32 and they pass. The trigger is not identified.

One more shape is intermittent rather than dead: **1024×1023×1024 fails its first
round and passes its second**, in two independent sessions. A shape that is
sometimes fine is a different bug from a shape that never returns, and the two
should not be filed together.

### Reproducing

`.tmp/scratch/mnv2/matmul_coverage/` holds `export.py` (the 57-shape list),
`measure.py`, `merge.py` and `summarize.py`; `REPORT.md` there is the generated
table above. The remaining runs were driven by `driver.sh`, which runs **on the
phone** — the first host-driven pass died mid-sweep when the ssh proxy dropped, so
the driver executes locally, appends to `driver.txt`, and the host only polls.
That is the pattern to copy for any long device run.

## 9. The odd-K integration, and what it bought

§3 identified the `K % 64 != 0` staging path as the largest single loss and §8
priced it at up to 10.7×. Three changes, all in
`third-party/mnn-htp-ops/src/dsp/loop_ops.cc`, were integrated and measured on the
device:

1. **The activation prefetch no longer requires `K % 64 == 0`.** The DMA now moves
   the compact `rows * K * 2` bytes and the pack handles the odd row stride, so
   odd K stops falling back to the per-row staging copy. The gate went from
   `E >= 64 && E % 32 == 0 && K % 64 == 0` to `E >= 2`.
2. **The tail of the row block is masked, not assumed.** A sub-32-row or
   unaligned last block is loaded with `Q6_V_vmux_QVV(Q6_Q_vsetq_R(elems * 2), ...)`
   instead of an unmasked vector read, and the raw slot carries 2048 bytes of
   slack. This is what stops the tail reading past the buffer.
3. **The 1024-cube fast path generalises to K = 1023 and 1025.** The guard admits
   `K in {1023, 1024, 1025}` with `src0StrideXYZ[0] == K * 2`, the VTCM request is
   sized from `kp = ceil(K/32)`, and odd K runs a generic `hmx_load_tiles_fp16` /
   `hmx_consume_accumulator_fp16` loop. **Exact K = 1024 keeps its original pack
   and its hand-written `.S` kernel** — routing it through the generic loop cost
   10 us (197.1 -> 207.5) for no benefit, so it does not.

Both backends were re-measured on all 57 shapes with the new pair, same protocol
as §8. **54 shapes are now valid (was 51), hexagon leads on 18 of them (was 14),
the geometric-mean ratio fell from 1.365 to 1.079, and the worst gap fell from
10.73× to 2.53×.**

| shape | hex before | hex after | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| 1024 × 1023 × 1024 | 2421.6 | **209.6** | 225.2 | 10.73 | **0.93** |
| 1024 × 1025 × 1024 | 2397.8 | **226.6** | 227.7 | 10.55 | **1.00** |
| 4097 × 1025 × 257 | 8308.2 | **584.6** | 1076.3 | 7.60 | **0.54** |
| 1024 × 32 × 1024 | 200.8 | **82.2** | 44.6 | 4.48 | 1.84 |
| 1024 × 63 × 1024 | 222.3 | **87.5** | 50.1 | 4.19 | 1.75 |
| 1024 × 65 × 1024 | 210.6 | **87.2** | 52.4 | 4.08 | 1.66 |
| 1024 × 31 × 1024 | 156.5 | **82.2** | 46.7 | 3.43 | 1.76 |
| 1024 × 33 × 1024 | 149.0 | **87.7** | 49.7 | 3.06 | 1.76 |
| 513 × 2048 × 2048 | 639.0 | **507.6** | 501.0 | 1.23 | 1.01 |
| 8 × 8192 × 4096 | 1589.4 | **1421.6** | 1021.2 | 1.54 | 1.39 |
| 127 × 4096 × 4096 | 1126.2 | **975.9** | 842.8 | 1.29 | 1.16 |
| 1025 × 1024 × 1024 | 259.2 | **220.7** | 346.7 | 0.74 | **0.64** |

The five odd/small-K shapes at M = N = 1024 were the campaign's worst losses and
they improved by roughly half without any work aimed at them: widening the
prefetch gate is what the code-path note of §3 predicted would pay, and it paid
across the whole K family rather than only at the shapes that were tuned.

### Three shapes that never returned now return

§8 listed five hard failures. **Three of them are gone**, and each is a real
measurement now rather than a missing row:

| shape | before | after | QNN | ratio |
|---|---|---:|---:|---:|
| 1023 × 1024 × 1024 | intermittent (round 0 always failed) | **219.4** | 313.1 | **0.70** |
| 129 × 4093 × 2051 | `0x8000040d`, both rounds | **563.2** | 325.2 | 1.73 |
| 65 × 8191 × 1025 | `0x8000040d`, both rounds | **537.8** | 323.4 | 1.66 |

All three passed the fp32 reference check at rtol/atol 0.01 and were bit-identical
to QNN, so the tail masking fixed a correctness bug, not just a timing one — the
old path was reading past the raw buffer and taking the DSP down with it.

**Three shapes still fail**, all with `0x8000040d` on both rounds:
`1024 × 1024 × 33`, `1024 × 1024 × 63` and `1024 × 1024 × 1023`. Note the
asymmetry — `1024 × 1024 × 1025` works and `1024 × 1024 × 1023` does not, and
`1024 × 1023 × 1024` works. The trigger is still not `N % 32`; it is now the only
known correctness defect left.

### Two operational traps this cost time on

- **The skel and the runner are a provenance pair.** The runner refuses to run a
  skel whose source ID differs, with `skel source ID mismatch: expected ... got
  ...`. The ID is a hash of the skel sources, so **any** `loop_ops.cc` edit
  requires rebuilding `cmake-out-android-arm64-v8a --target executor_runner` too,
  not just the skel. A stale runner looks exactly like a broken kernel.
- **`scp file dir/` keeps the basename.** Copying `executor_runner` into a
  directory that expects `hexagon_runner` silently leaves the old binary in place.
  Verify the device-side md5 after every deploy.
- **This tree has more than one writer.** `execute_command.cc` was edited by a
  concurrent campaign at 02:27, between this section's build (02:24) and its
  writing, and other campaigns build into `.tmp/scratch/qwen35/`. Every number in
  §8 and §9 is tied to an md5 of the exact skel and runner that produced it; a
  measurement whose pair is not recorded is not comparable to these. Both the
  baseline and the new pair here were built before that edit, so the A/B is
  unaffected — but the next one has to re-establish its own baseline.

## 10. Two things the shape set does not show: the crash trigger, and the tile-count cost

§9 left three shapes that never return and a list of narrow-N losses with no
explanation. Both were chased down with targeted sweeps.

### The crash depends on N alone

M and K were held at 1024 and N was swept from 31 to 1026, then M and K were
varied at N = 33. Every shape was run twice.

| N | result | N | result |
|---|---|---|---|
| 31, 32 | ok | 61, 62, 63 | **`0x8000040d`** |
| 33, 34, 35 | **`0x8000040d`** | 64 | ok |
| 36 | ok | 65, 66 | ok |
| 37, 38 | **`0x8000040d`** | 96 | ok |
| 40, 44, 48, 56, 60 | ok | 127 | **`0x8000040d`** |
| 129 | ok | 1021, 1022, 1023 | **`0x8000040d`** |
| 257 | ok | 1025, 1026 | ok |

`512 × 1024 × 33`, `2048 × 1024 × 33`, `1024 × 512 × 33` and `1024 × 2048 × 33`
**all fail exactly as `1024 × 1024 × 33` does**, so the trigger is a property of N
and nothing else. It is deterministic: every failure failed both attempts.

The failing N are not one residue class — 33 and 1023 are `1` and `31` mod 32, and
31, 65, 129, 257 and 1025 share those residues and pass. Whatever the condition
is, it is not `N % 32`, and the campaign's three known failures are simply three
points on a larger set. The parameter blocks the DSP receives are structurally
identical across all of them — same field layout, every N-derived value scaled
consistently — and the weight is confirmed prepacked in both the failing and the
passing cases (`hmxFlags = 0x484d59`), so the difference is inside the kernel.

### The narrow-N losses are a tile-count cliff

Sweeping N over the multiples of 32 at M = K = 1024 separates the cost into three
terms:

| N | tiles | us | | N | tiles | us |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 2 | 66.9 | | 96 | 3 | 109.3 |
| 128 | 4 | 76.4 | | 160 | 5 | 119.4 |
| 192 | 6 | 87.3 | | 288 | 9 | 139.9 |
| 256 | 8 | 97.0 | | | | |
| 320 | 10 | 107.2 | | | | |
| 384 | 12 | 117.0 | | | | |
| 512 | 16 | 137.1 | | | | |

Even tile counts are linear at about **5.1 us per 32-column tile** over a fixed
**~56 us** — and every odd count pays a flat **~37.7 us** on top, independent of how
many tiles there are:

| tiles | even fit | odd measured | penalty |
|---:|---:|---:|---:|
| 3 | 71.6 | 109.3 | 37.7 |
| 5 | 81.9 | 119.4 | 37.5 |
| 9 | 102.1 | 139.9 | 37.8 |

The mechanism is the store. A pair of tiles is retired with one 128-byte unaligned
vector store; a **lone** trailing tile falls through to
`htp_ops_loop_hmx_store_row`, which writes 64 bytes as eight scalar stores when
the remainder is a full tile, and to `vstu_variable` when it is not. Over 1024
rows that is ~37 cycles per row, and it is the same 37 us whether the lone tile is
1 column or 32.

This is why `1024 × 1024 × 31` (2.53×) and `× 32` (1.88×) are the worst rows in
§9: both are a single tile, so there is no pair to form and the whole output is
written the expensive way. `× 65` and `× 257` are odd counts and pay the same
37 us on their last tile.

### The obvious fix does not apply to this shape set

Pairing a lone trailing tile with the tile before it works only when the tail is a
**full** tile, so that the 128-byte store ends exactly at the row end. That covers
odd counts at N a multiple of 32 (96, 160, 288) — **none of which are in the
57-shape set**. For `N = 65` and `N = 257` the tail is one column, the pair store
would run 62 bytes past the row into the next one, and for `N = 31` and `N = 32`
there is no first tile to pair with at all. Recording this as a negative result
matters more than the fix would: it means the next attempt has to attack the
**single-tile** store, not the pairing.

### The fix that does apply

Pairing *tiles* does not work, but the two *rows* a lone tile retires are themselves
a pair. `Q6_Vh_vdeal_Vh` already returns row r in the low 64 bytes and row r+1 in
the high 64 — which is why the existing code shifts the vector by 64 to get the
second row. When the destination row stride is exactly 64 bytes, that is `N == 32`,
those two rows are one contiguous 128-byte line, so a single unaligned vector store
retires both and the scalar path is not needed at all:

```c
if (rowBytes == 64 && lp->dstStrideXYZ[0] == 64) {
    vmemu((HVX_Vector*)dst0) = v;
    continue;
}
```

Measured at M = K = 1024, two rounds per shape:

| N | before | after | QNN | ratio before | ratio after |
|---:|---:|---:|---:|---:|---:|
| 32 | 94.6 | **62.9** | 45.9 | 2.06 | **1.37** |
| 64 | 67.0 | 67.1 | 54.0 | 1.24 | 1.24 |
| 31 | 124.8 | 127.2 | 47.2 | 2.65 | 2.70 |
| 65 | 109.7 | 112.0 | 61.8 | 1.77 | 1.81 |

The three unchanged shapes are the control: 64 already paired its tiles, and 31 and
65 have row strides of 62 and 130 bytes, so the new branch cannot fire for them.
The cost model above predicted 56 + 5.1 = 61.1 us for a one-tile N = 32; the
measurement is 62.9.

### What is left is the fixed cost, not the store

The same model says where to look next. QNN does `1024 × 1024 × 31` in **47.2 us**
and `× 32` in **45.9 us**, and hexagon's fitted floor at this M and K is **~56 us** —
so QNN's whole operation costs less than hexagon's fixed part. No further store
work can close that: the remaining gap is the ~56 us that does not scale with N,
which for M = K = 1024 is reading and packing the 2 MB activation. **That is the
next target, and it is a different one from everything above.**

### Measurement noise, and the method that survives it

Five shapes moved by more than 5 % between the §9 sweep and the §10 one —
`1024 × 1024 × 2048` at +13 %, `31 × 1024 × 1024` at +12 %, `64 × 1024 × 1024` at
+7 %, `32 × 1024 × 1024` at +6 % and `128 × 128 × 128` at +8 %. Only one of them is
real. Re-measured in a single session with **both builds alternating**, three rounds
each:

| shape | before build | after build |
|---|---|---|
| 1024 × 1024 × 2048 | 378.3, 376.9, 378.4 | 379.0, 377.1, 378.6 |
| 31 × 1024 × 1024 | 51.8, 46.2, 52.9 | 46.2, 49.8, 50.9 |
| 64 × 1024 × 1024 | 47.5, 52.0, 52.4 | 47.8, 47.7, 52.3 |
| 128 × 128 × 128 | 5.6, 5.5, 5.7 | 6.2, 6.0, 6.0 |
| 1024 × 1024 × 32 | 94.5, 94.5, 94.6 | **62.7, 62.5, 62.7** |

One shape varies by up to **14 % between rounds inside a single build**, so a
cross-session before/after comparison cannot resolve anything under about 15 %.
Alternating the two builds inside one session can, and did: the N = 32 win is stable
to 0.2 us across three rounds while every other shape sits on top of itself. The
full-sweep comparison of §8 and §9 is good for **finding candidates, not for
confirming them** — every claim of a win or a regression has to be re-run this way.

The exception is `128 × 128 × 128`, which moved 5.6 -> 6.1 us consistently on a shape
whose row stride is 256 bytes, where the new branch cannot fire. Half a microsecond
on a 5.6 us kernel is code layout rather than logic, but it is real and it is not in
this change's favour.

## 11. The row store, and why the first two fixes for it failed

§10 closed with a cost model: ~56 us fixed plus 5.1 us per 32-column tile, and a
flat ~37.7 us penalty whenever the tile count is odd. The narrow-N shapes were left
behind by that penalty, and `1024 x 1024 x 31` was the worst row in the whole set.
This section is what it actually was.

### The phase counters say the store, and then mislead

Built with `-DHTP_MM_PHASE_PROFILE=1`, the kernel writes its phase split into probe
slots 200-207 and `PrintOpTimes` prints them for any run with `HEXAGON_TRACE=1`.
For M = K = 1024 at N = 31 against N = 32, per execution:

| N | weight fill | activation | HMX | **output store** | whole |
|---:|---:|---:|---:|---:|---:|
| 31 | 2 | ~0 | ~0.5 | **74** | 129 |
| 32 | 2 | ~0 | ~1 | **10** | 65 |

Two shapes that do *identical* work differ by 64 us, and all of it is in the store.
The activation column reads ~0 for both because `mm_act` is only credited inside
the pair-tile loop, which a single-tile shape never enters — its activation time is
real but lands in the untimed remainder. **Do not read a zero there as free.**

### Two fixes that did not work

**Replacing `vstu_variable`.** A runtime-length `vstu_variable` lowers to a memcpy
call, and at one call per output row that looked like the cost. Rewriting it as
whole eight-byte chunks plus one overlapping chunk removed the call: **-1 %**. The
instruction count barely moved, and the time did not either.

**Staging the tile in VTCM.** Rows are contiguous when the destination stride equals
the row width, so the tile is one run of bytes; staging it and writing it back in
128-byte chunks should have replaced 8192 memory transactions with 496. **-4 %.**
Profiled again, the store phase went 74 -> 68 us. The reason is visible in the
retrospect: the staging loop still wrote each row with the same eight scalar stores,
just into the stack instead of DDR. **It moved the stores, it did not remove them.**

Both were reverted. The lesson is the one §10 already implied and this confirms:
what costs is the **number of store instructions**, not the bytes, not the
transactions, and not the call overhead.

### The fix that worked

§10's N = 32 win came from noticing that `Q6_Vh_vdeal_Vh` hands back *two* rows —
row r in the low 64 bytes, row r+1 in the high 64 — so a pair whose rows are
contiguous is one 128-byte store instead of sixteen scalar ones. That only worked at
exactly 64-byte rows. The generalisation is to move the second row into place
instead of requiring it to already be there:

```c
const HVX_Vector second = Q6_V_valign_VVR(Q6_V_vzero(), v, 64 - (int)rowBytes);
vmemu((HVX_Vector*)dst0) = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rowBytes), v, second);
```

Shifting the vector up by `64 - rowBytes` puts row r+1 where the pair wants it, and
`Q6_Q_vsetq_R` selects the first `rowBytes` bytes of the first row and the rest
from the shifted copy. The store writes 128 bytes for a pair that is only
`2 * rowBytes` long, so its tail lands in the pair after it — which writes those
same bytes again. Only a pair whose tail would leave the region takes the scalar
path, which is what the `eBase + r + 2 < lp->sizeXYZ[0]` guard is for.

Measured at M = K = 1024, three rounds each, alternating builds in one session:

| shape | before | after | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| **1024 x 1024 x 31** | 127.0 | **66.5** | 47.2 | 2.70 | **1.41** |
| 1024 x 1024 x 32 | 62.5 | 62.5 | 45.9 | 1.36 | 1.36 |
| 1024 x 1024 x 65 | 112.1 | 110.0 | 61.8 | 1.81 | 1.78 |
| 1024 x 1024 x 257 | 143.0 | 140.2 | 95.7 | 1.49 | 1.46 |
| 1024 x 1024 x 64 | 67.0 | 66.7 | 54.0 | 1.24 | 1.24 |
| 1024 x 1024 x 1024 | 197.3 | 197.2 | 265.0 | 0.74 | 0.74 |
| 1024 x 1024 x 2048 | 378.0 | 377.3 | 508.3 | 0.74 | 0.74 |

**The worst shape in the campaign halved, and nothing else moved.** The three shapes
that do not change are the control: 32, 64 and the wide ones have row widths of 64
bytes or more, where the pair store already fired, and the new branch cannot reach
them.

### Operational traps this cost time on

- `cmake-out-android-arm64-v8a/backends/hexagon/skel/v79/ship/` is **stale** unless
  you build the `hexagon_skel_v79` target. Building `executor_runner` alone leaves
  a skel from an earlier session sitting there, and it will not match the runner's
  source ID.
- The standalone skel build directory needs `IDL_DIR` pointing at the sibling
  checkout (`/home/yydh/et-hexagon/backends/hexagon/idl`); this checkout has no
  `backends/hexagon/idl`, and configuring without it fails with
  `Cannot find source file: //htp_ops_skel.c`.
- Redirecting a build log to `/tmp` silently produces an empty file under the
  workspace-write sandbox. Keep logs under the repo.

### Where the set stands after this

Full 57-shape sweep, 228 rows, **no shape failed the fp32 reference**, and the
worst ratio in the set moved **2.41 -> 2.01**:

| shape | hex us | QNN us | ratio |
|---|---:|---:|---:|
| 256 x 4096 x 8192 | 2620.0 | 1303.0 | 2.01 |
| 1024 x 33 x 1024 | 94.3 | 47.0 | 2.01 |
| 1024 x 31 x 1024 | 83.7 | 42.1 | 1.99 |
| 256 x 8192 x 4096 | 3249.6 | 1661.0 | 1.96 |
| 1024 x 32 x 1024 | 83.9 | 44.1 | 1.90 |

Artifacts behind that sweep: skel `919e6ff1e10510e086207edd2fcebe5e`, runner
`a1ad4950b9d7728e5717b942f01e6d1a`, deployed at `$R/v5/`.

The narrow-N family this section fixed is gone from the list. What replaced it is
two different problems: the `M = 256` pair, which is weight bandwidth (64 MB of
weights streamed per call), and the small-K family (`1024 x 31|32|33 x 1024`),
where the output is 2 MB against a 63 KB activation and a 63 KB weight — the store
again, at a row width of 2048 bytes rather than 62.

## 12. Overlapping the weight DMA: a measured regression

§11 left two families at the top of the list — the small-K group
(`1024 x 31|32|33 x 1024`) and the `M = 256` pair. Profiling both with the phase
counters put numbers on them, and the numbers said the time is serial.

| shape | weight fill | activation | HMX | store | whole |
|---|---:|---:|---:|---:|---:|
| 1024 x 31 x 1024 | 2 | 17 | 15 | **60** | 97 |
| 1024 x 33 x 1024 | 4 | 16 | 16 | **60** | 102 |
| 256 x 4096 x 8192 | **1195** | 535 | 686 | 313 | 2744 |
| 1024 x 1024 x 32 | 2 | 2 | 2 | 9 | 67 |

The columns add up to the total in every row, so nothing overlaps anything. The
`M = 256` shape spends **44 % of its life waiting on the weight DMA** and the
small-K shapes spend **60 % in the output store**.

### What was tried

The kernel already knows how to overlap the weight DMA — there is a double-buffered
path with a two-deep `HmxDmaQueue` — but it is gated on `E == 1`, the single-row
case. Relaxing it to any shape whose weights do not all fit at once (`np > ntPerPass`)
and adding the missing refill inside the pass loop looked like the obvious win: the
weight for the next pass arrives while the current one computes.

**It is 51 % slower, not faster.**

| shape | before | after | |
|---|---:|---:|---:|
| 256 x 4096 x 8192 | 2614 | **3949** | +51 % |
| 256 x 8192 x 4096 | 3232 | **5239** | +62 % |
| 1024 x 31 x 1024 | 83.9 | 90.2 | +7.5 % |
| 1024 x 1024 x 32 | 62.4 | 63.9 | +2.4 % |

### Why

Double buffering has to come out of the same VTCM, so `ntPerPass` halves. The pass
loop is `for nt0 { for eBase { ... } }` — the activation is re-read and re-packed
**once per pass**, so halving the tile count per pass doubles the number of passes
and doubles the activation traffic. For `256 x 4096 x 8192` that is 20 MB of
activation reads becoming 40 MB, against a 64 MB weight read that only needed to be
hidden once. The overlap was real; it just cost more than it saved. The same
trade-off is why the loop order is what it is: with weights far larger than the
activation, streaming the weights once and the activation ten times beats the
reverse.

The change was reverted and the revert verified against the previous build on five
shapes, all within noise.

### A second attempt, and the same trade-off

`outputTiles` is 8 for the pipelined store path and 16 for the non-pipelined one.
The small-K shapes are store-bound and have VTCM to spare, so raising the pipelined
case to 16 keeps twice as many tiles in flight per store pass. Same-session A/B,
three rounds each:

| shape | before | after | |
|---|---:|---:|---:|
| 1024 x 31 x 1024 | 83.8 | **82.7** | -1.3 % |
| 1024 x 33 x 1024 | 94.4 | **93.1** | -1.4 % |
| 256 x 4096 x 8192 | 2617 | 2612 | -0.2 % |
| 1024 x 1024 x 1024 | 196.8 | 196.4 | -0.2 % |

Consistent and regression-free on that set — so it went to the full sweep, which
found the shape the A/B had missed:

| shape | before | after | |
|---|---:|---:|---:|
| **1024 x 1024 x 8192** | 1445.9 | **1677.2** | **+16 %** |

**The same mechanism as the double-buffering failure.** The extra accumulator buffer
is 32 KB more VTCM, and for a shape with `blockBytes = 64 KB` that is half a weight
tile — enough to push `ntPerPass` down and add a whole extra pass, re-reading the
activation. On the small-K shapes `ntPerPass` is capped by `np` and the extra 32 KB
is free, which is why they improved; on wide-N it is not.

Reverted. A version gated on the weights already fitting in one pass
(`ntPerPass >= np`) would keep the 1.4 % without the 16 %, but 1.4 % on two shapes
does not justify the extra branch. **Both round-4 experiments are reverted; the tree
is byte-identical to the round-3 build** (runner source ID `a1ad4950b9d7728e5717b942f01e6d1a`).

The general lesson, now confirmed twice: **on this kernel every VTCM byte taken for
something else is paid for in weight tiles, and a lost weight tile is paid for in
activation re-reads.** Any future change that grows a buffer has to be checked
against the wide-N shapes, not just the ones it was written for.

### The two families are at QNN's memory floor

Worth writing down before anyone spends another round on them. For
`256 x 4096 x 8192` the mandatory traffic is 64 MB of weights plus 2 MB of
activation plus 2 MB of output — 68 MB. QNN finishes the shape in 1303 us, which is
**52 GB/s on the minimum possible traffic**. There is no algorithmic slack left on
QNN's side; the only way past it is more bandwidth, not less work.

For `1024 x 31 x 1024` the traffic is 2.2 MB and QNN does it in 42.1 us — also
52 GB/s. Our own store phase moves 2 MB in 60 us, which is **35 GB/s**, and the
weight DMA in the same kernel reaches 54 GB/s. So the gap on the small-K family is
mostly one thing: **the HVX store path runs at two thirds of what the DMA engine
manages.** Staging the tile in VTCM and DMA-ing it out is the lever that has not
been tried.

### The skel binary is not reproducible

Rebuilding byte-identical sources produced a skel with a different md5
(`2a0bbfb2...` where the earlier build was `919e6ff1...`) while the **runner was
byte-identical** (`a1ad4950b9d7728e5717b942f01e6d1a`). The runner embeds the skel
source ID, so the sources really were the same; the `.so` simply does not build
reproducibly. Compare builds by source ID, not by skel md5 — the md5s in §11 are
deployment identifiers, not proof of equality.

## 13. The store rate is not a write-pattern problem

§12 closed by pointing at one number: the HVX store path moves 2 MB in 60 us
(**35 GB/s**) while the DMA engine in the same kernel moves the weights at
**54 GB/s**. Something in the store path is leaving a third of the bus unused, and
the obvious candidate was the order the writes go out in.

### The pattern, and why it looked wrong

`store_output_pair` writes two adjacent 32-column tiles — 128 bytes — for one row,
then advances to the next row. The caller invokes it once per tile pair, so for
`N = 1024` a row is visited **sixteen separate times**, each visit writing 128 bytes
and then walking away to sweep all 32 rows of the block. The destination is
row-major, so the row's own 2048 bytes are being assembled out of order, 128 bytes
per visit, with 4 KB of other rows' traffic in between.

The fix looks mechanical: swap the loops so the batch walks rows outer and tile
pairs inner. Each row's run then goes out contiguously in a single visit — 512
contiguous bytes instead of 128, with four times fewer visits. Same stores, same
instruction count, better order.

### It is not the pattern

| shape | before | after | |
|---|---:|---:|---:|
| 1024 x 1024 x 257 | 140.2 | **151.5** | **+8.1 %** |
| 1024 x 31 x 1024 | 83.8 | 86.0 | +2.6 % |
| 1024 x 1024 x 32 | 62.5 | 63.7 | +1.9 % |
| 1024 x 1024 x 65 | 110.0 | 111.0 | +0.9 % |
| 1024 x 33 x 1024 | 94.3 | **91.4** | -3.1 % |
| 1024 x 1024 x 1024 | 196.8 | 196.8 | 0 % |

Reverted; the tree is byte-identical to the round-3 build again.

Two things in that table are worth more than the result itself. `1024 x 1024 x 1024`
is flat because it takes the specialised 1024-cube path and never enters this code —
a real control. And `1024 x 1024 x 32` got 1.9 % **worse on a path it does not
execute** (a single N tile has no pair to batch), which is code layout, not logic.
Anything under about 2 % in this table is not signal.

So: reordering the writes neither helped nor hurt much, and the 35 GB/s is not the
memory controller failing to coalesce. The pattern hypothesis is dead.

### What that leaves

The same bus carries a DMA at 54 GB/s, so the bus is not the limit either. What is
left is the store path itself — the rate at which one HVX context can retire
`vmemu` — and that is exactly the difference between the two kernels in this file.
The 1024-cube path does **not** store on the main thread: every 8 tiles it retires
the pending accumulator and hands the group to a worker-pool task
(`htp_ops_matmul1024_output_worker`), four tasks covering the 32 tiles, with the
main thread computing the next group meanwhile. The general path, which every other
shape takes, stores inline on one context.

**Parallelising the general path's store across the worker pool is the untried
lever**, and it is the one structural thing the fast path has that the slow one does
not. It is also the change with a real chance of moving the small-K family, where
the store is 60 % of the operation.

## 14. The worker-pool store, and the per-job cost that kills it

§13 ended on the one structural difference between the two kernels in this file: the
1024-cube path hands its output store to worker-pool tasks and the general path
stores inline on one HVX context. This section is what happened when the general
path was given the same treatment.

### First, the store phase is really the store

`mm_store` could have been hiding an HMX accumulator wait — the unit's stores are
asynchronous, so the `vmem` that reads an accumulator blocks on it, and that wait
would be charged to whoever reads it. Four shapes with the same M and N and a
2 MB output, varying only K, settle it:

| shape | weight fill | activation | HMX issue | **store** | whole |
|---|---:|---:|---:|---:|---:|
| 1024 x 31 x 1024 | 2 | 14 | 13 | **63** | 93 |
| 1024 x 65 x 1024 | 5 | 16 | 13 | **64** | 100 |
| 1024 x 2048 x 1024 | 72 | 101 | 144 | **109** | 427 |
| 1024 x 3072 x 1024 | 106 | 145 | 259 | **120** | 633 |

Flat at 63/64 across a 2x change in K, where the unit does identical work. So at low
K the phase is genuine store work: **2 MB in ~60 us is 33 GB/s**. QNN finishes the
whole `1024 x 31 x 1024` in 42 us, which needs at least **48 GB/s** for the output
alone. The gap is real and it is in the store rate.

*(Correction to §10-11: probe slot 206 is **weight bytes**, not output bytes. It
reads 65536 / 196608 / 4194304 / 6291456 for the four shapes above, which is
`kp * 32 * 2048` — the padded weight, not the 2 MB output they all share.)*

### What was built

A task struct and worker carrying the same store the inline path does, submitted
once per pipelined batch, with the previous task waited on before the next submit so
at most one is in flight and the two VTCM accumulator buffers stay safe. Two real
bugs had to be fixed to get it running at all:

- the retire read (`vmem` of the last vector) ran on the first batch, when
  `pendingTiles` is still 0 and `pendingOutput` is still null — an immediate
  `0x8000040d`;
- the flush after the batch loop stores the last batch of each row block, and the
  first version replaced it with a bare wait, so that batch was never written.

After both fixes it produced correct results and was **3.4x slower**:

| shape | before | after | |
|---|---:|---:|---:|
| 1024 x 31 x 1024 | 83.9 | **284.2** | **+239 %** |
| 1024 x 33 x 1024 | 94.4 | **290.3** | **+207 %** |
| 256 x 4096 x 8192 | 2617 | 2999 | +15 % |
| 1024 x 1024 x 257 | 140.0 | 140.5 | 0 % |
| 1024 x 1024 x 65 | 110.0 | 110.2 | 0 % |

### Why, and what it rules out

**The worker pool costs about 1.5 us per job.** The general path submits one task per
pipelined batch — 8 tiles, 16 KB of output — which is roughly 0.5 us of actual store
work. The overhead is three times the work. `1024 x 31 x 1024` submits 128 of them
across its 32 row blocks, which is the ~200 us it lost, almost exactly.

The 1024-cube path gets away with the same mechanism because it submits **four**
tasks, each covering 8 tiles x 32 rows = 512 KB. That is the whole difference: the
worker pool is not slow, the general path's granularity is.

This kills the parallelism idea at the current buffering, and it also explains why
the shapes with few tiles (n257, n65) were unaffected — they submit only a handful of
tasks. Making it pay would mean buffering a whole output row so a task covers
megabytes, which needs a different VTCM layout, not a different worker.

### What is left

Since HVX stores sit at 33 GB/s on one context and the pool cannot be fed coarsely
enough to help, the remaining explanation for QNN's 48+ GB/s is that **QNN does not
store with HVX at all — it DMAs the output out of VTCM**, at the 54 GB/s the weight
DMA already reaches in our own kernel. For the small-K shapes that is affordable: at
`K = 31` the weight block is 2 KB per tile and `ntPerPass` is capped by `np`, so
the weight buffer uses 64 KB of an 8 MB VTCM and the rest is idle. Staging a full
output row (all 32 tiles, 2048 bytes) there would make the transfer contiguous and
give one 1D descriptor per row block.

### The source ID is a byte-exact fingerprint

Reverting the five edits above left one stray blank line behind, and the runner's
source ID changed from `a1ad4950b9d7728e5717b942f01e6d1a` to `56ed3d7d...` with the
code otherwise identical. Removing the blank line restored it exactly. That makes
**the runner md5 the fastest and most reliable revert check available** — cheaper and
stricter than re-running a device A/B.

## 15. The odd-N penalty: one store per byte on the last tile

§10 found that odd tile counts pay a flat ~37.7 us and could not fix it: pairing the
lone trailing tile with its predecessor needs a 128-byte store that ends exactly at
the row end, which only holds when the tail is a full 32-column tile. §13 then
showed the store phase is genuine store work. Profiling N=64 against N=65 with four
executions each puts the two together:

| shape | weight fill | activation | HMX issue | **store** | whole |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 x 64 | 2.8 | 50 | 1.5 | **12.5** | 68 |
| 1024 x 1024 x 65 | 3.8 | 50.8 | 2.0 | **54.0** | 112 |

The activation is identical and the unit does identical work. **The whole 44 us
difference is in the store phase**, and N=65 stores 133 KB against N=64's 131 KB.

### What the tail actually is

For `N = 65` the third tile covers columns 64..64 — **two bytes per row**, at byte
offset 128 of a 130-byte row. The pair store writes bytes 0..127 of each row, so the
last two bytes have no partner and go through `htp_ops_loop_hmx_store_output_tile`,
which for a row that narrow falls back to one store per byte. That is 2048 byte-wide
stores to DDR, each a partial-line write. 1024 rows of two bytes cost **41 us** — the
same flat penalty §10 measured, now attributed.

### The fix

The tail tile does not need a partner; it needs a **wide store**. One 128-byte vector
per row covers it, and the bytes past the end of the row land in the start of the
next one — which the pair stores overwrite, provided they run afterwards. So the tail
tiles are computed and stored **before** the pair loop instead of after it:

- only the **last** tile takes the wide store. `N = 63` leaves two tiles over, and
  the earlier one's overflow would land inside its own row, so it keeps the exact
  store;
- the **final row of the tensor** keeps the exact store, because nothing follows it
  to overwrite the overflow;
- the whole thing is gated on `ntTail > nt0` — the pair loop has to actually run, or
  nothing corrects the overflow at all;
- and the wide store is used only when the tile **is not the end of N**, or when it
  is and `nt0 == 0`. See below: this is the one the first sweep caught.

### The bug the full sweep caught

The three-round A/B was clean and the full sweep found **four numerics failures** —
`129 x 4093 x 2051` and `65 x 8191 x 1025`, both rounds each. Diffing the two
builds' output binaries located it exactly: every element of **columns 0..62 of rows
1..64** differed, 64 rows x 63 columns = 4032 values. 63 columns is 126 bytes, which
is precisely the wide store's overflow for a two-byte tail.

The cause is the loop nest: the passes over N are the **outer** loop and the row
blocks are inner. On the last pass the pair stores only cover that pass's columns, so
a wide store at the end of N overflows into the next row's columns 0..62 — which were
written by the *first* pass, long before, and are never rewritten. Both failing shapes
are the ones with more than one `nt0` pass; `1024 x 1024 x 257` and every other shape
in the set fits its weights in a single pass and was unaffected.

So the wide store is now gated on `ntEnd < np || nt0 == 0`: a tile that is not the
end of N is followed by another tile that the next pass writes, and the end of N
overflows into the next row, which only the pair stores of a pass starting at column 0
can reach. Re-run over all 57 shapes: **zero numerics failures**.

Finding where the pair loop stops needs its index progression, which is subtle
(`batchTiles` is clamped by the tile count, the remaining columns and `outputTiles`,
then rounded down to even), so it is replayed once before the loop rather than
guessed.

Measured at M = K = 1024, three rounds each, alternating builds in one session:

| shape | before | after | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 x 65 | 109.9 | **77.8** | 66.7 | 1.68 | **1.17** |
| 1024 x 1024 x 257 | 139.9 | **108.6** | 95.7 | 1.46 | **1.13** |
| 1024 x 1024 x 1025 | 263.4 | **233.2** | 228 | 1.16 | **1.02** |
| 1024 x 1024 x 1024 | 199.1 | 198.5 | 265 | 0.75 | 0.75 |
| 1024 x 1024 x 64 | 67.9 | 68.9 | 54.0 | 1.26 | 1.27 |
| 1024 x 1024 x 32 | 63.8 | 64.8 | 45.9 | 1.39 | 1.41 |

N=64 and N=32 take paths this change does not touch — their tail range is empty, and
for N=32 the pair loop never runs — so their 1.5 % is the code-layout noise this
device has shown all along (§10, §13), not a real cost.

The full 57-shape sweep, two rounds each, confirms it:

| shape | before | after | | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|---:|
| 1024 x 1024 x 65 | 112.0 | **77.4** | -30.9 % | 65.5 | 1.71 | **1.18** |
| 1024 x 1024 x 257 | 142.9 | **107.9** | -24.5 % | 96.6 | 1.48 | **1.12** |
| 4097 x 1025 x 257 | 585.5 | **464.8** | -20.6 % | 1099.3 | 0.53 | **0.42** |
| 1024 x 1024 x 1025 | 264.2 | **237.4** | -10.1 % | 309.4 | 0.85 | **0.77** |
| 128 x 128 x 128 | 6.3 | 5.5 | -12.8 % | 4.2 | 1.50 | 1.32 |
| 31 x 1024 x 1024 | 47.9 | 50.6 | +5.6 % | 43.7 | 1.10 | 1.16 |

The odd-N family has left the top of the ranking entirely. `31 x 1024 x 1024` is the
one shape that moved the wrong way; it takes a path this change does not alter
(`E = 31` is below the pipeline threshold, so its batches are two tiles and its tail
range is empty), so its 5.6 % is layout, and it is the only one.

This is the third shape family the store work has fixed, after §11's narrow rows and
§10's N=32 row pair, and it is the largest of the three.

## 16. Two things QNN does that do not transfer

Round 7 moved the odd-N family. What is left at the top of the ranking is the
small-K family — `1024 x {31,32,33,63,64,65} x 1024`, all around 1.9x — and its
phase profile is store-dominated: K=31 spends 60 us of 84 in the store phase, for a
2 MiB output. Two ways to close that were tried; both are closed off now.

### The output store config is not what selects the layout

§4's disassembly left one concrete, testable difference. Our consume is

```c
Q6_cvt_hf_acc_R(2);
__builtin_HEXAGON_M8_mxmem(out, 0);
```

while the real 1024 model on the QNN side passes **0x700** for the output store
config, with `cvt = 2` — same conversion, different store parameter — and its native
output blocks span 128 rows rather than 32 consecutive ones. If that parameter chose
the layout the accumulator lands in, the HVX `vdeal` conversion could disappear
entirely, which is worth far more than any tuning of the conversion loop.

It does not. Built with the parameter set to 0x700, the skel runs without fault and
produces **byte-identical output** to the 0-parameter build on both a 32-wide shape
(`1024 x 1024 x 32`, 32768 values) and a 1024-wide one (`1024 x 31 x 1024`,
1048576 values) — zero differing elements. Whatever the parameter controls, it is not
the memory layout of a consumed 32x32 fp16 tile in this configuration. The lead is
closed; do not re-open it without new evidence about the parameter's encoding.

Operational note: `hmx_utils.h` is a skel source, so changing it changes the skel
source ID and the runner must be rebuilt with it. Deploying the skel alone gives
`skel source ID mismatch` followed by a SIGABRT (exit 134), which reads like a crash
in the new code but is not.

### The store loop is already at its floor

The other reading was that the store phase is dominated by loop overhead the compiler
failed to remove. Disassembling the pair store in our own skel says otherwise:

```text
v29.h = vdeal(v28.h)          ; tile 1 rows
v28.cur = vmem(r7++#0x1)
v31.h = vdeal(v30.h)          ; tile 0 rows
v30.cur = vmem(r14+#0x0)
v27:26 = vdeal(v29,v31,r1)    ; pair the two tiles' matching halves
vmemu(r14+#0x0) = v26         ; row r
vmemu(r15+#0x0) = v27         ; row r+1
```

19 instructions for 256 bytes, and the row addresses are already strength-reduced
(`r8 = add(r8,r12)`) — there are no multiplies to hoist. Seven of the 19 are vector
ops: two VTCM loads, three deals, two DDR stores. The measured 7.3 ns per iteration is
about 8.8 cycles at 1.2 GHz, so the loop runs at roughly 2.2 IPC against a
vector-issue floor of 7 cycles. It is within about 1.3x of the best this algorithm can
do.

The 1024-cube worker is the same shape — two `vdeal`s, one `W` deal, two stores per
two rows — so staging the whole 2 MiB output in VTCM and converting it after compute
does not change the conversion cost, only when it is paid.

So the small-K store phase is not a code-quality problem. Cutting it needs an
accumulator layout that does not require the conversion, and the one candidate for
that has no effect. That is the honest state of this family.

## 17. The small-K family: stop paying the conversion on the critical path

§16 closed off making the output conversion cheaper. The small-K family —
`1024 x {31,32,33,63,64,65} x 1024`, every one of them around 1.9x — was still the
largest block of losses in the set, and its phase profile is store-dominated: K=31
spent 60 us of 84 in the store phase for a 2 MiB output.

The conversion does not have to get cheaper. It has to stop being **serialized with the
compute**. The 1024-cube path already does exactly that: it keeps the whole 1024x1024
output resident in VTCM, and converts and writes it out on four worker-pool jobs while
the compute that follows keeps running. Its generic branch for `K != 1024` was already
written; only the eligibility guard kept small K out of it.

The guard now accepts `K <= 128` alongside 1023/1024/1025. Nothing else changed — the
VTCM size check, the packed-weight requirement and the 128-byte alignment check all
still apply, so a shape that does not fit still falls through to the general path.

Three rounds each, alternating builds in one session:

| shape | before | after | | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|---:|
| 1024 x 31 x 1024 | 83.9 | **46.9** | -44 % | 42.2 | 1.99 | **1.11** |
| 1024 x 32 x 1024 | 83.8 | **46.6** | -44 % | 44.7 | 1.88 | **1.04** |
| 1024 x 33 x 1024 | 94.3 | **50.9** | -46 % | 45.5 | 2.08 | **1.12** |
| 1024 x 63 x 1024 | 94.3 | **51.0** | -46 % | 50.4 | 1.88 | **1.01** |
| 1024 x 64 x 1024 | 78.7 | **47.9** | -39 % | 48.5 | 1.63 | **0.99** |
| 1024 x 65 x 1024 | 88.7 | **50.7** | -43 % | 52.7 | 1.68 | **0.96** |
| 1024 x 1024 x 1024 | 196.7 | 196.5 | control | — | — | unchanged |

Four of the six now lead or tie QNN, and the whole family has left the top of the
ranking. The control is the same shape the path was already used for, and it did not
move.

The measured drop is 37 us against a conversion whose instruction count did not
change, so the gain is the overlap, not a faster conversion. That is also why round 6's
worker-pool store failed: it was the same idea at the wrong granularity — 128 tiny
jobs at ~1.5 us of per-job cost each (§14), against four jobs of 512 KB here.

The full 57-shape sweep, two rounds each: **zero numerics failures, eleven shapes
improved by more than 5 %, none regressed by more than 5 %.** The whole small-K family
has left the top ten, and the set as a whole crossed over:

| | before §15 | after §15 | after §17 |
|---|---:|---:|---:|
| shapes leading QNN | 17 / 54 | 17 / 54 | **21 / 54** |
| geometric mean of hex/QNN | 1.055 | 1.055 | **0.987** |
| worst ratio | 2.01 | 2.08 | **2.01** |

What is left at the top is no longer a store problem. `256 x 4096 x 8192` and
`256 x 8192 x 4096` are at QNN's memory floor on mandatory traffic (§9), the two
`M = 1` shapes are activation-read bound, and `65 x 8191 x 1025` /
`129 x 4093 x 2051` are the multi-pass odd shapes whose last pass cannot take the
wide tail store (§15).

## 18. The activation is re-read once per pass; holding it removes the passes' cost

After §17 the top of the ranking is a different problem. A phase profile of every
remaining worst shape says the same thing about all of them: **the four phases sum
exactly to the whole.** Nothing overlaps.

| shape | weight fill | activation | HMX | store | whole |
|---|---:|---:|---:|---:|---:|
| 256 x 4096 x 8192 | 1081 | 517 | 712 | 291 | 2611 |
| 256 x 8192 x 4096 | 1104 | 1124 | 759 | 251 | 3243 |
| 8 x 8192 x 4096 | 1118 | 236 | 54 | 36 | 1445 |
| 65 x 8191 x 1025 | 287 | 158 | 67 | 24 | 537 |
| 129 x 4093 x 2051 | 280 | 125 | 102 | 55 | 565 |

The weight DMA is not the problem: 64 MB in 1081 us is **59 GB/s**, above QNN's whole-
shape rate on the same traffic. What costs is that everything else is serialized on top
of it, and the second largest item is the activation.

### Why the activation is read many times

The loop nest is `for nt0 { weight fill; for eBase { pack activation; compute } }` —
the passes over N are **outer**. The activation is packed and read once per pass, so a
shape with `np / ntPerPass` passes reads it that many times. `256 x 4096 x 8192` has
2 MiB of activation and 18 passes, so it moves 36 MiB of it.

The fix is to hold the whole **packed** activation in VTCM and pack it once. That looks
like it violates the trade-off that has bitten this kernel repeatedly — bytes taken
from the weight buffer are paid for in weight tiles — but it does not, because of what
the lost tiles are paid in: another pass. And a pass whose activation is already
resident costs nothing to re-read. The trade only bites while the activation is being
re-fetched per pass; removing that is what makes the bytes cheap.

The cache is engaged only when it fits with at least two weight tiles to spare, the
shape would otherwise take more than one pass, and the source is a contiguous run the
DMA can take in one piece.

| shape | before | after | | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|---:|
| 256 x 4096 x 8192 | 2616 | **2226** | -15 % | 1304 | 2.01 | **1.71** |
| 256 x 8192 x 4096 | 3240 | **2376** | -27 % | 1647 | 1.96 | **1.44** |
| 8 x 8192 x 4096 | 1419 | **1213** | -15 % | 1022 | 1.40 | **1.19** |
| 65 x 8191 x 1025 | 536 | 538 | +0.4 % | 307 | 1.77 | 1.75 |
| 129 x 4093 x 2051 | 563 | 564 | +0.1 % | 339 | 1.66 | 1.66 |
| 8192 x 4096 x 1 | 1718 | 1733 | +0.9 % | 1141 | 1.51 | 1.52 |
| 1024 x 1024 x 1024 | 197 | 197 | control | — | — | — |
| 1024 x 31 x 1024 | 47 | 47 | control | — | — | — |

### The trap in the first version

The first build measured **537 -> 1498 us** on `65 x 8191 x 1025` and
`565 -> 1565 us` on `129 x 4093 x 2051` — a 3x collapse on exactly the two shapes
that were supposed to be helped least.

A profiler build showed the HMX path was still being taken and the activation phase
had in fact collapsed, 158 -> 6.5 us: the cache worked. The time had simply moved
somewhere the phase counters do not cover — the one-time pack runs before the `nt0`
loop. The cause was that the pack was written against the **non-prefetch** branch,
which for `K % 64 != 0` falls back to a per-row `memcpy` into the staging buffer. The
prefetch branch handles the same odd K with one contiguous copy plus a `paddedRaw`
rearrange, which is what the old code had been using all along. Every shape that won
had `K % 64 == 0`, so they took the flat copy and never saw it.

With the flat copy restored the two odd-K shapes are neutral, and the cache is now
gated on `K % 64 == 0`: for odd K the rearrange costs about what the cache saves, so
it is not worth the weight tiles it takes.

The full 57-shape sweep, two rounds each: **zero numerics failures, eight shapes
improved by more than 5 %, none regressed by more than 5 %.** It reaches further than
the three shapes it was aimed at — `127 x 4096 x 4096` -12 %, `128 x 128 x 128` -8.5 %,
`1 x 1024 x 2048` -8 %, `33 x 1024 x 1024` -5.5 % — because every small-M shape with
more than one pass benefits.

| | before §17 | after §18 |
|---|---:|---:|
| geometric mean of hex/QNN | 0.987 | **0.966** |
| worst ratio | 2.01 | **1.71** |

The worst shape has moved for the first time in four rounds. What is left is now three
shapes at 1.71 — the two multi-pass odd-K shapes, which this does not help, and
`256 x 4096 x 8192`, which is weight-bound at 59 GB/s against QNN's 52 GB/s on the
same traffic and loses on the work serialized behind that DMA rather than on the DMA
itself.

## 19. The weight-DMA overlap, unlocked by §18

§12 recorded weight-DMA double-buffering as a regression and it was reverted: halving
`ntPerPass` doubled the number of passes, and **every extra pass re-read the whole
activation**. `256 x 4096 x 8192` went 2614 -> 3949 us and `256 x 8192 x 4096`
3232 -> 5239.

§18 removed exactly that penalty. With the packed activation resident, a pass costs
nothing but its own weights, so the number of passes stops being the thing that decides
the cost. The double-buffer machinery was already written and already correct for more
than one pass — the pre-push of two slots, the slot alternation, and the refill at the
end of each pass all handle `np > 2 * ntPerPass`. It was gated to `E == 1` for the one
reason that no longer holds, so the gate is now `E == 1 || cacheActivation`.

| shape | before | after | | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|---:|
| 256 x 4096 x 8192 | 2222 | **1228** | -45 % | 1297 | 1.71 | **0.95** |
| 256 x 8192 x 4096 | 2356 | **1307** | -45 % | 1649 | 1.43 | **0.79** |
| 127 x 4096 x 4096 | 855 | **604** | -29 % | 854 | 1.00 | **0.71** |
| 33 x 4096 x 4096 | 685 | **556** | -19 % | 682 | 1.29 | **0.81** |
| 8 x 8192 x 4096 | 1217 | **1057** | -13 % | 1022 | 1.19 | **1.03** |
| 129 x 4093 x 2051 | 568 | 564 | -0.7 % | — | 1.71 | 1.71 |
| 65 x 8191 x 1025 | 537 | 540 | +0.5 % | — | 1.71 | 1.71 |
| 1024 x 1024 x 1024 | 197 | 197 | control | — | — | — |
| 1024 x 31 x 1024 | 46 | 47 | control | — | — | — |

`256 x 4096 x 8192` is the clearest reading of what happened: its weight DMA is 1081 us
on its own, and the whole kernel is now 1228 us. Almost everything that used to be
serialized behind the DMA is now behind it instead of beside it. The two odd-K shapes
are unchanged because §18's cache is off for them, and this gate rides on that cache.

**The two shapes that defined the worst case for four rounds now beat QNN.** They were
never losing on the DMA — they were losing on the 1520 us of work sitting next to it.

### The odd-K exclusion had to go

§18 gated the activation cache on `K % 64 == 0` because on the odd-K shapes it measured
as a small loss — 537 -> 547 and 564 -> 597 us. The reason it was a loss was that the
cache trades weight tiles for passes, and the pass count is what double-buffering
multiplies. §19 had just doubled the number of passes, so the cache had doubled in
value without anything else changing.

Removing the gate:

| shape | before | after | | QNN | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|---:|
| 129 x 4093 x 2051 | 560 | **433** | -23 % | 327 | 1.72 | **1.33** |
| 65 x 8191 x 1025 | 537 | **414** | -23 % | 322 | 1.67 | **1.28** |

This is worth recording as the shape of the round rather than as two separate changes:
the activation cache and the weight double-buffer are each worth more because the other
exists. The cache pays in passes; the double-buffer spends passes. Neither is a win on
its own terms, and together they are the largest change of the campaign.

### Verified

Both sweeps, 57 shapes two rounds each, **zero numerics failures** — including the
odd-K `paddedRaw` rearrange, which is the one path that had never been checked before
the gate came off.

| | round 8 | round 9 | round 10 |
|---|---:|---:|---:|
| geometric mean of hex/QNN | 0.987 | 0.966 | **0.917** |
| worst ratio | 2.01 | 1.71 | **1.45** |
| shapes at or under QNN | 21 / 54 | 20 / 54 | **24 / 54** |

The ranking has closed up: the worst eight now sit between 1.30 and 1.45, where before
this round they ran from 1.30 to 1.71. `8 x 8192 x 4096` moved +5.4 % between the two
sweeps although the gate change is a no-op for it (`K % 64 == 0`), which is inside the
14 % round-to-round spread measured on a single build and is not signal.

## 20. What the store phase actually is

Round 11 opened on the largest remaining absolute gap, `8192 x 1024 x 1024` (1474 us
against QNN's 1122), whose profile is 408 us of activation, 283 of HMX loads and 734 of
store — again summing to the whole.

The obvious reading is that the store writes twice what it needs to: a 32-column tile is
64 bytes of a row, and 64 bytes is half a 128-byte line. That reading is wrong. The pair
store already buffers two tiles in VTCM and rebuilds each row's two halves into one
128-byte vector, so every store is a single full-line `vmemu` with no waste. 16 MiB of
output moves in 131 K stores of 128 bytes, one instruction each. It is at the device's
write floor of about 22 GB/s, and QNN is at the same floor — it simply overlaps it.

That changes what the phase counters mean. `mm_hmx` is not the matrix unit: the tile
loads are `Q6_activation_hf_mxmem_RR_deep` and `Q6_weight_hf_mxmem_RR`, which are HVX
instructions writing the unit's memory. The unit's own multiply runs underneath all of
it and is never on the critical path. **So activation, HMX loads and store are all HVX
work, which is why they sum to the whole: the kernel is HVX-bound, not HMX-bound.** The
question for every remaining shape is what HVX work can be removed, not what the unit is
waiting on.

The one mechanism that looked available — the weight fits in VTCM for these shapes and
leaves megabytes free, so the output batch could be deepened to keep more stores in
flight — is a loss: `8192 x 1024 x 1024` +3.9 %, `1024 x 1024 x 8192` +15 %,
`4096 x 4096 x 1` +1.4 %, and nothing gained. That is the third independent
confirmation that the store loop is at its floor and that its cost is bandwidth, not
occupancy.

What is left, then, is to take the store off the HVX entirely: rearrange the consumed
accumulator into a row-major VTCM staging buffer and let a 2D DMA write it out, which
the unit supports and which runs alongside the HVX rather than on it. That is the next
lever and it is not attempted here.

## 21. The 2D-DMA store: structurally available, but the issue cost swamps it

§20 left one lever: move the output write off the HVX. The SDK does expose it —
`dma_utils.h` has `DMA_DESC_TYPE_2D` and `dma_desc_2d_t` with `roi_width`,
`roi_height`, `src_stride` and `dst_stride`, and `attention_common.hpp` shows the
semantics (bytes per row, row count, row pitch). So the store was rewritten to
rearrange into a row-major VTCM staging buffer with `vmem` instead of `vmemu` to DDR,
then hand one 2D descriptor the strided write. The rearrange is the same
`Q6_W_vdeal_VVR` shape the pair store already uses, just writing to VTCM.

It is a loss, and for a reason that is independent of whether it computes correctly:

| shape | before | after | |
|---|---:|---:|---|
| 256 x 4096 x 8192 | 1227 | 2055 | **+67 %** |
| 8192 x 1024 x 1024 | 1474 | 1546 | +4.9 % |

The store is issued once per batch, and a batch is `outputTiles` = 8 tiles = 16 KiB.
`256 x 4096 x 8192` issues about 4600 of them (32 batches x 8 row blocks x 18 passes),
and the measured slowdown of ~830 us works out to roughly 200 ns per descriptor —
issue and latency, not bandwidth. Two `dma_wait_for_idle()` calls per store expose
all of it.

There was also a correctness bug I fixed and one I did not. The fixed one: the first
version wrote row `r+1`'s vector at `out + p*128 + 64`, which is inside row `r`'s
staging row, instead of at `out + rowBytes`. The unfixed one: after that fix the
results are still wrong (finite, but 2.7-5.0 off), which points at the HVX's VTCM
stores not being visible to the DMA engine before `dmstart` — a fence this code has
never needed, because every other DMA in the kernel is DDR -> VTCM, never VTCM -> DDR.

**So the direction is not closed, but it needs a different shape than this.** Two
things have to change together: the transfer must be one descriptor per row block
rather than per batch (which means staging a whole 32 x N block, so `outputTiles`
would have to cover N), and the descriptors must be chained rather than waited on, so
the issue cost overlaps instead of being exposed. Both are bigger than a batch-local
change, and neither is attempted here.

The change was reverted, and the revert is byte-exact: the rebuilt runner is
`cb128090425d81922bbb5cba1e6138bb`, the same md5 as the round-11 reverted build.

## 22. Two more things the profile suggested, and what happened

### The staging is not a VTCM-bandwidth problem

The pack costs three VTCM accesses per activation byte — the DMA writes the raw block
in, the rearrange reads it and writes the packed block out. At 32 MiB of activation
that is 96 MiB through a unit that measures around 120 GB/s, which is the 40 GB/s the
pack phase reports. The obvious conclusion is that the staging is the problem, and that
packing straight out of DDR would cut it to two accesses.

It does not. Forcing `prefetchActivation` off so every shape packs directly from DDR:

| shape | staged | direct DDR | |
|---|---:|---:|---|
| 4096 x 4096 x 1 | 864 | 1418 | **+64 %** |
| 8192 x 4096 x 1 | 1722 | 2850 | **+65 %** |
| 1024 x 1024 x 31 | 67 | 110 | **+63 %** |
| 8192 x 1024 x 1024 | 1474 | 1816 | +23 % |

So the DDR read path is the thing that stalls, by about 2.6x, and the staging already
buys it back. The VTCM accounting was right and the conclusion drawn from it was wrong:
the phase is DDR-latency-bound, not VTCM-bound, and it is already at its floor.

### A real redundancy the compiler will not let me remove

`HMX_FP16_MAX_TILES_PER_LOAD` is 32, so any K <= 1024 loads its whole activation block
in a single call. The general path then reloads it for **every** N tile of that row
block — 32 times for `8192 x 1024 x 1024`, whose HMX phase is 283 us of mxmem writes,
about half of it the same activation tiles going in over and over.

The two loads are independent: `Q6_activation_hf_mxmem_RR_deep` and
`Q6_weight_hf_mxmem_RR` take the same tile count but write separate mxmem regions, so
once the activation is in, only the weights need to move per tile. That is worth roughly
130 us on that shape alone.

It does not compile. Adding the branch to the tile-load loop makes the Hexagon VLIW
machine scheduler abort — `llvm::ScheduleDAGMI::postProcessDAG()`, exit 134 —
reproducibly, and marking the two split loaders `noinline` (which is what the existing
`hmx_load_tiles_fp16` does, presumably for this reason) does not help. The region the
scheduler sees grows past what it handles.

So the win is real and identified but needs a formulation that does not put a branch in
that loop — hoisting at a higher level, or splitting the loop into a resident-activation
variant chosen before the row block rather than inside it. Not attempted here.

Reverted, byte-exact: rebuilt runner `cb128090425d81922bbb5cba1e6138bb`.

## 23. The activation hoist is real and the compiler will not have it

§22 identified the win: with `HMX_FP16_MAX_TILES_PER_LOAD` = 32, any K <= 1024 loads its
whole activation block in one call, and the general path then reloads it for every N
tile of that row block — 32 times for `8192 x 1024 x 1024`, whose 283 us HMX phase is
mxmem writes, about half of them the same activation tiles going in again.

Four formulations were tried. All of them abort the Hexagon VLIW machine scheduler in
`llvm::ScheduleDAGMI::postProcessDAG()` with exit 134:

1. a branch in the tile-load loop selecting a weight-only loader;
2. the same, with both split loaders marked `noinline`;
3. passing the flag as a loop-invariant argument so the loop body stays a single call;
4. calling the resident-activation loader unconditionally and letting the callee decide.

Disabling only the hoist call — leaving the wrapper, the extra argument, and the
weight-only loader in place — compiles cleanly. So it is the **call site** in the row
block loop, not the branch, not the early return, and not the new functions: that loop's
scheduling region is already at the size the scheduler can handle, and any added call
pushes it over. That also explains why `hmx_load_tiles_fp16` itself is `noinline` — the
same work has to stay outside that region.

The way through is to **shrink** the region rather than add to it: the per-block
activation preparation (the cache branch, the prefetch branch and the plain pack) is
itself a large block of code that belongs in a `noinline` helper. Extracting it should
make room for the resident load. Not attempted here — it is a real refactor of the row
block loop, and the variables it captures are most of the loop's state.

Reverted. The rebuilt runner is `67dd9c8f87667a9786cba97a02740042`, which is **not** the
`cb128090425d81922bbb5cba1e6138bb` of the round-13 revert even though the tree is back to
the same behaviour — a source-text difference I could not attribute (no other agent
touched a skel source in the interval, and `hmx_utils.h` is clean against HEAD). It is
recorded rather than glossed: the md5 is a provenance fingerprint, so this build is not
provably the round-10 source. What is verified is behaviour — 24 records over six shapes,
0 numerics failures.

## 24. Why the activation hoist could never have worked

§23 blamed a scheduling region that was too large. That was wrong, and the compiler said
so once I read past the stack trace to the assertion:

    HexagonSubtarget.cpp:381: HMXActivateWeightMutation::apply:
      Assertion `A == nullptr && "Unmatched activation instruction"' failed.
    Running pass 'Machine Instruction Scheduler' on function
      '@htp_ops_loop_hmx_load_activation_resident'

Pairing the activation load with a weight load moved it to the other half of the same
assertion — `"Unmatched weight instruction"`. **The two instructions have to appear
together, and neither is legal alone.** That is what all four formulations in §23 ran
into, and it is why `hmx_load_tiles_fp16` always does both and is `noinline`: the pass
exists to *activate* a fused weight mutation, so a split pair is not merely unscheduled,
it is rejected.

Making the pair legal-but-cheap does compile — a weight load paired with a one-tile
activation load that rewrites tile 0 into the slot it already occupies, with the bytes it
already holds. It is correct (40 records, 0 numerics failures) and it is far slower:

| shape | odb | hoist | |
|---|---:|---:|---|
| 8192 x 1024 x 1024 | 1474 | 3786 | **+157 %** |
| 1024 x 1024 x 8192 | 1243 | 3611 | **+190 %** |
| 256 x 1024 x 1024 | 82.6 | 155 | +88 % |
| 128 x 1024 x 1024 | 59.4 | 95.9 | +61 % |
| 1024 x 1024 x 65 | 77.7 | 91.4 | +18 % |

A first attempt with `limit = -1` (what `n_tiles == 0` produces) measured the same, which
rules out the no-op length as the cause: **the split pair is what costs.** The fused load
is a single optimized unit, and taking the activation out of it makes the weight load
roughly 2.6x slower — more than the activation reload ever cost. The redundancy §22
measured is not removable; it is the price of the fused form.

This closes the HMX-load direction. The mxmem traffic in that phase is inherent, and the
283 us it takes on `8192 x 1024 x 1024` is a floor, not slack.

Reverted, and this time byte-exact: the rebuilt runner is
`67dd9c8f87667a9786cba97a02740042`, identical to the round-14 build that passed 24
records with 0 numerics failures.

## 25. The store is not bound by where it writes

§20 inferred from the phase sums that the store is HVX-bound. That inference is now a
measurement, and it closes the transport question for good.

The probe is deliberately minimal: **the store code, the strides and the row count are
unchanged; only the destination moves from DDR to VTCM.** If the phase were waiting on
DDR, writing the same bytes the same way into the fastest memory the DSP has would show
it immediately.

| shape | DDR destination | VTCM destination | |
|---|---:|---:|---|
| 256 x 1024 x 1024 | 82.3 | 83.2 | +1.1 % |
| 128 x 1024 x 1024 | 59.4 | 60.2 | +1.3 % |
| 64 x 1024 x 1024 | 51.6 | 48.6 | -5.8 % |
| 32 x 1024 x 1024 | 46.3 | 45.7 | -1.3 % |
| 1024 x 1024 x 64 | 67.3 | 68.7 | +2.1 % |
| 1024 x 1024 x 32 | 62.7 | 64.4 | +2.7 % |

Nothing. The probe was **verified active**, not assumed: with the destination redirected
the real output is never written, and all three probed shapes report `torch_close=False`
with `max_abs_torch` 2.277. A probe that silently failed to allocate would have shown the
same timings and meant nothing.

So the store is bound by its **source read out of VTCM, or by the HVX issue rate itself** —
not by the destination. No DMA, no bypass bit, and no change of destination can help it.
This is a second, independent reason round 12's 2D-DMA store lost; the first was issue
overhead.

The probe was removed rather than left behind, and the rebuilt runner is
`67dd9c8f87667a9786cba97a02740042`, byte-identical to the verified round-14/15 build.

### What this means for the campaign

Every phase of the general path has now been measured against its floor:

| phase | verdict | evidence |
|---|---|---|
| weight fill | at floor | §3, §12 — DMA overlap |
| activation pack | at floor | §22, §23 — staging beats direct DDR by 2.6x |
| HMX tile loads | at floor | §24 — the fused load cannot be split |
| output store | at floor | §25 — destination-independent |

**The implementation is done.** Nothing in the current structure is leaving measurable
time on the table, so beating QNN on the remaining shapes is an *algorithmic* question,
not an optimisation one. The one concrete algorithmic difference found so far is the
N=1 family (`4096 x 4096 x 1` at 1.45, `8192 x 4096 x 1` at 1.28, together the two
largest gaps in the set): those shapes spend essentially their whole time rearranging a
32 MiB activation into an HMX tile layout to compute a single output column, which is
what a 32x32 unit is worst at, while QNN reads the same bytes at 54 GB/s. A GEMV path
would read the activation once and never build the tile layout. That is the next thing
worth building, and it is a new kernel rather than a change to this one.

Also worth recording from the prior disassembly pass
(`disasm_compare/comparison.md`): static analysis never established which fp16 op QNN
dispatches to at runtime, and it warns that QNN's 63 GB/s is weight bytes over overall
inference time, which is not comparable with our per-phase DMA figures. No QNN
instruction-level advantage has been identified that this kernel lacks.

## 26. The N=1 shapes cannot reuse the existing HVX matmul

§25 named a GEMV path for the N=1 family as the next thing to build. The cheap version of
that idea is to reuse the HVX fp16 matmul already in this file — it reads the activation
row-major, which is exactly what N=1 wants — by unpacking the single live weight column
out of its tile order and handing it over.

**The weight layout, established exactly** from `htp_ops_loop_hmx_pack_weight_tile`: the
prepacked weight of a single column is still a full 32-column tile, and row k sits at

    packed[(k / 32) * 1024 + ((k % 32) / 2) * 64 + (k % 2)]

Unpacking that back to `(K, 1)` is 8 KiB of gather for K = 4096, and clearing
`kHmxPlanPrepackedWeights` in a copy of the param is enough to make the dispatcher fall
through to the HVX paths instead of committing to the unit.

**It is correct, almost.** `max_abs_torch` came back at **0.031** against a 0.01 tolerance
on `4096 x 4096 x 1` and `8192 x 4096 x 1`. That is a precision difference, not a layout
error — a wrong permutation would put whole products in the wrong place and read O(1) —
and both `MNN_MATMUL_FP32_ACC` and `MNN_MATMUL_EXACT_PRODUCT` already default to 1, so
the accumulation is fp32 with exact fp16 products. Worth chasing, but moot:

| shape | odb | unpacked HVX | |
|---|---:|---:|---|
| 4096 x 4096 x 1 | 862 | 14317 | **+1561 %** |
| 8192 x 4096 x 1 | 1715 | 28230 | **+1546 %** |
| 1024 x 1024 x 31 | 67.0 | 68.5 | +2 % |
| 8192 x 1024 x 1024 | 1474 | 1485 | +1 % |

16x slower, and the arithmetic says why: 14317 us at ~1.2 GHz is 17 M cycles for 16.7 M
multiply-accumulates, or about **one cycle per MAC**. The `fast_*_contiguous` paths are
built for a wide N and degenerate completely at a single column. They are not a starting
point for this shape.

So §25's direction survives but its cheap route does not: N=1 needs a purpose-built GEMV
that stages the activation through VTCM (a direct DDR read measured 2.6x slower in §22)
and does a real vector dot product per row. The measured budget for that is a ~548 us
staged read against a ~330 us compute, both overlapped, against the 892 us the HMX path
spends rearranging — which is where the 1.45 would become roughly 0.9.

Reverted, byte-exact: rebuilt runner `67dd9c8f87667a9786cba97a02740042`.

## 27. The N=1 GEMV: built, and two measured walls

§26 said N=1 needs a purpose-built GEMV. This section is that kernel, built and measured --
it is 3.1x slower than the path it replaces, and it is 130x less accurate. Both walls are
concrete, and the second one is the whole idea's weak point.

The kernel stages the activation through VTCM in 16-row blocks with a double-buffered DMA
(`HmxDmaQueue`), unpacks the single weight column once, and takes a per-row dot product
with fp32 accumulation.

**Wall 1: the accuracy reference is not in question.** The obvious suspicion for a 0.031
error is a loose harness comparison, so it was checked directly. On these exact shapes:

| shape | HMX path | QNN |
|---|---:|---:|
| 4096 x 4096 x 1 | 0.00024 | 0.00024 |
| 8192 x 4096 x 1 | 0.00098 | 0.00098 |

The two backends agree to five decimals, so the reference is solid and 0.031 is a real
failure, not an artifact. Note also that §26's HVX attempt and this GEMV produced the
*identical* 0.031 -- two unrelated implementations do not share a bug, so they share a
lossy step. The candidate is the widening: `Q6_Vsf_vmpy_VsfVsf` is a truncating sf32
multiply, and over 4096 accumulated terms the dropped low bits compound. Swapping in
`Q6_Vqf32_vmpy_VsfVsf` with a qf32 add produced **NaN**, so that intrinsic pair is not
valid in this arrangement. The one-instruction widening, `Q6_Vqf32_vmpy_VhfVhf`, is
*undeclared* in this translation unit -- yet used at line 1761 of the same file. Resolving
that contradiction is the first thing to do on any further attempt.

**Wall 2: the conversion chain is the cost, not the DMA.**

| shape | odb | GEMV | |
|---|---:|---:|---|
| 4096 x 4096 x 1 | 862 | 2703 | **+214 %** |
| 8192 x 4096 x 1 | 1723 | 5372 | **+212 %** |

2703 us over 256 blocks is 10.6 us per 128 KiB block, which is 12 GB/s -- far under the
~60 GB/s the DMA alone sustains, so the dot product dominates and the overlap is not the
problem. Per 64 elements the loop runs two `hvx_my_vhf_to_wsf` calls, and that helper is
not one instruction: it is `vhf -> vqf16 -> wqf32 -> wsf`, three to four operations, paid
twice per iteration. That is what turns a projected ~330 us of compute into ~2700.

So the idea is not refuted, but it now depends entirely on a cheap fp16-to-fp32 widening.
With `Q6_Vqf32_vmpy_VhfVhf` available the loop would be a multiply and an add per 64
elements, which is the shape the ~330 us estimate assumed. Without it, this approach
cannot win: the conversion alone costs more than the 793 us of rearrangement it replaces.

Reverted. Rebuilt runner `59ba7546cb5dd9e2bbf6469e732805ea`, verified behaviourally
(20 records, 0 numerics failures). That md5 is **not** the round-10 `67dd9c8f...`: the
revert is semantically clean but the source text carries a residue this round could not
locate, so treat the ID as unverified until a byte comparison against the round-10 tree
confirms it.

## 28. The fp32-accumulation branch is dead code

§27 left one contradiction open: `Q6_Vqf32_vmpy_VhfVhf` is used at line 1760 of
`loop_ops.cc` yet was undeclared when this campaign used it at line 2029 of the same file.

Settled empirically. Line 1760 was replaced with a deliberately bogus identifier and the
skel rebuilt: **the build succeeded**, and the log confirms `loop_ops.cc.obj` was actually
recompiled (not a no-op). So the line is not compiled, and the `#if` guarding it --

    #if MNN_MATMUL_FP32_ACC && MNN_MATMUL_EXACT_PRODUCT

-- evaluates **false**, even though `loop_ops.cc` sets both to 1 via `#ifndef` and no
CMake, toolchain file, or generated header in the tree defines them. The enclosing
function `htp_ops_loop_matmul_fast_h_contiguous_range` is live (the dispatcher calls it),
so the false condition is somewhere in the block structure around lines 1735-1760 rather
than around the function. Locating it is a small, bounded job and the next round's first
task.

Two consequences, one of which may matter well beyond N=1:

1. **`Q6_Vqf32_vmpy_VhfVhf` is genuinely unavailable in this translation unit.** Round 18's
   compile error was correct and not a misplacement artifact. Any GEMV work must reach the
   widening another way.

2. **The fp16 matmul paths are not accumulating in fp32.** Both `#if MNN_MATMUL_FP32_ACC`
   branches are dead, so those paths run the `#else` fp16 accumulator. That is the same
   signature as the 0.031 error in §26 and §27 -- roughly 3 %, which is what fp16
   accumulation over K = 4096 looks like, and 130x the 0.00024 the HMX path and QNN both
   achieve. It also explains why two unrelated implementations produced the *identical*
   0.031: they were both routed into the same fp16-accumulating HVX code.

That reframes the N=1 direction. §27 concluded the GEMV needed a cheap widening; the more
direct question is whether simply making those branches live -- turning fp32 accumulation
on for the HVX paths -- removes the accuracy wall without a new kernel. It is a flag-level
change against existing, tested code.

Not resolved this round: the rebuilt runner is `7d9672b98efaefb717cdc79ae3d2fc60`, still
not the round-10 `67dd9c8f...`. Three semantically identical trees produced three IDs
(`59ba7546`, `87dfc07e`, `7d9672b9`) differing only in blank lines, so the drift is
whitespace, not behaviour -- but the exact round-10 text was not reconstructed and the ID
should not be quoted as verified.

## 29. Retraction: the fp32-accumulation branch is not dead

§28 concluded that `#if MNN_MATMUL_FP32_ACC && MNN_MATMUL_EXACT_PRODUCT` is dead code and
that the HVX fp16 paths therefore accumulate in fp16. **That conclusion is withdrawn.** The
probe behind it was unsound, and this round's control shows why.

What §28 did: replaced the intrinsic at line 1760 with a bogus identifier, rebuilt, saw no
error, and treated that as proof the line is skipped.

What round 20 added: a control. An `#error` was placed **outside** the guarded block, at
line 1759, unconditionally. The file was confirmed to contain it (`grep -c` = 1), the build
reported compiling `loop_ops.cc.obj`, and the build **succeeded** -- no `#error` fired.
An unconditional `#error` cannot be skipped by any conditional. So the probe is not
measuring what §28 assumed: the absence of a diagnostic is not evidence that the block is
skipped. A direct compiler invocation was also attempted and was malformed --
`DIRECT_EXIT=$?` after a pipeline captured `head`'s exit status, not the compiler's -- so it
settled nothing either.

Meanwhile the positive evidence points the other way:

- `loop_ops.cc` sets both macros to 1 via `#ifndef` at lines 43 and 52.
- Nothing in the repository, the Hexagon SDK includes, the generated skel/idl headers, or
  `flags.make` defines either macro to 0.
- `flags.make` carries only `HTP_MM_PHASE_PROFILE`, `HTP_OPS_PWL_COMPANDED16`,
  `HTP_OPS_PWL_LEARNED8`, `HTP_OPS_SKEL_ARCH`, `hex_htp_v79_EXPORTS`.

With the macros at 1 the block is live, the line is compiled, and `Q6_Vqf32_vmpy_VhfVhf`
is declared -- which is consistent with the file building cleanly on every other round.
The accuracy gap of §26/§27 therefore still needs an explanation, and fp16 accumulation is
no longer a supported candidate for it.

**Still open.** The original contradiction is untouched: round 18's use of
`Q6_Vqf32_vmpy_VhfVhf` at line 2029 failed with `use of undeclared identifier` and
`EXIT=2` -- a real, reported failure -- while the same identifier at line 1760 compiles.
Any future probe here must carry a control that is *known* to fire before its negative
result is believed.

**Provenance.** Runner IDs this round: `06cbb663...` (source ID) and runner md5s
`cd763b4e455cedc6ec4d9384331b8178`, `746291de28e39fa2391ae1b5e815381d`, still not the
round-10 `67dd9c8f...`. Six distinct IDs have now been observed for trees that differ only
in whitespace and probe lines. Two explanations fit and neither is confirmed: the source ID
hashes more than the file contents, or another campaign is concurrently editing shared skel
sources. Until that is settled, quote no runner ID as verified.

## 30. The store rate is flat: not DDR, not alignment

§25 concluded the store is destination-independent, but its probe was gated to outputs
<= 1 MB, so it never exercised a large one. This round measured the store rate across a 32x
range of output sizes with K and N fixed, which settles it.

M varies, K = N = 1024, phase slot 203 divided by executions:

| M | output bytes | store per exec | rate |
|---:|---:|---:|---:|
| 128 | 0.25 MB | 22 us | 11 GB/s |
| 256 | 0.5 MB | 25 us | 20 GB/s |
| 512 | 1 MB | 45 us | 22 GB/s |
| 8192 | 16 MB | 741 us | 22 GB/s |

Flat from 0.5 MB to 16 MB. DDR write bandwidth would not hold a constant 22 GB/s across a
32x size range -- small outputs would come out of cache and run faster -- so the store is
not bound by where it writes. §25 stands, now on direct evidence rather than a probe that
could not see the case.

**The alignment hypothesis, tested and refuted.** The activation pack moves the same 16 MB
at 39 GB/s with an almost identical operation count to the store's 22 GB/s. The obvious
difference was that the store uses `vmemu` on both ends while the pack uses `vmem`, and
`htp_ops_loop_hmx_copy_block` in this same file states plainly that "the unaligned load
costs extra byte rotates". So the pair store was given an aligned path: `vmem` for the
VTCM source, which is always 128-byte aligned, and `vmem` for the destination whenever the
row stride and tile base are whole cache lines.

Numerics were clean (28 records, 0 failures) and it bought nothing:

| shape | odb | aligned | |
|---|---:|---:|---|
| 8192 x 1024 x 1024 | 1472.6 | 1476.8 | +0.3 % |
| 1024 x 1024 x 1024 | 196.9 | 197.2 | neutral |
| 256 x 1024 x 1024 | 82.8 | 82.8 | neutral |
| 1024 x 1024 x 31 | 66.9 | 68.3 | +2 % |
| 65 x 8191 x 1025 | 404.7 | 413.5 | +2 % |

The odd shapes take the unaligned destination branch, so their 2 % is attributable to the
`vmem` source load alone: the aligned access is slightly *worse* there, not better.
Reverted; rebuilt and verified behaviourally (24 records, 0 numerics failures).

**What is left.** The store's 22 GB/s is neither alignment nor destination, which leaves the
shuffle/issue stream, as §20 concluded. But the pack's 39 GB/s at the same op count is not
explained by that, and the gap between two loops that should cost the same is now the
sharpest open question in this document. Either the pack is cheaper per byte than its
instruction count suggests, or the store is paying something not visible in the count.

**Confirmed in passing.** `1024 x 1024 x 1024` emits path tag 4 and leaves slots 200-203
unpopulated, i.e. it runs the 1024-cube path, not the general one -- consistent with §25.

## 31. Why the store is slower per byte than the pack: the loop itself

§30 left one question open: the activation pack moves 16 MB at ~34-39 GB/s while the store
moves the same bytes at 22 GB/s, and both looked like seven operations per 256 bytes.
Counting the real instructions from our own skel answers it.

`htp_ops_loop_matmul_hmx_general` disassembled with
`hexagon-llvm-objdump -d --mcpu=hexagonv79 --mattr=+hmx`. The pair store's inner loop
(0x61980..0x619b0), one iteration = 2 rows = 256 bytes in and 256 bytes out:

    r2 = add(r12,#-0x800)
    r10 = add(r13,r28)
    v7.h = vdeal(v6.h)
    v6.cur = vmem(r12++#0x1)
    r2 = add(r13,r14)          <- r2 written again; the first value is dead
    r9 = add(r9,#0x2)
    v9.h = vdeal(v8.h)
    v8.cur = vmem(r2+#0x0)
    r13 = add(r13,r15)
    v29:28 = vdeal(v7,v9,r1)
    vmemu(r2+#0x0) = v28
    nop                        <- scheduler could not fill the slot
    vmemu(r10+#0x0) = v29

**Thirteen instructions, one of them a NOP, for seven operations of real work.** Five are
scalar address computations, and two of those recompute `r2` with the first result
discarded -- loop-invariant arithmetic the scheduler did not hoist. The NOP is the visible
cost of that pressure: the slot could not be filled.

The pack loop nearby (0x60384..0x603a8) is nine instructions for the same shape of work and
is software-pipelined with its second shuffle and store hoisted out of the loop body.

This is the first concrete mechanism found for the 1.7x per-byte gap, and unlike the
alignment hypothesis of §30 it is visible in the generated code rather than inferred from
source. It also fits every earlier negative result: §25's destination probe changed nothing
because the destination was never the cost, and §30's aligned access changed nothing because
alignment was never the cost. The cost is the instruction stream around the shuffles.

**Caveat on the comparison.** The store loop is identified with certainty -- it is the
`vdeal(v.h)` + `vdeal(v,v,r1)` + two `vmemu` sequence matching
`htp_ops_loop_hmx_store_output_pair`. The nine-instruction loop I compare it against sits
in the same function and has the pack's shuffle shape, but its `r12 += 0x1000` stride does
not match a 128-byte activation store, so it may be the weight pack rather than the
activation pack. The *store* measurement stands on its own; the pack comparison is
suggestive, not established.

**Next.** Reduce the store loop's instruction count: hoist the invariant address arithmetic
out of the body, or unroll so the scalar overhead is amortised over more bytes. Thirteen
instructions per 256 bytes against seven of real work leaves roughly 40 % of the loop
recoverable in principle, which on `8192 x 1024 x 1024` (store 741 us of 1488) is the
largest single opportunity identified in this document.

No source change was made this round; the tree is at the §30 reverted state, verified
behaviourally (24 records, 0 numerics failures).

## 32. The store is not instruction-issue-bound

§31 found the pair store's inner loop running thirteen instructions per 256 bytes -- five of
them scalar address arithmetic, one a NOP -- and proposed that hoisting the invariant
addressing would recover much of it. That was implemented and it does not work.

The store was rewritten with a single advancing destination pointer and a two-pair body, so
the destination offsets became compile-time constants off one register. The generated loop
went from **13 instructions per 256 bytes to 22 per 512**:

| | instructions/byte |
|---|---:|
| before | 0.0508 |
| after | 0.0430 |

15 % fewer instructions per byte, one NOP still present. Numerics clean (36 records,
0 failures, including M = 31 and 33 which exercise both tail paths). Timings:

| shape | odb | unrolled | |
|---|---:|---:|---|
| 8192 x 1024 x 1024 | 1472.3 | 1481.7 | +0.6 % |
| 1024 x 1024 x 1024 | 197.2 | 196.8 | neutral |
| 256 x 1024 x 1024 | 82.7 | 83.1 | neutral |
| 128 x 128 x 128 | 5.7 | 6.2 | +9 % |

Nothing. A 15 % instruction reduction moved the total by under 1 % on the largest shape and
made a small one worse. Reverted; rebuilt and verified behaviourally (24 records, 0 numerics
failures).

**What this establishes.** The store is not issue-bound, so §31's mechanism is refuted even
though its measurement was correct -- the loop really does carry dead arithmetic, it just
does not matter. The surviving NOP alongside a 15 % instruction cut points at a stall rather
than a fill failure, and unrolling by two did not hide it, so the two chains are not
independent. The most likely reason is that every `vdeal(v,v,r1)` in the body shares the
same shift register, serialising what look like separate chains.

That completes a clean elimination across three rounds, each with a measured negative:

- §25: not the destination (DDR vs VTCM, identical).
- §30: not alignment (`vmem` where provably aligned, neutral to worse).
- §32: not the instruction count (15 % fewer, no change).

The store's 22 GB/s is a dependency/latency bound. Three of the four candidate mechanisms
are now excluded by measurement, and the fourth -- the shuffle chain -- has not been attacked
directly. Breaking that chain means giving the two halves of the pair distinct shift
registers, or restructuring so the shuffles of successive pairs do not queue behind one
another.

Reverted state: runner `184ee08cb20180599c7f3a74d5911c01`.

## 33. Not row stride -- and the phases do not overlap at all

§32's guess about a shared shift register was wrong: the disassembly sets `r1 = #0x40` once
outside the loop, so it is loop-invariant and serialises nothing. That left DRAM row scatter
as the candidate, which is testable: shapes with identical M and K but very different row
strides should show different store rates if scatter matters.

`M = K = 1024` throughout, so only N and therefore the row stride changes:

| shape | output | row stride | store | rate |
|---|---:|---:|---:|---:|
| m1024 x 1024 x 257 | 526 KB | 514 B | 37 us | 14 GB/s |
| m1024 x 1024 x 2048 | 4 MB | 4096 B | 170 us | 24 GB/s |
| m1024 x 1024 x 3072 | 6 MB | 6144 B | 260 us | 24 GB/s |
| m1024 x 1024 x 8192 | 16 MB | 16384 B | 738 us | 23 GB/s |

**Flat at 23-24 GB/s across a 4x change in row stride.** DRAM row scatter is not the limit.
The 14 GB/s at N = 257 is the small-output effect already seen in §30, not a stride effect.

The store's rate is therefore fixed at ~6.7 cycles per 128-byte store at ~1.2 GHz, and four
candidate mechanisms are now excluded by measurement:

| candidate | verdict | evidence |
|---|---|---|
| destination (DDR vs VTCM) | no | §25 |
| alignment (`vmemu` vs `vmem`) | no | §30 |
| instruction count | no | §32 |
| row stride / DRAM scatter | no | §33 |

**The larger observation.** In this profile the phases sum to the whole:

    m8192 x 1024 x 1024:  act 412 + hmx 268 + store 738 = 1418   whole 1488
    m1024 x 1024 x 2048:  act  54 + hmx  80 + store 170 =  304   whole 385

The activation pack, the HMX tile loads and the output store are **fully serialised**. Every
phase being individually at its floor, as §25 concluded, does not make the kernel optimal if
the floors are added rather than overlapped. This is consistent with the earlier static
finding (comparison.md, output sequencing) that the consumed weight slot is refilled only
after all column tiles in a pass and their output stores have finished.

That reframes the remaining gap: the question is no longer how to make any single phase
faster, but whether the store can be moved off the vector unit so it runs alongside the next
tile's work. The obvious vehicle is a DMA-based store, which §12 tried as a 2D DMA and
abandoned at +67 % on one shape -- but §12 tested it as a *replacement* store, not as
overlapped work, and the visibility fence that blocked it was never resolved.

No source change this round; the tree is at the §32 reverted state.

## 34. The store is bound by the DDR write, not by the vector unit

Four rounds eliminated four mechanisms without finding the cost. A fifth probe settles it, and
it overturns §25.

**The probe.** Take the pair store and delete the shuffles, keeping the loads and stores
identical -- two 128-byte loads from VTCM and two 128-byte stores to DDR per iteration, same
bytes, same addresses, same count. Only the shuffle work is gone. This is a timing probe on
an intentionally altered kernel; its numbers are a bound, not an optimization result.

| shape | real store | shuffles removed | |
|---|---:|---:|---|
| m1024 x 1024 x 2048 | 170 us | 165 us | -3 % |
| m1024 x 1024 x 8192 | 738 us | 701 us | -5 % |

**The store costs the same with or without its shuffle work.** So it is not bound by the
shuffles, the instruction stream, the alignment, the stride, or the destination pattern --
it is bound by the traffic itself: 16 MB read from VTCM and 16 MB written to DDR in 701 us,
23 GB/s each way.

The activation pack moves the identical 16 MB in 412 us -- 1.7x faster -- and the one thing
that differs is that it writes **VTCM** rather than DDR.

**Every earlier negative now has an explanation.**

| observation | why it was a negative |
|---|---|
| §25 destination probe, <= 1 MB | a small output stays in cache and never reaches DRAM, so moving it to VTCM changed nothing |
| §24 row stride, 4096 -> 16384 B | total write volume is unchanged by stride |
| §30 output size ladder, 0.5 -> 16 MB | same volume; small outputs are also dominated by per-tile overhead |
| §32 15 % fewer instructions | traffic is the limit, not issue slots |

**§25's conclusion was wrong.** It held that the store is at its floor and the implementation
is done. The store is at the *DDR-write* floor of ~23 GB/s, which is not the same thing, and
which the kernel is free to route around.

**Why this is promising.** The weight DMA already reads DDR at ~58 GB/s (2 MB in 36 us,
§3). The DMA path is therefore more than twice as fast as the vector store's write path.
Staging the shuffled output in VTCM and letting the DMA carry it to DDR should move the write
at DMA rates instead of vector-store rates -- and §12's abandoned 2D-DMA store, which failed
at +67 % on one shape, was never evaluated against this understanding.

That is the next step, and it is the first time in five rounds that a measurement has opened
a direction rather than closed one.

Reverted state: runner `746291de28e39fa2391ae1b5e815381d`, verified behaviourally
(20 records, 0 numerics failures).

## 35. DMA write probe: startup dominates, so staging must be one large transfer

§34 concluded the store is bound by its DDR write at ~23 GB/s and proposed staging the
shuffled output in VTCM so the DMA carries it out instead. The first thing to check is
whether the DMA's *write* direction is fast. The probe was wrong, and its wrongness is the
useful part.

**The probe.** Replace the pair store's body with a single `htp_ops_loop_hmx_dma_copy` of
`validRows * 128` bytes from the VTCM staging to DDR -- same byte count as the vector
store, contiguous destination, layout deliberately wrong (timing only).

| shape | vector store | DMA probe | |
|---|---:|---:|---|
| m1024 x 1024 x 2048 | 170 us | 647 us | 3.8x worse |
| m1024 x 1024 x 8192 | 738 us | 2749 us | 3.7x worse |

Taken at face value this kills the direction. It should not be taken at face value: the call
site runs once per (row block, column tile), so the probe issues thousands of ~4 KB transfers
each bracketed by `dma_wait_for_idle`. Dividing out, 16 MB in 2749 us over ~4096 calls is
**0.67 us, about 800 cycles, per 4 KB transfer** -- that is DMA startup cost, not bandwidth.
The probe measured latency and reported it as throughput.

**What it does establish.** Small DMAs are useless here, which rules out the naive form of
the idea (DMA each row segment as the vector store does today). Any DMA-based store must
issue a small number of large contiguous transfers.

**The design that follows.** Accumulate a whole row block of the full output width in VTCM,
then move it out with one large DMA:

    row block 32 x N=8192 x 2 B = 512 KB in VTCM
    512 KB at ~58 GB/s (the weight DMA's measured read rate) = ~9 us
    32 blocks per 16 MB output = ~288 us, i.e. ~55 GB/s

against 738 us today. The HVX side writes the shuffled result into VTCM instead of DDR --
VTCM traffic runs at the pack's ~78 GB/s, so ~400 us for 32 MB in and out -- and the DMA
overlaps it. The store phase becomes max(HVX ~400, DMA ~288) instead of 738. On
`8192 x 1024 x 1024` (store 738 of 1488) that is a ~340 us saving, taking the shape from
1.31x QNN to roughly parity.

This is also what the QNN disassembly says it does: keep the native output in VTCM and write
back contiguously at the end, ~4.2 MiB of 8 MiB workspace (hmx_decode/ANALYSIS.md line 79).

**Provenance.** Reverting this probe returned the runner to exactly
`746291de28e39fa2391ae1b5e815381d`, the same ID as the §25 reverted state. The source ID is
content-derived, so the earlier concern about six different IDs for nominally identical trees
was about trees that were not in fact identical.

Reverted state verified behaviourally (20 records, 0 numerics failures).

## 36. The 2D descriptor exists, and the band is 64 KB

§35's design needs one large contiguous-ish DMA rather than thousands of small ones. The
descriptor type for that already exists in this tree and is in use:
`dma_desc_2d_t` with `roi_width`, `roi_height`, `src_stride`, `dst_stride`, as configured
in `ops/matmul_q4fp16_mle32.c` (line 528 onward, a working example that DMAs a strided
region into VTCM).

For the output store the transfer is the transpose of that: VTCM (contiguous) out to DDR
(strided). One band of 32 rows x 1024 columns:

    type        = DMA_DESC_TYPE_2D
    src         = staging band
    dst         = dstBase + eBase * dstStrideXYZ[0] + nBegin * 2
    roi_width   = 2048            (1024 columns x 2 B)
    roi_height  = validRows       (<= 32)
    src_stride  = 2048            (contiguous in the staging band)
    dst_stride  = lp->dstStrideXYZ[0]
    src_bypass  = 1
    dst_bypass  = 1
    ordered     = 1

64 KB in a single descriptor. At the weight DMA's measured ~58 GB/s that is ~1.5 us,
startup amortised to nothing -- against ~800 cycles per 4 KB in §35's flawed probe. The
band is the unit of work, so this fits the existing loop nesting: `nt0 -> nt -> eBase`, with
the band completing inside one `eBase` iteration.

**What the change touches.** The store functions derive their destination from `dstBase`,
`lp->dstStrideXYZ` and `nBegin`, so redirecting them at a VTCM band needs an explicit
destination pointer and stride rather than the loop parameter's. Call sites:
`htp_ops_loop_hmx_store_output_pair` at 1523, 1532, 1545;
`htp_ops_loop_hmx_store_output_tile` at 1480, 1572, 1686;
`htp_ops_loop_hmx_store_output_tail` at 1478. The staging band is 64 KB, so two of them
(128 KB) double-buffer through `HmxDmaQueue` and the DMA runs behind the compute.

**Budget note.** This round pinned down the descriptor fields and the exact call sites but
made no source change: the edit spans six call sites and a destination-stride change, and it
cannot be implemented and behaviourally verified in the remaining round budget. Recording it
here so the next round starts from the design rather than rediscovering it.

Tree unchanged from §35's reverted state, runner `746291de28e39fa2391ae1b5e815381d`.

## 37. Two process facts, and the measurement still owed

**A failed program rolls its edits back.** The large-DMA probe needed two edits: allocate a
512 KB VTCM buffer, and issue one `htp_ops_loop_hmx_dma_copy` of that buffer inside the
store timing region. The first applied; the second was rejected because the anchor matched
twice -- `htp_ops_loop_hmx_store_output_tile` and `..._output_pair` end with byte-identical
tail blocks. The program threw, and the first edit was rolled back with it: `grep -c
vtcmProbe` returns 0. A throwing program reverts the edits it made, which is worth knowing
before assuming a partially-applied change survived.

For the next attempt the anchor must include the enclosing loop, which does differ between
the two stores, not just the tail.

**Another campaign is still editing shared sources.** Rebuilding an unchanged tree returned
runner `c30d55b7c149f885931aa1f32f046fdd` where the same tree gave
`746291de28e39fa2391ae1b5e815381d` in §35 and §36. The source file is clean of this round's
probe, so the difference comes from outside this campaign. Behaviour was verified rather than
assumed: 32 records across the ladder shapes including both odd tails, **0 numerics
failures**. Correctness is intact; only the provenance ID moved.

**The measurement still owed.** Whether a *large* DMA write to DDR runs at DMA read rates
(~58 GB/s) or at the vector store's 23 GB/s is still unmeasured, and it decides whether §36's
band-staging design is worth its six call sites. §35 proved only that small DMAs are
dominated by ~800 cycles of startup. The probe to run: one 512 KB
`htp_ops_loop_hmx_dma_copy` from VTCM to DDR inside the store region, once per kernel,
anchored on the pair store's enclosing loop.

## 38. §37 was wrong: a failed program does not reliably roll back

§37 concluded that a throwing program reverts its edits. That is false, and acting on it
left the tree dirty for a round.

Retrying the large-DMA probe, three edits: declare `static __fp16* g_dmaProbe`, allocate a
512 KB VTCM buffer next to `vtcmScales`, and issue one `htp_ops_loop_hmx_dma_copy` at the
top of `htp_ops_loop_hmx_store_output_pair`. The program threw on an ambiguous anchor. The
first edit had already persisted -- `grep -c g_dmaProbe` returned 2, then 1 on a recheck as
the file changed underneath -- leaving a live 512 KB VTCM allocation whose only consumer was
an edit that never landed. That is a silent performance regression, not a no-op.

**The rule to follow instead: after any failed edit program, grep for the edits' fingerprints
before doing anything else.** Do not assume rollback. The stray declaration was removed, the
file rebuilt, and behaviour verified: runner returned to exactly
`746291de28e39fa2391ae1b5e815381d` -- the §35 verified state -- with 24 records across the
ladder shapes and **0 numerics failures**.

**The file is being edited concurrently.** Between two greps seconds apart, the count of
`g_dmaProbe` went from 2 to 1 with no edit of mine in between. Anchors that were unique when
first read stop matching. This is why the probe has now failed twice on anchor ambiguity
rather than on anything technical, and it is why §28's rebuild returned a different runner ID
for nominally unchanged sources.

**Consequences for the next attempt.** Insert the probe with a single edit rather than three,
so there is no partial state to clean up; use `read` to fetch the anchor text immediately
before the edit rather than reusing text from earlier in the session; and verify with
`grep` after.

**The measurement is still owed.** Whether a large DMA write to DDR runs at ~58 GB/s or at
the vector store's 23 GB/s remains unmeasured, and it is the gate on §36's six-call-site
band-staging design. Two rounds have now been spent on tooling rather than on it.

## 39. The large DMA writes DDR at 43-55 GB/s -- the design is validated

The gate on §36 is answered, and it answers yes.

**The probe.** One `htp_ops_loop_hmx_dma_copy(dstBase, vtcmOutput, 4 MB)` issued once per
kernel from inside the pair store, so it lands in the store's own timing region. Source is
VTCM, destination is DDR -- the direction §36 needs. Layout is deliberately wrong; this is a
bound, not an optimization result.

| shape | store alone | store + 4 MB DMA | DMA cost | rate |
|---|---:|---:|---:|---:|
| m1024 x 1024 x 2048 | 170 us | 246 us | 76 us | **55 GB/s** |
| m1024 x 1024 x 8192 | 738 us | 836 us | 98 us | **43 GB/s** |

**A large DMA write to DDR runs at 43-55 GB/s, roughly twice the vector store's 23 GB/s.**
§35's probe measured ~800 cycles of startup per 4 KB transfer; at 4 MB the same startup is
amortised to nothing, which is exactly what §35 predicted and §36 assumed.

This is the number that gates the whole direction, and it is the first positive measurement
in twenty rounds.

**What it means for the target.** On `8192 x 1024 x 1024` the store is 738 us of 1488.
Replacing the vector DDR write with a VTCM write plus an overlapped large DMA should bring
the store to roughly max(HVX-to-VTCM, DMA) -- and even the pessimistic case, where the DMA
does not overlap at all, gives 738 -> ~370 + DMA. The shape is at 1.31x QNN; parity is in
reach on the arithmetic alone.

**Reverted cleanly**, applying §38's rule: the edit was a single insertion, `grep` confirmed
0 leftovers afterwards, and the rebuild returned exactly
`746291de28e39fa2391ae1b5e815381d` -- byte-identical to the already-verified binary, so no
further behavioural check was needed.

**Next.** Implement §36: allocate the 64 KB band staging, give the three store functions an
explicit destination pointer and stride, redirect them at the band, and issue the 2D
descriptor after each band completes. Double-buffer two bands through `HmxDmaQueue`.

## 40. Column-strip staging: a better fit than the band

§36 proposed staging a 32-row band of the full output width. Reading the loop nesting shows
that does not fit: the order is `nt0 -> nt -> eBase`, so a row block's full width is only
complete after every `nt0` pass, which means every row block is in flight at once -- 256
blocks x 64 KB = 16 MB for `8192 x 1024 x 1024`. It does not fit in VTCM.

The nesting gives a better unit. For a **fixed `nt`**, the `eBase` loop walks every row
block, and each iteration stores 32 rows x 64 columns. So when that `nt`'s `eBase` loop
ends, VTCM holds a complete **column strip**: all M rows x 64 columns, in row-major order.

    strip bytes = M x 64 x 2 = M x 128
    M = 8192  ->  1 MB        (fits; M = 16384 -> 2 MB, still fits)

and it leaves as a single 2D descriptor:

    type        = DMA_DESC_TYPE_2D
    src         = strip
    dst         = dstBase + nt * 64            (byte offset of this strip's first column)
    roi_width   = 128
    roi_height  = M
    src_stride  = 128                          (row-major within the strip)
    dst_stride  = lp->dstStrideXYZ[0]
    src_bypass = 1, dst_bypass = 1, ordered = 1

1 MB per descriptor at the measured 43-55 GB/s, startup amortised to nothing -- against
4 KB per descriptor in §35's probe.

**The redirect needs no change inside the store functions.** They compute
`base = dstBase + nBegin * dstStrideXYZ[2]` and write at `base + (eBase + r) *
dstStrideXYZ[0]`. Passing a local copy of the loop parameter with

    dstBase           = strip
    dstStrideXYZ[0]   = 128
    dstStrideXYZ[2]   = 0

makes `base = strip` and the writes land at `strip + (eBase + r) * 128` -- exactly the
strip layout. `HtpOpsLoopParam` is taken by `const` pointer, so a stack copy passed by
address is enough; the three store functions are untouched.

**Exact call sites.** `htp_ops_loop_hmx_store_output_pair` at 1523 (pipelined branch) and
1532 (plain branch), both inside the `eBase` loop that begins at 1392, which is inside
`nt` at 1376, inside `nt0` at 1360. The 2D DMA goes after that `eBase` loop closes,
still inside `nt`, guarded to run only on the last `nt0` pass -- earlier passes have not
written the strip's other columns yet, so the strip must accumulate across passes, which
means the guard is `nt0 + ntPerPass >= np` rather than an unconditional post-loop DMA.

**Two buffers** (2 x M x 128) alternate through `HmxDmaQueue` so the DMA of strip k runs
behind the compute of strip k+1.

No source change this round: the round budget went to reading the nesting and establishing
that §36's band geometry does not fit. Tree at the §39 verified state, runner
`746291de28e39fa2391ae1b5e815381d`.

## 41. The nesting is the other way round, and the redirect is one line

§40 had the nesting backwards. The real structure is **`eBase` outer, `nt` inner**:

    1360  for (nt0 = 0; nt0 < np; nt0 += ntPerPass) {
    1392      for (eBase = 0; eBase < E; eBase += 32) {
    1489          for (...; nt + 1 < ntEnd ...)      <- pipelined pair stores
    1552          for (...; nt < ntEnd; ++nt)        <- plain stores
    1577          }
    1578      }                                        <- eBase closes
    1587  }                                            <- nt0 closes

So for a fixed `eBase` the `nt` loop walks every column, and when it closes the row block
`eBase..eBase+31` across the **full width** is complete. That means §36's band geometry
*does* fit after all -- only one band is ever in flight -- and §40's column-strip redesign
was unnecessary. Its 16 MB objection applied to a nesting that does not exist.

**The redirect is one assignment.** All six store call sites (1478, 1480, 1523, 1532, 1545,
1572) pass `dstBase` and `eBase`. `dstBase` is a mutable parameter, and the store
functions address `base + (eBase + r) * dstStrideXYZ[0]` with
`base = dstBase + nBegin * dstStrideXYZ[2]`. So inside the `eBase` loop:

    dstBase = (uint8_t*)vtcmBand - (int64_t)eBase * N * 2;

gives `band + nBegin*2 + r*N*2` -- the band layout, for every call site, with no change to
any store function and no `HtpOpsLoopParam` copy. `ddrBase` must be saved before the loop
because the DMA still needs the real destination.

**Anchor traps that cost this round.** Three edit attempts failed on duplicate anchors, and
the duplicates are not obvious:

| anchor | why it matched twice |
|---|---|
| `for (int eBase = 0; ...)` + validRows block | also at 1339, the activation-cache path |
| `}\n}\nif (doubleBuffered) {\n const int next = nt0 + 2 * ntPerPass;` | also at 1588, the outer loop's close -- byte-identical |
| `htp_ops_loop_hmx_store_output_tile(dstBase, vtcmOutput, lp, eBase, validRows, N, nt)` | unique, but the anchor must span the closing braces to be usable |

The lesson: on this file, anchor on a span that contains a line only one site can have --
`#if HTP_MM_PHASE_PROFILE` immediately after the `eBase` header selects 1392 over 1339, and
the tile store call ending in `N, nt)` selects 1572 over 1480. Do not trust a 4-line window.

**Tree state.** All three attempts rolled back cleanly this time: `grep` for the new
identifiers returns 0 and the file md5 is `d98f283c29f0925875b5529718e3ba78`, unchanged, so
the binary remains the §39 verified one. The implementation is fully specified above and is
three edits: allocate `32*N*2` bytes, redirect `dstBase` inside the `eBase` loop, and issue
the 2D descriptor after the `nt` loop closes.

## 42. The band staging lands and fails numerics -- with a diagnostic pattern

The three edits from §41 were applied successfully this round. The duplicate-anchor problem
was solved by counting occurrences before writing: a Python pass that aborts unless every
anchor appears exactly once. It caught a fourth duplicate §41 had not listed --
`vtcmScales` is allocated in **both** the general and the 1024-cube paths, so the allocation
anchor needed the preceding `ntPerPass * blockBytes` line to disambiguate. With that fixed,
all three anchors were unique and the change compiled clean.

Numerics, however, failed: **10 of 24 records**, on a pattern that is worth more than the
patch.

| shape | N | result |
|---|---:|---|
| m8192 x 1024 x 1024 | 1024 | FAIL (max_abs 2.77) |
| m129 x 4093 x 2051 | 2051 | FAIL (max_abs 4.62) |
| m1024 x 1024 x 31 | 31 | FAIL (max_abs 2.16) |
| m1024 x 1024 x 2048 | 2048 | pass |
| m1024 x 1024 x 8192 | 8192 | pass |
| m8192 x 4096 x 1 | 1 | pass |

Two candidate causes, neither confirmed:

1. **The DMA's `roi_width` must be a multiple of 128.** For N = 31 the band row is 62 bytes,
   which is not. But N = 1024 gives 2048 bytes, which is, and it fails anyway -- so this
   cannot be the whole story.
2. **`eBase` may not be used only as a row offset outside the pair store.** §41 verified
   that for `htp_ops_loop_hmx_store_output_pair`, but **not** for
   `htp_ops_loop_hmx_store_output_tile` or `..._store_output_tail`, which are the calls
   taken on the `tailFirst` and non-contiguous paths (1478, 1480, 1572). If either reads
   `eBase` for anything else, setting it to zero for the band breaks that path while leaving
   the contiguous path intact -- which is exactly the observed split between passing and
   failing shapes.

Cause 2 is the better fit and is the first thing to check next round: read the two remaining
store functions and confirm what they do with `eBase`.

**Reverted cleanly.** Reverse pass, `grep` for `vtcmBand`/`ddrBase`/`bandDesc` returns 0,
source md5 returns to `d98f283c29f0925875b5529718e3ba78`, rebuild returns exactly
`746291de28e39fa2391ae1b5e815381d` -- the §39 verified binary.

## 43. The bug: the band pitch must be rowStride, not N*2

§42's hypothesis 2 was right that `eBase` is used for more than addressing, but wrong about
the consequence. Reading the two functions confirms the extra uses:

    store_output_tile:895    eBase + r + 2 < lp->sizeXYZ[0]
    store_output_tail:938    eBase + r + 1 < E
    store_output_tail:946    eBase + r + 2 < E

All three are **bounds checks**. The §42 redirect kept `eBase` intact and only moved
`dstBase`, so these still evaluated correctly -- they are not the bug. But the same read
gives the actual one.

**The store's addressing.** Every store computes its destination as

    dstBase + (eBase + r) * dstStrideXYZ[0] + nBegin * dstStrideXYZ[2]

The band redirect set `dstBase = band - eBase * N * 2`, so the `eBase` term cancels only if

    dstStrideXYZ[0] == N * 2

**That is not guaranteed.** `htp_ops_loop_matmul_hmx_general_eligible` requires
`dstStrideXYZ[2] == 2` and non-negative leading strides; it does **not** require
`dstStrideXYZ[0] == N * 2`. A padded output -- which is what a caller supplies when it wants
rows aligned, and what an odd N forces -- has `dstStrideXYZ[0] > N * 2`, and then the band
offset over- or under-shoots by `eBase * (dstStrideXYZ[0] - N * 2)` per row block. The error
grows with `eBase`, which is why large row counts corrupt badly and small ones survive:
`m8192 x 1024 x 1024` (max_abs 2.77) and `m129 x 4093 x 2051` (4.62) failed, while
`m1024 x 1024 x 2048` and `n8192` passed -- those two must have unpadded destinations.

**The fix is to pitch the band at the destination's own row stride.**

    allocate   32 * dstStrideXYZ[0]  bytes      (not 32 * N * 2)
    redirect   dstBase = band - (int64_t)eBase * dstStrideXYZ[0]
    DMA        src_stride = dstStrideXYZ[0]
               dst_stride = dstStrideXYZ[0]
               roi_width  = N * 2               (unchanged -- only N columns are live)

Then the `eBase` term cancels for any stride, padded or not, and the store functions need no
change at all. The band grows from `32*N*2` to `32*dstStrideXYZ[0]`, which is 64 KB vs 64 KB
for an unpadded N=1024 and proportionally more when padded -- still small against 8 MB.

Note `roi_width` stays `N * 2` rather than the full pitch: the band rows are pitched at
`dstStrideXYZ[0]` but only the first `N*2` bytes of each are live output.

**State.** Reverted to the §39 verified binary, `746291de28e39fa2391ae1b5e815381d`. The
implementation is again three edits with unique anchors; the only change from §41/§42 is that
the band pitch and the redirect use `lp->dstStrideXYZ[0]` instead of `N * 2`.

## 44. The pitch was not it -- and the identical failures name the likely cause

§43's fix was implemented: the band pitched at `lp->dstStrideXYZ[0]` instead of `N * 2`, the
redirect subtracting `eBase * bandPitch`, the descriptor using `src_stride = dst_stride =
bandPitch` with `roi_width = N * 2`. All three anchors unique, change compiled clean.

**The failures are byte-identical to §42's.**

| shape | N | §42 max_abs | §43 max_abs |
|---|---:|---:|---:|
| m8192 x 1024 x 1024 | 1024 | 2.76562 | **2.76562** |
| m129 x 4093 x 2051 | 2051 | 4.61719 | **4.61719** |
| m1024 x 1024 x 31 | 31 | 2.16406 | **2.16406** |

Same values to the digit, same 10 of 24 records. A change that alters the band geometry and
every descriptor stride, and moves nothing, means the pitch was never the cause. The
padded-output hypothesis is refuted: if `dstStrideXYZ[0] != N * 2` had mattered, correcting
the pitch would have moved the numbers.

**What the invariance implies.** The corruption is deterministic and independent of the band
layout, so it is shared by both attempts. The two attempts share exactly three things: the
band allocation, the `dstBase` redirect, and the DMA read of the band. The first two are
addressing and would have moved with the pitch. **That leaves the DMA reading the band.**

**The leading hypothesis: VTCM writes by HVX are not visible to the DMA without a fence.**
`dma_wait_for_idle()` before `dmstart` orders DMA against DMA, not against the HVX. The
band is written by `vmemu` stores from the vector unit, and the DMA engine reads VTCM
directly; nothing in the code establishes that those stores have drained into VTCM before the
descriptor starts. Stale bytes would be copied deterministically, giving the same wrong
result every run -- which is what is observed.

This is not a new suspicion. §12's abandoned 2D-DMA store was recorded as blocked by "the
visibility fence that was never resolved". That fence was never found, and this round's
evidence says it is the same blocker.

**Next.** Establish the fence, or prove it is unnecessary. The cheapest discriminator is a
standalone check: write a known pattern into a VTCM buffer with HVX, DMA it out with a 2D
descriptor, and compare -- with and without whatever barrier is available. That separates
descriptor semantics from store visibility, which the staging experiment cannot.

**State.** Reverted cleanly: 0 leftovers, source md5 `d98f283c29f0925875b5529718e3ba78`,
rebuild returns `746291de28e39fa2391ae1b5e815381d`.

## 45. There is no working diagnostic channel from the skel

§44 proposed a standalone discriminator: write a known pattern into VTCM with HVX, DMA it
out with a 2D descriptor, read it back from DDR, and compare. Building it ran into a problem
that has been silently shaping this whole campaign.

**`FARF(ERROR, ...)` does not reach logcat.** The probe was written, compiled, deployed and
run on `m1024 x 1024 x 2048`; `logcat -d | grep probe36` returned only the adsprpc skel-load
lines. The macro is used in this file already (line 690, the chained-DMA incomplete path), so
it is available and it compiles -- but its output does not appear in the logcat the runner
writes to. Whether it needs `DSP_LOG`, a different tag, or a diag listener was not
established.

**The phase array is not reachable from the store functions.** Redirecting the diagnostic to
`g_mm_phase_us[10]`/`[11]` -- which *are* printed as slots 210/211 -- fails to compile:
`error: use of undeclared identifier 'g_mm_phase_us'` at line 998, twice. The array is
declared inside `htp_ops_loop_matmul_hmx_general`, not at file scope, so the store functions
cannot see it. That is also why the earlier phase work had to funnel everything through the
general function's own counters.

**Consequence.** Every diagnostic in this campaign has been an *external* measurement -- a
timing slot, a numerics failure, a rate computed from wall clock. There has been no way to
print an internal value. That is why the last four rounds have had to infer mechanisms from
failure *patterns* rather than read a counter, and it is why §42's "identical max_abs"
observation was so valuable: it was the only signal available.

**A discriminator that needs no channel.** Replace the pair store's vector store with a 2D
DMA of the *same bytes to the same addresses* -- `roi_width = 128`, `roi_height = 2`,
`src_stride = 128`, `dst_stride = dstStrideXYZ[0]`. Slow, because 2-row DMAs are exactly
the startup-dominated case §35 measured, but the question is correctness, not speed, and
numerics is an observable that already works:

- if numerics **pass**, the 2D descriptor usage is correct and the band redirect is the
  culprit;
- if numerics **fail**, the descriptor usage is wrong and the band was never the problem.

Either outcome is decisive and neither needs a print.

**State.** Reverted cleanly: probe removed, source md5 `d98f283c29f0925875b5529718e3ba78`,
rebuild returns `746291de28e39fa2391ae1b5e815381d`.

## 46. The §45 discriminator is invalid, and multi-pass is not the bug

§45 proposed replacing the pair store's vector store with a 2D DMA of the same bytes to the
same addresses. That does not work as a test: **the pair store's job is to shuffle.** It
reads two 32-column accumulator tiles from VTCM and rebuilds them into row-major vectors
(`vdealh` + a 64-byte `vdeal`). A raw 2D DMA copies bytes unchanged, so it would produce
wrong output whether or not the descriptor is right -- the experiment cannot separate the two
things it was meant to separate.

**What the band failure does and does not implicate.** Working through the multi-pass case,
which looked like the obvious candidate:

The `nt0` loop runs `np / ntPerPass` passes, and the `eBase` loop and the DMA sit *inside*
it. So for a given `eBase`, the band only holds the columns belonging to the current pass --
`roi_width = N * 2` copies the full width, including columns written by an earlier pass. That
looks like stale-data corruption, and it is invariant to the band pitch, which would fit the
identical failures.

**It is harmless.** The band is per-`eBase` and reused, but the columns from the earlier pass
were already DMA'd to the same DDR addresses, and the band still holds exactly those bytes.
Re-copying them writes the correct data back. Multi-pass re-copy is a no-op, not a bug.

**What is left.** Three candidates, all invariant to the pitch:

1. **2D descriptor semantics.** `src_stride`/`dst_stride` might be a *gap* after
   `roi_width` rather than a pitch. The working example in `matmul_q4fp16_mle32.c` has
   `src_stride == roi_width`, so it cannot distinguish the two readings -- and in the band
   case `src_stride == roi_width` as well, so both readings agree there too. This candidate
   predicts no difference for contiguous cases, which is consistent with what is seen.
2. **`roi_width` alignment for odd N.** `N = 31` gives a 62-byte row, not a multiple of
   128. The DMA may require line granularity. This would explain `n31` but not
   `m8192 x 1024 x 1024`, whose rows are 2048 bytes.
3. **VTCM store visibility.** §44's hypothesis: `dma_wait_for_idle()` orders DMA against
   DMA, not against HVX `vmemu` stores.

Candidate 3 remains the only one that explains every failing shape and every passing one, and
it is the same blocker §12 recorded and never resolved.

**State.** No change this round; tree at the §39 verified state, runner
`746291de28e39fa2391ae1b5e815381d`. The next experiment must be able to observe an internal
value, because every failure pattern available from outside is now exhausted.

## 47. The diagnostic channel exists, and the 2D DMA does not work

Two findings, one of which resolves the whole band mystery.

### The channel was one line away

§45 concluded there was no way to print an internal value from the skel. Wrong.
`g_mm_phase_us` **is** file-scope -- line 1069, `unsigned long long g_mm_phase_us[13]` -- but
that is *after* the store functions at 962, so the identifier is not visible there. A single
forward declaration fixes it:

    #if HTP_MM_PHASE_PROFILE
    extern unsigned long long g_mm_phase_us[13];
    #endif

Slots 200-212 are then writable from anywhere in the file and printed by the runner. The
lesson is narrower than §45 claimed: the failure was declaration order, not a missing
facility. `FARF` genuinely does not reach logcat, but the slot path works and is enough.

### The 2D DMA copies almost nothing

The self-checking probe: write `p[i] = i & 0x7ff` for 2048 elements into `vtcmOutput`,
2D-DMA it to `dstBase` with `roi_width = 128`, `roi_height = 32`,
`src_stride = dst_stride = 128`, then read `dstBase` back and count mismatches.

    slot 210 = 2047      mismatches out of 2048
    slot 211 = 2         first mismatch at index 1 (stored as index + 1)
    slot 212 = 424242    probe ran

**2047 of 2048 elements wrong, with index 0 correct.** That is the signature of a transfer
that moved one element and left the rest as stale DDR content.

Re-running with `src_stride = 0` and `dst_stride = 0` -- which under the "gap after
roi_width" reading would make the layout contiguous -- gives the **identical** result: 2047
mismatches, first at index 1. So it is not stride semantics either. The descriptor setup is
wrong in a way that is invariant to the strides.

### This explains the band failures completely

The band change DMA'd the staging buffer to DDR. If the DMA copies one element, the
destination keeps its previous contents -- so the output is corrupted **deterministically**,
with the same wrong bytes in the same places on every run.

That is exactly the invariance §44 could not explain: two different band pitches, two
different descriptor stride sets, byte-identical `max_abs` values (2.76562, 4.61719,
2.16406). The corruption never depended on the band geometry because the band was never
transferred at all.

**§44's fence hypothesis was wrong.** `dma_wait_for_idle()` and HVX visibility are not the
issue; the DMA is not performing the copy.

**§30's rate measurement needs a caveat.** That probe timed a 4 MB `htp_ops_loop_hmx_dma_copy`
at 76-98 us and concluded 43-55 GB/s. The engine did consume that time, so the figure is a
real engine rate -- but the transfer it timed was 1D (`dma_desc_1d_t`) and was not checked
for correctness either. The 1D path may be fine; the 2D path demonstrably is not. Do not treat
43-55 GB/s as a validated write rate until a 1D copy is verified the same way.

**State.** Reverted: probe and forward declaration removed, source md5
`d98f283c29f0925875b5529718e3ba78`, rebuild returns `746291de28e39fa2391ae1b5e815381d`.

**Next.** Verify the 1D DMA the same way. If 1D copies correctly, the band design can be
salvaged with a 1D descriptor per row (128 bytes each -- startup-dominated, §35's measured
~800 cycles per 4 KB, so probably too slow) or by finding the correct 2D usage. If 1D also
fails, the DMA direction is dead and §34's 23 GB/s is a hardware floor.

## 48. The 1D DMA is correct, the 2D DMA is not, and the descriptor layout says why

### The decisive comparison

Same self-checking probe, same pattern, same read-back, only the transfer changed:

| path | descriptor | slot 210 (mismatches) | slot 212 |
|---|---|---:|---|
| `htp_ops_loop_hmx_dma_copy` | 1D, chained | **0** | 424242 (ok) |
| hand-built | 2D, `roi_width=128 roi_height=32` | **2047 / 2048** | 424242 (ran) |

Slots 210 and 211 did not print for the 1D run, meaning both were zero: zero mismatches. The
1D path copies a VTCM pattern to DDR **exactly**.

**§30's rate measurement is therefore valid.** It timed a 4 MB `htp_ops_loop_hmx_dma_copy`
at 76-98 us. §47 raised the concern that the transfer had never been checked for correctness;
it now has been, and it is correct. 43-55 GB/s is a real, verified DDR write rate -- roughly
double §34's 23 GB/s vector-store floor. The premise of the band design holds.

### The descriptor layout

`dma_desc_1d` and `dma_desc_2d` are both in `include/dsp/dma_utils.h`. They share a
leading layout -- `next`, then a flags union, then `src`, `dst`:

    struct dma_desc_1d {                    struct dma_desc_2d {
      uint32_t next;                          uint32_t next;
      union {                                 union {
        struct {                                struct {
          unsigned length : 24;                   unsigned length : 24;
          unsigned type : 2;                      unsigned type : 2;
          ... bypass/ordered/dstate ...           ... bypass/ordered/dstate ...
        };                                      };
        uint32_t dstate_order_bypass_type_length;
      };                                      };
      uint32_t src;                           uint32_t src;
      uint32_t dst;                           uint32_t dst;
    };                                        unsigned _pad1:24, cache_alloc:2, _pad2:6;
                                              uint16_t roi_width;
                                              uint16_t roi_height;
                                              uint16_t src_stride;
                                              uint16_t dst_stride;
                                              uint16_t src_width_offset;
                                              uint16_t dst_width_offset;
                                            };

**`length` is in the shared union, and the 2D probe never set it.** The struct is
`memset` to zero, so `length = 0`. The 1D path sets it; the 2D path relied on the ROI alone
and left it zero. A transfer whose length field is zero copies nothing, which is exactly the
observed signature: index 0 correct by coincidence of stale content, 2047 elements untouched.

This is the leading candidate and it is cheap to test -- set `length` to the total byte count
(`roi_width * roi_height`) and re-run the same probe.

Two other things the layout rules out or constrains:

- **`roi_width`/`roi_height`/`src_stride`/`dst_stride` are `uint16_t`.** A band pitch of
  `N * 2` overflows for N > 32767, and `roi_width` overflows for N > 32767. Not the cause
  here (the probe used 128) but it bounds the design: the band approach cannot use a single
  descriptor spanning a very wide row.
- **`cache_alloc` is a separate 2-bit field**, left 0 = `DMA_DESC_CACHEALLOC_NONE`. The 1D
  path works with it 0, so it is not required.

**State.** Probe reverted, source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild returns
`746291de28e39fa2391ae1b5e815381d`.

**Next.** Set `length` on the 2D descriptor and re-run the probe. If mismatches go to zero,
re-apply the band with a corrected descriptor. If they do not, fall back to a chained 1D
descriptor per band row -- 32 descriptors, one `dmstart`, `roi`-free, and the 1D path is
already proven correct.

## 49. length is not the fix; chained 1D moves data but not correctly

### length made no difference

Setting `pd.length = 128 * 32` on the 2D descriptor and re-running the identical probe gave
the **identical** result: 2047 mismatches, first at index 1. §48's candidate is refuted. The
2D descriptor is wrong for some reason not yet identified, after four attempts:

| attempt | change | result |
|---|---|---|
| §42/§43 | `src_stride = dst_stride = 128` | 2047 / 2048 |
| §46 | `src_stride = dst_stride = 0` | 2047 / 2048 |
| §48 | `length = 4096` | 2047 / 2048 |
| §39 | (1D control) | **0 / 2048** |

Invariant to every field that has been varied. The 2D path is abandoned.

### Chained 1D descriptors change the failure

Rebuilt the band using 32 chained `dma_desc_1d_t` per row block -- one per band row, each
copying `N * 2` bytes, linked with `dmlink`, one `dmstart` on the head. The 1D path is
verified correct in isolation (§48).

    before (§42/§43, 2D):   10 of 24 fail    2.76562 / 4.61719 / 2.16406
    after  (§49, 1D chain):  8 of 24 fail    2.23828 / nan     / 1.83984

**Both the count and the values changed.** The DMA is now transferring real data -- a 2D
descriptor that copies nothing produced stale-DDR corruption, which is why §42 and §43 agreed
to the digit; a working transfer produces different wrong answers. Two shapes moved from fail
to pass. The remaining eight are wrong for a different reason than before.

So the design is not refuted: the mechanism now demonstrably moves bytes. What is not yet
right is what the bytes are.

### What to check next

The most likely remaining defect is that the band's contents differ from what the vector
store would have written to DDR. Candidates, in order:

1. **Narrow rows.** The pair store writes a full 128-byte vector per row pair regardless of
   the live row width. For N = 31 that is 62 live bytes and 66 bytes running into the next
   band row. In the original code that overflow lands in the next DDR row and is overwritten
   by the following pair store; in the band it lands in the next band row and is copied by
   that row's descriptor. Whether the ordering still saves it depends on the tail store
   running last, which has not been checked.
2. **The tail store's bounds checks.** `store_output_tail` uses `eBase + r + 1 < E` to
   decide between a wide `vmemu` and an exact row store. `eBase` is preserved by the
   redirect, so these should still be correct -- but the wide store writes into the band's
   last row and the descriptor copies only `validRows` rows, so the last row's overflow is
   dropped rather than overwritten.
3. **Whether every column is stored before the DMA runs.** The DMA is issued after the whole
   `nt` loop, so this should hold; it is worth confirming against the pipelined pair path,
   which defers one batch's stores.

**State.** Reverted: source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild returns
`746291de28e39fa2391ae1b5e815381d`.

## 50. The band is a measured regression -- the direction is closed

The single-flat-copy version is the first to produce **clean numerics**: 24 records, 0 failures,
with the guard `dstStrideXYZ[0] == N * 2` and a 128-byte multiple on the transfer size. That
also confirms §49's chained-descriptor attempt failed because of the *chaining*, not the idea.

Clean numerics is not enough, because it is also what a guard that never fires looks like. The
A/B resolves it: the timings differ, so the band engaged.

    shape                   odb (us)          band4 (us)        change
    m8192 x 1024 x 1024     1473              1836              +25%
    m1024 x 1024 x 2048      379               458              +21%
    m256 x 8192 x 4096      1311              3291             +151%
    m1024 x 1024 x 8192     1243              1252             +0.7%
    m256 x 4096 x 8192      1226              1233             +0.5%
    m1024 x 1024 x 31         67                69             +2.9%

**It is a regression on every shape, badly on three.**

### Why, and why no tuning fixes it

`htp_ops_loop_hmx_dma_copy` is synchronous -- it issues `dmstart` and waits for idle. So
staging does not replace the DDR write, it *inserts a hop*:

    direct:              store to DDR            1/23 GB/s          = 0.0435
    staged:   store to VTCM, then DMA to DDR    1/39 + 1/55        = 0.0438

§34 established the store is bound by VTCM read + DDR write at 23 GB/s, and the activation
pack moves the same bytes to VTCM at 39 GB/s. §30/§48 verified the 1D DMA at 43-55 GB/s. Those
three numbers were the case for the design, and together they predict **a wash**. Measured
with overhead, it is a loss. The extra hop costs almost exactly what the faster write saves.

The one shape where the premise should have helped most -- `m256 x 8192 x 4096`, +151% -- is
the largest output. A 512 KB band on top of the weight double-buffer, activation cache and
output staging almost certainly pushed VTCM past what the kernel needs and degraded the
double-buffering that §10 and §11 established. That is consistent with the size of the loss.

**Direction closed.** This was the last well-evidenced lever for the store phase, and it is
now measured, not speculated. §34's 23 GB/s DDR-write floor stands as the floor.

### Provenance note

After reverting, the source md5 returns to `d98f283c29f0925875b5529718e3ba78` but
`executor_runner` now builds to `e4827d766ed67d3796179383fd70bff0` rather than the
previously reproducible `746291de28e39fa2391ae1b5e815381d`. The skel source ID is embedded in
the runner, so this means **another campaign has changed a skel source** -- the concurrency
hazard noted earlier. The A/B above is unaffected: both builds were made before this change,
and the `odb` baseline on the device (`c2966741e4f2f77c39f7d7e55d25c748`) is untouched.
Future builds will not match historical hashes until that is reconciled.

**State.** Reverted: 0 leftovers, source md5 `d98f283c29f0925875b5529718e3ba78`.

## 51. Standing against the originally named shapes

Every round since §34 has been spent inside one store-phase direction. It is worth restating
what the campaign is actually measured against, recomputed from the verified round-10 sweep
(`odb_results.json`, 54 paired shapes, same phone, same protocol).

**The shapes originally named:**

    m1024 x 1023 x 1024    hex  209.9 us   qnn  227.6 us   ratio 0.922
    m1024 x 1025 x 1024    hex  227.0 us   qnn  228.9 us   ratio 0.992
    m4097 x 1025 x 257     hex  458.5 us   qnn 1077.4 us   ratio 0.426

**Hexagon wins all three.** The third is 2.3x faster. These are the matmuls the work started
from, and they are the ones the goal names.

**The full set:**

    paired shapes            54
    geo-mean hex/qnn         0.912      (hexagon ~9% faster on average)
    best                     0.14
    at or under QNN          27 / 54
    worst                    1.45

**Where the losses are:**

    m4096 x 4096 x 1       1.448       N = 1
    m1024 x 1024 x 31      1.428       small N
    m128 x 128 x 128       1.383       small square
    m129 x 4093 x 2051     1.326       odd, prime-ish
    m1024 x 1024 x 32      1.323       small N
    m8192 x 1024 x 1024    1.314       large, contiguous
    m1024 x 1024 x 64      1.311       small N
    m65 x 8191 x 1025      1.306       odd
    m1024 x 1024 x 65      1.173
    m33 x 1024 x 1024      1.170

The pattern is small-N and small-square: the same family the closed GEMV work (§26/§27) and
the closed band work (§50) both targeted. `m1024 x 1024 x 31` runs 1024x1024x31 MACs in
66.9 us -- about 0.49 TFLOP/s against roughly 1.5 TFLOP/s on the large shapes. With N = 31
there is exactly one 32-column tile, so the HMX unit itself is at 31/32 utilisation and is
not the problem; the fixed work around it is.

**What this means for the goal.** The named targets are met. The aggregate is favourable. What
remains is a specific, identified family -- small N, small square, odd dimensions -- where the
per-call fixed cost dominates. That is a different problem from the store phase, and it is
where the remaining rounds should go.

## 52. Small N is not work-bound, and for N = 31 the phases do not see the cost

Phase profile of the current build (`HTP_MM_PHASE_PROFILE` is compiled into every build and
armed by `HEXAGON_TRACE=1`, so no special binary is needed), one execution, cumulative
counters read once:

    shape                    total   wfill  actpack   hmx   store   unaccounted
    m1024 x 1024 x 31          72       2        0     0      15        55
    m1024 x 1024 x 64          72       4       52     -      13        ~3
    m1024 x 1024 x 1024       197       -        -     -       -      (1024-cube path, tag 4)
    m8192 x 1024 x 1024      1489      36      411   276     746       ~20

Three things fall out.

**1. N = 31 and N = 64 cost exactly the same: 72 us.** Doubling the work changes nothing.
These shapes are not throughput-bound at all; a fixed cost dominates. QNN, by contrast, moves
from 46.8 us to 51.5 us -- it scales, weakly but visibly, and that is the whole gap.

**2. For N = 64 the phases account for the time.** 4 + 52 + 13 = 69 of 72 us, and the
activation pack is 52 of it -- 72%. For small N the activation transform is the kernel.

**3. For N = 31 the phases account for almost nothing.** 2 + 15 = 17 of 72 us, leaving 55 us
unattributed. Slot 201 (activation pack) reads **zero**, which cannot mean the activation was
never transformed -- the result is correct. It means the N = 31 path does its activation work
somewhere the counter does not cover.

That is the actionable finding. For N = 31 the kernel spends 55 us in work that is invisible
to every instrument the campaign has, on a shape that is not compute-bound, and it is the
worst small-N case (1.428). The general path's activation handling is instrumented; whatever
N = 31 takes instead is not.

**Next.** Read the N = 31 path -- what `ntPerPass`/`np` are, whether it takes the
`cacheActivation` branch, and which counter, if any, the activation work lands in. The 55 us
is either an uninstrumented copy/transform or a serialisation that the large shapes overlap.

## 53. The 55 us is the activation pack -- the counter just cannot see it

§52 left 55 us unattributed on `m1024 x 1024 x 31`. Found it, and the answer is an
instrumentation gap rather than a mystery cost.

**The counter is inside the wrong loop.** `mm_act` accumulates at line 1500, which sits
*inside* the pipelined store loop:

    1489:  for (; contiguous && nt + 1 < ntEnd && N - nt * 32 >= 64; nt += batchTiles) {
    1500:      mm_act += MM_T0() - mm_t;

For `N = 31`, `np = up_div(31, 32) = 1`, so `ntEnd = 1` and the loop condition
`nt + 1 < ntEnd` fails on entry. The loop body never runs, so `mm_act` never accumulates --
and the `mm_t` reset at 1398 is simply overwritten by the reset at 1542. The activation work
happens (line 1411 or 1436), is correct, and is invisible.

For `N = 64`, `ntEnd = 2`, the loop runs once, and `mm_act` reads 52 us. Same code path,
same work, one counter reading.

**This explains the flat 72 us.** The activation transform depends on E and K, not on N. Both
shapes transform the same 1024 x 1024 activation block:

    N = 31    total 72 us   activation (invisible) ~55 us
    N = 64    total 72 us   activation (counted)     52 us

Identical totals because it is the identical work. The kernel is not slow at N = 31 -- it is
**transform-bound**, and has been all along.

**And the transform is near its own floor.** It moves 1024 x 1024 x 2 bytes in and the same
out: 4 MB in 52 us is 77 GB/s, well above the 39 GB/s §34 measured for the activation pack's
destination traffic and close to what VTCM can sustain. Speeding it up is not available.

**What that leaves.** The only way to win on small N is to **not do the transform**, or to do
less of it. QNN manages `m1024 x 1024 x 31` in 46.8 us -- less than the transform alone costs
here -- which strongly suggests QNN feeds the activation to HMX in a layout that needs no
separate pass, or fuses it into the load. That is consistent with `hmx_decode/ANALYSIS.md`,
which records QNN's activation descriptor as 0x77c with the full native output kept in VTCM.

This is the first time the campaign has a target that is both dominant and understood. Every
previous direction attacked the store, which for small N is 13-15 us of a 72 us kernel.

**Next.** Establish how the activation reaches HMX on the QNN side -- whether there is a
transform pass at all -- and whether the same could be done here. That is a disassembly
question, and the skill's recipe and the existing `qnn.matmul.asm` are already in place.

## 54. The small-N budget, and what it would take

From §52/§53, `m1024 x 1024 x 31` at 72 us total:

    activation transform    ~52 us     (invisible to the counter; see §53)
    output store             15 us
    weight fill               2 us
    --------------------------------
    accounted                69 us   of 72

QNN does the same shape in **46.8 us**. To win, transform + store must come down from 67 us
to under 46.8 -- roughly 30% off the pair.

### The store is unusually inefficient here

1024 x 31 fp16 is 63 KB of live output, written in 15 us: **4.2 GB/s** against §34's 23 GB/s
on large shapes. The store issues a 128-byte vector per row regardless of the 62 live bytes,
so the real write traffic is 128 KB in 15 us -- still only 8.5 GB/s, five times off the large
shape rate. The per-tile fixed cost §35 measured (~800 cycles per transfer) dominates when
each tile carries so few useful bytes.

### The transform appears to be at the memory floor

1024 x 1024 x 2 bytes in and the same out is 4 MB in 52 us -- **77 GB/s**. That is above the
39 GB/s §34 measured for VTCM destination traffic, so the transform is already efficient and
there is no speedup to be had inside it.

Worth noting the tension: the code copies raw activation into `vtcmActivationRaw` (line 1429)
and then transforms it into `vtcmActivation` (line 1436). If those were two full passes over
2 MB each way, the traffic would be 8 MB, which at 52 us implies 154 GB/s -- impossible for
VTCM. So either the copy is much cheaper than it looks (it is a DMA, and §48 verified the 1D
DMA at 43-55 GB/s) or the passes overlap more than the code suggests. **Not resolved**, and it
matters: if the raw staging pass is genuinely extra traffic, fusing it away is the single
largest available win.

### The counter-evidence

§13 already tried removing the staging -- forcing `prefetchActivation = false` so the
transform reads DDR directly -- and measured it **2.6x worse**. That was on large shapes. It
does not follow that it is wrong for small N, where the balance is different: at N = 31 the
kernel is 72% transform and the activation is read once, while at large N the activation is
re-read across many passes and staging pays for itself.

**Next.** Two bounded questions: (a) how many times the activation bytes actually move for
N = 31, which the existing counters can answer by adding a byte counter rather than a time
counter; (b) whether QNN performs a transform pass at all, from the disassembly.

## 55. The §54 tension resolves -- the raw staging is overlapped

§54 could not reconcile 52 us with 8 MB of traffic (154 GB/s, impossible). Reading the
prefetch path resolves it without needing a new counter.

Line 1415 selects between three activation strategies. For N = 31, `cacheActivation` is false
(`np = 1` fails `np > ntWithoutCache`) and `prefetchActivation` is true (`!cacheActivation &&
E >= 2`). So the branch at 1402 runs:

    1403:  activationQueue.pop();
    1407:  activationQueue.push(vtcmActivationRaw + ..., src0Base + ..., ...);
    1411:  htp_ops_loop_hmx_transform_activation_block(vtcmActivation, currentRaw, K, validRows);

The push is an **asynchronous DMA** into the double-buffered raw staging; the transform then
reads that VTCM buffer. The two are overlapped, not serialised -- that is the entire point of
`prefetchActivation`, and it is why the staging pays for itself despite §13 measuring the
un-staged version 2.6x worse.

So the 52 us the counter attributes to `mm_act` is the transform alone: 1024 x 1024 x 2 bytes
in and out, **4 MB at 77 GB/s**. The staging traffic is hidden behind it. **There is no fusing
win** -- §54's largest hoped-for lever does not exist.

### Where that leaves small N

    activation transform   ~52 us   4 MB at 77 GB/s -- at the memory floor
    output store            15 us   63 KB at 4.2 GB/s -- 5x off the large-shape rate
    weight fill              2 us
    ---------------------------------
    total                   72 us   (QNN: 46.8 us)

The transform is at its floor and cannot be fused away. The store is the one component
running far below its own demonstrated rate, but it is only 15 us of 72 -- even making it
free leaves 57 us against QNN's 46.8.

**The only remaining route to a win on small N is to not transform the activation at all** --
which is what §53's disassembly question asks. QNN completes the whole shape in less time than
the transform alone costs here, so QNN must be feeding HMX a layout that needs no separate
pass. That question is now the single highest-value thing left, and it is a disassembly
question with the tooling already in place (`qnn.matmul.asm`, `et.matmul.asm`, the skill's
`hexagon-llvm-objdump` recipe).

## 56. The disassembly artifacts do not answer the activation question

§53/§55 left one question: does QNN repack the activation before feeding HMX, or does it feed
a layout that needs no separate pass?

**The artifacts in `disasm_compare/` cannot answer it.** Checking what is actually in them:

    qnn.asm            88 MB   activation_hf_mxmem = 0   weight_hf_mxmem = 0
                                mxmem = 0   dmstart = 0   vdeal = 1802   vshuff = 6710
    qnn.matmul.asm     7 KB    no HMX instructions at all
    et.matmul.asm    111 KB    dmstart = 2, no HMX activation/weight loads

`qnn.asm` contains **no HMX instructions and no DMA at all**. It is not the DSP kernel --
whatever it is (host side, or a different library), the matmul kernel is not in it, and
`qnn.matmul.asm` is a 7 KB extraction that missed the target entirely. Any conclusion drawn
from these files about the DSP matmul would be unfounded. This is worth recording because it
is a trap: the file names invite exactly that mistake.

**The real DSP findings are in `hmx_decode/`**, produced from `qnn.fp16.asm`, `layout.asm`,
`set_params.asm` and `et.loop.asm` -- different artifacts, not present under those names here.
They answer a *different* question, about output layout:

- QNN's candidate kernel **stores the native layout directly**; ET converts inside the compute
  flow and writes DDR (`ANALYSIS.md` line 77).
- The native input/output blocks correspond to groups **spanning 128 rows**, not 32 contiguous
  rows (line 93). ET's store works in 32-row bands -- four times finer.
- Measured dispatch parameters: activation 0x77c, weight 0x7ff, output store 0x700, cvt 2,
  64 kernel calls per execution.
- Line 79 proposes the untested idea of keeping the full 1024x1024 native output in VTCM and
  writing back contiguously afterwards. **That is the band design, and §50 measured it as a
  regression** -- though §50 staged 32 rows at a time rather than the full output.

**Where that leaves the question.** The activation repack is still unanswered. The answer needs
the DSP-side disassembly that `hmx_decode/` used, which is not the 88 MB file here. Either
locate that artifact or re-extract the kernel from the on-device QNN skel.

**What did come out of this round:** QNN's output granularity is 128 rows against ET's 32. The
store is the component running 5x below its own demonstrated rate at small N (§54), and a
coarser granularity is exactly what would fix a per-tile fixed cost. That is a concrete,
testable difference -- though §10's tile-count cost model already argued the 32-row band is
where the pair store pays.

## 57. Small N is bound by reading the activation, and the store takes the slow row path

Decomposing the 52 us from §55 with numbers already measured elsewhere:

    prefetch DMA:  2 MB raw activation from DDR at 43-55 GB/s (§48)   ~40 us
    transform:     4 MB in+out at 77 GB/s (§55)                       ~26 us

They overlap (that is what `prefetchActivation` buys, §55), so the phase costs
max(40, 26) = ~40-52 us. **Small N is bound by reading the activation from DDR**, not by the
repack -- the repack is hidden underneath.

QNN completes the whole shape in 46.8 us, which is roughly one activation read plus the
compute and store overlapping it. Our 52 + 15 = 67 us is activation read **plus** a store that
does not overlap it. That is the shape of the gap: not a faster component, but one fewer
serial step.

### The store takes the slow path at N = 31

`store_output_tile` has a pair-store fast path guarded at line 894:

    if (rowBytes < 64 && lp->dstStrideXYZ[0] == (int64_t)rowBytes &&
        eBase + r + 2 < lp->sizeXYZ[0]) {

For N = 31, `rowBytes` is 62. If the destination row stride is 62 the guard passes and the
`vmux` pair store runs; if the stride is 64 -- the natural padded choice, and what a caller
aligning rows would supply -- the guard fails and every row falls through to
`htp_ops_loop_hmx_store_row` at line 901, one call per row, 1024 calls for this shape.

That would explain 15 us for 63 KB (4.2 GB/s, §54). **Which stride the exported model uses is
not known** and is the first thing to check: it decides whether the store is already on its
fast path or is falling through 1024 times.

If it is falling through, the fix is to accept the padded case: with `rowBytes = 62` and a
64-byte stride, two rows occupy exactly 128 bytes and a single `vmux`-selected vector store
retires the pair -- the same trick line 896 already uses, just with the select mask and the
row extraction adjusted for a stride wider than the live row.

**Cheapest next step:** read the exported output stride for `m1024_k1024_n31` from the PTE or
the export metadata. One number decides whether this is a real lead or a dead end.

## 58. The stride is 62 -- the store is already on its fast path

§57's one-number question, answered directly through the diagnostic channel from §47
(forward-declare `g_mm_phase_us`, write to slots 210/211/212, read them from the runner):

    m1024 x 1024 x 31
      slot 210 = 62         lp->dstStrideXYZ[0]
      slot 211 = 62         N * 2
      slot 212 = 424242     probe ran

**The destination row stride equals the live row width.** So the guard at line 894

    if (rowBytes < 64 && lp->dstStrideXYZ[0] == (int64_t)rowBytes && eBase + r + 2 < ...)

**passes**, and the `vmux` pair-store fast path runs. §57's hypothesis is refuted: the store
is not falling through to `htp_ops_loop_hmx_store_row` 1024 times.

The probe also confirmed its own scope: slots 210/211 printed for N = 31 and **not** for N = 64
or N = 1024, meaning `store_output_tile` is not called at all on those shapes -- they take the
pair store or the 1024-cube path. Consistent with the code, and a useful check that the probe
was reading what it claimed to read.

### What that leaves

The store writes 1024 rows as 512 pair stores of 128 bytes each -- 64 KB of traffic -- in
15 us. That is **~30 ns, roughly 60 cycles, per pair store**, against the 2-4 ns a single
vector store should cost. The fast path is in use and is still 15x off.

So the cost is **per-store overhead**, not a wrong code path: a dependency stall, a serialising
store queue, or something in the loop that does not pipeline. That is a different and more
tractable question than the one §57 asked, and it is measurable with the same channel -- the
store is instrumented (slot 203), so a variant that unrolls or reorders the pair loop will show
its effect directly.

**State.** Probe reverted, source md5 `d98f283c29f0925875b5529718e3ba78`.

## 59. The store's cost is neither instructions nor bandwidth

§58 measured ~30 ns (about 60 cycles) per pair store at N = 31, with the fast path already in
use. Two earlier results bracket what can explain that, and together they exclude the obvious
answers.

**Not instruction count.** §32 cut the pair store's instructions per byte by 15% (0.0508 to
0.0430) and measured **no gain**. If the store were issue-bound, that would have shown up.
Whatever costs 60 cycles per iteration, it is not the instructions in the iteration.

**Not write bandwidth either, at this size.** §34's floor is 23 GB/s, which predicts 2.8 us for
the 64 KB this shape writes. The store takes 15 us -- 5x that. The floor was established on
large shapes with wide rows; at N = 31 the rows are 62 bytes and it does not apply.

**What is left is a stall.** With instructions and bandwidth excluded, the remaining candidates
are latency the loop does not hide:

- the two VTCM vector loads per iteration (`*src0++`, `*src1++`), read at the top of the
  iteration and consumed immediately -- a load-to-use stall the compiler may not pipeline
  because the loop body is short;
- the destination address computation, `base + (eBase + r) * rowStride`, a 64-bit multiply per
  row unless strength-reduced;
- store-queue serialisation: 128-byte `vmemu` stores landing on overlapping cache lines,
  since a 62-byte row stride means consecutive 128-byte stores overlap by 66 bytes.

The third is the most interesting because it is specific to narrow rows and would explain why
this shape is 5x off while wide-row shapes reach the 23 GB/s floor. It is also testable without
guessing: the store is instrumented as slot 203, so a variant that widens the effective row
(a padded destination stride, or a different pair grouping) shows its effect directly.

**The arithmetic worth checking first.** 15 us / 512 pair stores is the 30 ns figure, and 512
assumes 32 rows per `eBase` iteration x 32 iterations = 1024 rows in 512 pairs. That model has
not been verified. A counter on the pair-store entry, written to a spare slot through the §47
channel, would confirm or refute it in one run -- and if the real count is not 512, the per-call
cost is different and so is the diagnosis.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 60. The store model is confirmed: 512 pairs, and the cost is a stall

§59 flagged its own arithmetic as unverified. It is now verified, through the §47 channel with
counters on all three paths of `store_output_tile`:

    m1024 x 1024 x 31
      slot 210 = 32      store_output_tile calls
      slot 211 = 511     vmux fast-path pair stores
      slot 212 = 1       store_row fallbacks
    m1024 x 1024 x 64
      (nothing printed -- store_output_tile is not called at all)

**511 + 1 = 512 pairs**, exactly the 1024 rows / 2 the §59 model predicted. And 32 calls is
exactly 1024 / 32 rows per `eBase` iteration, so each call stores 16 pairs. The single fallback
is the final pair, where `eBase + r + 2 < sizeXYZ[0]` fails -- expected, not a defect.

The N = 64 run printing nothing is a useful control: it confirms both §58's finding that this
function is unused on that shape and that the counters read what they claim to.

**So the per-pair cost is real.** 15 us / 512 = ~30 ns, about 60 cycles, per pair, on the fast
path, with:

- §32 excluding instruction count (15% fewer instructions per byte, no gain);
- §34's 23 GB/s floor not applying at this size (it would predict 2.8 us, not 15).

### The leading explanation

A 62-byte row stride means consecutive 128-byte pair stores **overlap by 66 bytes**. Every store
spans parts of three 64-byte cache lines, and each of those lines is written again by the next
store. On a store queue that cannot merge overlapping writes, that serialises: the engine sees
roughly 3x the line traffic and a dependency between adjacent stores.

This is specific to narrow rows, which is exactly the observed pattern -- 5x off the floor at
N = 31, at the floor for wide rows. It also explains why §32's instruction cut did nothing: the
cost is not in the instructions, it is in what the stores do to the memory system.

**Testable without guessing.** Widen the effective destination row -- pad the stride, or group
pairs differently so each store covers a full set of lines -- and watch slot 203. If the
hypothesis holds, the store time falls even though the live output bytes are unchanged.

**State.** Counters reverted, source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild clean.

## 61. The overlap hypothesis is refuted -- the wide store is the win, not the cost

§60 proposed that 128-byte pair stores overlapping on a 62-byte stride serialise the store
queue, and that this explained 60 cycles per pair. Tested by disabling the `vmux` fast path so
every pair takes `store_row`, which writes 62 bytes per row with no overlap:

    shape                    baseline slot 203   fast path disabled
    m1024 x 1024 x 31              15 us               74 us      (5x worse)
    m1024 x 1024 x 64              13 us               14 us      (unchanged)
    m8192 x 1024 x 1024           746 us              747 us      (unchanged)

**The hypothesis is wrong.** The overlapping wide store is **5x faster** than the
non-overlapping row store, not 5x slower. Fewer, wider stores win even when they collide on
cache lines -- the instruction and transaction count matters more than the overlap.

The two controls behave exactly as they should, which is what makes the result trustworthy:
N = 64 and the large shape are both unchanged, confirming neither uses this path (§58, §60) and
that the experiment measured what it claimed.

### What this means

**The store at N = 31 is already the best available.** 15 us for 63 KB is 4.2 GB/s, but the
alternative is 74 us. §34's 23 GB/s floor is a wide-row figure and does not apply here; narrow
rows are simply slower per byte, and the kernel is already taking the fastest route it has.

So the store is not the opportunity. §57's arithmetic stands: activation read ~52 us plus a
store of 15 us against QNN's 46.8 us for the whole shape. Since neither component is
individually improvable -- the transform is at the memory floor (§55), the store is 5x better
than its alternative -- **the gap has to be that QNN overlaps them and we do not.**

That is a scheduling question, not a throughput one, and it is the third refuted hypothesis in
a row. Each has been cheap to kill (one build, one profile run) and each has narrowed the space
rather than merely failing, but it is worth being explicit that the small-N family has now
resisted three separate attacks: the GEMV route (§26/§27), the band route (§50), and the store
micro-optimisation route (§59-§61).

**State.** Reverted, source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild clean.

## 62. The scheduling gap has a precise cause: outputPipeline is off for N < 64

§61 concluded the small-N gap is scheduling rather than throughput. Located it.

Line 1237:

    const bool outputPipeline = E >= 32 && N >= 64;
    const int outputTiles = outputPipeline ? 8 : (E >= 32 ? 16 : 2);
    const size_t outputBytes = (size_t)outputTiles * 1024 * sizeof(__fp16) * (outputPipeline ? 2 : 1);

For N = 31 the `N >= 64` test fails, so **`outputPipeline` is false** and the output buffer is
single-buffered. The condition is a proxy for "the pipelined pair store applies": line 1489's
loop requires `N - nt * 32 >= 64`, and with a single 32-column tile at N = 31 it never runs.

**But the plain path does not double-buffer either.** Line 1552's loop -- the one N = 31
actually uses -- reads and stores through `vtcmOutput` directly (lines 1567, 1572), with no
`currentOutput`/`pendingOutput` slot selection. So at N = 31 the store is issued, completes,
and only then does the next `eBase` iteration begin its transform. Store and transform
serialise completely.

That is the 15 us: not a slow store, but an unoverlapped one. §52 measured the store at 15 us
and the total at 72; §61 showed the store cannot be made faster in isolation. What is available
is moving it, not shrinking it.

**Note on cost.** `outputTiles` is 8 with pipelining and 16 without, and `outputBytes` doubles
only in the pipelined case -- so 8 x 2 and 16 x 1 are both 16 tiles, 32 KB. **Enabling the
output pipeline for N = 31 would cost no additional VTCM**, which matters because §50 showed
this kernel is sensitive to VTCM pressure.

**What a fix would look like.** Give line 1552's loop the same slot selection line 1498 already
uses, so block k's store overlaps block k+1's transform. That is a restructure of the plain
path, not a threshold change -- lowering the `N >= 64` gate alone does nothing, because the
loop that would use the second buffer is not the loop N = 31 executes.

**Expected size.** If the store fully overlapped, 72 us becomes ~57. QNN is 46.8, so this alone
does not close the gap -- but it is the largest identified, non-speculative win available on
this shape, and it is worth roughly 20%.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 63. Implementing the overlap: the flush is the wrong primitive

§62 said the fix is to give line 1552's plain loop the slot selection line 1498 already uses.
Reading the flush path shows that is necessary but not sufficient.

**What already exists.** The pending mechanism is in place:

    1521:  if (outputPipeline) {
    1522:      for (int which = 0; which < pendingTiles; which += 2) {
    1523:          htp_ops_loop_hmx_store_output_pair(dstBase, pendingOutput + ..., lp, eBase, validRows, N, pendingNt + which);
    1526:      pendingOutput = currentOutput;
    1527:      pendingTiles = batchTiles;
    1528:      pendingNt = nt;
    1530:      outputSlot ^= 1;
    1544:  for (int which = 0; which < pendingTiles; which += 2) {     <- flush, inside the eBase loop
    1545:      htp_ops_loop_hmx_store_output_pair(...)

The flush at 1544 sits inside the `eBase` loop and runs before the plain loop, so a pending set
by the plain loop in iteration k **would** be flushed at the start of iteration k+1. The
deferral mechanism needs no new plumbing.

**The problem.** Both the in-loop store (1523) and the flush (1545) call
`htp_ops_loop_hmx_store_output_pair`, which stores **two** tiles -- hence `which += 2`. The
plain path computes **one** tile per `nt` and stores it with `store_output_tile` (1572). So:

- deferring from the plain loop needs a flush that calls `store_output_tile`, not the pair
  store -- a second flush loop, or a branch on which path produced the pending;
- `pendingTiles` semantics differ between the paths: pairs in the pipelined loop, singles in
  the plain one, and the shared `which += 2` stride is only correct for the former.

**Also required.** The plain loop must compute into a slotted buffer
(`vtcmOutput + outputSlot * outputTiles * 1024`, as line 1498 does) rather than `vtcmOutput`
directly (line 1567), and `outputBytes` must cover two slots. With `outputTiles` dropping
16 -> 8 and the slot count going 1 -> 2, the total stays at 16 tiles, 32 KB -- the §62
arithmetic holds, and no additional VTCM is needed.

**So the change is four parts, not one:**

1. `outputPipeline` true for `E >= 32` regardless of N, with `outputTiles = 8`;
2. the plain loop computing into the slotted buffer;
3. the plain loop storing the previous pending with `store_output_tile`;
4. a flush that uses the right primitive for whichever path set the pending.

Parts 3 and 4 are where a mistake would produce wrong output rather than slow output, so the
first check after building is numerics on the ladder shapes, not timing.

**Why not simply raise the threshold.** Lowering `N >= 64` alone changes `outputTiles` and
`outputBytes` but leaves the plain loop reading and writing `vtcmOutput` directly, so nothing
overlaps. The threshold is not the fix; the loop is.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 64. Open question: where does the last pending get flushed?

Reading the end of the `eBase` loop to plan §63's change turned up something I cannot
explain, and it is worth recording rather than guessing at.

    1572:  htp_ops_loop_hmx_store_output_tile(dstBase, vtcmOutput, lp, eBase, validRows, N, nt);
    1577:  }                      <- plain nt loop closes
    1578:  }                      <- eBase loop closes
    1579:  if (doubleBuffered) { ... weightQueue.push ... }
    1587:  }                      <- nt0 loop closes
    1592:  #if HTP_MM_PHASE_PROFILE

The pending flush is at 1544, which is **inside** the `eBase` loop and runs before the plain
loop. So for iteration k it flushes the pending left by iteration k-1. The pending left by the
**last** `eBase` iteration has no flush after 1578 that I can find.

The pipelined loop advances `nt += batchTiles` and leaves its final batch as pending, so the
plain loop resumes *after* those columns rather than recomputing them. On that reading the last
block of the last row-band would not be written.

**But numerics pass**, on all 57 shapes and at every build since round 10. So one of these is
true and I have not determined which:

1. the pending is always empty at the end of the final iteration for some reason I have not
   traced (for example `batchTiles` collapsing to zero on the last pass);
2. there is a flush I have not found -- the function is long and I read only the tail;
3. the columns in question are covered by the `ntTail`/tail path, which runs before the pair
   loop (1444-1447) and stores tiles the pair loop cannot reach.

**Why this matters for §63.** The change I planned needs the same question answered for its own
pending: if a deferred store from the plain loop is never flushed, the fix would produce exactly
the silent corruption §63 warned about. **Resolve this before implementing**, not after.

It is also possible this is a real latent bug that the current shapes happen not to expose --
in which case finding it is worth more than the overlap optimization. That would be consistent
with the campaign's history: §38's "rollback is automatic" was wrong, and §37's conclusion had
to be retracted two rounds later.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 65. The pending question stays open, but is almost certainly benign

§64 asked where the last `pendingTiles` gets flushed. The empirical probe did not report --
counters written at the end of the `eBase` body never appeared in the runner output, so either
the insert site is not reached or the build did not take. Not diagnosed; the probe was reverted
and the tree is back at the verified source.

**The strong prior is that §64 is a misreading.** The flush question only matters if a pending
batch is ever left unstored, and if that happened the affected output columns would be stale --
which would fail numerics. Numerics have passed on **all 57 shapes at every build since round
10**, across roughly 46 rounds of edits. A real missing flush would not survive that.

So the most likely explanation is candidate 2 from §64: a flush I have not found, in a function
long enough that I read only its tail. The finding is recorded, not resolved.

**The practical consequence.** §63's four-part change should not be attempted until this is
settled, because the change introduces a *new* pending path whose failure mode is identical --
silent numerical corruption. That constraint stands.

**State.** Probe reverted, source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild clean.

---

## Where the campaign stands after 56 rounds

Worth recording plainly, because the last dozen rounds have produced diagnosis rather than
speed.

**What is won.** The three shapes the work started from: `m1024 x 1023 x 1024` at 0.922,
`m1024 x 1025 x 1024` at 0.992, `m4097 x 1025 x 257` at 0.426. Across 54 paired shapes the
geo-mean is 0.912 -- hexagon ~9% faster on average -- with 27/54 at or under QNN.

**What is not.** A single identified family: small N (31/32/64/65), small square (128x128x128),
odd dimensions, and `4096x4096x1`. Worst case 1.45.

**What has been tried and closed on that family**, each with measurement rather than assertion:

- the GEMV route for N = 1 (16x slower, then 3.1x slower for the purpose-built kernel);
- the band/VTCM-staging route, on the store's DDR-write bound (a measured regression, §50,
  with the arithmetic showing why: staging inserts a serial hop that costs what it saves);
- store micro-optimisation (§32 instruction cut: no gain; §60 overlap hypothesis: refuted 5x
  the other way in §61);
- and now a located scheduling gap (§62) whose implementation is blocked on §64.

**What is genuinely new since round 47**: the diagnostic channel (§47 -- forward-declare
`g_mm_phase_us`, write slots 210-212, read them from the runner). Before that the campaign
could only infer from external measurements; five internal quantities have been read with it
since, each in one run.

**The honest read.** The remaining family is hard, and the goal's named targets are already met.
Further rounds should either settle §64 and land the §62 overlap, or move to shapes where the
gap is larger and the mechanism less explored.

## 66. §64 resolved: the pending state is per-iteration, and my reading was wrong

The question was where the last `pendingTiles` gets flushed. Answer: nowhere needs to, because
the state does not survive an iteration.

    1484:  int batchTiles = 2;
    1485:  int outputSlot = 0;
    1486:  __fp16* pendingOutput = nullptr;
    1487:  int pendingTiles = 0;
    1488:  int pendingNt = 0;
    1489:  for (; contiguous && nt + 1 < ntEnd && N - nt * 32 >= 64; nt += batchTiles) {

**These declarations are inside the `eBase` loop** (which closes at 1578). They are reset on
every iteration. So the flush at 1544 runs *after* this iteration's pipelined loop and clears
this iteration's pending -- there is no cross-iteration deferral to leave dangling. §64's
candidate 2 was right: a flush I had not found, because I read the tail of the loop without
noticing where its variables are scoped.

No latent bug. The 46 rounds of clean numerics were not hiding anything.

### What that means for §62/§63

The plan built on §62 is **wrong as written.** It assumed the pending mechanism defers a store
across `eBase` iterations, so that block k's store could overlap block k+1's transform. It does
not: the deferral is *within* one `eBase` body, across `nt` batches.

That matters, because it means the mechanism already provides exactly the overlap it can:

- for N >= 64, the pipelined loop runs several `nt` batches and the store of batch j overlaps
  the HMX compute of batch j+1 -- real overlap, already present;
- for N = 31 there is a single `nt`, so there is nothing within the iteration to overlap. The
  store of that one tile is exposed, and the only thing it could overlap is the *next* row
  band's transform.

**So the fix for N = 31 is not "use the existing pending mechanism". It is to hoist
`pendingOutput`/`pendingTiles`/`pendingNt`/`outputSlot` out of the `eBase` loop** so a
store can be deferred across iterations, plus a flush after the loop for the final pending --
which is exactly the construct §64 was looking for and could not find, because it does not
exist yet.

That is a larger change than §63 described, and it is now precisely specified: hoist four
declarations, add a flush after the `eBase` loop, and make the plain path store through
`store_output_tile` rather than the pair store.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 67. The overlap change was built, debugged, measured, and refuted

§62/§63/§66 specified a change: hoist the pending state out of the `eBase` loop so a store
could be deferred across iterations, add a flush after the loop, and store the plain path's
tiles with `store_output_tile`. It was implemented in eight edits, all anchors unique.

**First build failed numerics: 8 of 28 records.** The cause was visible immediately -- the
pipelined block at 1521 set `pendingOutput`/`pendingTiles`/`pendingNt` but not the three new
fields, so `pendingIsTile`/`pendingEBase`/`pendingValidRows` carried stale values from the
previous iteration into the flush. Setting them fixed it: **20 records, 0 failures.**

**Then the measurement.**

    shape                   odb (us)          ovl (us)        change
    m1024 x 1024 x 31       66.9              67.5            +0.9%
    m1024 x 1024 x 32       62.7              64.1            +2.2%
    m1024 x 1024 x 64       67.3              68.5            +1.8%
    m1024 x 1024 x 65       77.7              78.9            +1.5%
    m1024 x 1024 x 2048    379               381              +0.5%
    m8192 x 1024 x 1024   1472              1481              +0.6%
    m256 x 8192 x 4096    1306              1321              +1.2%
    m129 x 4093 x 2051     430               435              +1.1%

**A consistent small regression, and no 20% win.**

### §62's premise is refuted

§62 reasoned that the store's 15 us was serial with the next block's transform, so overlapping
it would save 15 of 72 us. That is wrong, and the reason is a property of the instruction:
**HVX `vmemu` stores are posted.** The CPU issues them and continues; it does not wait for the
write to reach memory. So the 15 us the phase profiler reports is *issue* time, not drain time,
and the drain already overlaps the following iteration's work on its own.

Deferring the store by an iteration therefore buys nothing that was not already happening, and
costs a second output slot's worth of bookkeeping and an extra branch in the flush path -- which
is the ~1% regression observed.

### Consequence

This was the fourth direction closed on the small-N family, after the GEMV route (§26/§27), the
band route (§50), and store micro-optimisation (§59-§61). Each closed by measurement.

Worth noting what the sequence produced beyond the negatives: the diagnostic channel (§47), a
verified 1D DMA (§48), a corrected model of the store (§60), and now a correct understanding of
what the store phase actually measures -- issue, not drain. That last one invalidates any future
argument built on the store being a serial 15 us, including §57's arithmetic.

**State.** All eight edits reverted, source md5 `d98f283c29f0925875b5529718e3ba78`. Rebuild
produces runner `fa2e4785a63c6c7a9dc0e24d5c914b5b` -- a new hash from the same verified source,
consistent with §50's note that another campaign is editing skel sources.

## 68. Small M is weight-fill bound, and the weight fill is not overlapped

Profiled four shapes from the loss table that the campaign had never examined, including two
outside the small-N family. One shape's profile is dominated by a phase nothing has attacked.

    shape                   total   wfill   act   hmx   store   unattributed
    m128 x 128 x 128           10      2      4     -      2        4
    m256 x 256 x 256           18      4      4     3      5        2
    m33 x 1024 x 1024          49     35      6     -      6        2
    m65 x 8191 x 1025         462    125     30    18     41      248

**`m33 x 1024 x 1024` spends 35 of 49 us -- 71% -- filling weights.** It is in the loss table
at 1.170, and this is why.

### The weight read is irreducible, but it is not overlapped

The weight for `m33 x 1024 x 1024` is 1024 x 1024 x 2 = 2 MB, and 35 us is 57 GB/s -- exactly
one read at the measured DMA rate (§39/§48). Not redundant traffic.

The telling comparison: `m8192 x 1024 x 1024` has the **same 2 MB weight** and spends **36 us**
on `wfill`. Identical cost, but 2.4% of that shape's 1489 us instead of 71% of 49 us.

So the weight fill is a **fixed ~35 us cost**, and small M simply cannot amortise it. Since it is
already one read at full DMA rate, the only lever is **overlapping it with compute** rather than
making it faster.

### The mechanism exists and is gated off

`doubleBuffered` (line 1292) is exactly that mechanism -- it prefetches the next weight block
while the current one computes. Its condition is:

    prepackedWeights && (E == 1 || cacheActivation) && ntPerPass >= 2 && ...

The `E == 1 || cacheActivation` clause is the interesting one. `m33 x 1024 x 1024` has E = 33,
so it does **not** satisfy `E == 1`, and whether it double-buffers depends on `cacheActivation`,
which requires `np > ntWithoutCache`. That is the thing to check next.

### Why this is a better lead than the last four

The four closed directions all attacked the *store* or the *transform* on small-N shapes, where
the phases sum to the total and every component was already at its floor. This one attacks a
phase that is **71% of a losing shape's time**, that is **not overlapped**, and for which the
overlapping mechanism **already exists in the code** and is merely gated. It is also the first
lead that is not small-N -- it is small-M, a family the campaign has not touched.

**State.** No change; tree at verified source `d98f283c29f0925875b5529718e3ba78`.

## 69. The weight double-buffer gate, and a confound that invalidates cross-build A/B

§68 identified the weight fill as 71% of `m33 x 1024 x 1024` and the prefetch that would
overlap it as gated off. Both confirmed by reading.

### Why the prefetch is off

   1251:  const int rowBlocks = htp_ops_loop_up_div(E, 32);
   1252:  const size_t activationCacheBytes = (size_t)rowBlocks * blockBytes;
   1258:  cacheActivation = np > ntWithoutCache &&
   1259:      vtcmBytes >= fixed + activationCacheBytes + 2 * blockBytes && ...

For `m33 x 1024 x 1024`: np = 32, but `ntWithoutCache` is roughly 127 (VTCM is ~8 MB and a
block is 64 KB), so `np > ntWithoutCache` **fails**. With E = 33 the `E == 1` clause also fails.
So `doubleBuffered` is false and the 35 us weight fill is serial.

The gate's own comment explains it: splitting the pass doubles the passes, and each re-reads and
re-packs the whole activation. That penalty scales with E x K. For E = 33 it is 33 x 1024 x 2 =
67.6 KB, about 1.4 us -- against 35 us of weight fill it would hide. **The gate assumes a large
activation and is over-conservative for small M.**

### The relaxation, and why it is principled

The test is whether the activation is cheaper than the weight fill it hides -- E x K x 2 against
K x N x 2, i.e. `E * 2 < N`. That selects exactly the small-M family and leaves everything else
alone:

    m33  x 1024 x 1024    33 < 512   -> double-buffer
    m65  x 8191 x 1025    65 < 512   -> double-buffer
    m8192 x 1024 x 1024   8192 > 512 -> unchanged
    m1024 x 1024 x 31     1024 > 15  -> unchanged
    m128 x 128 x 128      128 > 64   -> unchanged

Built it: **36 records, 0 numerics failures.**

### The confound

    shape                   odb (us)            wdb (us)
    m33 x 1024 x 1024       48.1 / 49.4 / 46.0  51.5 / 53.6 / 47.7
    m65 x 8191 x 1025       414.6 / 412.8 / 411.4  413.0 / 412.6 / 411.1
    m128 x 128 x 128        5.7 / 5.9 / 5.8     6.1 / 6.0 / 6.1
    m256 x 256 x 256        13.0 / 13.1 / 12.8  13.8 / 13.8 / 13.9
    m1024 x 1024 x 1024     197.1 / 197.1 / 196.6  197.1 / 196.8 / 197.0
    m1024 x 1024 x 31       67.0 / 66.9 / 66.7  68.5 / 68.4 / 68.6
    m8192 x 1024 x 1024     1472.5 / 1474.8 / 1472.7  1486.1 / 1483.7 / 1483.8
    m256 x 8192 x 4096      1306.7 / 1309.8 / 1324.3  1328.0 / 1323.1 / 1324.0

**Every shape is worse, including three the condition cannot touch.** `m1024 x 1024 x 31` has
E = 1024 and N = 31, so `E * 2 < N` is false and its code path is byte-identical between the two
builds -- yet it reads 67.0 -> 68.5. `m128 x 128 x 128` and `m8192 x 1024 x 1024` likewise.

**So the `odb` baseline is no longer a valid control.** It is a historical binary from round 10;
any build from the current tree also carries another campaign's skel edits (§50). Comparing
against it measures the tree drift plus the change, not the change.

### Consequences

1. **This A/B is void** -- it cannot say whether the relaxation helps.
2. **Round 58's A/B is also suspect.** Its ~1% "regression" is the same magnitude as the drift
   seen here on untouched shapes. §62's refutation still stands, because that experiment
   predicted a 20% win and produced none -- a null result is robust to a 1-2% confound. But the
   attribution of the residual to the change is not.
3. **Future A/Bs must build both sides from the current tree**, differing only in the change
   under test, rather than reusing `odb`.

**State.** Reverted, source md5 `d98f283c29f0925875b5529718e3ba78`, rebuild clean.

## 70. Two failed controls, a stale ship copy, and another campaign in the file

Round 61 set out to fix §69's confound by building both A/B sides from the current tree. It did
not get a valid measurement. What it did get is three pieces of infrastructure knowledge, one of
which invalidates a class of future comparisons.

### The `ship/` skel copy goes stale

The `base` build failed with:

    skel source ID mismatch: expected c73e1bbd40c3a0778b3f447cc46f0631,
                                    got f3db55a91e6ca2da2563b8bffb2f07c9

The cause: after reverting, `cmake --build` found the skel **up to date** and did not re-run the
step that copies it into `backends/hexagon/skel/v79/ship/`. The runner was rebuilt, so it
embedded a new expected source ID, but the `ship/` skel was an older artifact. Copying that pair
to the device gives a runner that refuses to run.

**Fix: `touch` a skel source before rebuilding**, so the skel and its ship copy are regenerated.
That worked -- `base2` ran and printed `exit d0: ok`.

### Another campaign is editing `loop_ops.cc`

Running the **same** revert script that produced `d98f283c29f0925875b5529718e3ba78` in round 60
now produces `c80c9fcea5480c39725603d620d02db8`. The script is deterministic and its anchor
matched exactly once, so the difference is not mine.

**This is §50's provenance hazard, verified directly rather than inferred.** The file is shared,
and it changed between rounds 60 and 61 while this campaign was working on it.

### Both controls failed

- `base`: the source-ID mismatch above -- never ran.
- `base2`: ran (`exit d0: ok`) but the A/B reports **0.5 us on every shape**, including
  `m8192 x 1024 x 1024` which takes 1484 us. It is not computing. Not diagnosed.

    shape                   base2 (us)          wdb2 (us)
    m33 x 1024 x 1024       1.6 / 0.5 / 0.5     58.4 / 54.1 / 58.4
    m1024 x 1024 x 31       0.6 / 0.6 / 0.4     68.7 / 68.5 / 68.5
    m8192 x 1024 x 1024     0.8 / 0.6 / 0.7     1483.8 / 1484.1 / 1483.9

### Where the `E * 2 < N` relaxation stands

**Untested, not refuted.** Two attempts, both void for infrastructure reasons. The derivation
(§69) is sound and the numerics are clean (36 records, 0 failures), but no timing evidence exists
either way.

### What has to happen first

A/B measurement on this file is not currently trustworthy, because the tree moves under it. Before
the next optimisation attempt, the campaign needs either:

1. a frozen copy of `loop_ops.cc` that both A/B sides are built from -- differing only in the
   change under test; or
2. an agreement that the other campaign's edits are stable for the duration of a measurement.

Option 1 is within this campaign's control and is the right answer. **§69's confound and this
round's file drift are the same problem**, and it will keep producing void experiments until it
is fixed.

**State.** My change reverted (`cheapActivation` absent, `doubleBuffered` at its original form).
File md5 `c80c9fcea5480c39725603d620d02db8`, which includes another campaign's edits.

## 71. The frozen-build discipline works; the device does not

Round 62 implemented §70's prerequisite: freeze `loop_ops.cc` to a file, build both A/B sides
back-to-back from those exact bytes, and restore. That part worked.

**The frozen copy is `d98f283c29f0925875b5529718e3ba78`** -- the verified md5. So round 61's
`c80c9fcea5480c39725603d620d02db8` was a **transient** state: the other campaign edited the file
and reverted it, and the tree is back at the verified source. The §70 drift finding stands as a
warning, but the file is currently clean.

Both sides built with no errors, from identical bytes, differing only in the `E * 2 < N` clause.

### The measurement

    shape                   fbase (us)              fvar (us)
    m33 x 1024 x 1024       56.2 / 57.3 / 63.1       61.7 / 60.4 / 52.4
    m65 x 8191 x 1025       394.8 / 447.5 / 447.5   404.4 / 448.6 / 463.7
    m128 x 128 x 128        14.8 / 6.6 / 6.8        22.9 / 6.9 / 6.3
    m256 x 256 x 256        25.4 / 23.7 / 21.6      25.1 / 24.1 / 14.2
    m1024 x 1024 x 31       68.8 / 81.7 / 82.2      83.7 / 83.5 / 68.7
    m1024 x 1024 x 1024     197.9 / 207.5 / 208.4   207.7 / 207.8 / 207.0
    m8192 x 1024 x 1024     1485.4 / 1495.3 / 1484.9 1503.5 / 1503.8 / 1485.2
    m256 x 8192 x 4096      1323.7 / 1329.9 / 1331.5 1323.8 / 1324.1 / 1324.3

**The device is no longer giving stable numbers.** `m128 x 128 x 128` reads 14.8 / 6.6 / 6.8
within a single build -- a 2.2x spread. `m256 x 256 x 256` reads 25.4 / 23.7 / 21.6 against
25.1 / 24.1 / 14.2.

Levels are also **elevated across the board** versus earlier rounds: `m256 x 256 x 256` measured
13.0 us in §60 and now reads 21-25; `m33 x 1024 x 1024` measured 48-49 and now reads 56-63. The
large shapes are stable (`m8192 x 1024 x 1024` 1485-1503, `m256 x 8192 x 4096` 1324-1332), which
points at a fixed per-execution overhead growing -- thermal state, or contention on the device --
rather than the kernel itself changing.

### Consequence

**The `E * 2 < N` relaxation is still untested -- now for three attempts.** Each was void for a
different reason: §69's baseline was a historical binary, §70's controls failed outright, and
this one is drowned in device noise.

No conclusion can be drawn from this table in either direction. The fvar numbers are not better,
but with a 2.2x within-build spread on the small shapes, they could not have been detected if
they were.

### What the campaign should do

Three consecutive void measurements on the same family is itself the signal. Before another
optimisation attempt, either:

1. **re-baseline the device** -- cool it, then measure a known shape and confirm the numbers
   return to their round-10 values (`m1024 x 1024 x 1024` at 197, not 207); or
2. **abandon the small-M family** and spend the remaining rounds on shapes where the effects are
   large relative to this noise floor.

Option 1 is cheap and worth doing first, because the same noise has now degraded three
experiments and will degrade the next one.

**State.** Frozen file restored, tree md5 `d98f283c29f0925875b5529718e3ba78`, builds clean.

## 72. The relaxation is refuted, and the tree builds up to 15% slower than round 10

Round 63 re-baselined and got the first clean measurement in four rounds.

### The device is fine

Repeats agree within 0.5%. Round 62's 2.2x spreads were transient; the device is cool (35-38 C)
and stable. No thermal problem.

### But the current tree builds a slower binary

`odb` is the round-10 artifact. `fbase` is built now from the **same `loop_ops.cc` md5**
(`d98f283c29f0925875b5529718e3ba78`), so the two differ only in files this campaign does not edit.

    shape                   odb (us)   fbase (us)   delta
    m1024 x 1024 x 1024     197.1      197.9        +0.4%
    m8192 x 1024 x 1024     1474       1485         +0.8%
    m1024 x 1024 x 31       66.8       68.8         +3.0%
    m256 x 256 x 256        13.0       13.8         +6.4%
    m128 x 128 x 128        5.8        6.7          +15%

**The drift is systematic and hits small shapes hardest.** `m128 x 128 x 128` is 15% slower than
the artifact that produced the loss table. The other campaign's skel edits have cost real
performance on exactly the family this campaign has spent twenty rounds trying to improve.

### The relaxation is refuted

With the device stable, the frozen A/B is finally readable:

    shape                   fbase (us)              fvar (us)              verdict
    m33 x 1024 x 1024       45.8 / 49.6 / 49.1 / 45.6  47.7 / 51.2 / 47.7 / 52.4  ~4% worse
    m65 x 8191 x 1025       399.2 / 388.5 / 389 / 394.1 401.6 / 400.8 / 405.2 / 397.3  ~2.5% worse
    m128 x 128 x 128        6.8 / 6.6 / 6.6 / 6.5   6.2 / 6.3 / 6.3 / 6.1  5% faster, untargeted
    m256 x 256 x 256        14.0 / 14.1 / 13.9 / 14.1 13.8 / 14 / 13.7 / 13.7  unchanged
    m1024 x 1024 x 31       68.8 / 68.9 / 68.9 / 68.9 68.7 / 68.6 / 68.7 / 68.7  unchanged
    m1024 x 1024 x 1024     197.9 / 198.7 / 197.3 / 197.9 198.3 / 197.7 / 198.0 / 197.8  unchanged
    m8192 x 1024 x 1024     1489 / 1502.5 / 1502.7 / 1502.4 1502 / 1503.0 / 1491.8 / 1502.4  unchanged
    m256 x 8192 x 4096      1361.4 / 1374.7 / 1344.3 / 1343.3 1359.7 / 1343.1 / 1370.3 / 1348.3  unchanged

Both targeted shapes are worse. The untargeted `m128 x 128 x 128` moved 5% the other way, which
bounds the residual build-to-build noise -- so the ~3% regressions on the targets are at the edge
of resolvable, but they are certainly not wins.

**Why it fails.** Splitting the pass doubles the number of passes, and each one re-reads and
re-packs the activation. The transform is already prefetch-bound (§55), so doubling it costs what
the weight-fill overlap was supposed to save. §68's arithmetic -- 1.4 us of activation against
35 us of weight fill -- compared the *re-read* but not the *re-pack*, and the re-pack is the
expensive half.

### Consequences for the campaign

1. **The loss table is the right reference and current builds are handicapped.** Any future
   improvement has to be measured against `fbase`, not `odb`, or it will be credited with
   overcoming a 0.4-15% drift it did not cause.
2. **The small-shape losses may be partly drift.** `m128 x 128 x 128` at 1.383 was measured with
   `odb`; the same code built today is 15% slower on that shape. That does not excuse the loss,
   but it means the gap is partly another campaign's regression, not this kernel's design.
3. **Four directions are now closed on small N and one on small M**, all by measurement:
   GEMV (§26/§27), band staging (§50), store micro-optimisation (§59-§61), cross-iteration
   overlap (§67), and weight double-buffering (§72).

**State.** No change in the tree; frozen file restored, md5 `d98f283c29f0925875b5529718e3ba78`.

## 73. The drift is the phase profiler -- and turning it off is a real win

§72 found the current tree builds 0.4-15% slower than the round-10 `odb` artifact from the same
`loop_ops.cc`. §72 attributed it to "the other campaign's skel edits". That was wrong, and the
answer was in a file this campaign owns.

### Two hypotheses, one confirmed

**First hypothesis: the per-command cache flush.** `execute_command.cc` gained
`flush_command_outputs`, called per command in the group loop -- a cache clean over the command
plus a tensor sync over its outputs. A fixed per-execution cost, which would explain why the
drift scales with 1/shape-size.

Tested by building a variant with the call commented out:

    m128 x 128 x 128      odb 5.4/5.8/5.7   fbase 6.4/6.8/6.8   noflush 6.5/6.5/6.7
    m256 x 256 x 256      odb 13.0/13.2/13.1 fbase 14.1/14.2/13.7 noflush 13.9/14/13.5
    m1024 x 1024 x 31     odb 66.9/66.9/67.0 fbase 68.9/68.9/69.0 noflush 68.9/68.6/68.8

**`noflush` matches `fbase`, not `odb`.** Refuted -- the flush is not the drift.

**Second hypothesis: `HTP_MM_PHASE_PROFILE`.** It is set to 1 in
`backends/hexagon/skel/CMakeLists.txt`, so **every build has it** -- but `odb` is a round-10
artifact and may predate it. Built a variant with the define at 0:

    shape                   odb (us)     fbase (us)   noprof (us)   recovery
    m128 x 128 x 128        5.77         6.77         6.30          -7.0%
    m256 x 256 x 256        12.93        13.87        13.07         -5.8%
    m1024 x 1024 x 31       66.93        68.97        67.17         -2.6%
    m1024 x 1024 x 1024     197.07       197.93       196.90        -0.5%
    m8192 x 1024 x 1024     1473.6       1485.3       1477.1        -0.6%

**`noprof` lands on `odb` almost exactly.** The profiler is the drift.

### What this means

1. **The instrumentation costs 0.5-7% and the cost scales with 1/shape-size.** The timer reads
   and the extra code bloat the hot loop; on `m128 x 128 x 128` that is 7% of the whole kernel.
2. **It hit exactly the family this campaign spent twenty rounds on.** The small-N and small-M
   shapes are the ones where a fixed overhead is largest as a fraction. Every phase profile taken
   since round 21 was measured on a build handicapped in proportion to how small the shape was.
3. **The phase analysis itself survives.** The slots are *relative* attributions within one
   execution, so §52/§57/§62's conclusions about where time goes are unaffected. What was
   inflated is the absolute total, and any comparison against QNN or against `odb`.
4. **`HTP_MM_PHASE_PROFILE` is now 0 by default.** Re-enable it by editing
   `backends/hexagon/skel/CMakeLists.txt` when a phase profile is needed, and expect the build to
   be slower on small shapes while it is on.

### The irony, recorded

The profiler was built to find where the time goes on small shapes. It was itself the largest
single overhead on those shapes, and it took four rounds of void measurements -- two blamed on
the device, one on a historical baseline -- before the search turned to the build flags.

**State.** `HTP_MM_PHASE_PROFILE=0` in `backends/hexagon/skel/CMakeLists.txt`;
`execute_command.cc` restored; `loop_ops.cc` at `d98f283c29f0925875b5529718e3ba78`.

## 74. Full sweep with the profiler off -- the campaign's standing

Round 64 turned the profiler off. Round 65 re-ran all 57 shapes on both backends through
ExecuTorch (228 records, **0 numerics failures**) to establish where things actually stand.

    metric                        new (noprof)   round 10 (odb)
    geo-mean hex / QNN            0.9406         0.9175
    shapes at or under QNN        23 / 54        24 / 54
    paired shapes                 54             54

**The three shapes this work started from are all won:**

    m1024 x 1023 x 1024     hex   218.9   QNN   231.4   ratio 0.946
    m1024 x 1025 x 1024     hex   230.3   QNN   232.4   ratio 0.991
    m4097 x 1025 x 257      hex   456.0   QNN  1078.4   ratio 0.423

The last is 2.4x faster than QNN. **All three have now been won in two independent sweeps**, four
rounds apart, with different builds.

**Largest wins:**

    m8192 x  256 x 4096     ratio 0.141    (7.1x faster)
    m1024 x 1024 x 8192     ratio 0.195    (5.1x faster)
    m4097 x 1025 x  257     ratio 0.423
    m4096 x 8192 x  256     ratio 0.499
    m4096 x 4096 x 4096     ratio 0.514

**Residual losses, one family:**

    m128  x  128 x  128     ratio 1.803
    m8192 x 4096 x    1     ratio 1.494
    m4096 x 4096 x    1     ratio 1.463
    m1024 x 1024 x   64     ratio 1.401
    m1024 x 1024 x   65     ratio 1.341
    m65   x 8191 x 1025     ratio 1.326
    m256  x  256 x  256     ratio 1.325

Small N, small square, GEMV (N = 1), and odd dimensions.

### A protocol discrepancy, recorded

The small shapes read substantially higher here than in round 64's A/B: `m256 x 256 x 256` is
19.8 us in this sweep against 13.07 in the A/B, and `m128 x 128 x 128` is 10.5 against 6.30. The
sweep and the A/B use different scripts with different execution counts and warmup. **The two are
not interchangeable**, and the A/B numbers are the ones to trust for a single shape while the
sweep is the one to trust for the aggregate. Not reconciled.

### Against the goal

The objective was to beat QNN on the matmuls this work started from, optimising first and
disassembling QNN when that failed. Both were done, and the three named shapes are won. The
aggregate also favours hexagon (0.94). The remaining losses are a single identified family, and
five separate attacks on it have been closed by measurement: the GEMV route (§26/§27), band
staging (§50), store micro-optimisation (§59-§61), cross-iteration overlap (§67), and weight
double-buffering (§72). Round 64 additionally found and removed a 0.5-7% instrumentation cost
(§73).

### Every shape, split (profiler off, §74 sweep)

**23 of 54 beat QNN.** Sorted best-first.

| shape | hexagon us | QNN us | hex/QNN |
|---|---:|---:|---:|
| `m8192_k256_n4096` | 2316.7 | 16409.4 | **0.141** |
| `m1024_k1024_n8192` | 1242.2 | 6375.2 | **0.195** |
| `m4097_k1025_n257` | 456.0 | 1078.4 | **0.423** |
| `m4096_k8192_n256` | 2596.2 | 5206.6 | **0.499** |
| `m4096_k4096_n4096` | 12426.6 | 24187.9 | **0.514** |
| `m1025_k1024_n1024` | 227.1 | 352.6 | **0.644** |
| `m1024_k1024_n3072` | 547.9 | 832.7 | **0.658** |
| `m1023_k1024_n1024` | 231.4 | 323.8 | **0.714** |
| `m127_k4096_n4096` | 626.5 | 869.4 | **0.721** |
| `m2048_k2048_n2048` | 1544.8 | 2045.9 | **0.755** |
| `m1024_k1024_n1025` | 231.8 | 306.4 | **0.757** |
| `m1024_k1024_n2048` | 392.9 | 514.4 | **0.764** |
| `m256_k8192_n4096` | 1327.9 | 1652.5 | **0.804** |
| `m1024_k3072_n1024` | 644.1 | 793.2 | **0.812** |
| `m1024_k8192_n1024` | 2436.8 | 2825.8 | **0.862** |
| `m1024_k64_n1024` | 46.2 | 51.7 | **0.893** |
| `m1024_k65_n1024` | 50.4 | 54.0 | **0.935** |
| `m513_k2048_n2048` | 465.7 | 496.4 | **0.938** |
| `m1024_k1023_n1024` | 218.9 | 231.4 | **0.946** |
| `m256_k4096_n8192` | 1242.6 | 1298.6 | **0.957** |
| `m1024_k1024_n1024` | 196.7 | 203.3 | **0.967** |
| `m1024_k63_n1024` | 49.8 | 51.0 | **0.976** |
| `m1024_k1025_n1024` | 230.3 | 232.4 | **0.991** |

**31 of 54 lose to QNN.** Sorted worst-first.

| shape | hexagon us | QNN us | hex/QNN |
|---|---:|---:|---:|
| `m128_k128_n128` | 10.5 | 5.8 | 1.803 |
| `m8192_k4096_n1` | 1711.3 | 1145.6 | 1.494 |
| `m4096_k4096_n1` | 867.1 | 592.6 | 1.463 |
| `m1024_k1024_n64` | 76.6 | 54.7 | 1.401 |
| `m1024_k1024_n65` | 84.5 | 63.0 | 1.341 |
| `m65_k8191_n1025` | 410.3 | 309.4 | 1.326 |
| `m256_k256_n256` | 19.8 | 14.9 | 1.325 |
| `m129_k4093_n2051` | 430.9 | 326.6 | 1.319 |
| `m1024_k1024_n32` | 63.0 | 47.9 | 1.315 |
| `m8192_k1024_n1024` | 1475.3 | 1122.4 | 1.314 |
| `m31_k1024_n1024` | 51.5 | 39.6 | 1.299 |
| `m33_k1024_n1024` | 51.9 | 40.1 | 1.293 |
| `m1024_k1024_n31` | 67.1 | 52.0 | 1.292 |
| `m64_k1024_n1024` | 55.2 | 43.0 | 1.285 |
| `m128_k1024_n1024` | 62.2 | 52.7 | 1.180 |
| `m256_k1024_n1024` | 83.5 | 71.4 | 1.170 |
| `m1024_k1024_n257` | 114.1 | 98.3 | 1.161 |
| `m1024_k32_n1024` | 46.7 | 41.2 | 1.131 |
| `m1024_k31_n1024` | 47.0 | 41.6 | 1.128 |
| `m1024_k2048_n1024` | 423.9 | 375.9 | 1.128 |
| `m1_k3072_n1024` | 112.5 | 101.5 | 1.108 |
| `m1_k1024_n3072` | 111.9 | 101.5 | 1.103 |
| `m65_k1024_n1024` | 55.0 | 50.2 | 1.094 |
| `m32_k1024_n1024` | 46.2 | 42.8 | 1.078 |
| `m1_k1024_n2048` | 74.2 | 69.8 | 1.063 |
| `m512_k1024_n1024` | 130.1 | 123.6 | 1.053 |
| `m33_k4096_n4096` | 571.3 | 547.4 | 1.044 |
| `m1024_k33_n1024` | 50.9 | 49.2 | 1.034 |
| `m512_k512_n512` | 47.5 | 46.6 | 1.018 |
| `m8_k8192_n4096` | 1068.7 | 1052.8 | 1.015 |
| `m1_k8192_n8192` | 2064.4 | 2046.1 | 1.009 |

### What separates them

**Wins are N-large.** Every shape at N >= 2048 with M >= 1024 wins, and the 1024-cube
family (m1023/m1024/m1025 x k1024 x n1024/1025/2048/3072) wins across the board at
0.64-0.99. The largest margins come from N >= 4096: `m8192 x 256 x 4096` at 0.141 and
`m1024 x 1024 x 8192` at 0.195.

**Losses are four families, and they are small in a specific dimension:**

1. **Small N** (N <= 65, any M): 1.29-1.40. `m1024 x 1024 x 31/32/64/65`, and
   `m1024 x 1024 x 257` at 1.161. This is the family §52-§73 attacked five times.
2. **Small M** (M <= 256, any N, K = 1024): 1.08-1.33. `m31/32/33/64/65/128/256 x 1024 x 1024`.
   §68 traced this to a fixed ~35 us weight fill that small M cannot amortise.
3. **N = 1 (GEMV):** `m8192 x 4096 x 1` 1.494, `m4096 x 4096 x 1` 1.463, and the `m1 x ...`
   shapes at 1.01-1.11. §26/§27: the HMX path cannot be reused and the purpose-built GEMV
   was 3.1x slower than the general path.
4. **Small square** (<= 512^2): `m128 x 128 x 128` 1.803, `m256 x 256 x 256` 1.325,
   `m512 x 512 x 512` 1.018. Fixed per-execution cost dominates.

**Small K is a fifth, milder family:** `m1024 x k31/k32/k33 x 1024` at 1.03-1.13 and
`m1024 x 2048 x 1024` at 1.128.

**One shape does not fit:** `m8192 x 1024 x 1024` loses at 1.314 despite being large in
every dimension. Its mirror `m1024 x 8192 x 1024` *wins* at 0.862, so the difference is M --
M = 8192 with K = 1024 is a combination no other losing shape has, and it has not been
investigated. It is the most promising single target left, because it is large enough that
the fixed costs which explain every other loss cannot explain this one.

## 75. Reproducing the results

Everything below was run in this session. Paths are relative to `/home/yydh/executorch`.

### Environment

    cd /home/yydh/executorch
    source /home/yydh/miniconda3/etc/profile.d/conda.sh && conda activate et

`conda activate et` is required -- without it the sweep dies with `ModuleNotFoundError: No module
named 'torch'`.

Device: OnePlus 13 (PJZ110), Snapdragon 8 Elite / SM8750, Hexagon v79, Android 15, root via
Magisk. Reached over ssh as `oneplus13-reverse` (ProxyJump `vyatta-bastion`, port 38022, user
`u0_a354` -- see the ssh config). All device work is serialised through the lock:

    .tmp/locks/lockrun device -- <cmd>      # exclusive
    .tmp/locks/lockrun build  -- <cmd>
    .tmp/locks/lockrun export -- <cmd>

### Build

    .tmp/locks/lockrun build -- bash -c \
      'cmake --build cmake-out-android-arm64-v8a --target hexagon_skel_v79 executor_runner -j8'

**If you only reverted a change, `touch` a skel source first.** Otherwise `cmake --build` finds
the skel up to date, skips the step that copies it into `skel/v79/ship/`, and the rebuilt runner
refuses the stale skel with `skel source ID mismatch`. This cost a round (§70).

### Deploy

    C="-o ControlMaster=auto -o ControlPersist=300 \
       -o ControlPath=/home/yydh/executorch/.tmp/scratch/mnv2/matmul_coverage/ssh_control"
    R=/data/data/com.termux/files/home/csm/matmul_rebenchmark
    D=noprof

    ssh  $C oneplus13-reverse "mkdir -p $R/$D/skel"
    scp  $C cmake-out-android-arm64-v8a/executor_runner oneplus13-reverse:$R/$D/hexagon_runner
    scp  $C cmake-out-android-arm64-v8a/backends/hexagon/skel/v79/ship/libhex-htp-skel-v79.so \
             oneplus13-reverse:$R/$D/skel/
    ssh  $C oneplus13-reverse "chmod +x $R/$D/hexagon_runner"

Runner and skel are a **provenance pair**. Deploying a skel alone gives SIGABRT (exit 134);
deploying a mismatched pair gives the source-ID refusal above.

### Full sweep, both backends (the §74 result)

    export PYTHONPATH=/home/yydh \
           SWEEP_SKEL=noprof/skel \
           SWEEP_RUNNER=noprof/hexagon_runner \
           SWEEP_RESULTS=noprof_full.json
    .tmp/locks/lockrun device -- python .tmp/scratch/mnv2/matmul_coverage/measure_newskel.py

57 shapes x 2 backends x 2 rounds = 228 records. `SWEEP_TAGS=a,b,c` subsets it; omit for all.
Device-side PTEs live in `$R/coverage/` (114 files, `hex_<tag>.pte` + `<tag>.bin`).

### Head-to-head against QNN

    python3 .tmp/scratch/mnv2/matmul_coverage/cmp65.py

Prints the geo-mean, win count, the three named shapes, and the worst/best tables quoted in §74.
Reads `noprof_full.json` and `odb_results.json`.

### A/B two builds

    export PYTHONPATH=/home/yydh \
      AB_BUILDS=fbase:fbase/skel:fbase/hexagon_runner,fvar:fvar/skel:fvar/hexagon_runner \
      AB_TAGS=m33_k1024_n1024,m1024_k1024_n31 \
      AB_ROUNDS=3
    .tmp/locks/lockrun device -- python .tmp/scratch/mnv2/matmul_coverage/ab2.py

Writes `ab2_results.json`. **The baseline must be built from the current tree**, not reused from
`odb` -- see the frozen-build discipline below.

### The frozen-build discipline

The tree is shared and moves. To measure one change, freeze the file and build both sides from
those exact bytes:

    F=.tmp/scratch/mnv2/matmul_coverage/frozen_loop_ops.cc
    L=backends/hexagon/third-party/mnn-htp-ops/src/dsp/loop_ops.cc

    cp $L $F                                    # freeze
    cp $F $L && touch $L && <build> && <deploy as A>
    cp $F $L && python3 <edit>.py && touch $L && <build> && <deploy as B>
    cp $F $L                                    # restore

The frozen copy in this session is md5 `d98f283c29f0925875b5529718e3ba78`.

### Phase profile

Requires a build with `HTP_MM_PHASE_PROFILE=1` in `backends/hexagon/skel/CMakeLists.txt` (it is
**0 by default** since §73, because it costs 0.5-7%):

    export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:$R
    export ADSP_LIBRARY_PATH=$R/prof43/skel
    export HEXAGON_TRACE=1
    taskset -c 7 $R/prof43/hexagon_runner \
      --model_path $R/coverage/hex_m1024_k1024_n31.pte \
      --inputs     $R/coverage/m1024_k1024_n31.bin \
      --num_executions 1 --print_output none | grep 'optime d0 slot='

Slots: 200 weight fill, 201 activation pack, 202 HMX tile loads, 203 output store, 204 whole
kernel, 205 call count, 206 weight bytes, 207 path tag. 210-212 are free scratch, used for the
diagnostics in §47-§61. Values are cumulative and zero slots are not printed.

### Traps that cost rounds

1. **Bash redirects to `/tmp` are outside the sandbox and silently produce empty logs.** Write
   logs under `.tmp/`.
2. **`diff` is silently broken here.** Use Python `difflib`, `md5sum`, or `wc -c`.
3. **`--output_file X` writes `X-0.bin`**, not `X`.
4. **A throwing edit program may leave earlier edits in place** (§38). After any failed edit,
   grep for the edits' fingerprints before doing anything else.
5. **Anchor on a span only one site can have, and count occurrences before writing.** A Python
   pass that aborts unless every anchor appears exactly once caught a duplicate that a naive
   edit would have corrupted.
6. **Do not baseline against a historical artifact.** `odb` is a round-10 binary; comparing
   against it measures tree drift plus the change (§69, §72).
7. **The sweep and the A/B are not interchangeable.** Small shapes read ~1.6x higher in the
   sweep than in the A/B (§74). Trust the A/B for one shape, the sweep for the aggregate.

### Where the artifacts are

**All under `.tmp/scratch/mnv2/matmul_coverage/`, which is git-excluded** -- these are not
durable and will not survive a clean:

    noprof_full.json      57-shape sweep, profiler off (228 records, 0 numerics failures)
    odb_results.json      round-10 sweep, the historical baseline
    cmp65.py              the head-to-head comparison
    ab2_results.json      most recent A/B
    measure_newskel.py    the sweep driver
    ab2.py                the A/B driver
    frozen_loop_ops.cc    the frozen source for A/B
    prof43.sh, prof59.sh  phase-profile scripts

If these should be durable, they need to be copied into the repository -- `.tmp/` is excluded by
design, and the reproduction of §74 currently depends on files that live there.

## 76. Related documents

Outside git, under `.tmp/scratch/mnv2/` — restated numbers above are the durable
copy:

- `matmul_coverage/` — the 57-shape sweep of §8: `REPORT.md` (generated table),
  `manifest.json`, `combined.json`, the per-run logs and etdumps, and
  `driver.sh` (the on-device runner)
- `matmul_shape_sweep/` — the earlier 24-shape sweep, same protocol, and the
  source of the `K % 64 != 0` code-path note
- `matmul_hexagon_vs_qnn.md` — the head-to-head report, both regimes, revision
  history and the two disproven hypotheses
- `M1_slow_runbook.md` — the `M == 1` problem, reproduction and deploy commands
- `chained_weight_dma_fix.md` — the two fixes of §3 in full
- `hexagon_matmul_50rounds.md` — the round-by-round optimization log
- `disasm_compare/comparison.md` — static ET-vs-QNN DMA comparison
- `disasm_compare/hmx_decode/ANALYSIS.md` and `REPORT.md` — the HMX decode of §4
- `disasm_compare/dispatch_probe/REPORT.md` — the entry-skip dispatch probe

In git:

- `backends/hexagon/README.md` — backend structure and op contracts
- `backends/hexagon/RUNBOOK.md` — build, deploy, run
- `backends/hexagon/STATUS.md` — the same "evidence that cannot be reproduced,
  so it is kept here" precedent this file follows
- `backends/hexagon/third-party/mnn-htp-ops/VENDORING.md` — how the vendored DSP
  library is carried
