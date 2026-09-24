# mnn-htp-ops

Hexagon DSP op library ported from MNN. This directory is a vendored copy; the
only ExecuTorch-specific code lives outside it.

## Provenance

| | |
|---|---|
| Upstream | `MNN/source/backend/hexagon/htp-ops-lib` |
| Revision | `b8c533a58926a45d9e893565958701db2c7f09fe` (2026-09-15) |
| Copied | `include/`, `src/dsp/`, `schema/` |
| Dropped | `src/host/`, `src/dsp/ops/*` build dirs, MNN CMake |

## License

MNN is Apache-2.0. The vendored sources carry no per-file license headers
upstream, so the license is retained here as `LICENSE`, copied verbatim from
the upstream tree at the revision above, and the attribution required by section
4 of the license is in `NOTICE`.

Nothing here has been relicensed. Of the four modifications listed under "Local
modifications", the first two are build-environment fixes and do not change
behavior; the other two are ExecuTorch's own -- one instrumentation that only
runs when the host passes a profile buffer, and one correctness fix to the scalar
tails of three kernels. Those last two are the only changes this PR makes to the
vendored sources. Separately, one defect was found in this snapshot and is
described under "Defects found in this snapshot"; that one is not fixed here, and
nothing in this directory was changed for it.

The pin above is a revision of `htp-ops-lib`, not a byte-for-byte description of
this directory, and this PR did not make it one: 19 of the 70 files under
`include/` and `src/dsp/` differ from it, 2125 changed lines in all, of which the
four items below account for a small part. The per-file counts and what is
attributed to whom are under "How much of this directory is not the pin"; an
unattributed difference is not this PR's change and not a defect anyone here
found, but a reviewer should know it is there before reading a diff of this
directory.

## Why the DSP side is kept verbatim

The upstream op library is already a standalone DSP library: the op entry points
in `src/dsp/*_ops.cc` are plain C functions taking raw pointers and integers,
and the only externally visible contract is the FlatBuffers command format in
`schema/Command.fbs`. Nothing in `src/dsp/` references the MNN runtime, so
ExecuTorch consumes it through the same RPC surface MNN does:

    htp_ops_init_backend()          start the backend (power, VTCM, HMX, workers)
    htp_ops_get_skel_arch()         arch of the loaded skel
    htp_ops_execute_command_group() run one batch of ops

The ExecuTorch-specific work is therefore entirely on the host side: the AOT
emits the command descriptors and the runtime drives the RPC.

## Host/DSP contract

One `execute_command_group` call carries a whole batch:

    group buffer (fd, offset):
        [8 bytes header][count x 3 int32: (cmd_fd, cmd_offset, cmd_size)]
    cmd_size <= 0   the DSP invalidates the region before use
    cmd_size  > 0   the host already cleaned it

Each command is a FlatBuffers `DSPCOMMAND::Command` at `(cmd_fd, cmd_offset)`:

    table Command   { type:int; inputs:[Tensor]; outputs:[Tensor]; params:[int]; }
    table Tensor    { fd:int; offset:int; size:int; }
    table SyncGroup { inputs:[Tensor]; outputs:[Tensor]; }

`SyncGroup` names the tensors to invalidate before and flush after the batch.

## Local modifications

Four. The first two are forced by the build environment, the third is
instrumentation that only runs when the host asks for it, and the fourth is a
correctness fix. The first two arrived with this directory; the third and fourth
are the two changes this PR makes to it, and nothing else under `include/` or
`src/dsp/` was edited by this PR:

1. `src/dsp/vtcm_mgr.cc` and `include/dsp/vtcm_mgr.h` — three changes, of which
   the second and third are not upstream anywhere: the
   `#include "flatbuffers/flatbuffers.h"` is removed (the file never references
   the `flatbuffers::` namespace, and keeping the dependency for a dead include
   would couple the skel to flatbuffers); `reset()` grows an `#ifdef
   HTP_OPS_KEEP_VTCM` early return, an experiment that is off by default because
   holding the reservation across delegates changes what the next delegate
   measures; and the header declares `vtcm_manager_get_vtcm_alloc_end()`, which
   `src/dsp/loop_ops.cc:976` reads to keep a planned HMX tile's working set
   inside the end of the reservation. Measured against the pin: 10 lines added
   and 1 removed in the `.cc`, 9 added in the `.h`.
2. `schema/current/Command_generated.h` is checked in rather than generated.
   Upstream generates it with `flatc` from `schema/Command.fbs`; the checked-in
   header is that schema generated with flatc 24.3.25 -- the version its own
   static assertion requires -- and declares the same seven accessors
   (`type`, `inputs`, `outputs`, `params`, `fd`, `offset`, `size`) as the header
   in the upstream tree. It is not byte-identical to that header: the older flatc
   upstream used writes its own scaffolding (raw field offsets rather than `VT_`
   constants, `MiniReflectTypeTable` rather than builders). `flatc` is not part of
   the Hexagon SDK. `schema/current/Command.fbs` is kept as the source of truth,
   byte-identical to upstream's, so the header can be regenerated whenever flatc
   is available.
3. The profile buffer (`htp_ops_execute_command_group_profile`, and the probe
   records and stage writes in `src/dsp/execute_command.cc` plus the `htp_probe_stage`
   calls in the attention kernels) is ExecuTorch's, not upstream's. It exists
   because a DSP fault takes the RPC session down with it and the probe survives
   it. Nothing in it does anything unless the host passes a profile buffer, and
   the op-selection paths are untouched, so a call without one behaves as
   upstream. It now also records each command's kernel microseconds in the last
   int of its probe record and marks the header with a version, which is what
   the host-side profiler reads to attribute time to individual ops. This PR's
   whole change to that file is 24 lines added and 7 removed against the commit
   the PR starts from (the file differs from the pin by more, see below).
4. Three kernels answered a NaN differently in the scalar tail they finish their
   vector loop with than the vector loop itself does, and differently from
   torch. `htp_ops_clamp_fp16_chunk` (`src/dsp/unary_ops.cc`, the clamp behind
   relu, relu6 and hardtanh) returned the bound, `htp_ops_binary_relu_fp16_scalar`
   (`src/dsp/eltwise_ops.cc`, the scalar half of `add_relu`) returned zero, and
   the maximum fold in `htp_ops_reduce_fp16_scalar_range` (`src/dsp/eltwise_ops.cc`)
   returned the value it was reduced against. Each now asks for the NaN by the
   `(|x| & 0x7fff) > 0x7c00` bit test the vector half already used and leaves the
   input in place, so a NaN that lands in the tail comes back a NaN. The three
   are covered on hexagon-sim by `backends/hexagon/test/test_blob_on_sim.py`,
   whose clamp sweep and `test_the_other_two_tails_keep_a_nan_as_well` go red
   without them. Replaying this on an upstream update means the bit tests in
   those three functions, and the `htp_ops_fp16_bits_are_nan` helper the
   reduction fold reads (`src/dsp/eltwise_ops.cc`; the clamp tail and `add_relu`
   write the same test out inline instead of calling it), and nothing else in
   either file changed. Against the commit this PR starts from that is 13 lines
   added and 3 removed in `src/dsp/unary_ops.cc`, and 27 added and 2 removed in
   `src/dsp/eltwise_ops.cc`; both files differ from the pin by much more than
   that, and the rest of it is not this PR's (see below).

   Two of the three are partial, in the same way and for the same reason.
   `Q6_Vhf_vmax_VhfVhf` answers a NaN whose sign bit is set by returning the
   other operand, and that instruction is the vector half of the reduction's fold
   *and* the rectifier behind `add_relu`'s `max(a + b, 0)`. Both vector halves
   therefore still differ from torch, and from their now-fixed tails, for that
   value: driven on hexagon-sim with operands of `0xfe00`, a vector-lane element
   of `add_relu` comes back `0.0` where torch and the tail keep the NaN. The
   clamp is the one of the three whose two halves agree on every NaN, because its
   vector loop restores the input when the bit test matches rather than comparing
   -- measured by driving its sweep with `0xfe00` in place of the NaN it carries,
   which no case in this checkout asserts.
   Left as it is rather than masked over, because keeping a NaN in the max means
   carrying a predicate through the fold and through the five vector rotations
   that finish it, which is a different change from this one.

## How much of this directory is not the pin

Every file under `include/` and `src/dsp/` diffed against the pin at `b8c533a`,
changed lines counted:

| file | added | removed |
| --- | --- | --- |
| `src/dsp/loop_ops.cc` | 603 | 40 |
| `src/dsp/eltwise_ops.cc` | 384 | 14 |
| `src/dsp/unary_ops.cc` | 187 | 3 |
| `src/dsp/execute_command.cc` | 178 | 10 |
| `src/dsp/attention_sync_process.cc` | 164 | 23 |
| `src/dsp/attention_hmx.cc` | 159 | 27 |
| `src/dsp/attention_push_kv.cc` | 91 | 3 |
| `src/dsp/attention_common.hpp` | 56 | 2 |
| `src/dsp/attention_entry.cc` | 46 | 4 |
| `src/dsp/attention_private.hpp` | 30 | 2 |
| `src/dsp/hmx_queue.cc` | 29 | 0 |
| `src/dsp/worker_pool.cc` | 15 | 0 |
| `src/dsp/commu.cc` | 11 | 0 |
| `src/dsp/vtcm_mgr.cc` | 10 | 1 |
| `src/dsp/attention_sync_setup.cc` | 9 | 7 |
| `include/dsp/vtcm_mgr.h` | 9 | 0 |
| `src/dsp/region_ops.h` | 5 | 0 |
| `include/dsp/hmx_utils.h` | 2 | 0 |
| `include/dsp/worker_pool.h` | 1 | 0 |

51 of the 70 files are byte-identical; those 19 are the rest, 2125 changed lines
in all. Five of them are the modifications above: `src/dsp/vtcm_mgr.cc` and
`include/dsp/vtcm_mgr.h` (1), `src/dsp/execute_command.cc` (3),
`src/dsp/unary_ops.cc` and `src/dsp/eltwise_ops.cc` (4). Even in those five, most
of the difference is not ours -- this PR's own change is 24/7, 13/3 and 27/2
lines against the commit it starts from, against 178/10, 187/3 and 384/14 here.

The other 14 files are not attributed: `loop_ops.cc`, the seven `attention_*`
files, `hmx_queue.cc`, `worker_pool.cc`, `commu.cc`, `region_ops.h`,
`hmx_utils.h` and `worker_pool.h`. They carry entry-point parameters and symbols
the pin's own history has no record of -- `MNN_ATTN_SRC_BYPASS`,
`MNN_ATTN_WEIGHT_CACHEALLOC`, `g_attn_dma_fault`, `HMX_FP16_MAX_TILES_PER_LOAD`,
`hmxTileBudget`, `hmxFlags`, `worker_pool_debug_state` -- so this directory came
from a tree that is not exactly the pin, or from the pin plus changes nobody
recorded. **This PR did not edit any of them**, and none of them is a defect
found here; a reviewer reading a diff of this directory should know which 14
files are in that position before asking what ExecuTorch changed in them.

## Defects found in this snapshot

One, and it is not fixed here. **No file in this directory was changed for it**;
the only vendored changes in this PR are items 3 and 4 above.

The pin above is not a byte-for-byte description of this directory, and this
branch did not make it one (the per-file counts are in the section above): of
the 70 files under `include/` and `src/dsp/` that the pin has, 51 are
byte-identical and 19 differ, 2125 changed lines in all, and the four items above
account for only part of that. The attention kernels also carry entry-point
parameters, a DMA-fault probe and compile-time bypass and cache-allocation knobs
(`MNN_ATTN_SRC_BYPASS`, `MNN_ATTN_WEIGHT_CACHEALLOC`, `g_attn_dma_fault`), and
the HMX path carries a tile budget (`hmxFlags`, `hmxTileBudget`,
`HMX_FP16_MAX_TILES_PER_LOAD`) that `hexagon_ops.py` plans for, a
`worker_pool_debug_state` entry and a VTCM alloc-end guard -- none of which the
list above describes. Re-vendoring this directory means diffing it against the
pinned revision rather than trusting the pin; attributing the rest of those 19
files is left to whoever does that.

### `store_output_tile_fp16` puts the wrong channel tile's bias in a ragged position tile

| | |
|---|---|
| Where | `src/dsp/im2col_convolution_fp16.cc:15`, called from the single-tile path at `:1748` |
| Also reached by | `htp_ops_conv1x1_direct_fp16` (`:1840`), which is this same entry point |
| Trigger | `mp` and `np` in the command's params are zero, or one: one position tile and one channel tile per pass, **and** the position count is not a multiple of 32, **and** the tensor has more than one channel tile |
| Expected | every output channel gets its own bias |
| Actual | the last position's upper 32 channels come back off by exactly `bias[l] - bias[32 + l]` for `l` in 0..31: they carry the bias of the channel tile 32 lanes up instead of their own |

The reproduction is in this repository and needs no device:

    pytest backends/hexagon/test/test_conv_sim.py -k single_tile

It runs a 3x3 stride-1 pad-1 convolution with 64 input and 64 output channels over
a 5x5 input -- 25 positions, so the position tiles are ragged -- once with
`mp = np = 1` and once with `mp = 1, np = 2`, and asserts which elements of the
first differ from torch and by how much. The criterion is worth more than the
mismatch on its own: the difference is *exactly* `bias[l] - bias[32 + l]`, which
says the accumulator was right and the bias that reached the second half came from
the neighbouring channel tile. A bare "the numbers disagree" does not distinguish
a store bug from a staging or layout bug; this does.

The values pin down which tile's bias arrives. Which line applies it was not
instrumented, because finding it means changing files in this directory; the
leftover branch of the single-tile store adds its bias before the vector is
rotated (`:105`-`:124`), and both the loop above it and
`store_output_tile_pair_fp16` (`:147`, whose leftover branch is at `:180`)
post-process both halves of a vector, so the two paths are not written the same
way. The pair store is the reference: it is exact on every geometry
`test/sim/conv_runner.cpp` runs.

### What this costs the caller, and how it is avoided here

The emitters pass `mp = 1, np = 2` on every im2col command, so no tile reaches the
single-tile store. The price is VTCM: the kernel asks for `np * kp * 2048` bytes
of weight staging and `mp * kp * 2048` of activation staging
(`:1783`-`:1786`), where `kp` is `kernelY * kernelX * ceil(ic / 32)`. Two channel
tiles is three kp-sized staging tiles where one would be two, and that is what
bounds the widest reduction that fits: `kp <= 1364`, which for a 3x3 window is
4832 input channels. The backend refuses convolutions past that bound instead of
emitting a command whose buffers do not exist, and the bound is arithmetic on the
allocations above and on the 8192 KiB a manager hands out -- it has not been
measured against a device's real budget.

Beyond that bound, no measured cost. The pair store is the path the kernel takes
when it is asked for two channel tiles, and every geometry in
`test/sim/conv_runner.cpp` and every convolution blob in
`test/test_blob_on_sim.py` is bit-exact through it.

### A fix upstream could make

Make the single-tile store's leftover branch apply the current channel tile's bias
to both halves of the vector it writes, the way the pair store's leftover branch
does; nothing here changes if it does, because this checkout uses the pair either
way.

## Build

Built by `htp/CMakeLists.txt` as the per-arch skel, one `.so` per Hexagon arch:

    hexagon-clang++ -mv79 -mhvx -mhvx-length=128b -mhmx -O2 -fPIC -std=c++17
    # kernels in src/dsp/ops/ are C, everything else is C++17
    # link: -L<toolchain>/target/hexagon/lib/v79/G0/pic -lc++ -lc++abi

Compile definitions:

    HTP_OPS_SKEL_ARCH=0x79          required; the kernels gate on the arch
    HTP_OPS_PWL_COMPANDED16=1
    HTP_OPS_PWL_LEARNED8=1          the `learned8` PWL variant, upstream default
    -mhvx-ieee-fp                   additionally required for arch < 79
