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

Nothing here has been relicensed. The two modifications listed under "Local
modifications" are build-environment fixes and do not change behavior.

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

Two, both forced by the build environment rather than by logic:

1. `src/dsp/vtcm_mgr.cc` — one `#include "flatbuffers/flatbuffers.h"` removed.
   The file never references the `flatbuffers::` namespace, and adding the
   dependency for a dead include would couple the skel to flatbuffers.
2. `schema/current/Command_generated.h` is checked in rather than generated.
   Upstream generates it with `flatc` from `schema/Command.fbs`; the checked-in
   header is byte-identical to upstream's, and `flatc` is not part of the
   Hexagon SDK. `schema/current/Command.fbs` is kept as the source of truth, so
   the header can be regenerated whenever flatc is available.

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
