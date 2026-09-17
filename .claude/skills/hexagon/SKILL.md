---
name: hexagon
description: Build, test, or develop the Hexagon DSP backend. Use when working on backends/hexagon/, building the skel or the Android runner (backends/hexagon/build.sh), pushing to a device, or debugging a Hexagon run that fails to load, aborts, or returns wrong numbers. Trigger on "Hexagon backend", "hexagon skel", "execute_command_group", "0x8000040d", "0x00000047", "0x00000049", "DelegateInvalidCompatibility", "fastrpc", "HTP", "HVX/HMX", "DSP crash", or "DSP output doesn't match CPU".
---

# Hexagon DSP Backend

Runs delegated subgraphs on the Qualcomm Hexagon DSP over FastRPC. No QNN
anywhere in the path: the DSP side is the MNN op library vendored under
`third-party/mnn-htp-ops`, and both halves are generated from `htp_ops.idl` and
the command schema by `qaic`, so the wire format cannot drift.

## The two artifacts

| | built by | runs on | loaded by |
|---|---|---|---|
| **skel** `libhex-htp-skel-v<arch>.so` | Hexagon toolchain | DSP | `ADSP_LIBRARY_PATH` |
| **runner** `executor_runner` | Android NDK | phone CPU | `LD_LIBRARY_PATH` |

`backends/hexagon/build.sh` builds both: `skel`, `runner`, `all`, or `deploy`.
`backends/hexagon/RUNBOOK.md` has the manual equivalent and the on-device run.

## Reach for a tiny model first

Almost every bug in this backend reproduces on a model small enough to export in
seconds. The 4.2 GB LLM here costs 25 minutes of export plus 9 minutes of upload
per iteration; a 68 KB hand-written model that exercises the same op costs one
second and no measurable memory. Both go through the same emitter, the same skel
kernel, the same FastRPC path and the same dispatcher, so the tiny model answers
the same question — and when it fails its failure is diagnosable, where the big
one's is a single line naming a delegate index and an error code.

Build the tiny model to isolate one thing: one op, one shape, one command. This
paid off three times in one session — `tiny_sdpa.pte` (68 KB) to ask whether the
attention emitter fix works, a six-node fp32 island to prove the narrow/widen
path is bit-exact, and a hand-built blob to reproduce a shape bug in two seconds.
The island is the instructive one: it isolated the narrow/widen contract that the
big model's error could never have named, and gave an upper bound on the error
(one fp16 ulp) that no amount of staring at the 4.2 GB model would have produced.

Use the big model only for the claim that needs the big model — end-to-end text,
or a real speed number — and make that run count by fixing every known defect
first, so one export carries them all. Two 25-minute exports that could have been
one is the expensive version of the same mistake, and running them concurrently
on a 13 GB host will OOM the machine.

Corollary when delegating: a subtask phrased as "verify the emitter fix" does not
need the big model, and must not be handed exclusive use of the export slot. Ask
for the tiny reproduction instead; grant the big export to whoever can spend it
once, on the step that genuinely requires it.
## Build flags that are load-bearing

Three settings are not preferences; each one fails in a way that does not point
at the build.

**1. The skel must be optimized.** `skel/CMakeLists.txt` sets `-O2`. The kernels
are one function per operator kind with every branch of that kind inline;
unoptimized, one fp16 branch chain wants a 14848-byte frame
(`allocframe(#0x3a00)`), which overflows the RPC thread's stack. The fault is in
the prologue, so it fires before the kernel reads a byte and surfaces as
`0x8000040d` from `remote_handle64_invoke`, taking the user PD with it.
`CMAKE_BUILD_TYPE` is empty for this build, so nothing else adds optimization.
Verify with:

```bash
grep -m1 CXX_FLAGS <skelbuild>/CMakeFiles/hex-htp-v79.dir/flags.make   # want ... -mhmx -O2
```

Note `-fstack-usage` reports 512 bytes for the same function; only the
disassembly is truthful.

**2. The toolchain file selects the compiler.** Without
`-DCMAKE_TOOLCHAIN_FILE=$HEXAGON_SDK_ROOT/build/cmake/hexagon_toolchain.cmake`,
CMake picks the host compiler and rejects `-mhvx` / `-mhmx`.

**3. Pin the flatbuffers root.** `-DHEXAGON_FLATBUFFERS_ROOT=$ET/third-party/flatbuffers`.
The default can resolve to an older copy elsewhere on the machine, and the
generated `Command_generated.h` then fails to compile on a three-argument
`VerifyField`.

For the runner, `EXECUTORCH_BUILD_HEXAGON=ON` needs `--whole-archive` or the
linker drops the static self-registration and every model fails with
`Backend HexagonBackend is not registered` (`0x20`).

## Debugging a failed run

**stderr is nearly empty on purpose, and that is the trap.** Almost all runtime
logging is `ET_LOG`, which on Android goes to logcat, not stderr. A failed run
can print a single line and abort, which reads as "it died immediately" when the
real error is one logcat line you have not looked at yet.

```bash
./executor_runner --model_path m.pte > run.log 2>&1; echo "EXIT=$?"
logcat -d | grep -i ExecuTorch | tail -20      # this is where the error is
```

`EXIT=134` is SIGABRT, and the tombstone carries the abort message, which names
the failing assert and an ExecuTorch `Error` code.

### Error codes seen so far

| code | meaning | where to look |
|---|---|---|
| `0x20` | backend not registered | runner linked without `--whole-archive` |
| `0x30` | `DelegateInvalidCompatibility` — `init()` rejected the blob | the `hexagon: ...` ET_LOG line names which check; 12 sites return this |
| `0x47` (71) | command group entries misaligned | entry base must be `group_ptr + 8` |
| `0x49` (73) | malformed command flatbuffer | `FlatBufferBuilder` reuse |
| `0x8000040d` | DSP faulted in `execute_command_group` | skel built without `-O2`; check `flags.make` |
| `0x0c` | RPC call exceeded the armed RPC timeout (`AEE_EEXPIRED`) | `rpc timeout armed` line at startup; only after 10 s, so a *known-good* model hitting it means the timeout is too small |

### A hang is silent by default

FastRPC ships with **no RPC timeout**, so an invoke that never returns produces no
error, no log line, and no exit — the runner just sits there and only a reboot
clears whatever state is behind it. The driver now arms a 10 s per-PD timeout in
`HexagonDriver::Open()` (`remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, …)`
with `remote_rpc_notif_register_v2.timeout`, ms), so the same hang comes back as
`AEE_EEXPIRED` (`0x0c`) plus a `USERPD_TIMEOUT` notification. `backends/hexagon/RPC_TIMEOUT_EXPERIMENT.md`
has the mechanism, the platform gate (the SDK only promises it "after Kaanapali"),
and the three-step judgment package.

Framework operations are the exception: the SDK says a stuck mmap/unmap/pd-kill
triggers SSR after **5 s**. That is why some stalls self-heal and a plain invoke
hang never did — not a difference in severity, just a different missing timer.

**Do not read numbers out of `--print_output`.** It wraps the listing, and
parsing it with a regex silently misaligns: a comparison that looked like "the
first 100 elements match and the rest are garbage" was purely a print-format
artifact. Use `--output_file` and read the raw tensor:

```bash
./executor_runner --model_path m.pte --output_file devout --print_output none
# devout-0.bin is numel * sizeof(dtype)
```

## Four bugs worth knowing by shape

All four ran clean on the host and only failed on a device, and all four are in
the runtime rather than the emitters. Three crash. The fourth produces plausible
but wrong numbers with a clean exit, which is the worst failure mode here.

**Blob bounds check counting arena budgets.** Only weights and activations are in
the blob; input and output sizes are arena budgets the runtime allocates
against. Adding them overstates the blob and rejects every delegate that has an
input:

```cpp
// wrong
sections_total = weights_bytes + inputs_bytes + activations_bytes + outputs_bytes;
// right
sections_total = weights_bytes + activations_bytes;
```

**Command group entry base.** The DSP reads `int* commands = (int*)(group_ptr + 8)`,
so entries start at int index **2**. Writing from index 1 shifts every
`(fd, offset)` pair by four bytes:

```cpp
group[2 + i * 3 + 0] = arena_fd;
group[2 + i * 3 + 1] = offset + arena_bias;
group[2 + i * 3 + 2] = 0;   // 0 is intentional: the DSP invalidates then reads
```

When in doubt about these contracts, read the DSP side
(`third-party/mnn-htp-ops/src/dsp/execute_command.cc`) rather than trusting the
host writer.

**FlatBufferBuilder cleared too late, in two places.** `Clear()` must come
before the Tensor offsets are created, not after. Clearing later leaves
`CreateCommand` holding offsets into a buffer that no longer exists:

```cpp
builder.Clear();                                    // first
auto t = MakeTensor(builder, ...);
builder.Finish(DSPCOMMAND::CreateCommand(builder, ..., builder.CreateVector(t), ...));
```

Fixing the command loop is not enough. The sync group builds its offsets the
same way and had the same late `Clear()`, and because the sync group is what
tells the DSP which buffers to invalidate on the way in and flush on the way
out, corrupting it does **not** fault: the model loads, runs, exits 0, and
returns numbers that look plausible and are wrong. With the sync group fixed,
a 1024-element output matched the host reference to a maximum absolute
difference of 6.3e-3 with none of them off by more than 0.01. When a run
succeeds but the numbers are off, suspect cache maintenance first.

Grep for every `builder.Clear()` and confirm each one precedes that builder's
first `MakeTensor`.

## Coverage is easy to misread

A delegate census counts *delegates*, not operators. An op the partitioner
declines does not become another delegate — it silently stays on the portable
CPU kernel, so the census looks clean while the work is elsewhere. Count the
operators that did not get taken:

```python
for n in edge._edge_programs["forward"].graph.nodes:
    if "executorch_call_delegate" in str(n.target):
        print("hexagon", n)
```

On a small LLM-shaped model this showed 33 Hexagon subgraphs beside 477
operators left on the CPU, of which 272 were layout ops (`view_copy`,
`permute_copy`, `getitem`, `_to_dim_order_copy`, `slice_copy`). Those are cheap
on the CPU but they fragment the graph and cost the DSP larger fused subgraphs.

## Contracts that fail silently

Op parameters are positional and unchecked: a wrong order or a wrong tensor
count produces wrong numbers, not an error. The table is in the README. Two
worth repeating: absent operands need `fd = -1` rather than size 0 (a zero-size
tensor still maps to a live address and is read as data), and layer norm reads
epsilon through a `float*` cast over the int params, so the emitter bit-casts
rather than rounds.
