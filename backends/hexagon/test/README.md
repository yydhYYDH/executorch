# Hexagon Backend Tests

Three things live here beyond the ordinary pytest files. `test_blob_on_sim.py`,
`test_hexagon_sim.py`, `test_conv_sim.py` and `test_gemv_on_sim.py` build a real
blob, compile the vendored `mnn-htp-ops` DSP sources for v79 with `hexagon_sim.py`,
and run them under `hexagon-sim` against the same fixtures. `blob_interpreter.py`
is a host model of the same command stream, so every case is compared three ways:
torch, the host model, and the kernel as `hexagon-sim` runs it -- never on
silicon, which no case in this directory has touched.

Run them the way the rest of the backend is run:

```
pytest backends/hexagon/test
```

They skip, with a reason, when the Hexagon SDK and `hexagon-sim` are not on the
machine; a skip is not a pass.

**That "never on silicon" clause still holds, and it is worth keeping explicit
rather than letting it be read as stale.** The device run `../README.md` now
records was made with `executor_runner` and a `.pte`; no case in this directory
ran on a phone, and `pytest` still has no path that touches one. A device result
elsewhere is not a substitute for a case here, because the two do not look at the
same thing. Where the same command has now been seen both ways it agreed -- a
sign-bit NaN is answered `0.0` on silicon exactly as `hexagon-sim` answers it --
and where it has not, the reason was a case reaching a different function rather
than a disagreement: this directory's `amax` NaN cases (`_AmaxAt`, tags `BJ` and
`BL`) reduce the axis that leaves `inside = 100 >= 64`, so the kernel takes
`htp_ops_reduce_fp16_inside_vector_range`, where a device run of a
`torch.amax(x, dim=1)` on a two-dimensional tensor has `inside = 1` and takes
`htp_ops_reduce_max_fp16_inside1_hvx` instead. Read a green case here as a
statement about the function it happens to reach.

**The scaffold these cases run on has since been checked against the phone
directly, and that is the answer the paragraph above was working around.** Nine
blobs, one per kernel family, taken out of the `.pte` the phone ran and handed to
`hexagon_sim.run` unchanged, agree with the phone bit for bit: the reduction over
a run-time-patched span, `amax` over a 100-wide fold with a NaN in it and
without, the flat binary add, the row gather, the max pool, the depthwise
convolution and the 3x3 im2col convolution. The convolution is the one worth
naming, because it goes through HMX and `--mhmx=3` is a second implementation of
that unit rather than a recompilation of it, and because the case was run with
fp16 weights drawn from a normal distribution: it differs from torch at 1152 of
4096 outputs by up to one ULP and agrees with the phone at all 4096. So on those
seven paths a green case here is evidence about the DSP, not about a model of it.
The caveat above still governs which function that evidence is about, and nothing
outside those seven paths is upgraded.

## Simulator buffers must be 128-byte aligned, explicitly

The runners hand the kernels raw byte arenas. An HVX vector load or store wants
a 128-byte-aligned address, and the kernels address their operands that way, so
an unaligned buffer is not a performance problem -- **it is wrong data**. Keep
this in mind whenever you touch a runner in `sim/` -- `blob_runner.cpp`,
`ops_runner.cpp`, `conv_runner.cpp` or `gemv_runner.cpp` -- or add a fixture:

- **The symptom is a wrong number, not a crash.** Reads and writes land at the
  wrong addresses, so a case comes back with plausible-looking garbage: a
  pass-through range returning a period-4 pattern, the last eight lanes of a
  128-element tensor zeroed, one case in the file failing where its neighbours
  pass. Every one of those reads exactly like a kernel bug or a misread layout,
  and the first version of the simulator work here reported one as such. Before
  blaming a kernel, print `(uintptr_t)arena % 128`.
- **The alignment is decided by the shared object, not by the declaration.** The
  arena was a plain `static uint8_t` array sized by a value the fixture header
  computes, and a byte array's alignment is whatever the link layout gives it --
  so **adding or removing one fixture can flip it**, in a diff that says nothing
  about memory. That is the dangerous part: a case could pass today, fail
  tomorrow, and the change that moved it was a test someone added elsewhere in
  the file.
- **So it is declared, and enforced at compile time.** All four runners write
  `__attribute__((aligned(128)))` on the arena and on every operand buffer a
  kernel reads or writes, and each declaration is followed by a `static_assert`
  on `__alignof__(buffer) == 128` that names the buffer -- spelled out in
  `blob_runner.cpp`, `ops_runner.cpp` and `gemv_runner.cpp`, and wrapped in
  `VECTOR_ALIGNED` and `REQUIRE_VECTOR_ALIGNMENT` in `conv_runner.cpp`. Delete
  the attribute and the build stops -- the compiler reports `expression evaluates
  to '1 == 128'` -- instead of a suite that quietly goes back to depending on
  link layout. The assertion has to be written that way: a `static_assert` on the
  address is not a constant expression, and `alignof` on the array's *type* is
  its element's alignment, so neither of those forms compiles or reports. Such a
  failure stays visible because `hexagon_sim.BuildFailed` is not
  `hexagon_sim.Unavailable`: a machine without the SDK skips these tests, a
  translation unit that does not compile fails them. Do the same for anything new
  that a kernel dereferences. The generated fixture arrays are the exception, and
  only because the runner `memcpy`s them into the arena: a scratch runner that
  hands its own array straight to a kernel -- the way
  `htp_ops_matmul_q4a16_gemv_i8` was driven while investigating the quantized GEMV
  entries -- has to align that array itself. The one thing that must not happen
  is a passing suite that depends on the compiler happening to place an array
  where the HVX wants it.

## What the simulator cannot run

- **Worker-pool kernels, permanently.** QuRT under `hexagon-sim` refuses to start
  a worker pool: `qurt_cb_fwk_worker_init` returns -4 and the runtime aborts,
  which kills the whole run rather than the one case. `DSP_OP_FLASH_ATTN` (18) --
  the entry point behind `llama.custom_sdpa` -- is such a kernel, so it is kept
  out of the fixture entirely and nothing about it can be settled here. Kernel
  selection in the dispatcher (`htp_ops_unary_pick_task_count`) also reads a
  count while holding a worker context, so keep single-command fixtures small:
  below 2048 fp16 elements it picks one task and stays on the caller's thread.
- **VTCM is not acquired for you.** Kernels that size themselves from
  `vtcm_manager_get_vtcm_size()` -- the two quantized GEMV entries, among others
  -- see zero unless the delegate's own setup runs first: this harness's `main()`
  is not the delegate, so call `vtcm_manager_setup()` and `vtcm_manager_acquire()`
  before driving them, otherwise they return -1 and write nothing. With that, the
  simulator reports the same 8 MiB a device has.
- **Everything a functional model cannot see.** Timing, cache coherence, real
  HMX behaviour, and the size of the workspace an allocation actually hands over
  are outside what a run here can establish, however green the comparison is.
