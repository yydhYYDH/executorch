# How a wrong result in this backend can look like a right one

This is not a list of missing ops. Every entry is a place where a check, a count
or a file can report success while the thing that matters is wrong, and each one
names the command that catches it. The catalogue and its measurements live in
`.tmp/hex_workstreams/TRUSTCAT-report.md`; this file is the part that has to
outlive that report.

## The three shapes

A quantity that does not move cannot see a change. A quantity that moves the
wrong way hides an improvement. A check that returns what it was asked for can
still be blind to the thing that matters. Most of what follows is one of those
three wearing a different hat.

## A delegate count is not a coverage number

Measured on this tree, through the real partitioner, with the blob decoded:

| graph | delegates | commands the DSP got |
|---|---|---|
| `relu(x)` | 1 | 1 |
| `relu(a+b)` | 1 | 2 |
| `relu(a+b) * relu(a+b)` | 1 | 5 |
| `erf(x)` | 0 | none |

The count is 1 in every supported case. It is constant across a fivefold change
in what the DSP computes, so "delegates: 1" in a report says nothing about
coverage, and grouping more nodes into fewer delegates lowers the count while
raising coverage, which makes an improvement read as a regression. A count of 0
is equally "nothing was supported" and "the partitioner formed no delegate", and
those differ by an order of magnitude of work.

The count is also not replaceable by the delegate inner graph. For
`relu(erf(x))` with `erf` unsupported, the delegate holds `aten_erf_default`
among its nodes while the command stream is a single `DSP_OP_UNARY` and the erf
runs on the CPU inside the delegate module. The node set over-reports; only the
decoded command list is about the DSP.

```
PYTHONPATH=src:backends/hexagon/test python backends/hexagon/scripts/trust_probe.py
```

## A green simulator run can mean the simulator never ran

`hexagon_sim.run` raises the same `Unavailable` for "no SDK" and for "the
simulator did not run the binary", and a fixture that catches it and skips turns
a DSP that computed nothing into a green suite. On this tree, with the simulator
healthy, `test_blob_on_sim.py` is 21 passed and 0 skipped; with the simulator
unable to load the runner it is 13 passed and 8 skipped. Both exit 0. Read the
skip reasons, or the count:

```
export HEXAGON_SDK_ROOT=/home/yydh/hexagon-tutorial/tools/hexagon-sdk
pytest -q -rsx backends/hexagon/test/test_blob_on_sim.py
```

A skip whose reason mentions "hexagon-sim did not run the runner" is a failure
wearing a skip. This backend does have a real simulator tier: 174 of the 1498
collected node ids depend on a `hexagon_sim.run`. The other 1324 are host-side
and use a numpy model.

## A refusal asserted with a bare support object asserts the signature instead

`HexagonOperatorSupport()` is given the program signature so it can tell a
constant weight from a method input. Built without it, `data_names` is empty and
the weight-reading gates refuse everything, at the geometries the backend
supports as much as at the ones it refuses. Measured: a supported `convolution`
and a supported `embedding` are both refused by the bare object and both accepted
by the correctly built one.

So `assert not HexagonOperatorSupport().is_node_supported({}, node)` is green in
the tree where the gate works and green in the tree where the gate is gone.
The suite builds the bare object 28 times and the parameterised one 9. The
control is what keeps this honest: on the weightless families the bare object
still accepts `relu`, `add`, `sum` and `silu` and still refuses `expm1` and
`erf`, so the blindness is scoped rather than global.

Pair every negative with a positive at the same geometry, as
`test_refused_targets.py` does at lines 451 and 464.

## A scan that reports zero and a broken scan report the same thing

The raster-blit prefix scan returned zero five times before its control existed,
and every zero was a bug in the scan. Later, with the control in place, the scan
still reported zero on a tree it had previously read, because a commit rewrote
the fixture into local names the recogniser no longer matched. The control still
passed. Both answers were correct and together they meant nothing.

```
PYTHONPATH=src python backends/hexagon/scripts/trust_probe.py ; echo $?
```

The control is a commit whose fixture is known to hold the stale form, and the
probe refuses to report a tree result unless the control still finds its five
entries. A scan with no control is a list.

## The generated table cannot see an error it shares with its generator

`gen_op_support.py --check` compares `render()` with `OP_SUPPORT.md`. Both come
from the same rows, so a row that is wrong is wrong on both sides of the
comparison and the check passes. Agreement is not correctness, and the only
oracle that survives is a real lowering decoded to a command list.

```
PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py --check ; echo $?
```

`OP_SUPPORT.md` is generated and guarded. `OP_GAPS.md` is written by hand and
nothing watches it, so after any change to what a path supports, grep the claim
across the backend rather than regenerating and trusting.

## On the device, an output file proves that a file exists

The arena is zero-filled and a refused or unwritten command leaves that fill
behind, which the suite itself asserts: a refused case is expected to answer with
all zeros. A file that is present and the right length is therefore what a run
in which the command group never wrote also looks like, and when the correct
answer is legitimately all zeros the two are byte-identical.

```
./executor_runner --model_path X.pte --output_file out --print_output none ; echo RUNNER_EXIT=$?
ls -l out-0.bin
cmp out-0.bin ref-0.bin && echo SAME || echo DIFFERENT
```

The exit code is not the verdict, the length is not the verdict, and the printed
output is not the verdict. Read the binary, at the dtype the model produces.

## Index narrowing happens on the host and only for int64

`narrow_indices_to_int32` checks every index value against the vocabulary the
command carries and refuses one outside it. It is guarded on the caller being
`Long`, so an int32 index tensor is copied in unchecked and the kernel clears
the row rather than raising. A device run that verified an int32 gather therefore
says nothing about an int64 one, and the divergence lives in code that never
executes on the DSP.

```
grep -n narrow_indices_to_int32 backends/hexagon/runtime/hexagon_backend.cpp
grep -rn narrow_indices backends/hexagon/test/
```

The second command is the measurement: no host test names the C++ function, so
nothing in the host tier runs it. The model in `blob_interpreter.py` mirrors it,
which means a simulator test on int64 indices exercises the model of the
narrowing, never the narrowing.

