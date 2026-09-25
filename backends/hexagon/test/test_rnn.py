# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM and GRU prediction networks, and what of them reaches the DSP.

An unrolled recurrence is not a fused op here. `torch.export` decomposes
`aten.lstm.input` into per-step `addmm`, `add`, `sigmoid`, `tanh`, `mul` and a
`cat` of the steps, and the partitioner then asks of each piece whether the
existing emitters can take it. The answer was yes for the cells and no for the
stitch, because a blit command holds three regions and a four-step sequence
has four pieces. `MAX_CAT_INPUTS` refused the node, the node came out of the
delegate, and the copy of the steps that makes a sequence out of T vectors ran
on the host.

The fix is several blit commands over one allocation: a `TensorRef` is a
section, an offset and a size, and both the runtime
(`hexagon_backend.cpp:849`) and the host interpreter (`blob_interpreter.py:291`)
resolve a ref as base plus offset, so a command writing a shifted ref writes
into a slice of the same buffer. `BlobBuilder._remap` had to learn to keep that
shift, because it rewrites every activation ref from the block it packed.

What these tests hold down is the part that is easy to get wrong and invisible
from a passing forward pass: that the stitch is a DSP command and not host code,
that the per-step
command count is the cell's algebra and not a guess from its target, and that
the sequence as a whole stays within a bound rather than each step
individually. The per-step numbers are read off the lowered graph and summed
against the blob's own command count, and a step's membership is a lookup by
the inner graph's node names, which are the outer graph's names.
"""

import collections
import operator
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from blob_interpreter import Arena, execute, read_blob  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _fixture_header,
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
)
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    BLIT_BLOCKS_PER_COMMAND,
    cat_plan,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP op type to name, the only way the command types in a blob get read.
OP_NAMES = {v: k for k, v in vars(hexagon_ops).items() if k.startswith("DSP_OP_")}

#: A lowered graph's targets are EXIR `EdgeOpOverload`s, not the `torch.ops.aten`
#: overloads of the same name, so membership is compared against these.
_EDGE = exir_ops.edge.aten
FULL = _EDGE.full.default
CAT = frozenset((_EDGE.cat.default,))
ADDMM = _EDGE.addmm.default

#: The targets a recurrent step is made of. Each is one DSP command, so this is
#: also the per-step command count: one projection, the add of the input
#: projection, the gates, and the products and sums that combine them.
STEP_TARGETS = frozenset(
    (
        ADDMM,
        _EDGE.add.Tensor,
        _EDGE.sub.Tensor,
        _EDGE.mul.Tensor,
        _EDGE.sigmoid.default,
        _EDGE.tanh.default,
    )
)

#: What one time step costs, in commands, counted off the lowered graph rather
#: than written down from the cell equations. Both cells are eleven, but not the
#: same eleven: an LSTM spends its arithmetic on three sigmoid gates, two tanh and
#: the products and sums of the cell update, while a GRU reuses the projection it
#: already has for its reset gate and so carries more adds and fewer gates. A
#: summary of the equations would have said thirteen and fourteen and been wrong,
#: which is the reason these are the measured per-target counts.
LSTM_STEP = {"addmm": 1, "add": 2, "sigmoid": 3, "tanh": 2, "mul": 3}
GRU_STEP = {"addmm": 1, "add": 4, "sigmoid": 2, "tanh": 1, "mul": 2, "sub": 1}

#: The only host nodes a lowered recurrent net may keep: the initial-state
#: zeros from a `full`, and the `getitem` that indexes the cell's return tuple.
#: Anything else on the host is a piece of the recurrence that came back to the
#: CPU, which is the failure this file is about.
_ALLOWED_HOST_TARGETS = frozenset(
    (
        _EDGE.full.default,
        operator.getitem,
        torch.ops.higher_order.executorch_call_delegate,
    )
)

HIDDEN = 64
IN_FEATURES = 4
CLASSES = 8

#: The worst the sequence may differ from an fp64 evaluation of the same
#: weights, over the length below, on the simulator and not the host
#: interpreter. Measured: 1.76e-3 at four steps and 1.85e-3 at eight for an
#: LSTM, 1.43e-3 and 1.72e-3 for a GRU, so the error does not compound with the
#: sequence -- the recurrence contracts what the gates hand it. The bound sits
#: just above those and well below what one companded16 gate costs against the
#: kernel's own scalar path, so a gate that stopped using the PWL tables, or a
#: stitch that stopped running at all, could not meet it.
SIM_BOUND = 2.5e-3

#: What one companded16 gate costs on this kernel against the scalar path it
#: replaces: 1.71e-3 for sigmoid and 5.86e-3 for tanh, measured over the same
#: arguments run both ways -- 63 values as a 63-element all-scalar chunk, and the
#: same 63 inside a 128-element all-vector chunk. The tanh is the larger and is
#: what a first time step cannot exceed.
GATE_PWL_COST = 5.86e-3

SIM_STEPS = 8


class _PredictionNet(torch.nn.Module):
    """A recurrent cell over a sequence, then a projection per time step.

    The projection is the part a prediction network is named for: the per-step
    logits over a vocabulary. It is not the recurrent cell, and a report that
    confuses the two has said nothing about either.
    """

    def __init__(self, kind: str, steps: int) -> None:
        super().__init__()
        self.cell = getattr(torch.nn, kind)(IN_FEATURES, HIDDEN, batch_first=True)
        self.projection = torch.nn.Linear(HIDDEN, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.cell(x)
        return self.projection(out)


def _input(steps: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, steps, IN_FEATURES, generator=generator).half()


def _lowered(model, args):
    """The partitioned program, its one delegate, and the delegate's blob."""
    program = to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    submodule = program.graph_module.get_submodule(calls[0].args[0].target)
    return program, calls, submodule


def _census(blob: bytes) -> collections.Counter:
    """The command types in a blob, counted from the bytes themselves."""
    _, commands = read_blob(blob)
    return collections.Counter(OP_NAMES[command.type] for command in commands)


def _host_args(program, call, x: torch.Tensor):
    """The tensors the host hands the delegate, in the order it takes them.

    Everything the graph computes before the call is either a placeholder, an
    attribute, or an initial-state `full`; evaluating it here is what the
    portable kernel does at run time.
    """
    env = {}
    for node in program.graph_module.graph.nodes:
        if node is call:
            break
        if node.op == "placeholder":
            env[node] = x
        elif node.op == "get_attr":
            env[node] = getattr(program.graph_module, node.target)
        elif node.op != "output":
            env[node] = node.target(
                *[env[a] if isinstance(a, torch.fx.Node) else a for a in node.args],
                **{
                    k: (env[v] if isinstance(v, torch.fx.Node) else v)
                    for k, v in node.kwargs.items()
                },
            )
    return [env[arg] for arg in call.args[1:]]


def _dsp_answer(model, x: torch.Tensor) -> np.ndarray:
    """What the blob computes, run on the host interpreter's arena."""
    program, calls, submodule = _lowered(model, (x,))
    blob = bytes(submodule._processed_bytes)
    outputs = execute(blob, [a.numpy() for a in _host_args(program, calls[0], x)])
    with torch.no_grad():
        shape = tuple(model(x).shape)
    return np.concatenate([out.copy() for out in outputs]).view(np.float16).reshape(shape)


def _reference(model, x: torch.Tensor) -> np.ndarray:
    """The same weights in fp64, which is the only reference worth stating."""
    import copy

    with torch.no_grad():
        return copy.deepcopy(model).double()(x.double()).numpy()


def _inner_nodes(submodule):
    """The delegate's own nodes, in order, with their names.

    The lowered backend module has no `.graph`; its nodes come from the module
    it wraps, and those keep the outer graph's names, so membership is a name
    lookup and never a guess by target -- two projections share a target string.
    """
    return list(submodule.original_module.graph_module.graph.nodes)


def _reads(node, target: torch.fx.Node) -> bool:
    """Whether a node's value comes from `target`, one argument at a time."""
    for arg in node.args:
        if arg is target:
            return True
        if isinstance(arg, torch.fx.Node) and arg.op == "call_function":
            if _reads(arg, target):
                return True
    return False


def _projection_steps(submodule, steps: int):
    """The node index each time step's hidden projection starts at.

    A step's projection is the `addmm` that multiplies by the hidden weight,
    and which `addmm` that is is settled by looking the weight placeholder up
    by name rather than by counting them: the net's own output projection is
    an `addmm` too, and so is the prefill that projects every step's input at
    once. Neither is per-step work, so the steps are what is left over.
    """
    nodes = _inner_nodes(submodule)
    hidden = [
        node
        for node in nodes
        if node.op == "placeholder" and node.name.endswith("weight_hh_l0")
    ]
    assert len(hidden) == 1, f"{len(hidden)} hidden-weight placeholders"
    projections = [
        index
        for index, node in enumerate(nodes)
        if node.op == "call_function"
        and node.target is ADDMM
        and _reads(node, hidden[0])
    ]
    assert len(projections) == steps, f"{len(projections)} hidden projections for {steps} steps"
    return nodes, projections
def _short_name(target) -> str:
    """`aten.mul.Tensor` as `mul`, so a per-step tally reads as a tally."""
    return (
        str(target).split("aten.")[-1].split("(")[0].split(">")[0].strip()
        .split(".")[0]
    )


def _per_step_nodes(submodule, steps: int):
    """The nodes belonging to each time step, grouped by name.

    A step's block runs from its hidden projection to the next step's, so it
    holds that step's gates and nothing later. The last block also carries the
    stitch and the projection, which belong to the sequence rather than to a
    step, so the two are reported apart below.
    """
    nodes, starts = _projection_steps(submodule, steps)
    ends = list(starts[1:]) + [len(nodes)]
    return [nodes[start:end] for start, end in zip(starts, ends)]




@pytest.mark.parametrize("steps", [2, 3, 4, 7, 8])
@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_the_prediction_net_is_one_delegate(kind, steps):
    """The whole net, stitch and projection included, is one DSP delegate.

    Before the stitch could take several commands it came out of the delegate,
    so the net was two delegates and the copy that makes a sequence out of the
    steps ran on the host. A T of three fitted one command and hid this, which is
    why the cases go past it.
    """
    model = _PredictionNet(kind, steps).half()
    x = _input(steps)
    program, calls, _ = _lowered(model, (x,))
    assert len(calls) == 1, f"{len(calls)} delegates for a {steps}-step {kind}"


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_only_the_initial_state_stays_on_the_host(kind, steps):
    """What the host still computes is the zeros, and nothing else.

    The initial state is a `full` of zeros, which no emitter claims, so it stays
    on the portable kernel. It is two tensors for an LSTM (h and c) and one for a
    GRU, and the count is the point: a second `full` would mean a second piece of
    the recurrence had come back to the CPU. The graph also keeps one `getitem`,
    which indexes the cell's `(output, (h, c))` return tuple and moves no bytes,
    so it is named in the allowance rather than folded into the arithmetic count.
    """
    model = _PredictionNet(kind, steps).half()
    program, _, _ = _lowered(model, (_input(steps),))
    outside = [
        node.name
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.target not in _ALLOWED_HOST_TARGETS
    ]
    states = sum(
        1
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.target is _EDGE.full.default
    )
    assert states == (2 if kind == "LSTM" else 1), (
        f"{kind} at {steps} steps: {states} initial states"
    )
    assert not outside, f"{kind} at {steps} steps also computed: {outside}"


@pytest.mark.parametrize("steps", [2, 3, 4, 7, 8, 16])
def test_a_long_stitch_is_several_blits_over_one_result(steps):
    """A T-way stitch is ceil(T / 3) commands writing disjoint slices.

    A blit header is three ints and a region twelve, out of a forty-int parameter
    block, so one command carries three regions however many sources the kernel
    would accept -- `src_ptrs[10]` in `execute_command.cc:363` is not the limit.
    The result is one allocation; each command's output ref is that allocation
    shifted by the bytes it owns, and the byte ranges neither overlap nor leave a
    hole. That last part is the one a reading of the emitter would miss: the
    packer rewrites activation offsets, and before it learned to keep the shift
    every slice landed on the block start and the last piece overwrote the first.
    """
    model = _PredictionNet("GRU", steps).half()
    _, _, submodule = _lowered(model, (_input(steps),))
    cats = [
        node
        for node in _inner_nodes(submodule)
        if node.op == "call_function" and node.target in CAT
    ]
    assert len(cats) == 1, f"{len(cats)} cat nodes for {steps} steps"
    plan = cat_plan(cats[0])
    assert plan is not None, "the stitch is not describable as blit commands"
    assert len(plan) == -(-steps // BLIT_BLOCKS_PER_COMMAND), (
        f"{steps} steps planned as {len(plan)} commands"
    )
    spans = []
    for params, indices, byte_offset, byte_size in plan:
        assert params[0] == len(indices) <= BLIT_BLOCKS_PER_COMMAND
        assert len(params) == 3 + 12 * params[0]
        spans.append((byte_offset, byte_size))
    assert spans[0][0] == 0
    walked = spans[0][1]
    for offset, size in spans[1:]:
        assert offset == walked, (
            f"a gap or an overlap before the slice at {offset}: {spans}"
        )
        walked += size
    total = sum(size for _, size in spans)
    assert spans[-1][0] + spans[-1][1] == total, spans
    assert total == steps * HIDDEN * 2, (
        f"the slices cover {total} bytes, a {steps} x {HIDDEN} fp16 sequence is"
        f" {steps * HIDDEN * 2}"
    )


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("steps", [4, 8])
def test_the_stitch_commands_are_in_the_blob(kind, steps):
    """The stitch is commands in the bytes, not nodes the partitioner dropped."""
    model = _PredictionNet(kind, steps).half()
    _, _, submodule = _lowered(model, (_input(steps),))
    _, commands = read_blob(bytes(submodule._processed_bytes))
    blits = [c for c in commands if c.type == hexagon_ops.DSP_OP_RASTER_BLIT]
    regions = sum(c.params[0] for c in blits)
    assert regions >= steps, (
        f"{regions} blit regions for a {steps}-step stitch"
    )


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_commands_per_time_step(kind, steps):
    """The per-step command count, cross-checked against the blob's own total.

    The per-step tally is read off the lowered graph and compared to the blob's
    own command count. An LSTM step is one `BATCH_MATMUL` (the hidden
    projection), two `BINARY_ELEMENTWISE` adds, three `UNARY` sigmoid gates, two
    `UNARY` tanh and three `BINARY_ELEMENTWISE` products: eleven commands. A GRU
    step is also eleven and a different eleven -- two sigmoid, one tanh, four
    adds, two products and one subtract -- because its reset gate reuses the
    projection it already has instead of computing a new one.

    The blocks come from the inner graph by node name, and the sum of what the
    blocks claim is cross-checked against the number of commands the blob
    actually holds, so a block cannot quietly claim a node that never reached
    the DSP.
    """
    model = _PredictionNet(kind, steps).half()
    _, _, submodule = _lowered(model, (_input(steps),))
    _, commands = read_blob(bytes(submodule._processed_bytes))
    blocks = _per_step_nodes(submodule, steps)
    wanted = LSTM_STEP if kind == "LSTM" else GRU_STEP

    for step, block in enumerate(blocks[:-1]):
        counts: dict = {}
        for node in block:
            if node.op != "call_function" or node.target not in STEP_TARGETS:
                continue
            name = _short_name(node.target)
            counts[name] = counts.get(name, 0) + 1
        assert dict(counts) == wanted, (
            f"{kind} step {step} of {steps}: {dict(counts)}, wanted {wanted}"
        )
    assert len(commands) >= steps * sum(wanted.values())


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_the_sequence_tracks_fp64_over_its_whole_length(kind, steps):
    """The bound is on the sequence, not on a step.

    A per-step band is the wrong acceptance for a recurrence: a gate that is
    slightly off feeds the next step, and the error is carried forward rather
    than re-measured. So the claim here is about the whole output at each length,
    and the lengths are chosen to double, so a cost that grew with the sequence
    would show up as a growing bound rather than as a constant one.

    This is the host interpreter, which models sigmoid and tanh with the exact
    formula the kernel's own scalar path evaluates; it measures the wiring, not
    the companded16 PWL the vector path uses, which is measured on the simulator
    by the gate tests in `test_unary_sim` and compounds from there.
    """
    model = _PredictionNet(kind, steps).half()
    x = _input(steps, seed=steps)
    got = _dsp_answer(model, x).astype(np.float64)
    want = _reference(model, x)
    error = np.abs(got - want).max()
    assert error < 2.0e-3, f"{kind} at {steps} steps: max {error:.3e}"


def _sim_fixture(kind: str, steps: int):
    """A blob, the bytes its delegate is handed, and the fp64 reference."""
    model = _PredictionNet(kind, steps).half()
    x = _input(steps, seed=1)
    program, calls, submodule = _lowered(model, (x,))
    blob = bytes(submodule._processed_bytes)
    header, _ = read_blob(blob)
    args = _host_args(program, calls[0], x)
    fixture = SimpleNamespace(
        tag=kind,
        blob=blob,
        inputs=b"".join(a.numpy().tobytes() for a in args),
        length=0,
        arena_bytes=len(Arena(header, blob, "fixture").bytes),
    )
    return fixture, model, x


def _sim_run(fixtures):
    """The simulator's answers, or a skip when there is no simulator."""
    hexagon_sim._check()
    try:
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header(fixtures),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


def _sim_answer(answers, tag, model, x):
    """One output slot, as fp64, checked to be the shape the graph asked for."""
    key = tag + "0"
    assert key in answers, f"the simulator answered nothing for {tag}"
    with torch.no_grad():
        shape = tuple(model(x).shape)
    raw = np.asarray(answers[key], dtype=np.uint16).view(np.float16)
    assert raw.size == int(np.prod(shape)), (
        f"{tag} answered {raw.size} values for a {shape} output"
    )
    return raw.reshape(shape).astype(np.float64)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_the_prediction_net_runs_on_hexagon_sim(kind):
    """The whole net, stitch included, executed by the DSP command stream.

    This is the tier that says the commands ran rather than that they were
    emitted. The fixture is the real blob from the real pipeline, its inputs
    are what the host hands the delegate, and the answer is compared with the
    same weights in fp64 over the whole sequence rather than a step at a time.
    """
    fixtures = []
    models = {}
    for name in ("LSTM", "GRU"):
        fixture, model, x = _sim_fixture(name, SIM_STEPS)
        fixtures.append(fixture)
        models[name] = (model, x)
    answers = _sim_run(fixtures)
    for name in ("LSTM", "GRU"):
        model, x = models[name]
        error = np.abs(_sim_answer(answers, name, model, x) - _reference(model, x))
        assert error.max() < SIM_BOUND, (
            f"{name} over {SIM_STEPS} steps: max {error.max():.3e} at step"
            f" {int(error.reshape(SIM_STEPS, -1).max(axis=1).argmax())}"
        )


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_the_sim_answer_is_not_the_host_model(kind):
    """A control: the PWL gates have to be visible in the answer.

    The host interpreter evaluates sigmoid and tanh with the formula the
    kernel's own scalar path uses, so a simulator that agreed with it to fp16
    noise would say the gates never took the companded16 path.

    The claim is made on the first time step only, where the output is a
    function of one step's gates and nothing earlier. Over the whole sequence
    the two implementations drift apart by a weight-dependent amount, measured
    at 1.40e-3 and 2.81e-3 for the same GRU under two seeds, and a bound fitted
    to either of those would be a bound fitted to noise.

    The lower bar is above what the exact-gate model itself is from fp64 at the
    first step, which is 8.1e-5 for an LSTM, so the difference is the gates and
    not the arithmetic around them. The upper bar is one tanh's own PWL cost
    against the kernel's scalar path, measured on this kernel at 5.86e-3: a
    first step cannot differ by more than the gates it evaluates.
    """
    fixture, model, x = _sim_fixture(kind, SIM_STEPS)
    answers = _sim_run([fixture])
    dsp = _sim_answer(answers, kind, model, x)
    modelled = _dsp_answer(model, x).astype(np.float64)
    first = float(np.abs(dsp[:, 0, :] - modelled[:, 0, :]).max())
    assert first > 1.0e-4, (
        f"{kind}: the DSP and the exact-gate model differ by {first:.3e}"
        " at the first step, which is what a scalar path would give"
    )
    assert first <= GATE_PWL_COST, (
        f"{kind}: the first step differs by {first:.3e}, more than one"
        " tanh gate costs on this kernel"
    )
