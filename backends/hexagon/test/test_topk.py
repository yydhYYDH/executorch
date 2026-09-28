# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""topk on the DSP, and the half of it that does not go there.

`htp_ops_topkv2_k1_fp16` (topk_ops.cc:48) is the one kernel in this backend that
answers two outputs: per row it writes a maximum and the position holding it.
The emitter takes the first and allocates scratch for the second, because the
position the kernel writes is not the position torch writes -- it is the *first*
occurrence of the maximum, while torch's own kernel returns whatever index its
partial sort lands on, which is neither the first nor the last of a tie. These
tests pin that measurement, the rule it justifies, and every argument combination
that must therefore stay on the portable kernels.
"""

import operator

import numpy as np
import pytest
import torch


import blob_interpreter
from blob_interpreter import Arena, execute, read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    TopkSpec,
    topk_is_emittable,
    topk_spec,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export

F16 = torch.float16

#: DSP_OP_TOPKV2_K1_FP16, the only command a topk emits.
_TOPK = 27


class _Topk(torch.nn.Module):
    def __init__(self, k=1, **kwargs) -> None:
        super().__init__()
        self.k = k
        self.kwargs = kwargs

    def forward(self, x):
        return torch.topk(x, self.k, **self.kwargs).values


class _Both(torch.nn.Module):
    """A topk whose positions the graph goes on to read."""

    def forward(self, x):
        values, indices = torch.topk(x, 1)
        return values + indices.to(F16)


class _Both(torch.nn.Module):
    """A topk whose positions the graph goes on to read."""

    def forward(self, x):
        values, indices = torch.topk(x, 1)
        return values + indices.to(F16)


class _Wrapper(torch.nn.Module):
    """A module whose forward is whatever it was handed."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _program(model, x, **export_kwargs):
    return to_edge_transform_and_lower(
        export(model, (x,), **export_kwargs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _lowered(model, x, **export_kwargs):
    """The one delegate's blob and its decoded command stream."""
    program = _program(model, x, **export_kwargs)
    calls = _delegates(program)
    assert len(calls) == 1, f"topk did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    raw = bytes(lowered._processed_bytes)
    _header, commands = read_blob(raw)
    return raw, commands


def _values(raw, x):
    return np.frombuffer(execute(raw, [x.numpy()])[0], dtype=np.float16)


def _scratch_indices(raw, x):
    """The positions the kernel wrote into the slot nothing reads.

    The graph never asks for them, so the only way to see what the kernel would
    have handed a graph that did is to read the arena the command wrote into,
    which is why `execute` is handed the arena to run in.
    """
    header, commands = read_blob(raw)
    arena = Arena(header, raw, "topk")
    execute(raw, [x.numpy()], arena=arena)
    topk = next(command for command in commands if command.type == _TOPK)
    return np.frombuffer(bytes(arena.view(topk.outputs[1])), dtype=np.int32)


def test_a_topk_is_one_command_and_the_row_maximum():
    """The command, its two params, and torch's own values.

    A maximum is one of the row's own elements, so the comparison is an equality
    rather than a tolerance: a kernel walking the wrong row length would answer
    with another row's maximum, and a wrong sign of the row/rowSize pair would
    answer with a transpose's.
    """
    for shape in ((8,), (2, 3, 4), (2, 5, 6), (3, 7), (2, 3, 4, 5), (64,), (1, 8)):
        x = torch.randn(*shape, dtype=F16)
        raw, commands = _lowered(_Topk(), x)
        assert [command.type for command in commands] == [_TOPK]
        assert list(commands[0].params) == [shape[-1], x.numel() // shape[-1]]
        assert _values(raw, x).tobytes() == (
            torch.topk(x, 1).values.reshape(-1).numpy().tobytes()
        )


def test_the_command_takes_one_input_and_two_outputs():
    """The shape `execute_command.cc:653` reads the pointers in.

    The dispatcher takes the values as `mapped_ptrs[inputs->size]` and the
    positions as the pointer after it, so the order is the whole contract: an
    emitter that wrote them the other way round would fill the graph's tensor
    with row indices and answer a plausibly-sized wrong tensor.
    """
    x = torch.randn(2, 3, 4, dtype=F16)
    raw, commands = _lowered(_Topk(), x)
    topk = commands[0]
    assert len(topk.inputs) == 1
    assert len(topk.outputs) == 2
    # The values go to the graph's output slot; the positions to an activation
    # the caller never sees, which is where the kernel's second write lands.
    assert topk.outputs[0].space.name == "OUTPUT"
    assert topk.outputs[0].size == 6 * 2
    assert topk.outputs[1].space.name == "ACTIVATION"
    assert topk.outputs[1].size == 6 * 4


def test_the_position_the_kernel_writes_is_not_the_one_torch_writes():
    """Why the indices output is refused, measured rather than asserted.

    torch's CPU kernel breaks a tie wherever its partial sort leaves it: 1 for a
    row whose first two elements are equal, 2 for a row of four equal values, and
    for 200 rows of quantized values it is neither the first nor the last
    occurrence on 175 of them and the last on 13 more, so it is not the first
    occurrence on 188 in all. This kernel returns the first
    occurrence of the maximum, every time. Handing a graph that position would put
    a number torch never produced into a tensor the model reads, so the rule keeps
    it out; this test is the evidence the rule rests on, and the position is read
    out of the blob's own scratch slot rather than out of a transcription of the
    kernel. The three counts and the identity between them are in
    `test_reason_one_refuses_the_node_and_reason_two_is_a_repairable_hazard`.
    """
    x = torch.tensor([[1.0, 1.0, 0.5, -2.0, 1.0, 0.5, 0.0]], dtype=F16)
    raw, _commands = _lowered(_Topk(), x)
    assert int(_scratch_indices(raw, x)[0]) == 0
    assert int(torch.topk(x, 1).indices[0][0]) == 1
    # The values agree, which is the half that is not a matter of which tie won.
    assert _values(raw, x).tobytes() == torch.topk(x, 1).values.numpy().tobytes()

    equal = torch.full((3, 4), 2.0, dtype=F16)
    raw, _commands = _lowered(_Topk(), equal)
    assert list(map(int, _scratch_indices(raw, equal))) == [0, 0, 0]
    assert list(map(int, torch.topk(equal, 1).indices.reshape(-1))) == [2, 2, 2]

    torch.manual_seed(0)
    quantized = torch.randint(0, 4, (200, 33)).to(F16)
    raw, _commands = _lowered(_Topk(), quantized)
    first = quantized.argmax(-1)
    last = (quantized.shape[1] - 1) - quantized.flip(-1).argmax(-1)
    theirs = torch.topk(quantized, 1).indices.reshape(-1)
    assert not torch.equal(theirs, first) and not torch.equal(theirs, last)
    assert list(map(int, _scratch_indices(raw, quantized))) == list(
        map(int, first.reshape(-1))
    )


def test_a_reader_of_the_indices_keeps_the_whole_node_on_the_host():
    """The two-output rule, on the graph the rule is about.

    Nothing has to be done to the emitter: the node is refused at the gate and
    the values getitem is refused with it, so the topk never becomes a command.
    The add downstream is still delegated, which is why the assertion is about
    which commands the delegate holds rather than how many delegates there are --
    and the values are checked, because a refusal that broke the graph would
    otherwise look like a pass.
    """

    def both(a):
        values, indices = torch.topk(a, 1)
        return values + indices.to(F16)

    x = torch.randn(2, 8, dtype=F16)
    edge = _edge_nodes(_Wrapper(both), x)
    support = HexagonOperatorSupport()
    assert [_name(node.target) for node in edge] == [
        "aten.topk.default",
        "operator.getitem",
        "operator.getitem",
        "dim_order_ops._to_dim_order_copy.default",
        "aten.add.Tensor",
    ]
    # The producer is refused by its own check, and the indices getitem by the
    # int64 gate -- but the *values* getitem is accepted, because a getitem is
    # judged without looking at its producer (`max.dim` is pinned the same way in
    # test_overload_census2). That verdict is not a partition: nothing forms one
    # across a tuple, so the reader stays on the host with the node it reads and
    # the check below is the one that says so.
    assert [support.is_node_supported({}, node) for node in edge] == [
        False,
        True,
        False,
        False,
        True,
    ]
    program = _program(_Wrapper(both), x)
    commands = _commands_of(program)
    assert [_TOPK in [command.type for command in stream] for stream in commands] == [
        False
    ]
    assert torch.equal(_Wrapper(both)(x), both(x))


@pytest.mark.parametrize(
    "k, kwargs",
    [
        (2, {}),  # the kernel holds one element per row, not two
        (3, {}),
        (1, {"dim": 1}),  # a 3-D operand: the reduction axis is not the last
        (1, {"dim": 0}),
        (1, {"dim": -2}),
        (1, {"largest": False}),  # there is no descending walk to select
    ],
)
def test_a_topk_the_kernel_does_not_answer_stays_on_the_host(k, kwargs):
    """Every argument the one command has no slot for, and the refusal for it."""
    x = torch.randn(2, 3, 4, dtype=F16)
    model = _Topk(k, **kwargs)
    program = _program(model, x)
    assert _delegates(program) == [], f"k={k} {kwargs} reached the delegate"
    # The graph still computes: a refusal falls back, it does not rewrite.
    assert model(x).shape == torch.topk(x, k, **kwargs).values.shape


def test_sorted_is_not_part_of_the_rule():
    """With one element there is no order to restore, so both spellings fit.

    torch takes `sorted` as a promise about the k returned values, and k == 1
    keeps it either way; the kernel has no argument for it and needs none.
    """
    x = torch.randn(2, 5, dtype=F16)
    for sorted_ in (True, False):
        model = _Topk(1, sorted=sorted_)
        raw, commands = _lowered(model, x)
        assert [command.type for command in commands] == [_TOPK]
        assert _values(raw, x).tobytes() == model(x).reshape(-1).numpy().tobytes()


def test_a_symbolic_extent_is_refused():
    """Both extents are integers in the command, so neither may be symbolic.

    `rowSize` is the stride between rows and `rows` counts them, and neither is
    patched from the run-time length here: a symbolic row would be a stride no
    command can carry, and a symbolic row count is only right if the run length
    happens to be the one the graph was exported at. The node stays where the
    portable kernel reads the caller's tensor.
    """
    x = torch.randn(4, 8, dtype=F16)
    model = _Topk(1, dim=1)
    for axis in (0, 1):
        dynamic = export(model, (x,), dynamic_shapes={"x": {axis: Dim("d")}})
        program = to_edge_transform_and_lower(
            dynamic,
            partitioner=[HexagonPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()
        assert _delegates(program) == [], f"a symbolic axis {axis} was delegated"


def test_a_non_contiguous_operand_is_refused():
    """The kernel reads one run of `rowSize` elements per row.

    A transposed operand's rows are strided, and the command has no stride to
    carry, so the node stays where the portable kernel can read it.
    """
    x = torch.randn(3, 8, dtype=F16).t()
    assert not x.is_contiguous()
    node = _topk_node(x)
    assert topk_spec(node) is None


def _topk_node(x, k=1, **kwargs):
    """A topk node carrying the value its own predicate reads."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = x
    node = graph.call_function(exir_ops.edge.aten.topk.default, args=(source, k))
    node.meta["val"] = (
        torch.empty(tuple(x.shape[:-1]) + (k,), dtype=F16),
        torch.empty(tuple(x.shape[:-1]) + (k,), dtype=torch.int64),
    )
    return node


def test_topk_spec_reads_the_command_out_of_a_node_that_fits():
    """The one place the params come from, on the shape the kernel walks."""
    assert topk_spec(_topk_node(torch.empty(2, 3, 4, dtype=F16))) == TopkSpec(
        row_size=4, rows=6
    )
    assert topk_spec(_topk_node(torch.empty(8, dtype=F16))) == TopkSpec(
        row_size=8, rows=1
    )


@pytest.mark.parametrize(
    "shape, k",
    [
        ((0, 4), 1),  # an empty row: the kernel returns -1 before it walks
        ((2, 0), 1),
        ((2, 3, 4), 2),  # one element per row is the whole kernel
    ],
)
def test_topk_spec_refuses_what_the_command_cannot_describe(shape, k):
    assert topk_spec(_topk_node(torch.empty(*shape, dtype=F16), k)) is None


def _edge_nodes(model, x):
    """The edge graph's call_function nodes, before any partitioning."""
    program = to_edge(
        export(model, (x,)), compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    return [
        node for node in program.graph_module.graph.nodes if node.op == "call_function"
    ]


def _commands_of(program):
    """The decoded command stream of every delegate in the lowered program."""
    streams = []
    for call in _delegates(program):
        lowered = program.graph_module.get_submodule(call.args[0].target)
        _header, commands = read_blob(bytes(lowered._processed_bytes))
        streams.append(commands)
    return streams


def test_the_predicate_takes_the_getitem_that_reads_the_values():
    """The getitem rule, on the two nodes it admits.

    The pair has to be read off the edge graph rather than off the lowered one:
    inside a delegate both nodes are already past the gate, so a predicate that
    had started refusing them would not show up there at all.
    """
    x = torch.randn(2, 8, dtype=F16)
    calls = _edge_nodes(_Topk(), x)
    support = HexagonOperatorSupport()
    assert [_name(node.target) for node in calls] == [
        "aten.topk.default",
        "operator.getitem",
    ]
    assert [support.is_node_supported({}, node) for node in calls] == [True, True]
    assert len(_delegates(_program(_Topk(), x))) == 1


def _name(target):
    return "operator.getitem" if target is operator.getitem else target.__name__


def test_the_host_interpreter_takes_the_maximum_the_way_the_kernel_does():
    """The interpreter's transcription, against an implementation sharing no code.

    Ties are where a maximum alone is not enough: the value says which of the
    equal elements was kept, so the scratch slot the kernel fills is read too,
    because the position is where a walk transcribed wrongly actually diverges.
    Every maximum here is a nonzero finite value, which is the one place the two
    are expected to agree exactly -- a maximum that is a zero has a sign torch
    and numpy pick differently, which is the caveat max and amax already carry.
    """
    for shape in ((8,), (2, 3, 7), (5, 64), (2, 3, 4, 5)):
        x = torch.randn(*shape, dtype=F16)
        raw, _commands = _lowered(_Topk(), x)
        assert _values(raw, x).tobytes() == (
            torch.topk(x, 1).values.reshape(-1).numpy().tobytes()
        )
    tied = torch.tensor(
        [[1.0, 1.0, 0.5], [3.0, 3.0, 3.0], [-1.5, -1.5, -2.0], [2.0, 1.0, 2.0]],
        dtype=F16,
    )
    raw, _commands = _lowered(_Topk(), tied)
    assert list(map(int, _scratch_indices(raw, tied))) == [0, 0, 0, 0]
    assert _values(raw, tied).tobytes() == (
        torch.topk(tied, 1).values.reshape(-1).numpy().tobytes()
    )


def test_topk_is_the_only_operator_this_file_put_on_the_dsp():
    """A guard on the command type, so a rename cannot silently change the test."""
    assert hexagon_ops.DSP_OP_TOPKV2_K1_FP16 == _TOPK
    assert blob_interpreter.TOPKV2_K1_FP16 == _TOPK


# --- The two stated reasons, resolved one at a time -----------------------------
#
# OP_GAPS section 3 gives the refusal two reasons: the position the kernel writes
# is not the one torch writes, and the node declares int64 where the kernel
# writes one int32 a row. They are separable, they were both asserted rather than
# measured, and they turn out to have different answers -- one is what refuses
# the node, the other is a repair the library already has a command for.
#
# The tests below are the measurements. Each brings a control, because a scan
# that finds nothing and a scan that is broken return the same thing.


def test_reason_one_refuses_the_node_and_reason_two_is_a_repairable_hazard():
    """Which of the two reasons refuses the node, established separately.

    Reason one -- the position -- is real, and it is what the gate refuses on.
    The controls are what make it a measurement rather than an assertion: the
    tied input has to be tied (200 of 200 rows, so the ceiling is not what caps
    the count), torch's answer has to be a genuine maximum at the index it names
    (a wrong index pointing at a non-maximum would make every count below
    meaningless), and the categories have to sum to the row count, because a
    decomposition with no arithmetic on it is a list rather than a census.

    Reason two -- the width -- is real as a fact and is NOT what refuses the
    node. The gate's body, read with the docstring stripped, names no dtype at
    all, so there is no width clause for the reason to be load-bearing on; what
    refuses the node is the reader rule. The width is a hazard the library can
    repair, and the repair is one region, measured in the next test.
    """
    import ast
    import inspect
    import textwrap

    # -- reason one, measured ------------------------------------------------
    torch.manual_seed(0)
    x = torch.randint(0, 4, (200, 33)).to(F16)
    flat = x.reshape(-1, 33)
    first = flat.argmax(-1)
    last = (flat.shape[-1] - 1) - flat.flip(-1).argmax(-1)
    theirs = torch.topk(x, 1).indices.reshape(-1)

    # CONTROL: torch's index must hold a maximum, or the counts below are about
    # something other than tie-breaking.
    assert torch.equal(
        flat.gather(-1, theirs.reshape(-1, 1)).reshape(-1), flat.max(-1).values
    )
    # CONTROL: only a tied row can carry a different answer at all, so the count
    # of tied rows is the ceiling the number below has to respect.
    tied = ((flat == flat.max(-1, keepdim=True).values).sum(-1) > 1).sum()
    assert int(tied) == 200, f"only {int(tied)} of 200 rows are tied"

    is_first = theirs == first
    is_last_only = (theirs == last) & ~is_first
    neither = (theirs != first) & (theirs != last)
    # CONTROL: the three categories partition the rows.
    assert int(is_first.sum()) + int(is_last_only.sum()) + int(neither.sum()) == 200
    # CONTROL: a difference is impossible on an untied row, so this count is a
    # property of ties and not of the harness.
    distinct = torch.randn(400, 64).to(F16)
    assert (
        int(
            (
                torch.topk(distinct, 1).indices.reshape(-1)
                != distinct.argmax(-1).reshape(-1)
            ).sum()
        )
        == 0
    )

    # The figure that refuses the node: how often the kernel's first occurrence
    # is not the position torch returns. 188, not the 175 four files quote --
    # 175 counts the rows that are neither first nor last, which leaves out the
    # 13 rows where torch lands on the last occurrence, and those are just as
    # wrong. Both numbers are kept, and the difference between them is 13.
    assert int((theirs != first).sum()) == 188
    assert int(neither.sum()) == 175
    assert int(is_last_only.sum()) == 13

    # CONTROL on the method: the endpoints are recomputed with an explicit
    # per-row loop, so a vectorised census that agreed with itself because of a
    # wrong axis or a flip of the wrong thing would disagree here instead. The
    # loop is the slow obvious answer and it is what the two figures rest on.
    loop_first, loop_last = [], []
    for row in x.reshape(-1, 33).tolist():
        row_max = max(row)
        loop_first.append(row.index(row_max))
        loop_last.append(len(row) - 1 - row[::-1].index(row_max))
    assert loop_first == list(map(int, first))
    assert loop_last == list(map(int, last))
    assert (
        sum(1 for t, f, l in zip(map(int, theirs), loop_first, loop_last) if t != f)
        == 188
    )

    # The kernel's own answer, read out of the blob's scratch slot rather than
    # out of a transcription of the kernel.
    raw, _commands = _lowered(_Topk(), x)
    assert list(map(int, _scratch_indices(raw, x))) == list(map(int, first))

    # -- reason two, read rather than measured -------------------------------
    # The node does declare int64, so the prose is right about the fact ...
    both = torch.randn(2, 8, dtype=F16)
    program = to_edge(
        export(_Both(), (both,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    topk = next(
        node
        for node in program.graph_module.graph.nodes
        if node.target is exir_ops.edge.aten.topk.default
    )
    assert [out.dtype for out in topk.meta["val"]] == [F16, torch.int64]

    # ... and the gate that refuses it names no width. Read from the parsed
    # source with the docstring stripped, so the prose cannot satisfy this.
    tree = ast.parse(textwrap.dedent(inspect.getsource(topk_is_emittable)))
    body = ast.unparse(ast.Module(body=tree.body[0].body[1:], type_ignores=[]))
    for word in ("int64", "int32", "INT32_BYTES", "dtype", "ScalarType"):
        assert word not in body, f"the gate now names {word}; reason two became real"
    # What is left is the reader rule, and that is what refuses the node.
    assert "topk_getitem" in body and "topk_spec" in body


def test_the_int64_position_width_is_one_raster_blit_region():
    """Reason two is repairable with a command the library already emits.

    The kernel writes one int32 per row and the slot is declared int64, so the
    upper half of every position would be left as the arena held it. A RASTER_BLIT
    region carries a stride per side at every level, which makes the widening a
    single region: the rows on the innermost level, ss2 of 4 for the int32 pitch
    and ds2 of 8 for the int64 one.

    The control is the int32-to-int32 copy at matching strides, and it is what
    separates "the widening is impossible" from "this probe wrote somewhere
    else". An earlier spelling set both the destination ref's offset and the
    region's dst_offset, which writes at 128 while the assertion reads at 64 --
    a row of zeros that looks exactly like a refutation. The region's offset is
    RELATIVE to the ref.
    """
    from executorch.backends.hexagon.serialization import blob as B
    from blob_interpreter import Arena, Command, _run_raster_blit

    activation = int(B.TensorSpace.ACTIVATION)
    rows = 4
    src = np.array([0, 7, 3, 11], dtype=np.int32)
    dst = 64

    def blit(src_stride, dst_stride, dst_bytes):
        arena = Arena.__new__(Arena)
        arena.name = "topk width probe"
        arena.base = {int(space): 0 for space in B.TensorSpace}
        arena.bytes = bytearray(1024)
        arena.bytes[0 : rows * 4] = src.tobytes()
        command = Command(
            type=blob_interpreter.RASTER_BLIT,
            inputs=[B.TensorRef(space=activation, offset=0, size=rows * 4)],
            outputs=[B.TensorRef(space=activation, offset=dst, size=rows * dst_bytes)],
            params=[1, 1, 1, 0, 0, 0, 1, 1, rows, 0, 0, src_stride, 0, 0, dst_stride],
            patch_param=-1,
            patch_input=-1,
            patch_scale=0,
            in_place=0,
        )
        _run_raster_blit(command, command.params, arena)
        return arena

    widened = blit(4, 8, 8)
    assert list(
        map(
            int,
            np.frombuffer(bytes(widened.bytes[dst : dst + rows * 8]), dtype=np.int64),
        )
    ) == src.tolist()
    # CONTROL: the same region at matching strides is a plain copy, and it has to
    # come out exact or the assertion above proves nothing.
    plain = blit(4, 4, 4)
    assert list(
        map(
            int,
            np.frombuffer(bytes(plain.bytes[dst : dst + rows * 4]), dtype=np.int32),
        )
    ) == src.tolist()
    # CONTROL: the region writes no byte past its own destination.
    assert set(bytes(widened.bytes[dst + rows * 8 : dst + rows * 8 + 16])) == {0}


def test_reason_one_is_the_topk_reason_and_not_the_argmax_one():
    """Why the position refusal is topk's and not the arg reductions'.

    A strict first-occurrence walk is what an arg reduction wants, and torch CPU's
    argmax is exactly that: argmax and max(x, -1).indices agree with a strict walk
    on every one of these tied rows, while topk does not. So the position the
    kernel writes is a correct answer for a reduction and the wrong answer for a
    partial sort, which is why the figure cannot be carried from one op to the
    other and why the two are refused for different reasons.

    The control is one input through all four, so a difference in the count is a
    property of the operation rather than of the data.
    """
    torch.manual_seed(0)
    x = torch.randint(0, 4, (200, 33)).to(F16)
    flat = x.reshape(-1, 33)
    first = flat.argmax(-1)

    def differs(indices):
        return int((indices.reshape(-1) != first).sum())

    assert differs(flat.argmax(-1)) == 0
    assert differs(flat.max(-1).indices) == 0
    assert differs(flat.max(-1, keepdim=True).indices) == 0
    assert differs(torch.topk(x, 1).indices) == 188
    # A row of equal values, where every answer is a maximum and only one is the
    # first occurrence: argmax takes it and topk does not.
    equal = torch.full((3, 4), 2.0, dtype=F16)
    assert equal.max(-1).indices.reshape(-1).tolist() == [0, 0, 0]
    assert torch.topk(equal, 1).indices.reshape(-1).tolist() == [2, 2, 2]
    # And the kernel's answer agrees with the reductions, which is the whole
    # reason the two families came apart.
    raw, _commands = _lowered(_Topk(), equal)
    assert list(map(int, _scratch_indices(raw, equal))) == [0, 0, 0]


def test_the_int64_narrowing_is_the_gathers_input_and_not_this_output():
    """What hexagon-int64idx (b204d36) covers, and why it does not reach topk.

    The question this answers is whether that commit unblocks topk's positions.
    It does not, and the reason is a direction rather than an omission: the
    narrowing converts a caller's int64 tensor into an int32 INPUT slot, while
    topk's positions are an OUTPUT the kernel writes. The commit's diff mentions
    topk nowhere.

    Two controls, because "the commit does not mention topk" would also be true
    of a commit that did not exist. The first is the mechanism itself: the
    runtime's narrowing table is sized by the input count, so an output index has
    no entry to narrow into. The second is the positive control -- the gather
    really does take an int64 index through that path today, so the three
    measured numbers are the working case and not an absence of testing.
    """
    import pathlib
    import re

    from executorch.backends.hexagon.partition.hexagon_partitioner import (
        HexagonPartitioner as _Partitioner,
    )
    from executorch.exir import to_edge_transform_and_lower as _lower
    from torch.export import export as _export

    runtime = (
        pathlib.Path(__file__).resolve().parents[1]
        / "runtime"
        / "hexagon_backend.cpp"
    ).read_text()

    # The table is indexed by method INPUT, which is the whole reason.
    assert "delegate->gather_indices.assign(header->n_inputs" in runtime
    assert re.search(r"gather_indices\.assign\(header->n_outputs", runtime) is None
    # And the narrowing clause is reached from the copy-in loop over inputs.
    assert "narrow_indices_to_int32(" in runtime
    assert "ScalarType::Long" in runtime

    # POSITIVE CONTROL: the gather takes an int64 index through that path now,
    # so the assertion above is about a direction and not about a missing commit.
    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(64, 32)

        def forward(self, tokens):
            return self.emb(tokens)

    def gather_commands(dtype):
        torch.manual_seed(0)
        model = Embedding().eval()
        tokens = torch.from_numpy(np.array([3, 63, 0, 17, 5], dtype=dtype))
        program = _lower(
            _export(model, (tokens,)),
            partitioner=[_Partitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()
        calls = _delegates(program)
        assert len(calls) == 1, f"the gather did not delegate for {np.dtype(dtype).name}"
        lowered = program.graph_module.get_submodule(calls[0].args[0].target)
        _header, commands = read_blob(bytes(lowered._processed_bytes))
        return commands[0]

    for dtype, width in ((np.int32, 4), (np.int64, 8)):
        command = gather_commands(dtype)
        assert command.type == hexagon_ops.DSP_OP_SHARED_GATHER
        # The slot is the kernel's own width whatever the caller handed over,
        # and the command says which the caller handed over.
        assert command.params[7] == width
        indices = command.inputs[0]
        assert indices.space is blob_interpreter.B.TensorSpace.INPUT
        assert indices.size == 5 * 4, "the slot is not the kernel's int32 width"

    # And the graph that reads topk's positions still reaches no command at all,
    # with or without the narrowing in the tree.
    program = _program(_Both(), torch.randn(2, 8, dtype=F16))
    assert [
        _TOPK in [command.type for command in stream]
        for stream in _commands_of(program)
    ] == [False]
