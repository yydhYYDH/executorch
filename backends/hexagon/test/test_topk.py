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
    for 200 rows of quantized values it is the first occurrence on some and the
    last on others and neither on 175 of them. This kernel returns the first
    occurrence of the maximum, every time. Handing a graph that position would put
    a number torch never produced into a tensor the model reads, so the rule keeps
    it out; this test is the evidence the rule rests on, and the position is read
    out of the blob's own scratch slot rather than out of a transcription of the
    kernel.
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
