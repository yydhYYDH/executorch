# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A comparison produces a bool, and no kernel here can read one.

`aten.gt.Tensor` and `aten.lt.Tensor` are refused for the plain reason that the
dtype gate admits fp16 and fp32 and a comparison's result is bool: the kernel
would write one element per value where the arena holds two, and the element
count is decided by the graph. What a comparison's *output* then does is the
other half of the question -- `x + (y > 0)` is an ordinary thing to write -- and
it is answered by the promotion torch puts in the way: the bool is converted to
the arena's width by a cast, and `dim_order_keeps_the_bytes` refuses that cast
for a non-fp16/fp32 source, so the comparison and its conversion stay on the
portable kernels and the element-wise op is handed a fp16 tensor.
"""

import os
import pathlib
import sys

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    operand_dtypes_are_readable,
    where_is_emittable,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_SELECT, and the tensor space a get_attr lands in.
_DSP_OP_SELECT = 26
_WEIGHTS = 0


class _Comparison(torch.nn.Module):
    """`x + (y > 0)`, which torch promotes for us."""

    def forward(self, x, y):
        return x + (y > 0.0)


class _ComparisonWithCast(torch.nn.Module):
    """The same thing with the conversion written out."""

    def forward(self, x, y):
        return x + (y > 0.0).to(torch.float16)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def test_the_comparison_and_its_conversion_stay_outside_the_delegate():
    """The bool never reaches the arena; the element-wise op still does.

    Both spellings lower the same way, which is the point: whichever the graph
    author writes, the bool is converted outside the delegate and the op that
    consumes it is handed fp16. The delegate still holds the add, so this is not
    a refusal of the whole graph.
    """
    x = torch.randn(4, 8, dtype=torch.float16)
    y = torch.randn(4, 8, dtype=torch.float16)
    for model in (_Comparison(), _ComparisonWithCast()):
        program = _program(model, (x, y))
        outer = {
            str(node.target)
            for node in program.graph_module.graph.nodes
            if node.op == "call_function"
        }
        assert any("gt" in target for target in outer), outer
        assert any("dim_order_copy" in target for target in outer), outer
        calls = _delegates(program)
        assert (
            len(calls) == 1
        ), f"the element-wise op did not reach the delegate: {calls}"
        lowered = program.graph_module.get_submodule(calls[0].args[0].target)
        blob = bytes(lowered._processed_bytes)
        _, commands = read_blob(blob)
        converted = (y > 0.0).to(torch.float16)
        got = np.frombuffer(
            execute(blob, [x.numpy(), converted.numpy()])[0], dtype=np.float16
        )
        expected = x + converted
        assert got.tobytes() == expected.numpy().reshape(-1).tobytes()


def _node_with_operand(dtype, size=(4, 8)):
    """An add whose second operand carries `dtype`."""
    graph = torch.fx.Graph()
    lhs = graph.placeholder("x")
    lhs.meta["val"] = torch.empty(*size, dtype=torch.float16)
    rhs = graph.placeholder("y")
    rhs.meta["val"] = torch.empty(*size, dtype=dtype)
    node = graph.call_function(exir_ops.edge.aten.add.Tensor, args=(lhs, rhs))
    node.meta["val"] = torch.empty(*size, dtype=torch.float16)
    return node


def test_a_bool_operand_is_refused_at_the_gate():
    """The guard, on the node it reads.

    The exporter does not produce this shape today -- it converts the bool first,
    as the test above shows -- but a hand-built graph can, and a kernel reading
    it would take the neighbouring slot's bytes as the value: no error, wrong
    numbers. The control is the same node with a fp16 operand, which is taken.
    """
    assert not operand_dtypes_are_readable(_node_with_operand(torch.bool))
    assert not HexagonOperatorSupport().is_node_supported(
        {}, _node_with_operand(torch.bool)
    )
    assert operand_dtypes_are_readable(_node_with_operand(torch.float16))


class _Where(torch.nn.Module):
    def forward(self, cond, a, b):
        return torch.where(cond, a, b)


class _WhereWithConstant(torch.nn.Module):
    """The condition as a buffer rather than an argument."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mask",
            torch.tensor([[True, False, True, False], [False, True, False, True]]),
        )

    def forward(self, a, b):
        return torch.where(self.mask, a, b)


def _select_case(model, args):
    """The one delegate's blob, its command, and the interpreter's answer."""
    program = _program(model, tuple(args))
    calls = _delegates(program)
    assert len(calls) == 1, f"the select did not reach the delegate: {calls}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [
        _DSP_OP_SELECT
    ], f"the delegate is not one select: {[command.type for command in commands]}"
    return blob, commands[0]


def test_the_select_takes_its_condition_at_one_byte_per_element():
    """The chain the kernel's guard was the open question about.

    A comparison's result is bool and no kernel here writes one, so the condition
    reaches the delegate as an argument and is the only operand in the blob that
    is one byte per element. Three separate things have to hold for that to be
    right, and the blob states all three: the command declares condBytes 1, the
    condition's slot is one byte per element rather than the two the runtime
    would otherwise copy, and the kernel reads the flag and not its neighbour's
    low half. The last is what running it settles, and `test_blob_on_sim.py`'s
    teeth case is where the encoding is falsified.
    """
    conditions = torch.tensor(
        [[True, False, True, False, True], [False, True, False, True, False]]
    )
    on = torch.arange(1, 11, dtype=torch.float16).reshape(2, 5)
    off = -on
    blob, command = _select_case(_Where(), (conditions, on, off))

    assert list(command.params[:6]) == [10, 10, 10, 10, 2, 1], list(command.params)
    assert (
        command.inputs[0].size == 10
    ), f"the condition's slot is {command.inputs[0].size} bytes for ten flags"
    assert [ref.size for ref in command.inputs[1:]] == [20, 20]

    got = np.frombuffer(
        execute(blob, [conditions.numpy(), on.numpy(), off.numpy()])[0],
        dtype=np.float16,
    )
    expected = torch.where(conditions, on, off)
    assert (
        got.tobytes() == expected.numpy().reshape(-1).tobytes()
    ), f"the select answered {got.tolist()} for {expected.flatten().tolist()}"


def test_the_select_reads_a_bool_buffer_at_that_width_too():
    """A condition that is a weight, which is the other way one arrives.

    A registered buffer is a get_attr and lands in the weights section, so its
    width is the emitter's rather than the runtime's -- and the blanket rule that
    a weight is fp16 would refuse it for being one byte. The exemption is the
    same one the operand gate makes and is on the same target, so this is where
    the two are held together: an fp16 constant on the same node stays refused.
    """
    on = torch.arange(1, 9, dtype=torch.float16).reshape(2, 4)
    off = -on
    blob, command = _select_case(_WhereWithConstant(), (on, off))

    assert list(command.params[:6]) == [8, 8, 8, 8, 2, 1], list(command.params)
    assert (
        command.inputs[0].space == _WEIGHTS
    ), f"the buffer did not land in the weights section: {command.inputs[0].space}"
    assert command.inputs[0].size == 8, (
        f"the buffer's slot is {command.inputs[0].size} bytes for eight flags, so "
        "the blob holds it at the arena's width rather than its own"
    )
    got = np.frombuffer(execute(blob, [on.numpy(), off.numpy()])[0], dtype=np.float16)
    expected = torch.where(_WhereWithConstant().mask, on, off)
    assert got.tobytes() == expected.numpy().reshape(-1).tobytes()


def _where_support(model, args):
    program = _program(model, tuple(args))
    return len(_delegates(program)), program


def test_a_where_the_kernel_cannot_walk_stays_portable():
    """The refusals, each of which would otherwise be a wrong number.

    Two of them are shapes the command cannot describe, and both have a
    kernel-side reason rather than a policy one: the command's own guard admits a
    condition that is the whole output or a single element, so a broadcast
    condition is out, and a bool *result* has no writer here at all because the
    arena holds two bytes per element. In both cases the node stays on the
    portable kernels, which is the whole graph and not an error.

    The third is the condition's dtype, and it is unreachable through `export`:
    `aten.where.self`'s own schema requires a bool predicate, so no graph the
    exporter produces can carry anything else. The gate still reads it, because
    a hand-built graph can, and because the kernel would not make torch's `!= 0`
    test over a two-byte flag in the first place.
    """
    on = torch.arange(1, 9, dtype=torch.float16).reshape(2, 4)
    off = -on

    class _BroadcastCondition(torch.nn.Module):
        """The condition is (2, 1) and the values are (2, 4)."""

        def forward(self, cond, a, b):
            return torch.where(cond, a, b)

    columns = torch.tensor([[True], [False]])
    assert (
        _where_support(_BroadcastCondition(), (columns, on, off))[0] == 0
    ), "a condition the command's own guard rejects reached the delegate"

    class _BoolResult(torch.nn.Module):
        def forward(self, cond):
            return torch.where(cond, cond, cond)

    flags = torch.tensor([[True, False, True, False], [False, True, False, True]])
    assert (
        _where_support(_BoolResult(), (flags,))[0] == 0
    ), "a bool result reached a kernel whose arena holds two bytes per element"

    assert where_is_emittable(_node_where(torch.bool))
    assert not where_is_emittable(
        _node_where(torch.float16)
    ), "a fp16 condition would be read one byte at a time by the kernel"

    # And the control: the shape all three are variations of is taken, so the
    # refusals above are about their own operand and not about `where` in
    # general.
    assert _where_support(_Where(), (flags, on, off))[0] == 1


def _node_where(cond_dtype):
    """A `where.self` node with a condition of this dtype, built by hand."""
    graph = torch.fx.Graph()
    args = []
    for name, dtype in (
        ("cond", cond_dtype),
        ("a", torch.float16),
        ("b", torch.float16),
    ):
        placeholder = graph.placeholder(name)
        placeholder.meta["val"] = torch.empty(2, 4, dtype=dtype)
        args.append(placeholder)
    node = graph.call_function(exir_ops.edge.aten.where.self, args=tuple(args))
    node.meta["val"] = torch.empty(2, 4, dtype=torch.float16)
    return node
