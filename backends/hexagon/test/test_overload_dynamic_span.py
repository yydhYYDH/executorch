# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The overloads wired beside a wired sibling inherit that sibling's patches.

`efe2a64` gave `_emit_reduction` the three `DynamicPatch` records its span params
need: a span that holds the run-time length is written as the exported bound plus
the patch that recomputes it. The reduction overloads `mean.default`,
`max.default`, `amax.default` and `mean.dim(dim=())` arrived separately, so what
keeps them covered is that each reaches `_emit_reduction` rather than emitting a
command of its own -- they are thin delegations, and the patches are the
callee's. That is a property of the wiring and not of the overloads, and
`test_reduction_dynamic_span.py` exercises only `sum` and `mean.dim(None)`, so a
later overload that stopped delegating would leave a run-time length unpatched
with nothing to say so.

The unary family is the same shape of thing one level down: relu, hardtanh and
`x ** 2` are new arrivals on DSP_OP_UNARY, whose param 0 is the element count,
and they are covered because that count is `_patch_dynamic_numel`'s in
`_emit_clamp_bounds`/`_unary`.

So each row pins, for one overload, that the delegate's node target is the target
that names it and that the command it produced carries the patch the run-time
length needs. A static graph is the control -- it must carry none, because
patching a param the graph had already settled would be as wrong as leaving a
run-time one unpatched. The host interpreter cannot model a patch, so what is read
here is the record the runtime applies.
"""

import struct

import numpy as np
import pytest
import torch


import blob_interpreter
from blob_interpreter import read_blob
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export

#: The two command types these overloads land on, and the two unary op types.
_REDUCTION = 29
_UNARY = 4
_SUM = 1
_MAXIMUM = 2
_MEAN = 3
_CLAMP = 15
_SQUARE = 9
_FP16_BYTES = 2

#: The trailer's magic, and the seven ints after it: version, two unused, the
#: longest length the export declared, then the patch and layout counts.
_TRAILER_MAGIC = 0x44594E48

#: The traced length and the longest one the symbol allows, so that the two
#: numbers a param could carry are distinguishable. The operand is [1, t, 8].
_TRACED = 6
_UPPER = 16
_ROW = 8

#: The reduction span as [outside][reduce][inside], over that operand: every dim
#: in one span, one row's worth of outside, and the last row in it.
_WHOLE = [1, _UPPER * _ROW, 1]
_MIDDLE = [1, _UPPER, _ROW]
_LAST = [_UPPER, _ROW, 1]

#: A param that is the symbol's bound stands for the traced length instead; the
#: two values it takes over the operand above, so a static row can be written as
#: the dynamic one with the bound replaced.
_AT_TRACED = {_UPPER: _TRACED, _UPPER * _ROW: _TRACED * _ROW}


class _Op(torch.nn.Module):
    """A module whose forward is one plain function of one tensor.

    Called from inside a real `forward` rather than assigned to one, so that the
    signature `torch.export` inspects is this method's and not a builtin's --
    `torch.mean` has none, which export reports as a ValueError.
    """

    def __init__(self, forward) -> None:
        super().__init__()
        self._forward = forward

    def forward(self, x):
        return self._forward(x)


def _sequence_shape():
    return {"x": {1: Dim("tokens", min=1, max=_UPPER)}}


def _reduction_params(span, kind):
    return [*span, kind, _FP16_BYTES]


def _unary_params(count, op_type, *bounds):
    return [count, op_type, _FP16_BYTES, *bounds]


def _lowered(model, x, dynamic):
    """The blob, and the edge targets that reached the delegate, by name."""
    program = to_edge_transform_and_lower(
        export(model, (x,), dynamic_shapes=dynamic),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the graph did not reach one delegate: {calls}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    return bytes(lowered._processed_bytes), [
        node.target.__name__
        for node in lowered._original_exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _command_of_type(blob, command_type):
    """That one command, and the patch records the trailer aims at it."""
    _, commands = read_blob(blob)
    found = [
        (index, command)
        for index, command in enumerate(commands)
        if command.type == command_type
    ]
    assert len(found) == 1, f"expected one command of type {command_type}: {found}"
    index, command = found[0]
    offset = blob.find(struct.pack("<I", _TRAILER_MAGIC))
    if offset < 0:
        return list(command.params), []
    header = struct.unpack_from("<7I", blob, offset)
    assert header[4] == _UPPER, f"the trailer's longest length is {header[4]}"
    records = [
        struct.unpack_from("<4i", blob, offset + 28 + record * 16)
        for record in range(header[5])
    ]
    return (
        list(command.params),
        [tuple(record[1:]) for record in records if record[0] == index],
    )


def _at_the_traced_length(params, spans):
    """The leading span params over the traced length: the bound is the shape.

    Only the leading span products are rewritten. What follows them is a kind, an
    element width, or a unary's clamp bounds -- fp16 bit patterns, which are not
    counts and must not be rescaled.
    """
    head = [_AT_TRACED.get(param, param) for param in params[:spans]]
    return head + list(params[spans:])


# --------------------------------------------------------------------------
# Rows: (label, forward, edge target, span, kind, the patch it must carry)
# --------------------------------------------------------------------------

_REDUCTIONS = [
    ("mean.default", torch.mean, "aten.mean.default", _WHOLE, _MEAN, (1, _ROW, 0)),
    ("max.default", torch.max, "aten.max.default", _WHOLE, _MAXIMUM, (1, _ROW, 0)),
    ("amax.default", torch.amax, "aten.amax.default", _WHOLE, _MAXIMUM, (1, _ROW, 0)),
    (
        "amax.default dim=()",
        lambda a: torch.amax(a, dim=()),
        "aten.amax.default",
        _WHOLE,
        _MAXIMUM,
        (1, _ROW, 0),
    ),
    (
        "mean.dim dim=()",
        lambda a: torch.mean(a, dim=()),
        "aten.mean.dim",
        _WHOLE,
        _MEAN,
        (1, _ROW, 0),
    ),
    (
        "mean.dim dim=None",
        lambda a: torch.mean(a, dim=None),
        "aten.mean.dim",
        _WHOLE,
        _MEAN,
        (1, _ROW, 0),
    ),
    (
        "mean.dim dim=1",
        lambda a: torch.mean(a, dim=1),
        "aten.mean.dim",
        _MIDDLE,
        _MEAN,
        (1, 1, 0),
    ),
    (
        "mean.dim dim=-1",
        lambda a: torch.mean(a, dim=-1),
        "aten.mean.dim",
        _LAST,
        _MEAN,
        (0, 1, 0),
    ),
    (
        "sum.dim_IntList dim=()",
        lambda a: torch.sum(a, dim=()),
        "aten.sum.dim_IntList",
        _WHOLE,
        _SUM,
        (1, _ROW, 0),
    ),
    (
        "sum.dim_IntList dim=None",
        lambda a: torch.sum(a, dim=None),
        "aten.sum.dim_IntList",
        _WHOLE,
        _SUM,
        (1, _ROW, 0),
    ),
    (
        "sum.dim_IntList dim=1",
        lambda a: torch.sum(a, dim=1),
        "aten.sum.dim_IntList",
        _MIDDLE,
        _SUM,
        (1, 1, 0),
    ),
    (
        "sum.dim_IntList dim=-1",
        lambda a: torch.sum(a, dim=-1),
        "aten.sum.dim_IntList",
        _LAST,
        _SUM,
        (0, 1, 0),
    ),
]

# --------------------------------------------------------------------------
# Rows: (label, forward, edge target, params, the patch it must carry)
# --------------------------------------------------------------------------

_UNARIES = [
    (
        "relu.default",
        torch.relu,
        "aten.relu.default",
        _unary_params(_UPPER * _ROW, _CLAMP, 0, 31744),
        (0, _ROW, 0),
    ),
    (
        "relu6 as hardtanh(0, 6)",
        torch.nn.functional.relu6,
        "aten.hardtanh.default",
        _unary_params(_UPPER * _ROW, _CLAMP, 0, 17920),
        (0, _ROW, 0),
    ),
    (
        "hardtanh.default",
        torch.nn.functional.hardtanh,
        "aten.hardtanh.default",
        _unary_params(_UPPER * _ROW, _CLAMP, 48128, 15360),
        (0, _ROW, 0),
    ),
    (
        "clamp.default",
        lambda a: torch.clamp(a, -1, 1),
        "aten.clamp.default",
        _unary_params(_UPPER * _ROW, _CLAMP, 48128, 15360),
        (0, _ROW, 0),
    ),
    (
        "pow.Tensor_Scalar x ** 2",
        lambda a: a**2,
        "aten.pow.Tensor_Scalar",
        _unary_params(_UPPER * _ROW, _SQUARE),
        (0, _ROW, 0),
    ),
]


#: The interpreter cases, which are about a value rather than a command.
_ANSWERS = (
    ("mean.default", torch.mean),
    ("max.default", torch.max),
    ("relu.default", torch.relu),
    ("pow.Tensor_Scalar x ** 2", lambda a: a**2),
)


def _ids(rows):
    return [row[0] for row in rows]


# One case per overload rather than a loop over the table: a row that stops being
# patched is then named by its own failure instead of stopping the rows after it
# from being checked at all.


@pytest.mark.parametrize("case", _REDUCTIONS, ids=_ids(_REDUCTIONS))
def test_a_wired_reduction_overload_carries_its_span_patches(case):
    """One reduction target, over a shape the run-time length runs through.

    The target is asserted by name, so a lost `EMITTERS` entry fails here as a
    missing delegate rather than as an unpatched command.
    """
    label, forward, target, span, kind, patch = case
    x = torch.randn(1, _TRACED, _ROW, dtype=torch.float16)
    blob, targets = _lowered(_Op(forward), x, _sequence_shape())
    assert targets == [target], f"{label} reached {targets}"
    got, patches = _command_of_type(blob, _REDUCTION)
    assert got == _reduction_params(span, kind), f"{label} emitted {got}"
    assert patches == [patch], f"{label} patched {patches}"


@pytest.mark.parametrize("case", _UNARIES, ids=_ids(_UNARIES))
def test_a_wired_unary_overload_carries_its_element_count_patch(case):
    """The same property for the unary family, whose param 0 is the count."""
    label, forward, target, params, patch = case
    x = torch.randn(1, _TRACED, _ROW, dtype=torch.float16)
    blob, targets = _lowered(_Op(forward), x, _sequence_shape())
    assert targets == [target], f"{label} reached {targets}"
    got, patches = _command_of_type(blob, _UNARY)
    assert got == params, f"{label} emitted {got}"
    assert patches == [patch], f"{label} patched {patches}"


@pytest.mark.parametrize("case", _REDUCTIONS, ids=_ids(_REDUCTIONS))
def test_a_static_reduction_patches_nothing(case):
    """The control: with no symbol in the graph there is nothing to recompute.

    A bound written as the shape and a patch aimed at it are a pair, so a patch
    left behind on a static param would scale a number the graph had settled.
    """
    label, forward, _, span, kind, _ = case
    x = torch.randn(1, _TRACED, _ROW, dtype=torch.float16)
    blob, _ = _lowered(_Op(forward), x, None)
    got, patches = _command_of_type(blob, _REDUCTION)
    assert patches == [], f"{label} patched a static shape: {patches}"
    want = _at_the_traced_length(_reduction_params(span, kind), 3)
    assert got == want, f"{label} emitted {got}"


@pytest.mark.parametrize("case", _UNARIES, ids=_ids(_UNARIES))
def test_a_static_unary_patches_nothing(case):
    """The same control for the family whose param 0 is the element count."""
    label, forward, _, params, _ = case
    x = torch.randn(1, _TRACED, _ROW, dtype=torch.float16)
    blob, _ = _lowered(_Op(forward), x, None)
    got, patches = _command_of_type(blob, _UNARY)
    assert patches == [], f"{label} patched a static shape: {patches}"
    assert got == _at_the_traced_length(params, 1), f"{label} emitted {got}"


@pytest.mark.parametrize("case", _ANSWERS, ids=_ids(_ANSWERS))
def test_the_interpreter_answers_at_the_bound_the_command_was_written_for(case):
    """The patch has no host model, so the bound is the length the blob runs at."""
    label, forward = case
    x = torch.randn(1, _UPPER, _ROW, dtype=torch.float16)
    blob, _ = _lowered(_Op(forward), x, _sequence_shape())
    got = np.frombuffer(
        blob_interpreter.execute(blob, [x.numpy()])[0], dtype=np.float16
    )
    want = forward(x).half().numpy().reshape(-1)
    assert np.array_equal(got, want), f"{label} answered {got}, torch {want}"
