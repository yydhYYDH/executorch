# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A reduction whose span holds the run-time length recomputes that length.

The three REDUCTION span params are products of the operand's shape, and a
product built from a dynamic shape is the exported *bound* plus a patch the
runtime applies -- `_patch_dynamic_product` is that pairing, and every emitter
that lets a `torch.SymInt` through uses it. A reduction that skipped it
described the arena's allocation instead of the run-time buffer: a `sum` over
the sequence, or a whole-tensor `mean`, folded in whatever the arena holds past
the run-time length and answered with it. The host interpreter cannot model a
patch, so what is pinned here is the record the runtime applies, the way
`test_dynamic_blob.py` pins the matmul's.
"""

import os
import pathlib
import struct
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

import blob_interpreter  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

#: DSP_OP_REDUCTION, and the kind each case below selects.
_REDUCTION = 29
_SUM = 1
_MEAN = 3

#: The trailer's magic, and the seven ints after it: version, two unused, the
#: longest length the export declared, then the patch and layout counts.
_TRAILER_MAGIC = 0x44594E48

#: The traced length and the longest one the symbol allows, so that the two
#: numbers a param could carry are distinguishable.
_TRACED = 6
_UPPER = 16
_FP16_BYTES = 2


class _Reduce(torch.nn.Module):
    def __init__(self, kind, dim) -> None:
        super().__init__()
        self.kind = kind
        self.dim = dim

    def forward(self, x):
        if self.kind == "mean":
            return torch.mean(x, dim=self.dim)
        return torch.sum(x, dim=self.dim)


class _ReduceAStaticWindow(torch.nn.Module):
    """A reduction whose operand is the same shape at every run-time length."""

    def forward(self, x):
        return torch.sum(x[:, :4, :], dim=-1)


def _sequence_shape(shape, minimum):
    return {"x": {1: Dim("tokens", min=minimum, max=_UPPER)}}


def _lowered(model, x, dynamic):
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
    assert len(calls) == 1, f"the reduction did not reach one delegate: {calls}"
    return bytes(
        program.graph_module.get_submodule(calls[0].args[0].target)._processed_bytes
    )


def _reduction_and_patches(blob):
    """The single reduction command, its params, and the patches aimed at it."""
    _, commands = read_blob(blob)
    reduction = [
        (index, command)
        for index, command in enumerate(commands)
        if command.type == _REDUCTION
    ]
    assert len(reduction) == 1, f"expected one reduction, got {reduction}"
    index, command = reduction[0]
    offset = blob.find(struct.pack("<I", _TRAILER_MAGIC))
    assert offset >= 0, "a dynamic graph emits a trailer"
    header = struct.unpack_from("<7I", blob, offset)
    assert header[4] == _UPPER, f"the trailer's longest length is {header[4]}"
    records = [
        struct.unpack_from("<4i", blob, offset + 28 + record * 16)
        for record in range(header[5])
    ]
    return (
        list(command.params),
        [record[1:] for record in records if record[0] == index],
        index,
    )


def test_a_span_over_the_run_time_length_is_patched():
    """The params that hold the length are recomputed for it.

    `sum` over the last axis makes the *outside* count the length; over the
    middle axis the length is the whole reduce span; over every axis it is that
    span divided by the row width, so the patch's scale is what is left. Each
    param is the exported bound in the blob, and each carries the patch that
    replaces it.
    """
    x = torch.randn(1, _TRACED, 8, dtype=torch.float16)
    for dim, params, patch in (
        (-1, [_UPPER, 8, 1, _SUM, _FP16_BYTES], (0, 1, 0)),
        (1, [1, _UPPER, 8, _SUM, _FP16_BYTES], (1, 1, 0)),
        (None, [1, _UPPER * 8, 1, _SUM, _FP16_BYTES], (1, 8, 0)),
    ):
        blob = _lowered(_Reduce("sum", dim), x, _sequence_shape(x.shape, minimum=1))
        got, patches, _ = _reduction_and_patches(blob)
        assert got == params, f"sum over dim {dim} emitted {got}"
        assert patches == [patch], f"sum over dim {dim} patched {patches}"


def test_a_whole_tensor_mean_is_patched_as_one_span():
    """The FIX workstream's mean.dim over every dim: one span, scaled by the rest."""
    x = torch.randn(1, _TRACED, 8, dtype=torch.float16)
    blob = _lowered(_Reduce("mean", None), x, _sequence_shape(x.shape, minimum=1))
    got, patches, _ = _reduction_and_patches(blob)
    assert got == [1, _UPPER * 8, 1, _MEAN, _FP16_BYTES], f"mean emitted {got}"
    assert patches == [(1, 8, 0)], f"mean patched {patches}"


def test_a_reduction_whose_spans_are_static_carries_no_patch():
    """The control: a dynamic program, a command that never mentions the length.

    Patching a param the graph had settled would be as wrong as leaving a
    run-time one unpatched, so the records are checked against this command
    rather than against the blob, which the slice's own blits also patch. The
    slice is what fixes the window's shape, so it is the one thing here that
    has to fit inside the symbol's smallest value.
    """
    x = torch.randn(1, _TRACED, 8, dtype=torch.float16)
    blob = _lowered(_ReduceAStaticWindow(), x, _sequence_shape(x.shape, minimum=4))
    got, patches, _ = _reduction_and_patches(blob)
    assert got == [4, 8, 1, _SUM, _FP16_BYTES], f"sum emitted {got}"
    assert patches == [], f"a static span was patched: {patches}"


def test_the_interpreter_runs_the_blob_at_the_length_it_was_written_for():
    """The emitted bound is the longest export, so that is the operand it takes."""
    x = torch.randn(1, _UPPER, 8, dtype=torch.float16)
    blob = _lowered(_Reduce("mean", None), x, _sequence_shape(x.shape, minimum=1))
    got = blob_interpreter.execute(blob, [x.numpy()])[0]
    assert np.frombuffer(got, dtype=np.float16)[:1] == np.float16(torch.mean(x).item())
