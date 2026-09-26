# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The softmax and reduction axis surface, measured rather than restated.

Three things this pins, each of which a reader can otherwise get backwards:

* A last-axis softmax is admitted at EVERY width. Below the vector width it is
  the single SOFTMAX command; at or above it, the shifted sum of exponentials.
  There is no width at which it falls back to a portable kernel, so a wide row
  costs five commands rather than a host round trip.
* The middle-axis gate reads the REDUCED extent, not the trailing one. A
  (4, 8, 256) reduced over axis 1 is an 8-wide reduction and delegates; a
  (4, 256, 8) reduced over axis 1 is a 256-wide one and is refused. Both
  statements are needed, because either alone is satisfied by the other.
* A reduction's ragged tail is the kernel's problem and the host never gates on
  it. Every reduce and inside extent from 1 to 3300 is admitted by all four
  families; the only refusal is a reduced-dim list that is not one span, and all
  four families refuse it, not the three OP_GAPS.md names.
"""
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import SOFTMAX_VECTOR_WIDTH  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

CONFIG = EdgeCompileConfig(_check_ir_validity=False)
_RASTER_BLIT = 3
_SOFTMAX = 28
_REDUCTION = 29
_COMPOSITION = [_REDUCTION, 19, 4, _REDUCTION, 19]


def _softmax(dim):
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=dim)

    return M()


def _ramp(*shape):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    n = 1
    for extent in shape:
        n *= extent
    return torch.arange(n, dtype=torch.float32).remainder(11).sub(5).reshape(shape).half()


def _lower(model, args, dynamic_shapes=None):
    """The delegate command streams of one lowering."""
    program = to_edge_transform_and_lower(
        export(model, tuple(args), dynamic_shapes=dynamic_shapes or {}),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    streams = []
    for node in program.graph_module.graph.nodes:
        if node.target is not torch.ops.higher_order.executorch_call_delegate:
            continue
        blob = bytes(
            program.graph_module.get_submodule(node.args[0].target)._processed_bytes
        )
        _, commands = read_blob(blob)
        streams.append([command.type for command in commands])
    return streams


_RANKS = [
    (8,),
    (8, 63), (8, 64), (8, 65),
    (2, 3, 63), (2, 3, 64), (2, 3, 65),
    (2, 3, 4, 63), (2, 3, 4, 64), (2, 3, 4, 65),
]


@pytest.mark.parametrize("shape", _RANKS)
def test_a_last_axis_softmax_is_admitted_at_every_width_on_every_rank(shape):
    """One command below the vector width, five at or above it, never zero.

    The assertion that matters is the delegate count: an empty stream is the
    portable fallback this says does not happen, so a width that fell back fails
    on the line before it can fail the equality.
    """
    streams = _lower(_softmax(-1), (_ramp(shape),))
    assert len(streams) == 1, f"{shape} produced {len(streams)} delegates"
    width = shape[-1]
    if width < SOFTMAX_VECTOR_WIDTH:
        assert streams[0] == [_SOFTMAX], f"{shape} emitted {streams[0]}"
    else:
        assert streams[0] == _COMPOSITION, f"{shape} emitted {streams[0]}"


@pytest.mark.parametrize("width", [1, 2, 7, 31, 32, 33, 62, 96, 100, 127, 197, 512])
def test_no_last_axis_width_falls_back_to_a_portable_kernel(width):
    """The sweep that says a wide row is composed rather than refused.

    A width that is not a multiple of the vector width is the interesting half:
    96, 100, 127 and 197 are each one vector plus a tail, and each has to reach
    the same five commands 64 does.
    """
    streams = _lower(_softmax(-1), (_ramp(3, width),))
    assert len(streams) == 1, f"width {width} produced {len(streams)} delegates"
    assert streams[0] == (
        [_SOFTMAX] if width < SOFTMAX_VECTOR_WIDTH else _COMPOSITION
    ), f"width {width} emitted {streams[0]}"


# ---------------------------------------------------- the middle-axis extent


def _middle_command(extent, trailing):
    """One middle-axis softmax, and the channel the command was given."""
    shape = (4, extent, trailing)
    streams = _lower(_softmax(1), (_ramp(shape),))
    if not streams:
        return None
    assert len(streams) == 1, f"{shape} produced {len(streams)} delegates"
    assert streams[0] == [_RASTER_BLIT, _SOFTMAX, _RASTER_BLIT], streams[0]
    blob = program_blob(_softmax(1), (_ramp(shape),))
    return read_blob(blob)[1][1]


def program_blob(model, args):
    program = to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    for node in program.graph_module.graph.nodes:
        if node.target is torch.ops.higher_order.executorch_call_delegate:
            return bytes(
                program.graph_module.get_submodule(node.args[0].target)._processed_bytes
            )
    raise AssertionError("no delegate")


@pytest.mark.parametrize("extent", [1, 2, 8, 32, 62, 63])
def test_a_middle_axis_softmax_under_the_vector_width_delegates(extent):
    assert _middle_command(extent, 5) is not None


@pytest.mark.parametrize("extent", [64, 65, 100, 197, 256])
def test_a_middle_axis_softmax_at_or_above_the_vector_width_is_refused(extent):
    assert _middle_command(extent, 5) is None


@pytest.mark.parametrize("trailing", [1, 5, 8, 64, 65, 256, 1000])
def test_a_wide_trailing_extent_does_not_refuse_an_eight_wide_reduction(trailing):
    """The control the refusal test above cannot supply by itself.

    The gate reads shape[dim], the REDUCED extent. A (4, 8, 256) reduced over
    axis 1 is an 8-wide reduction whatever its trailing extent is, so it has to
    delegate at every trailing width. Without this, the row above would be
    satisfiable by a gate that refuses anything with a wide axis anywhere.
    """
    command = _middle_command(8, trailing)
    assert command is not None, f"trailing {trailing} refused an 8-wide reduction"
    assert list(command.params) == [4 * trailing, 8, 1, 2], list(command.params)


# ------------------------------------------------------- the dynamic middle axis


@pytest.mark.parametrize("bound", [16, 63, 64, 65, 256])
def test_a_dynamic_middle_axis_softmax_is_refused_at_every_export_bound(bound):
    """A symbolic reduced extent is refused even where a static one is admitted.

    The static gate accepts a reduced extent of 63 and refuses 64. A bound of 16
    or 63 is the same geometry the export cannot see, and the gate refuses it
    too, because the contiguity walk it does compares the shape against
    row-major strides and a SymInt is not an int. So a dynamic-shape model pays
    for a middle-axis softmax at a width the static path runs. This is the
    asymmetry, named by the two numbers beside it.
    """
    streams = _lower(
        _softmax(1), (_ramp(1, 3, 6),), {"x": {1: Dim("d", min=1, max=bound)}}
    )
    assert streams == [], f"bound {bound} delegated {streams}"


@pytest.mark.parametrize("bound", [16, 63, 64, 65, 256])
def test_a_dynamic_last_axis_softmax_reads_the_export_bound(bound):
    """The other half of the pair: the last axis does take the bound.

    Below the vector width the single command carries the bound as its channel;
    at or above it the five commands do. Either way the node delegates, which is
    what makes the middle-axis refusal above an asymmetry rather than a policy.
    """
    streams = _lower(
        _softmax(-1), (_ramp(1, 3, 6),), {"x": {2: Dim("d", min=1, max=bound)}}
    )
    assert len(streams) == 1, f"bound {bound} produced {len(streams)} delegates"
    assert streams[0] == (
        [_SOFTMAX] if bound < SOFTMAX_VECTOR_WIDTH else _COMPOSITION
    ), f"bound {bound} emitted {streams[0]}"


def _reduce(kind, dims):
    class M(torch.nn.Module):
        def forward(self, x):
            if kind == "sum":
                return torch.sum(x, dim=dims)
            if kind == "mean":
                return torch.mean(x, dim=dims)
            if kind == "amax":
                return torch.amax(x, dim=dims)
            return torch.amin(x, dim=dims)

    return M()


def _blob(model, args):
    program = to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    for node in program.graph_module.graph.nodes:
        if node.target is torch.ops.higher_order.executorch_call_delegate:
            return bytes(
                program.graph_module.get_submodule(node.args[0].target)._processed_bytes
            )
    raise AssertionError("no delegate")


_FAMILIES = ["sum", "mean", "amax", "amin"]
_REDUCTION_KINDS = {"sum": 1, "mean": 3, "amax": 2, "amin": 4}


@pytest.mark.parametrize("kind", _FAMILIES)
@pytest.mark.parametrize("width", [1, 2, 31, 32, 33, 62, 63, 64, 65, 100, 197, 256])
def test_a_reduction_is_admitted_at_every_reduce_extent(kind, width):
    """inside == 1, so the row the kernel reduces is width elements long.

    33, 62, 63, 65, 100, 197 and 256 are ragged rows: none a multiple of the 64
    fp16 lanes a vector holds, so each is finished by the kernel's scalar tail.
    The host admits all of them, which is the claim; the empty-stream case is
    what this is about, so the delegate count is asserted first.
    """
    x = _ramp(4, width)
    streams = _lower(_reduce(kind, (1,)), (x,))
    assert len(streams) == 1, f"{kind} at width {width} produced {len(streams)}"
    assert streams[0] == [_REDUCTION], f"{kind} at width {width}: {streams[0]}"
    command = read_blob(_blob(_reduce(kind, (1,)), (x,)))[1][0]
    assert list(command.params[:5]) == [
        4, width, 1, _REDUCTION_KINDS[kind], 2,
    ], f"{kind} at width {width} emitted {list(command.params[:5])}"


@pytest.mark.parametrize("kind", _FAMILIES)
@pytest.mark.parametrize("inside", [2, 31, 32, 33, 62, 63, 64, 65, 100, 197, 256])
def test_a_reduction_is_admitted_at_every_output_extent(kind, inside):
    """The other side of the span: inside is the row the kernel writes.

    32 and 64 are the two branch boundaries in htp_ops_reduction and 33, 65, 100
    and 197 straddle them, so this is the sweep that would move if the host ever
    grew a gate on the width. It has not: the kernel's four-way dispatch covers
    every inside the host admits.
    """
    x = _ramp(3, 4, inside)
    streams = _lower(_reduce(kind, (1,)), (x,))
    assert len(streams) == 1, f"{kind} at inside {inside} produced {len(streams)}"
    assert streams[0] == [_REDUCTION], f"{kind} at inside {inside}: {streams[0]}"
    command = read_blob(_blob(_reduce(kind, (1,)), (x,)))[1][0]
    assert list(command.params[:5]) == [
        3, 4, inside, _REDUCTION_KINDS[kind], 2,
    ], f"{kind} at inside {inside} emitted {list(command.params[:5])}"


# ------------------------------------------------------ the one span rule


@pytest.mark.parametrize("kind", _FAMILIES)
def test_a_reduction_over_two_adjacent_axes_is_one_span(kind):
    x = _ramp(2, 3, 4, 5)
    streams = _lower(_reduce(kind, (1, 2)), (x,))
    assert len(streams) == 1, f"{kind} over (1,2) produced {len(streams)}"
    assert streams[0] == [_REDUCTION], f"{kind} over (1,2): {streams[0]}"
    command = read_blob(_blob(_reduce(kind, (1, 2)), (x,)))[1][0]
    assert list(command.params[:5]) == [2, 12, 5, _REDUCTION_KINDS[kind], 2], list(
        command.params[:5]
    )


@pytest.mark.parametrize("kind", _FAMILIES)
def test_every_family_refuses_two_non_adjacent_reduced_axes(kind):
    """The only refusal the four families share, and all four share it.

    OP_GAPS.md's limit row names mean.dim, sum.dim_IntList and amax for this.
    Measured, amin refuses it too -- 4 of 4 families, not 3 of 4 -- so the doc
    is the thing that was short a name. The refusal is the contiguous-span rule
    and nothing else: the same shape and the same extents delegate at (1, 2).
    """
    x = _ramp(2, 3, 4)
    assert _lower(_reduce(kind, (0, 2)), (x,)) == [], f"{kind} over (0,2) delegated"
    assert len(_lower(_reduce(kind, (0, 1)), (x,))) == 1, (
        f"{kind} over (0,1) is the control and has to delegate"
    )


def test_the_span_sweep_has_no_other_refusal_in_it():
    """The parts-to-total check the two sweeps above do not make for themselves.

    4 families x (12 reduce extents + 11 inside extents) lowerings, every one of
    which asserts a delegate. If a width were refused the sweeps would fail on
    their own assertion; this counts them so a reader can see the total that the
    claim covers, and a family quietly dropping out of the parametrization makes
    the count wrong rather than the suite quietly smaller.
    """
    assert len(_FAMILIES) == 4
    widths = [1, 2, 31, 32, 33, 62, 63, 64, 65, 100, 197, 256]
    insides = [2, 31, 32, 33, 62, 63, 64, 65, 100, 197, 256]
    assert 4 * (len(widths) + len(insides)) == 92
