# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The three ways a decoder-shaped block can be lowered, and what each costs.

The question these pin is the one a cost measurement has to answer first: if
the nearest upsample is refused, does the rest of the block still reach the
DSP, and in how many delegates? A refusal that took the convolutions with it
would measure the wrong thing, so the CPU-upsample case here is two
convolutions with one portable node between them, not a portable block.

The seam is the same one the fraction and bilinear refusals use:
``upsample_regions`` returning None is the refusal. The partitioner is @final
and builds its own support object, so this is the one place a single node's
verdict can be changed without a backend change, and it is the same decision
the emitter would make if it were asked.
"""

import contextlib
import os
import pathlib
import sys

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon.partition import (  # noqa: E402
    hexagon_partitioner as _partitioner,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_BLIT = 3
_CONV = 12


class _DecoderBlock(torch.nn.Module):
    """conv3x3, nearest 2x, conv3x3: the diffusers UpDecoderBlock2D shape."""

    def __init__(self, channels: int, upsample: bool = True) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = upsample

    def forward(self, x):
        import torch.nn.functional as F

        h = self.conv1(x)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        return self.conv2(h)


@contextlib.contextmanager
def _nearest_upsample_refused():
    original = _partitioner.upsample_regions
    _partitioner.upsample_regions = lambda node: None
    try:
        yield
    finally:
        _partitioner.upsample_regions = original


def _block(channels: int, upsample: bool = True) -> torch.nn.Module:
    torch.manual_seed(0)
    return _DecoderBlock(channels, upsample=upsample).half().eval()


def _input(channels: int, height: int, width: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randint(
            -1, 2, (1, channels, height, height if width is None else width),
            generator=generator,
        ).half()
    )


def _lower(module, inputs):
    return to_edge_transform_and_lower(
        export(module.eval(), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _commands(program):
    out = []
    for call in _delegates(program):
        lowered = program.graph_module.get_submodule(call.args[0].target)
        _, commands = read_blob(bytes(lowered._processed_bytes))
        out.append([command.type for command in commands])
    return out


def _regions(command):
    """(size, destination strides) for every region of a blit command."""
    count = command.params[0]
    for region in range(count):
        base = 3 + region * 12
        yield tuple(command.params[base + 3 : base + 6]), tuple(
            command.params[base + 9 : base + 12]
        )


def _upsample_phase_regions(program):
    """The four phase regions, found by the strided destination they write."""
    out = []
    for call in _delegates(program):
        lowered = program.graph_module.get_submodule(call.args[0].target)
        _, commands = read_blob(bytes(lowered._processed_bytes))
        for command in commands:
            if command.type != _BLIT:
                continue
            for size, dst in _regions(command):
                if dst[2] == 2:
                    out.append((size, dst))
    return out


def _upsample_nodes(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "upsample_nearest2d" in str(node.target)
    ]


def test_the_block_is_one_delegate_with_the_upsample_inside():
    """The default: one delegate whose command stream carries all of it."""
    module = _block(4)
    x = _input(4, 16)
    program = _lower(module, (x,))
    calls = _delegates(program)
    assert len(calls) == 1, calls
    types = _commands(program)[0]
    # Four blits for the upsample's four phases, and one convolution each.
    # Each convolution also brings a zero-fill and two arena copies.
    assert types.count(_BLIT) == 6, types
    assert types.count(_CONV) == 2, types
    assert not _upsample_nodes(program), "nothing should be left outside"


def test_a_refused_upsample_leaves_both_convolutions_on_the_dsp():
    """The CPU-upsample variant: two delegates, one portable node between."""
    module = _block(4)
    x = _input(4, 16)
    with _nearest_upsample_refused():
        program = _lower(module, (x,))
    calls = _delegates(program)
    assert len(calls) == 2, calls
    for types in _commands(program):
        # Each delegate is one convolution: a zero-fill, an arena copy in,
        # the convolution itself, and the copy the output slot needs.
        assert types.count(_CONV) == 1, types
        assert types.count(_BLIT) == 2, types
    outside = _upsample_nodes(program)
    assert len(outside) == 1, outside
    assert list(outside[0].meta["val"].shape) == [1, 4, 32, 32]


def test_without_the_upsample_the_floor_is_still_one_delegate():
    """The floor: the same two convolutions, no resize between them."""
    module = _block(4, upsample=False)
    x = _input(4, 16)
    program = _lower(module, (x,))
    assert len(_delegates(program)) == 1
    types = _commands(program)[0]
    assert types.count(_CONV) == 2, types
    assert types.count(_BLIT) == 4, types
    assert not _upsample_nodes(program)


@pytest.mark.parametrize("channels", [4, 8])
@pytest.mark.parametrize("height", [8, 16, 32])
def test_the_seam_costs_one_portable_node_and_nothing_else(channels, height):
    """The refusal is per node, so the delegate count is exactly two."""
    module = _block(channels)
    x = _input(channels, height)
    with _nearest_upsample_refused():
        program = _lower(module, (x,))
    assert len(_delegates(program)) == 2
    assert len(_upsample_nodes(program)) == 1
    # Both convolutions are still there, in order, one per delegate.
    refused = _commands(program)
    assert [c for types in refused for c in types if c == _CONV] == [
        _CONV,
        _CONV,
    ]
    whole = _lower(module, (x,))
    assert len(_delegates(whole)) == 1
    assert [c for c in _commands(whole)[0] if c == _CONV] == [_CONV, _CONV]


@pytest.mark.parametrize("width", [63, 64, 65])
def test_the_upsample_regions_walk_the_row_width_exactly(width):
    """63 is scalar, 64 is one vector, 65 is a vector and one element."""
    module = _block(4)
    x = _input(4, 8, width)
    program = _lower(module, (x,))
    assert len(_delegates(program)) == 1
    regions = _upsample_phase_regions(program)
    # Four phase regions, split two per command, each walking a whole row.
    assert len(regions) == 4, regions
    height, out_height, out_width = 8, 16, width * 2
    for size, dst in regions:
        assert size == (4, height, width), regions
        # A phase writes every other row and every other column of the output,
        # so its strides are the full output plane and twice the output row.
        assert dst == (out_height * out_width, 2 * out_width, 2), regions
    # The innermost loop is the contiguous run the kernel vectorises: its
    # vector part is column & -64 and the rest is scalar. The command stream
    # does not change shape across that boundary, which is the point of
    # sweeping it here rather than only at a multiple of 64.
    vector, scalar = width & -64, width % 64
    assert (vector, scalar) == ((0, 63), (64, 0), (64, 1))[width - 63]

