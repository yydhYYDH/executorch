# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A convolution whose height is the run-time length.

The height of a mel spectrogram is however many frames the caller captured, so a
convolution over it is the one thing an ASR export cannot spell as a constant.
What makes the height improvable is that it is the *bare* exported symbol rather
than an expression over it: an extent the length reaches by being multiplied by a
static factor is one the runtime rebuilds from the length alone
(`DynamicPatch`, serialization/blob.py), and a stride-1 convolution that leaves
the height alone is the case where every extent that reaches a command parameter
is that kind of product. A stride divides and a lopsided padding shifts, and
neither is affine in the length, so both stay on a portable kernel.

The params equality at the end is the check that the records are right, and it is
the only one that is *exact*: for each run length, the dynamic blob's records are
applied and the result has to equal, field for field, the blob the same weights
lower to when the height is static. Nothing is exempted from that comparison. The
first draft of it skipped the memset's single parameter on the reasoning that a
buffer's size is the allocation and not the run, which is exactly the parameter
that was wrong: `_emit_zero` writes the buffer's byte count, the runtime resizes
the buffer for the run, and a count left at the export's longest clears past the
end of the shorter arena -- a device answers that with `0x8000040d` and no output.
"""

import os
import pathlib
import sys

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import read_blob, read_dynamic_trailer  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

#: DSP_OP_ZERO, DSP_OP_RASTER_BLIT and DSP_OP_IM2COL_CONVOLUTION_FP16: what a
#: convolution over a mel reaches, because the plane is neither one element wide
#: nor already blocked, so it is packed, walked and unpacked.
_ZERO = 24
_BLIT = 3
_IM2COL = 12

#: The 64-channel block the blocked layout carries, and how many lanes of it a
#: buffer holds per element of the row-major one.
_BLOCK = 64
_FP16_BYTES = 2

_MEL = 128
_CHANNELS = 64
_TRACED = 64
_UPPER = 256

#: Put on the path by the module, which is also why the heights below are what
#: they are: a run length is a bare symbol to the emitter and the traced example
#: to everything the runtime does not rebuild.
_CALL_DELEGATE = torch.ops.higher_order.executorch_call_delegate


class _Mel(torch.nn.Module):
    def __init__(self, out_channels=_CHANNELS, stride=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            1, out_channels, (3, 3), stride=stride, padding=(1, 1), bias=True
        )

    def forward(self, mel):
        return self.conv(mel)


def _weights(seed=7):
    """Integer weights, so a comparison of two blobs is of the emitter alone."""
    torch.manual_seed(0)
    module = _Mel().half().eval()
    generator = torch.Generator().manual_seed(seed)
    weight = module.conv.weight
    weight.data = torch.randint(-1, 2, weight.shape, generator=generator).to(
        weight.dtype
    )
    bias = module.conv.bias
    bias.data = torch.randint(-32, 32, bias.shape, generator=generator).to(bias.dtype)
    return module


def _mel(height):
    generator = torch.Generator().manual_seed(11)
    return torch.randint(-4, 5, (1, 1, height, _MEL), generator=generator).half()


def _heights(dynamic):
    """The dynamic export's constraint over the height, or none for a static one."""
    if not dynamic:
        return None
    return {"mel": {2: Dim("frames", min=16, max=_UPPER)}}


def _graph(module, height, dynamic):
    return export(module, (_mel(height),), dynamic_shapes=_heights(dynamic))


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is _CALL_DELEGATE
    ]


def _blob(module, height, dynamic):
    """The one delegate's blob, from a program the partitioner had its say on."""
    program = to_edge_transform_and_lower(
        _graph(module, height, dynamic),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = _delegates(program)
    assert len(calls) == 1, f"the mel convolution did not reach one delegate: {calls}"
    return bytes(program.graph_module.get_submodule(calls[0].args[0].target)._processed_bytes)


def _commands(blob):
    return read_blob(blob)[1]


def _patched(blob, length):
    """The blob's commands with its records applied, the way the runtime applies them.

    The scale and the add are the whole of what a record can say
    (serialization/hexagon_schema.h), and `length * scale + add` is what
    runtime/hexagon_backend.cpp writes into the slot before it dispatches.
    """
    trailer = read_dynamic_trailer(blob)
    assert trailer is not None, "a graph with a moving height emits a trailer"
    commands = [list(command.params) for command in _commands(blob)]
    kinds = [command.type for command in _commands(blob)]
    for op_index, param_index, scale, add in trailer.patches:
        commands[op_index][param_index] = length * scale + add
    return kinds, commands


def test_a_mel_convolution_is_a_memset_a_pack_a_walk_and_an_unpack():
    """The four commands, their params, and the height in all four of them.

    The im2col kernel reads the packed plane and writes the other one, and both
    of its extent parameters are the height; the blit either side carries the
    plane in its region, as the size of the longest axis and as the pitches the
    two layouts disagree about. The memset's parameter is the one that is not a
    tensor extent at all: it is the byte count of the buffer it clears.
    """
    blob = _blob(_weights(), _TRACED, dynamic=True)
    commands = _commands(blob)
    assert [command.type for command in commands] == [_ZERO, _BLIT, _IM2COL, _BLIT]

    zero, pack, walk, unpack = commands
    # One row of the packed buffer: the mel's width, one block of it, in fp16.
    assert list(zero.params) == [_UPPER * _MEL * _BLOCK * _FP16_BYTES]
    # The region is the whole plane, so the row-major pitches are the plane and
    # the blocked ones are the block.
    assert list(pack.params) == [
        1,
        _FP16_BYTES,
        1,
        0,
        0,
        0,
        1,
        1,
        _TRACED * _MEL,
        _TRACED * _MEL,
        _TRACED * _MEL,
        1,
        _TRACED * _MEL * _BLOCK,
        1,
        _BLOCK,
    ]
    assert list(unpack.params) == [
        1,
        _FP16_BYTES,
        1,
        0,
        0,
        0,
        1,
        _CHANNELS,
        _TRACED * _MEL,
        _TRACED * _MEL * _BLOCK,
        1,
        _BLOCK,
        _CHANNELS * _TRACED * _MEL,
        _TRACED * _MEL,
        1,
    ]
    # The walk: the plane's height in and out, and the plane's element count as
    # the two places the kernel reads it, both of them times a whole block.
    assert list(walk.params)[6:18] == [
        3,
        3,
        0,
        9,
        _MEL,
        _TRACED,
        _MEL,
        _TRACED,
        _TRACED * _MEL * _BLOCK,
        _TRACED * _MEL,
        _BLOCK,
        _TRACED * _MEL * _BLOCK,
    ]


def test_the_trailer_names_the_height_and_where_to_read_it():
    """One input, the height's own axis, the export's longest, and the example."""
    trailer = read_dynamic_trailer(_blob(_weights(), _TRACED, dynamic=True))
    assert trailer.input_index == 0
    assert trailer.axis == 2
    assert trailer.max_length == _UPPER
    assert trailer.example_length == _TRACED


def test_every_extent_the_length_reaches_carries_a_record():
    """The whole record list, which is what says the extents are all affine.

    A length reaches a parameter either directly, as the height the kernel reads,
    or through one static factor: the mel's width, a block of 64 lanes, the
    channel count. Each of those is a distinct scale, which is why the list is
    this long rather than one record per command -- and why the memset's count is
    on it, since the buffer the count is of is the one the height moves.
    """
    trailer = read_dynamic_trailer(_blob(_weights(), _TRACED, dynamic=True))
    width = _MEL
    assert trailer.patches == [
        # The memset's byte count: one row of the packed buffer.
        (0, 0, width * _BLOCK * _FP16_BYTES, 0),
        # The pack blit: the plane as the region's longest axis, the row-major
        # pitches, and where the first block of the packed plane starts.
        (1, 8, width, 0),
        (1, 9, width, 0),
        (1, 10, width, 0),
        (1, 12, width * _BLOCK, 0),
        # The walk: the height in, the height out, and the plane's element count
        # at each of the two places the kernel reads it.
        (2, 11, 1, 0),
        (2, 13, 1, 0),
        (2, 14, width * _BLOCK, 0),
        (2, 17, width * _BLOCK, 0),
        # The unpack blit: the blocked pitches, then the row-major ones.
        (3, 8, width, 0),
        (3, 9, width * _BLOCK, 0),
        (3, 12, _CHANNELS * width, 0),
        (3, 13, width, 0),
    ]


@pytest.mark.parametrize("height", [16, 32, 64, 96, 128, 192, _UPPER])
def test_the_records_rebuild_the_params_a_static_height_would_give(height):
    """Patched, the dynamic blob is the blob that height lowers to. Nothing skipped.

    This is the claim the records make and the only one that is exact: whatever
    the emitter would have written had the height been this number, the runtime
    writes. Every command's type and every parameter of it is compared, including
    the memset's byte count, which is the one that is a buffer's size rather than
    a tensor's extent and the one an early version of this test talked itself out
    of checking.
    """
    dynamic = _blob(_weights(), _TRACED, dynamic=True)
    static = _blob(_weights(), height, dynamic=False)
    assert [command.type for command in _commands(dynamic)] == [
        command.type for command in _commands(static)
    ], "the two blobs are not the same commands"

    kinds, patched = _patched(dynamic, height)
    for index, command in enumerate(_commands(static)):
        assert kinds[index] == command.type, f"command {index} changed kind"
        assert patched[index] == list(command.params), (
            f"command {index} at height {height}: the records rebuilt "
            f"{patched[index]} where a static height writes {list(command.params)}"
        )


def test_the_memsets_count_is_the_buffer_the_run_gets():
    """The one parameter that is a byte count, and the trap that hid it.

    `htp_ops_zero` clears as many bytes as the parameter says, and the runtime
    resizes the buffer for the run rather than handing it the export's longest.
    A count left at the longest therefore clears past the end of the shorter
    arena, which a device reports as a failed command group and no output.
    """
    blob = _blob(_weights(), _TRACED, dynamic=True)
    zero = _commands(blob)[0]
    # The blob carries the longest: one row of the packed plane times the bound.
    assert list(zero.params) == [_UPPER * _MEL * _BLOCK * _FP16_BYTES]
    for height in (16, _TRACED, _UPPER):
        kinds, patched = _patched(blob, height)
        assert kinds[0] == _ZERO
        assert patched[0] == [height * _MEL * _BLOCK * _FP16_BYTES]


def test_a_height_a_stride_divides_is_not_a_height_this_patches():
    """The control: the gate has to keep refusing what no record can express.

    A stride-2 convolution's output height is the input height divided by two,
    and the record list has no divisor in it, so the emitter refuses the node and
    the graph keeps a portable convolution -- which on this backend is the whole
    graph, since a mel has no other op a portable kernel can carry.
    """
    module = _Mel(out_channels=64, stride=2).half().eval()
    module.conv.weight.data = _weights().conv.weight.data.clone()
    module.conv.bias.data = _weights().conv.bias.data.clone()
    program = to_edge_transform_and_lower(
        _graph(module, _TRACED, dynamic=True),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    assert _delegates(program) == []


def test_a_width_that_moves_is_not_a_height_that_moves():
    """The other control: the axis is not interchangeable with any other.

    A batch, a channel count or a window's width that moved would each need its
    own patch over geometry the emitter reads once, so only the height is
    admitted -- and only when nothing else in the graph's own shape moved with
    it.
    """
    module = _Mel().half().eval()
    module.conv.weight.data = _weights().conv.weight.data.clone()
    module.conv.bias.data = _weights().conv.bias.data.clone()
    program = to_edge_transform_and_lower(
        export(
            module,
            (_mel(_TRACED),),
            dynamic_shapes={"mel": {3: Dim("mel_bins", min=16, max=256)}},
        ),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    assert _delegates(program) == []
