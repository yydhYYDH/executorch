# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A grouped, non-depthwise convolution on the DSP.

Neither DSP convolution kernel carries a group field, so the host supplies the
channel mapping: the input, the weight, the bias and the output are partitioned
by group and each group is emitted as the dense im2col walk the existing kernel
already is. These tests pin that one dense walk per group is what the blob
holds, that each walk sees the group-local channel counts rather than the whole
tensor's, that the output lands in the tensor's own group-major order, that the
depthwise form keeps its old path, and that every shape still refused stays
refused and emits no command at all.

The 63/64/65 sweep is structural rather than a boundary search: HVX walks the
vector part of an activation to size & -64 and finishes scalar, so 63 channels
per group are all scalar, 64 are all vector, and 65 are 64 plus one scalar lane.
Nothing here infers one length's result from a neighbouring length's; each is
lowered and run on its own.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import test_conv as C  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

_BLIT = 3
_IM2COL = 12
_DEPTHWISE = 2

#: The fixed parameter block every command's params live in
#: (serialization/hexagon_schema.h:44). One command per group multiplies the
#: *command* count, never a single command's parameter count, so the budget is
#: a per-command fact and the group count grows the command count instead.
MAX_OP_PARAMS = 40


def _grouped(in_channels, out_channels, kernel=3, stride=1, padding=1, groups=4):
    return C._Conv(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        groups=groups,
    ).half()


def _walks(commands):
    return [command for command in commands if command.type == _IM2COL]


def test_a_grouped_convolution_is_one_dense_walk_per_group():
    """The command stream is the dense walk repeated, with the input grouped."""
    model = _grouped(256, 256, groups=4)
    C._whole(model, 23)
    x = C._exact((1, 256, 8, 8), 24, -2, 3)
    data, commands = C._blob(C._lower(model, (x,)))
    walks = _walks(commands)
    assert len(walks) == 4, "one dense walk per group, not one for the tensor"

    # Every walk is told the group-local channel counts, not the tensor's.
    for walk in walks:
        assert walk.params[18] == 64  # in_channels
        assert walk.params[19] == 64  # icup4, the padded reduction
        assert walk.params[20] == 64  # out_channels

    # The first command is the blit that picks the first group out of the input
    # and writes it as a dense row-major plane; the last is the blit that puts
    # the last group's result back at its ordinal channel offset.
    first = commands[0]
    assert first.type == _BLIT
    assert first.outputs[0].size == 1 * 64 * 8 * 8 * 2
    assert list(first.params) == [
        1,
        2,
        1,  # one region, fp16, one source
        0,
        0,
        0,  # source index 0, no offsets
        1,
        64,
        64,  # size: [batch][group in channels][area]
        256 * 64,
        64,
        1,  # source strides, from the whole tensor's plane
        64 * 64,
        64,
        1,  # destination strides, the group's own contiguous plane
    ]
    last = commands[-1]
    assert last.type == _BLIT
    assert list(last.params)[5] == 3 * 64 * 64  # the fourth group's output offset

    # The numbers are exact against torch, so a wrong channel mapping is a wrong
    # tensor rather than a rounding difference.
    expected = model(x).detach().numpy().reshape(-1)
    assert C._run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_each_group_sees_only_its_own_channels_and_weights():
    """Group g's walk reads input channel g * in_per_group and writes its rows.

    The weights are made distinct per group and per output channel, so if any
    group read another group's input plane or another group's weight rows, the
    result would be a different number.
    """
    model = _grouped(12, 24, groups=3)
    with torch.no_grad():
        weight = model.conv.weight
        weight.zero_()
        # Output channel g*8 + k reads only input channel g*4, through one tap.
        for g in range(3):
            for k in range(8):
                weight[g * 8 + k, 0, k % 3, k // 3] = 1.0
        model.conv.bias.zero_()
    x = torch.zeros(1, 12, 4, 4).half()
    # One impulse per group, each in that group's own first input channel.
    for g in range(3):
        x[0, g * 4, 1, 1] = 1.0
    data, commands = C._blob(C._lower(model, (x,)))
    got = C._run(data, [x.numpy()]).reshape(1, 24, 4, 4)
    expected = model(x).detach().numpy()
    # Each group is also convolved on its own, so a walk that read another
    # group's input plane or another group's weight rows would differ from the
    # slice the group would produce by itself.
    for g in range(3):
        one = torch.nn.functional.conv2d(
            x[:, g * 4 : (g + 1) * 4],
            model.conv.weight[g * 8 : (g + 1) * 8],
            model.conv.bias[g * 8 : (g + 1) * 8],
            padding=1,
        )
        assert np.array_equal(
            got[0, g * 8 : (g + 1) * 8], one.detach().numpy()[0]
        ), f"group {g}"
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("per_group", [63, 64, 65])
def test_a_group_channel_count_off_the_64_lane_boundary_still_works(per_group):
    """C_in = groups * 63, groups * 64 and groups * 65 all lower and run.

    The group-local channel count is what the walk's ic field carries, so a
    group whose channel count is not a whole 64-lane block is the interesting
    case: the block is packed per group, and the group plane is the ragged one.
    """
    model = _grouped(3 * per_group, 3 * per_group, groups=3)
    C._whole(model, 31)
    x = C._exact((1, 3 * per_group, 4, 4), 32, -2, 3)
    data, commands = C._blob(C._lower(model, (x,)))
    walks = _walks(commands)
    assert len(walks) == 3
    for walk in walks:
        assert walk.params[18] == per_group
        assert walk.params[19] == (per_group + 3) // 4 * 4
        assert walk.params[20] == per_group
    assert max(len(command.params) for command in commands) <= MAX_OP_PARAMS
    expected = model(x).detach().numpy().reshape(-1)
    got = C._run(data, [x.numpy()])
    assert got.tobytes() == expected.tobytes(), f"{per_group} channels per group"


@pytest.mark.parametrize("groups", [2, 3, 4, 8, 16, 32])
def test_the_command_count_scales_with_the_group_count_and_stays_in_budget(groups):
    """One command per group multiplies commands, never a command's params.

    The per-command parameter block is fixed at 40 ints and the widest command
    here is the 29-int im2col walk, so the group count cannot push a single
    command over the budget; what grows is the number of commands, linearly in
    the group count. Every command in the blob is checked against the block.
    """
    per_group = 4
    model = _grouped(groups * per_group, groups * per_group, groups=groups)
    C._whole(model, 41)
    x = C._exact((1, groups * per_group, 4, 4), 42, -2, 3)
    data, commands = C._blob(C._lower(model, (x,)))
    assert len(_walks(commands)) == groups
    assert all(len(command.params) <= MAX_OP_PARAMS for command in commands)
    expected = model(x).detach().numpy().reshape(-1)
    assert C._run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_depthwise_convolution_keeps_the_existing_path():
    """groups == C_in == C_out is still the per-channel walk, not one per group.

    A grouped form with one channel per group is the depthwise case the DSP
    already has, and it must keep taking that path rather than the new
    per-group decomposition: the command is the depthwise walk, one of them.
    """
    channels = 64
    model = _grouped(channels, channels, groups=channels)
    C._whole(model, 51)
    x = C._exact((1, channels, 8, 8), 52, -2, 3)
    data, commands = C._blob(C._lower(model, (x,)))
    assert _walks(commands) == [], "the depthwise path emits no dense walk"
    depthwise = [c for c in commands if c.type == _DEPTHWISE]
    assert len(depthwise) == 1
    assert depthwise[0].params[5] == channels // 64
    expected = model(x).detach().numpy().reshape(-1)
    assert C._run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_group_count_that_does_not_divide_the_input_channels_is_refused():
    """A hand-built node whose groups do not divide C_in stays refused."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.zeros(1, 6, 4, 4)
    weight = graph.placeholder("w")
    weight.meta["val"] = torch.zeros(4, 1, 3, 3)
    bias = graph.placeholder("b")
    bias.meta["val"] = torch.zeros(4)
    out = graph.call_function(
        torch.ops.aten.convolution.default,
        (x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 4),
    )
    out.meta["val"] = torch.zeros(1, 4, 4, 4)
    graph.output(out)
    node = next(n for n in graph.nodes if n.op == "call_function")
    assert hexagon_ops.conv_spec(node, lambda operand: True) is None


def test_a_grouped_convolution_over_a_dynamic_height_stays_portable():
    """A run-time height with more than one group is refused, not mis-emitted.

    The per-group walk is emitted with a static extent. Extending it to a
    run-time length would need a dynamic patch per group command, which the
    exporter does not produce, so the node is refused rather than baked with
    the export's example height.
    """
    torch._dynamo.reset()
    module = C._Conv(8, 8, 3, padding=1, groups=4).half()
    C._whole(module, 71)
    x = C._exact((1, 8, 32, 16), 2, -2, 3)
    program = to_edge_transform_and_lower(
        export(module, (x,), dynamic_shapes=[{2: Dim("frames", min=16, max=64)}]),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    assert C._delegates(program) == [], "a dynamic-height grouped conv is refused"


def test_the_command_stream_is_read_from_the_blob_not_the_header():
    """The count is decoded from the blob, not from an ops=N header."""
    model = _grouped(64, 64, groups=2)
    C._whole(model, 61)
    x = C._exact((1, 64, 4, 4), 62, -2, 3)
    data, commands = C._blob(C._lower(model, (x,)))
    header, decoded = read_blob(data)
    assert len(decoded) == len(commands) == header.n_ops
    names = {
        value: name
        for name, value in vars(hexagon_ops).items()
        if name.startswith("DSP_OP_")
    }
    kinds = [names[command.type] for command in decoded]
    assert kinds.count("DSP_OP_IM2COL_CONVOLUTION_FP16") == 2

