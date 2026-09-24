# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GroupNorm and InstanceNorm, which are the norm kernel over another view.

There is no group-norm kernel and no instance-norm kernel in the vendored
`mnn-htp-ops` tree, and neither is in upstream MNN's Hexagon backend: the
dispatch in `src/dsp/execute_command.cc` has no case for either name, and
`htp_command.h` declares no command for them. What the tree does have is
`htp_ops_layer_norm` (layer_norm_ops.cc), which reduces one mean and one
variance over each row of an `[outer][inner]` view of its input, and both ops are
that kernel over a view the op's own definition already names:

* `native_group_norm` computes one mean and one variance per group over that
  group's channels and the whole spatial block, which is one row per
  `(batch, group)` of an `[N*G][(C/G)*H*W]` view;
* InstanceNorm exports as `_native_batch_norm_legit.no_stats` over a
  `[1, N*C, *spatial]` view -- the exporter flattens the batch into the channel
  axis so that per-channel statistics become per-`(n, c)` ones -- which is one
  row per `(batch, channel)` of an `[N*C][H*W]` view.

Neither command carries a weight: the kernel's `gamma` operand is left null, as
it is on the layer-norm path, and the affine is emitted as the two element-wise
commands that op spends on its own. So a group norm is three commands and an
instance norm is three, and what these tests pin is the view each one reduces
over, the gate that keeps the ones the view cannot describe on the portable
kernels, and the numbers.
"""

import operator
import os
import pathlib
import struct
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import ABSENT, execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    _float_bits,
    BATCH_NORM_NO_STATS,
    batch_norm_normalizes_one_span,
    GETITEM,
    group_norm_normalizes_one_group_per_row,
    NATIVE_GROUP_NORM,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16

#: The trailer's magic, whose records carry one patch per (command, param).
_TRAILER_MAGIC = 0x44594E48

#: DSP_OP_LAYER_NORM and DSP_OP_BINARY_ELEMENTWISE: every norm here is one of the
#: first followed by one or two of the second.
_LAYER_NORM = 8
_BINARY = 19

#: BINARY_OP_MUL and BINARY_OP_ADD, the two the affine spends.
_MUL = 3
_ADD = 1

#: The kernel reduces in fp32 and numpy's order is its own, so the comparison is
#: a tolerance. Measured on the host model at two ulp of an answer of size one.
_TOLERANCE = 5e-3


class _GroupNorm(torch.nn.Module):
    def __init__(self, groups, channels, affine=True):
        super().__init__()
        self.norm = torch.nn.GroupNorm(groups, channels, affine=affine)

    def forward(self, x):
        return self.norm(x)


class _InstanceNorm(torch.nn.Module):
    def __init__(self, channels, affine=True):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(channels, affine=affine)

    def forward(self, x):
        return self.norm(x)


def _group_norm_reference(x, groups, weight, bias, eps):
    """The op's definition, written as the view it is: one row per group.

    Two-pass on purpose -- a mean, then the mean of the squared deviations -- so
    that the reference shares no arithmetic with the kernel's one-pass
    `E[x^2] - E[x]^2`. The two agree wherever the op is accurate and come apart
    where the one-pass form cancels, which is what the offset test measures.
    """
    shape = list(x.shape)
    rows = x.float().reshape(shape[0] * groups, -1)
    centered = rows - rows.mean(dim=1, keepdim=True)
    variance = (centered * centered).mean(dim=1, keepdim=True)
    normalized = centered / torch.sqrt(variance + eps)
    result = normalized.reshape(shape)
    if weight is not None:
        result = result * weight.float().reshape(1, -1, *([1] * (result.dim() - 2)))
    if bias is not None:
        result = result + bias.float().reshape(1, -1, *([1] * (result.dim() - 2)))
    return result


def _instance_norm_reference(x, weight, bias, eps):
    """The same definition with one row per (batch, channel).

    The affine comes in the form the command reads it, one value per row, which
    is what the op's per-channel weight becomes once the batch axis is folded
    into the rows: a row here is a (batch, channel) pair, not a channel.
    """
    batches, channels = x.shape[:2]
    rows = x.float().reshape(batches * channels, -1)
    centered = rows - rows.mean(dim=1, keepdim=True)
    variance = (centered * centered).mean(dim=1, keepdim=True)
    normalized = centered / torch.sqrt(variance + eps)
    if weight is not None:
        normalized = normalized * weight.float().reshape(-1, 1)
    if bias is not None:
        normalized = normalized + bias.float().reshape(-1, 1)
    return normalized.reshape(x.shape)


def _lowered(model, args, dynamic_shapes=None):
    """The single delegate a whole model lowers to, and its commands."""
    program = to_edge_transform_and_lower(
        export(model, tuple(args), dynamic_shapes=dynamic_shapes),
        partitioner=[HexagonPartitioner()]
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the model did not lower to one delegate: {calls}"
    inner = program.graph_module.get_submodule(calls[0].args[0].target)
    assert inner.backend_id == "HexagonBackend"
    blob = bytes(inner._processed_bytes)
    _, commands = read_blob(blob)
    # A dynamic graph hands the delegate the run-time length as an argument of
    # its own, which is an int rather than a tensor and has no shape to report.
    shapes = [
        tuple(argument.meta["val"].shape)
        for argument in calls[0].args[1:]
        if isinstance(argument.meta.get("val"), torch.Tensor)
    ]
    return blob, commands, shapes


def _host(blob, args):
    return np.frombuffer(execute(blob, [t.numpy() for t in args])[0], dtype=np.float16)


def test_group_norm_is_one_norm_command_and_its_affine():
    """Three commands: the statistics, the weight and the bias.

    The weight is a graph placeholder the backend owns, so it reaches the
    subgraph as a constant and the mul is a plain element-wise command against a
    broadcast operand -- the kernel's own gamma stays null, which is what the
    ABSENT operands say.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3).half()
    model = _GroupNorm(2, 4).half()
    blob, commands, shapes = _lowered(model, (x,))
    assert [command.type for command in commands] == [_LAYER_NORM, _BINARY, _BINARY]
    assert shapes == [(2, 4, 3, 3)], "the group norm's weights are delegate inputs"

    norm, multiply, add = commands
    assert list(norm.params) == [4, 18, _float_bits(1e-5), 0], list(norm.params)
    assert [ref.space for ref in norm.inputs[1:]] == [ABSENT, ABSENT], (
        "the kernel was handed a gamma it does not have a use for"
    )
    # One row per (batch, group) of (C/group) * H * W values, and the row the
    # weight is read under is a broadcast over everything but the channel axis:
    # its strides are zero everywhere except where the channel axis is.
    assert list(multiply.params[:4]) == [72, 72, 4, _MUL], list(multiply.params[:4])
    assert list(add.params[:4]) == [72, 72, 4, _ADD], list(add.params[:4])
    assert list(multiply.params[8:13]) == [4, 2, 4, 3, 3], (
        "the broadcast is not over the output's own rank and shape"
    )
    assert list(multiply.params[17:21]) == [36, 9, 3, 1], list(multiply.params[17:21])
    assert list(multiply.params[25:29]) == [0, 1, 0, 0], (
        "the weight is not read as one value per channel"
    )


def test_group_norm_matches_torch():
    torch.manual_seed(0)
    for shape, groups in (((2, 4, 3, 3), 2), ((1, 6, 4, 5), 3), ((4, 4, 16, 16), 1)):
        x = torch.randn(*shape).half()
        model = _GroupNorm(groups, shape[1]).half()
        blob, _, _ = _lowered(model, (x,))
        with torch.no_grad():
            expected = _group_norm_reference(
                x, groups, model.norm.weight, model.norm.bias, 1e-5
            )
        np.testing.assert_allclose(
            _host(blob, (x,)).astype(np.float32),
            expected.numpy().reshape(-1),
            rtol=2e-3,
            atol=_TOLERANCE,
        )


def test_group_norm_without_affine_drops_the_two_element_wise_commands():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3).half()
    model = _GroupNorm(2, 4, affine=False).half()
    blob, commands, _ = _lowered(model, (x,))
    assert [command.type for command in commands] == [_LAYER_NORM]
    expected = _group_norm_reference(x, 2, None, None, 1e-5)
    np.testing.assert_allclose(
        _host(blob, (x,)).astype(np.float32),
        expected.numpy().reshape(-1),
        rtol=2e-3,
        atol=_TOLERANCE,
    )


def test_instance_norm_is_one_norm_command_per_batch_channel_pair():
    """The batch axis the exporter flattened is the row the command reduces.

    The affine is per row rather than per channel of a nested axis: the graph's
    `repeat` out to `N*C` is left on the portable kernels, so the two weights
    arrive as delegate inputs already the length of the row count and are read
    with a single stride-zero broadcast.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3).half()
    model = _InstanceNorm(4).half()
    with torch.no_grad():
        model.norm.weight.copy_(torch.randn(4).half())
        model.norm.bias.copy_(torch.randn(4).half())
    blob, commands, shapes = _lowered(model, (x,))
    assert shapes == [(2, 4, 3, 3), (8,), (8,)], shapes
    assert [command.type for command in commands] == [
        _LAYER_NORM,
        _BINARY,
        _BINARY,
        3,
    ]
    norm, multiply = commands[0], commands[1]
    assert list(norm.params[:2]) == [8, 9], (
        "the instance norm does not reduce one row per (batch, channel)"
    )
    assert list(multiply.params[:4]) == [72, 72, 8, _MUL], list(multiply.params[:4])
    assert list(multiply.params[8:13]) == [4, 1, 8, 3, 3], list(multiply.params[8:13])
    assert list(multiply.params[17:21]) == [0, 9, 3, 1], (
        "the activation is not walked as [batch, channel, height, width]"
    )
    assert list(multiply.params[25:29]) == [0, 1, 0, 0], (
        "the weight is not read as one value per row"
    )
    with torch.no_grad():
        weight, bias = model.norm.weight, model.norm.bias
        repeated = [weight.detach().repeat(2), bias.detach().repeat(2)]
        expected = _instance_norm_reference(x, *repeated, 1e-5)
    np.testing.assert_allclose(
        _host(blob, (x, *repeated)).astype(np.float32),
        expected.numpy().reshape(-1),
        rtol=2e-3,
        atol=_TOLERANCE,
    )


def test_instance_norm_without_affine_is_the_norm_and_the_view_back():
    """One norm command and one blit: the view back to the exported shape.

    The view is where the result has to leave in the graph's own shape, and the
    alias path cannot take it because the delegate's result is written to a
    region of its own.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3).half()
    model = _InstanceNorm(4, affine=False).half()
    blob, commands, shapes = _lowered(model, (x,))
    assert shapes == [(2, 4, 3, 3)], shapes
    assert [command.type for command in commands] == [_LAYER_NORM, 3]
    expected = _instance_norm_reference(x, None, None, 1e-5)
    np.testing.assert_allclose(
        _host(blob, (x,)).astype(np.float32),
        expected.numpy().reshape(-1),
        rtol=2e-3,
        atol=_TOLERANCE,
    )


def test_a_norm_with_a_large_offset_loses_precision_that_is_the_shared_kernels():
    """The offset case degrades, and the degradation is the kernel's, not this
    emitter's.

    `htp_ops_layer_norm` accumulates `sum(x*x)` and `sum(x)` in fp32 and forms
    `var = sqsum/n - mean*mean`, which cancels when the mean is large next to the
    standard deviation: at an offset of 100 with a deviation of 0.1 the second
    moment is 10000 and the variance is 0.01, so the seven digits fp32 carries
    leave about two of them.

    A group norm over one group normalizes the trailing dims of each batch, which
    is the same span a layer norm over those dims reduces, so the two emit the
    very same command and answer with the very same bits. That is what makes the
    error the shared kernel's rather than this emitter's, and it is why the gate
    does not refuse an offset input: a bound on the mean would be inventing a
    limitation torch does not have.
    """
    torch.manual_seed(0)
    x = (torch.randn(2, 4, 3, 3) * 0.1 + 100.0).half()
    groups = _GroupNorm(1, 4, affine=False).half()
    layer = torch.nn.LayerNorm([4, 3, 3], elementwise_affine=False).half()

    group_blob, group_commands, _ = _lowered(groups, (x,))
    layer_blob, layer_commands, _ = _lowered(layer, (x,))
    assert list(group_commands[0].params) == list(layer_commands[0].params), (
        "the two paths no longer reduce the same span"
    )
    group_result = _host(group_blob, (x,))
    assert np.array_equal(group_result, _host(layer_blob, (x,))), (
        "the same command answered differently on the two paths"
    )
    want = _group_norm_reference(x, 1, None, None, 1e-5).numpy().reshape(-1)
    error = np.max(np.abs(group_result.astype(np.float32) - want))
    assert error > 0.1, f"the offset case did not degrade: {error}"


def _group_norm_node(
    shape, groups, eps=1e-5, channels=None, batch=None, spatial=None, weight=4
):
    """A native_group_norm with the values its gate reads, and a getitem on it."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=F16)
    scale = graph.placeholder("weight")
    scale.meta["val"] = torch.empty(weight, dtype=F16)
    offset = graph.placeholder("bias")
    offset.meta["val"] = torch.empty(weight, dtype=F16)
    inner = 1
    for extent in shape[2:]:
        inner *= extent
    norm = graph.call_function(
        NATIVE_GROUP_NORM,
        args=(
            source,
            scale,
            offset,
            shape[0] if batch is None else batch,
            shape[1] if channels is None else channels,
            inner if spatial is None else spatial,
            groups,
            eps,
        ),
    )
    norm.meta["val"] = (torch.empty(shape, dtype=F16), None, None)
    sink = graph.call_function(GETITEM, args=(norm, 0))
    sink.meta["val"] = torch.empty(shape, dtype=F16)
    return norm


def test_the_group_norm_gate_reads_the_view_the_command_takes():
    """The gate answers, and its answer is yes, on the shape the op names."""
    support = HexagonOperatorSupport()
    node = _group_norm_node((2, 4, 3, 3), 2)
    assert group_norm_normalizes_one_group_per_row(node)
    assert support.is_node_supported({}, node)


def test_a_group_count_that_does_not_divide_the_channels_is_refused():
    """The rows are whole groups, so a count that splits one is not a row.

    `C // group` would silently hand the kernel a row covering part of a group
    and part of the next -- a wrong answer rather than an error, which is the
    shape of failure this gate exists to prevent.
    """
    node = _group_norm_node((2, 4, 3, 3), 3)
    assert not group_norm_normalizes_one_group_per_row(node)
    assert not HexagonOperatorSupport().is_node_supported({}, node)


def test_a_group_norm_that_declares_other_extents_than_its_input_is_refused():
    """N, C and HxW are the row count and the row length, or the view lies.

    Each is a separate way to describe a row the operand's bytes do not have:
    the wrong row count reads past the end of the buffer, and the wrong row
    length normalizes over the wrong span.
    """
    support = HexagonOperatorSupport()
    for kwargs in (
        {"batch": 1},
        {"channels": 2},
        {"spatial": 4},
    ):
        node = _group_norm_node((2, 4, 3, 3), 2, **kwargs)
        assert not group_norm_normalizes_one_group_per_row(node), kwargs
        assert not support.is_node_supported({}, node), kwargs


def test_a_group_norm_whose_weight_is_not_one_per_channel_is_refused():
    """The affine is a broadcast over the channel axis and nothing else."""
    node = _group_norm_node((2, 4, 3, 3), 2, weight=6)
    assert not group_norm_normalizes_one_group_per_row(node)


def test_a_group_norm_over_a_transposed_operand_is_refused():
    """The rows are contiguous only when the operand is.

    The command walks [outer][inner] over the buffer as it lies, so an operand
    whose channel axis is not the second one would be reduced along whatever
    axis the strides happen to put there.
    """
    graph = torch.fx.Graph()
    value = torch.empty(2, 3, 4, 5, dtype=F16).transpose(1, 2)
    source = graph.placeholder("x")
    source.meta["val"] = value
    scale = graph.placeholder("weight")
    scale.meta["val"] = torch.empty(4, dtype=F16)
    norm = graph.call_function(
        NATIVE_GROUP_NORM,
        args=(source, scale, None, 2, 4, 5, 2, 1e-5),
    )
    norm.meta["val"] = (torch.empty(value.shape, dtype=F16), None, None)
    assert not value.is_contiguous()
    assert not group_norm_normalizes_one_group_per_row(norm)


def test_a_second_reader_of_the_group_norm_is_refused():
    """The result is the getitem's slot; a second reader has no slot to read.

    The gate asks for the norm's first output to be read by the getitem alone,
    which is what keeps the command's destination and the graph's value the same
    thing.
    """
    node = _group_norm_node((2, 4, 3, 3), 2)
    norm = node.args[0]
    reader = norm.graph.call_function(GETITEM, args=(norm, 0))
    reader.meta["val"] = torch.empty(2, 4, 3, 3, dtype=F16)
    norm.graph.output(reader)
    assert not group_norm_normalizes_one_group_per_row(norm)


def test_the_mean_of_a_group_norm_does_not_delegate():
    """The second output is the op's mean and the command writes no such thing."""
    node = _group_norm_node((2, 4, 3, 3), 2)
    norm = node.args[0]
    second = norm.graph.call_function(GETITEM, args=(norm, 1))
    second.meta["val"] = torch.empty(2, 2, dtype=F16)
    norm.graph.output(second)
    assert not HexagonOperatorSupport().is_node_supported({}, second)


def _batch_norm_node(shape, eps=1e-5, training=True, channels=None, rows=None):
    """A no-stats batch norm with the values its gate reads, and a getitem."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=F16)
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty(shape[1] if rows is None else rows, dtype=F16)
    norm = graph.call_function(
        BATCH_NORM_NO_STATS,
        args=(source, weight, None, training, 0.1, eps),
    )
    norm.meta["val"] = (torch.empty(shape, dtype=F16), None, None)
    sink = graph.call_function(GETITEM, args=(norm, 0))
    sink.meta["val"] = torch.empty(shape, dtype=F16)
    return norm


def test_the_batch_norm_gate_reads_one_row_per_channel():
    """The view instance_norm exports is the one the kernel can reduce."""
    node = _batch_norm_node((1, 8, 3, 3))
    assert batch_norm_normalizes_one_span(node)
    assert HexagonOperatorSupport().is_node_supported({}, node)


def test_a_batch_norm_over_more_than_one_batch_is_refused():
    """Per-channel statistics of a real batch are not one row per channel.

    With a batch of two the op normalizes each channel over both batches, while
    one command's row is a single contiguous span of the buffer. The command
    would answer the per-batch statistics of the first batch only -- which is
    what the control beside this test lowers without the gate and measures.
    """
    node = _batch_norm_node((2, 8, 3, 3))
    assert not batch_norm_normalizes_one_span(node)
    assert not HexagonOperatorSupport().is_node_supported({}, node)


def test_a_batch_norm_asked_for_its_training_statistics_is_refused():
    """The no-stats op takes the flag, and the batch's statistics are the graph's.

    `training=False` on this overload returns nothing at all in this torch build
    (a segfault, measured), so the graphs the exporter produces carry True and a
    False here means a caller the shape of whose statistics is not the batch's.
    """
    node = _batch_norm_node((1, 8, 3, 3), training=False)
    assert not batch_norm_normalizes_one_span(node)
    assert not HexagonOperatorSupport().is_node_supported({}, node)


def test_a_batch_norm_whose_weight_is_not_one_per_channel_is_refused():
    node = _batch_norm_node((1, 8, 3, 3), rows=4)
    assert not batch_norm_normalizes_one_span(node)


def _patches(blob):
    """{command: [(param, scale)]} from the trailer, as the runtime reads it."""
    offset = blob.find(struct.pack("<I", _TRAILER_MAGIC))
    assert offset >= 0, "a dynamic graph emits a trailer"
    header = struct.unpack_from("<7I", blob, offset)
    records = [
        struct.unpack_from("<4i", blob, offset + 28 + record * 16)
        for record in range(header[5])
    ]
    patches = {}
    for index, param, scale, _mode in records:
        patches.setdefault(index, []).append((param, scale))
    return header[4], patches


def test_a_dynamic_batch_axis_patches_the_row_count():
    """The rows are N*G, so the batch axis is a product the length scales.

    A graph whose batch axis is dynamic carries `sym_size` where a static one
    carries the op's own arguments, and the row count is built from the input's
    shape rather than from those arguments for exactly that reason: an argument
    that is a node has no bound to take. What the command then says is the
    export's bound plus a patch the runtime applies, which is the pairing every
    other shape-derived param here has.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3).half()
    blob, commands, _ = _lowered(
        _GroupNorm(2, 4).half(),
        (x,),
        dynamic_shapes={"x": {0: Dim("n", min=1, max=8)}},
    )
    assert [command.type for command in commands] == [_LAYER_NORM, _BINARY, _BINARY]
    # The bound is eight batches of two groups, so the row count is sixteen and
    # the runtime recomputes it as `scale * length` with the group count. The
    # inner span is (C/G)*HxW and does not hold the length, so the norm carries
    # no second record; the affine commands' own patches are the elementwise
    # emitter's, and they follow the same length.
    assert list(commands[0].params[:2]) == [16, 18], list(commands[0].params[:2])
    longest, patches = _patches(blob)
    assert longest == 8, longest
    assert {index: records for index, records in patches.items() if index == 0} == {
        0: [(0, 2)]
    }, patches


def test_a_dynamic_spatial_extent_patches_the_inner_count():
    """The inner span is (C/G)*HxW, so a dynamic spatial extent scales it too.

    Without the second patch the command would reduce the allocation rather than
    the buffer -- it would fold in whatever the arena holds past the run-time
    length -- which is a wrong answer rather than a failure.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6).half()
    blob, commands, _ = _lowered(
        _GroupNorm(2, 4).half(),
        (x,),
        dynamic_shapes={"x": {2: Dim("tokens", min=1, max=12)}},
    )
    assert [command.type for command in commands] == [_LAYER_NORM, _BINARY, _BINARY]
    # Two channels per group and the bound of twelve, so twenty-four values a
    # row; the row count is the batch and the group count, which the length does
    # not touch, so the norm carries no record for it.
    assert list(commands[0].params[:2]) == [4, 24], list(commands[0].params[:2])
    longest, patches = _patches(blob)
    assert longest == 12, longest
    assert {index: records for index, records in patches.items() if index == 0} == {
        0: [(1, 2)]
    }, patches


def _delegate_free_blob(model, args):
    """The blob this model's subgraph makes with the gate taken out of the way.

    `HexagonBackend.preprocess` emits every node it is given, so lowering the
    graph directly is exactly what a partitioner that admitted everything would
    do. What the command then says is the measure of what the gate is holding
    back.
    """
    program = to_edge(export(model, tuple(args))).exported_program()
    return bytes(HexagonBackend.preprocess(program, []).processed_bytes)


def test_the_batch_of_one_gate_is_load_bearing():
    """Without it the command describes half the operand and the half is one batch.

    A batch-of-two no-stats batch norm lowered straight to the command -- which
    is what the gate refuses -- emits `rows = 8, inner = 9`: 72 of the operand's
    144 elements, which is the first batch's own [C][H*W] block, left there
    because the row count is the channel axis and nothing else. The second batch
    is not described by the command at all, and the host model will not run it
    either: it reads an operand of 144 elements as a `[8][9]` view and says so.
    The two together are why the shape has to be refused rather than answered.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 3).half()
    model = torch.nn.BatchNorm2d(8, affine=False, track_running_stats=False).half()
    model.eval()
    assert not batch_norm_normalizes_one_span(_batch_norm_node((2, 8, 3, 3)))
    blob = _delegate_free_blob(model, (x,))
    _, commands = read_blob(blob)
    assert list(commands[0].params[:2]) == [8, 9], list(commands[0].params[:2])
    assert 8 * 9 == x.numel() // 2, "the command does not cover one of the two batches"
    with pytest.raises(ValueError, match="cannot reshape"):
        execute(blob, [x.numpy()])
