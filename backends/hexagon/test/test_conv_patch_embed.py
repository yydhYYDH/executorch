# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The patch-embed convolution rewrite: what it changes, and what it must not."""

import copy
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.conv_patch_embed import (  # noqa: E402
    DecomposePatchEmbed,
    patch_embed,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.test.blob_interpreter import (  # noqa: E402
    BATCH_MATMUL,
    BINARY_ELEMENTWISE,
    execute,
    read_blob,
)
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_CONVOLUTION = "aten.convolution.default"


def _decompose(program):
    """The pass, on a program obtained without a pass manager."""
    return DecomposePatchEmbed()(program).exported_program


class _PatchEmbed(torch.nn.Module):
    """The shape Qwen3VLVisionPatchEmbed has: flatten, convolve, flatten."""

    def __init__(self, in_channels=3, embed_dim=8, temporal=2, patch=4):
        super().__init__()
        self.in_channels = in_channels
        self.temporal = temporal
        self.patch = patch
        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv3d(
            in_channels,
            embed_dim,
            (temporal, patch, patch),
            stride=(temporal, patch, patch),
            bias=True,
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.temporal, self.patch, self.patch)
        return self.proj(x).view(-1, self.embed_dim)


def _edge_program(model, patches=2, dtype=torch.float32):
    model = model.eval().to(dtype)
    inner = model.in_channels * model.temporal * model.patch * model.patch
    x = torch.randn(patches, inner, dtype=dtype)
    return to_edge(export(model, (x,))).exported_program(), x


def _nodes(program):
    return [
        getattr(node.target, "__name__", str(node.target))
        for node in program.graph_module.graph.nodes
    ]


def _delegates(program):
    """The delegate submodules of a lowered graph, in node order."""
    graph_module = program.graph_module
    return [
        graph_module.get_submodule(str(node.args[0].target))
        for node in graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]


def _census(program):
    """Delegate counts by backend id, and the ops each one holds."""
    counts = {}
    ops = {}
    for delegate in _delegates(program):
        backend = delegate.backend_id
        counts[backend] = counts.get(backend, 0) + 1
        ops.setdefault(backend, []).extend(
            getattr(inner.target, "__name__", str(inner.target))
            for inner in delegate.original_module.graph_module.graph.nodes
            if inner.op == "call_function"
        )
    return counts, ops


def _delegate_inputs(delegate):
    """The placeholders of a delegate, in the order the runtime feeds them."""
    return [
        node
        for node in delegate.original_module.graph_module.graph.nodes
        if node.op == "placeholder"
    ]


def _matches(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if patch_embed(node, program) is not None
    ]


def test_the_rewrite_computes_the_same_convolution():
    """Each shape compared element by element against the convolution it replaces.

    In fp32 the two differ by the summation order of the matmul. In fp16, one
    ulp of the accumulation is all the difference there can be, so half an ulp
    more than that at this magnitude is the bar.
    """
    for model, patches, dtype, bound in (
        (_PatchEmbed(3, 8, 2, 4), 2, torch.float32, 1e-5),
        (_PatchEmbed(1, 3, 1, 2), 1, torch.float32, 1e-5),
        (_PatchEmbed(4, 5, 2, 3), 3, torch.float32, 1e-5),
        (_PatchEmbed(3, 8, 2, 4), 2, torch.float16, 2**-9),
        (_PatchEmbed(1, 3, 1, 2), 1, torch.float16, 2**-9),
    ):
        program, x = _edge_program(model, patches, dtype)
        assert len(_matches(program)) == 1, "the pattern was not matched"
        before = copy.deepcopy(program)
        program = _decompose(program)
        # It really is the rewritten graph being executed, not the same one.
        assert _CONVOLUTION not in _nodes(program)
        assert "aten.mm.default" in _nodes(program), _nodes(program)

        with torch.no_grad():
            expected = before.module()(x)
            got = program.module()(x)
        assert got.shape == expected.shape
        worst = float((got - expected).abs().max())
        assert worst < bound, f"{dtype} rewrite differs by {worst}"


def test_the_whole_patch_embed_lands_in_one_hexagon_partition():
    """Every op of the rewrite, counted by backend_id and not by delegate count."""
    model = _PatchEmbed(3, 1024, 2, 16).eval().to(torch.float16)
    x = torch.randn(2, 3 * 2 * 16 * 16, dtype=torch.float16)
    lowered = to_edge_transform_and_lower(
        export(model, (x,)),
        transform_passes=[DecomposePatchEmbed()],
        partitioner=[HexagonPartitioner()],
    ).exported_program()

    census, ops = _census(lowered)
    assert census == {"HexagonBackend": 1}, census
    assert _CONVOLUTION not in _nodes(lowered)
    assert all(_CONVOLUTION not in op for op in ops["HexagonBackend"])
    assert sorted(ops["HexagonBackend"]) == sorted(
        [
            "aten.view_copy.default",
            "aten.mm.default",
            "aten.add.Tensor",
            "aten.view_copy.default",
        ]
    ), ops["HexagonBackend"]


def test_the_unused_convolution_weight_is_dropped():
    """The weight the convolution read is gone; the bias stays an input."""
    program, _ = _edge_program(_PatchEmbed(3, 8, 2, 4))
    program = _decompose(program)
    names = [spec.target for spec in program.graph_signature.input_specs]
    assert "proj.weight" not in names, names
    assert "proj.bias" in names, names


@pytest.mark.parametrize(
    "model, spatial, one_window",
    [
        # Each of these keeps everything the pattern asks for except one
        # condition, so a match would be a rewrite over different arithmetic.
        (torch.nn.Conv3d(1, 4, 3, stride=2, bias=True), 4, True),  # kernel != stride
        (torch.nn.Conv3d(1, 4, 2, stride=2, padding=1, bias=True), 1, True),  # padding
        (
            torch.nn.Conv3d(1, 4, 2, stride=2, dilation=2, bias=True),
            3,
            True,
        ),  # dilation
        (torch.nn.Conv3d(4, 4, 2, stride=2, groups=2, bias=True), 2, True),  # groups
        (torch.nn.ConvTranspose3d(1, 4, 1, stride=1, bias=True), 1, True),  # transposed
        (
            torch.nn.Conv2d(1, 4, 2, stride=2, bias=True),
            2,
            True,
        ),  # not five-dimensional
        # Disjoint windows, kernel equal to stride -- but four of them, not one,
        # so the batch is not the row count this rewrite would matmul over.
        (torch.nn.Conv3d(1, 4, 2, stride=2, bias=True), 4, False),
    ],
)
def test_a_convolution_the_pattern_does_not_describe_is_left_alone(
    model, spatial, one_window
):
    """One condition short of the pattern: nothing may be rewritten."""
    x = torch.randn(1, model.in_channels, *([spatial] * (model.weight.dim() - 2)))
    program = to_edge(export(model.eval(), (x,))).exported_program()
    # The setup, before the pass: an edge convolution whose output window is one
    # patch wide is on the graph, so a match is only withheld by the arithmetic.
    convolutions = [
        node
        for node in program.graph_module.graph.nodes
        if getattr(node.target, "__name__", "") == _CONVOLUTION
    ]
    assert len(convolutions) == 1, _nodes(program)
    windows = convolutions[0].meta["val"].shape[2:]
    assert all(size == 1 for size in windows) == one_window, windows

    names = _nodes(program)
    program = _decompose(program)
    assert _nodes(program) == names, "the graph was rewritten anyway"
    assert _matches(program) == []


def test_a_convolution_over_a_caller_supplied_weight_is_left_alone():
    """An operand that arrives as an input is not a constant to fold."""

    class _DynamicWeight(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.conv3d(
                x, weight, bias, stride=(2, 2, 2), padding=0
            )

    x = torch.randn(1, 1, 2, 2, 2)
    weight = torch.randn(4, 1, 2, 2, 2)
    bias = torch.randn(4)
    program = to_edge(export(_DynamicWeight(), (x, weight, bias))).exported_program()
    names = _nodes(program)
    program = _decompose(program)
    assert _nodes(program) == names


def test_the_rewrite_survives_a_dynamic_patch_count():
    """A symbolic row count still delegates, on both call paths."""
    model = _PatchEmbed(3, 8, 2, 4).eval().to(torch.float16)
    x = torch.randn(2, 3 * 2 * 4 * 4, dtype=torch.float16)
    program = to_edge(
        export(
            model,
            (x,),
            dynamic_shapes=({0: torch.export.Dim("patches")},),
        )
    ).exported_program()
    program = _decompose(program)
    assert len(_matches(program)) == 0

    lowered = to_edge_transform_and_lower(
        export(model, (x,), dynamic_shapes=({0: torch.export.Dim("patches")},)),
        transform_passes=[DecomposePatchEmbed()],
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    census, ops = _census(lowered)
    assert census == {"HexagonBackend": 1}, census
    assert "aten.mm.default" in ops["HexagonBackend"], ops["HexagonBackend"]


def test_the_emitted_commands_are_the_matmul_and_the_bias():
    """The two commands, run through the host model of the DSP.

    Both commands are shared with paths already on the device; what this pins is
    that the rewrite reaches them, and with a bias whose broadcast strides the
    kernel can walk. The operands are built from the delegate's own placeholders
    rather than from a list written by hand: a delegate is fed positionally, so
    a wrong guess here compares the wrong tensor and still looks like a pass.
    """
    model = _PatchEmbed(3, 8, 2, 4).eval().to(torch.float16)
    inner = 3 * 2 * 4 * 4
    x = torch.randn(2, inner, dtype=torch.float16)
    lowered = to_edge_transform_and_lower(
        export(model, (x,)),
        transform_passes=[DecomposePatchEmbed()],
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    (delegate,) = _delegates(lowered)

    _, commands = read_blob(delegate.processed_bytes)
    assert [command.type for command in commands] == [BATCH_MATMUL, BINARY_ELEMENTWISE]

    weight = model.proj.weight.detach().float().reshape(8, inner).t().contiguous()
    placeholders = _delegate_inputs(delegate)
    names = [node.name for node in placeholders]
    # The transposed weight is the constant the pass lifted, under the generated
    # name the lift gives it (the only lifted constant in this graph).
    (lifted,) = [name for name in names if name.startswith("_lifted_tensor_constant")]
    by_name = {
        "p_proj_bias": model.proj.bias.detach().to(torch.float16).numpy(),
        lifted: weight.to(torch.float16).numpy(),
        "x": x.numpy(),
    }
    # What the order is, pinned: the constants the graph carries come ahead of
    # the activation. The mapping from name to tensor is the part that is read.
    assert names == ["p_proj_bias", lifted, "x"], names
    operands = [by_name[name] for name in names]
    assert [tuple(node.meta["val"].shape) for node in placeholders] == [
        operand.shape for operand in operands
    ]

    got = np.frombuffer(
        execute(delegate.processed_bytes, operands)[0], dtype=np.float16
    )
    # The DSP rounds the product to fp16 in an activation before the bias is
    # added, so the reference does the same rather than adding in fp32.
    expected = (
        ((x.float() @ weight).half().float() + model.proj.bias.detach().float())
        .half()
        .numpy()
        .reshape(-1)
    )
    assert np.array_equal(got, expected), (
        "the DSP model differs by "
        f"{np.abs(got.astype(np.float32) - expected.astype(np.float32)).max()}"
    )


def test_the_pass_refuses_a_program_that_still_holds_the_aten_convolution():
    """A silent no-op here would look exactly like a program with no matches."""
    model = _PatchEmbed(3, 8, 2, 4)
    program = export(model.eval(), (torch.randn(2, 96),))
    with pytest.raises(RuntimeError, match="edge program"):
        DecomposePatchEmbed()(program)


def test_a_graph_with_no_patch_embed_is_returned_untouched():
    """Nothing to match means nothing changes, not a rewritten graph."""

    class _Other(torch.nn.Module):
        def forward(self, a, b):
            return torch.mm(a, b)

    a = torch.randn(4, 8)
    b = torch.randn(8, 3)
    program = to_edge(export(_Other(), (a, b))).exported_program()
    before = _nodes(program)
    parameters = set(program.state_dict)
    program = _decompose(program)
    assert _nodes(program) == before
    assert set(program.state_dict) == parameters
