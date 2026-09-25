# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vision attention: the layout the kernel walks, and the command that does it.

Two things are checked, and separately. The addressing is checked against a
second derivation of the same index -- the kernel's own expression transcribed
from `attention_entry.cc`, against the products of the extents a row-major
`[batch, tokens, heads, headDim]` tensor implies -- because a transcription and
a formula that are wrong in the same way agree with each other perfectly. The
command is then checked end to end: a real export through the real fusion pass
and the real partitioner produces a delegate whose blob carries one command of
the right type with the right params and the right two outputs, and the host
interpreter runs it against torch's own answer.

What none of this covers is the DSP. The attention here is a second
implementation of what the vendored kernel does, and the layout is the one the
emitter believed the kernel walks; agreement means the source was read the same
way twice. The README's "unverified on device" list is the rest of it.
"""

import pathlib
import struct

import numpy as np
import pytest
import torch


import blob_interpreter
from blob_interpreter import (
    execute,
    read_blob,
    VISION_ATTENTION_FP16,
    vision_attention_kernel_offset,
    vision_attention_row_major_offset,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.backends.hexagon.vision_attention import (
    FuseVisionAttention,
    vision_attention_pattern,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from torch.export import Dim, export

BACKEND_ID = "HexagonBackend"

#: A ViT attention block's geometry with all four extents different, so a
#: reading that swaps any two of them produces different numbers.
BATCH, TOKENS, HEADS, HEAD_DIM = 2, 6, 4, 8
EMBED = HEADS * HEAD_DIM


class _Attention(torch.nn.Module):
    """The decomposition a vision tower exports: split, matmul, softmax, matmul.

    Written the way `CLIPAttention` and `SiglipAttention` write it, which is the
    form that carries the head transposes the fused op consumes.
    """

    def __init__(self, embed=EMBED, heads=HEADS, causal=False, masked=False):
        super().__init__()
        self.heads, self.head_dim = heads, embed // heads
        self.causal, self.masked = causal, masked
        self.query = torch.nn.Linear(embed, embed, bias=True)
        self.key = torch.nn.Linear(embed, embed, bias=True)
        self.value = torch.nn.Linear(embed, embed, bias=True)
        self.project = torch.nn.Linear(embed, embed, bias=True)
        self.scale = (embed // heads) ** -0.5

    def forward(self, x):
        batch, tokens, _ = x.shape
        head = lambda tensor: tensor.view(  # noqa: E731
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        query, key, value = head(self.query(x)), head(self.key(x)), head(self.value(x))
        logits = (query @ key.transpose(-2, -1)) * self.scale
        if self.causal:
            logits = logits + torch.triu(
                torch.full((tokens, tokens), float("-inf"), dtype=logits.dtype), 1
            )
        if self.masked:
            logits = logits + torch.full((tokens, tokens), 0.5, dtype=logits.dtype)
        weights = logits.softmax(dim=-1)
        return self.project(
            (weights @ value).transpose(1, 2).reshape(batch, tokens, -1)
        )


class _CrossAttention(torch.nn.Module):
    """A query run as long as the context is not the same function as a square one."""

    def __init__(self, embed=EMBED, heads=HEADS):
        super().__init__()
        self.heads, self.head_dim = heads, embed // heads
        self.query = torch.nn.Linear(embed, embed, bias=True)
        self.key = torch.nn.Linear(embed, embed, bias=True)
        self.value = torch.nn.Linear(embed, embed, bias=True)
        self.scale = (embed // heads) ** -0.5

    def forward(self, x, context):
        batch, tokens, _ = x.shape
        other = context.shape[1]
        head = lambda tensor, length: tensor.view(  # noqa: E731
            batch, length, self.heads, self.head_dim
        ).transpose(1, 2)
        query = head(self.query(x), tokens)
        key = head(self.key(context), other)
        value = head(self.value(context), other)
        weights = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return (weights @ value).transpose(1, 2).reshape(batch, tokens, -1)


class _HeadMajor(torch.nn.Module):
    """Operands that are already `[batch, heads, tokens, dim]`, with no split."""

    def __init__(self, head_dim=HEAD_DIM):
        super().__init__()
        self.scale = head_dim**-0.5

    def forward(self, query, key, value):
        weights = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return weights @ value


class _RuntimeScale(torch.nn.Module):
    """A scale the graph only learns at run time, which no param can carry."""

    def __init__(self, embed=EMBED, heads=HEADS):
        super().__init__()
        self.heads, self.head_dim = heads, embed // heads
        self.query = torch.nn.Linear(embed, embed, bias=True)
        self.key = torch.nn.Linear(embed, embed, bias=True)
        self.value = torch.nn.Linear(embed, embed, bias=True)

    def forward(self, x, scale):
        batch, tokens, _ = x.shape
        head = lambda tensor: tensor.view(  # noqa: E731
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        query, key, value = head(self.query(x)), head(self.key(x)), head(self.value(x))
        logits = (query @ key.transpose(-2, -1)) * scale
        return (
            (logits.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, tokens, -1)
        )


def _edge(model, inputs, dynamic_shapes=None):
    return to_edge(
        export(model.eval(), tuple(inputs), dynamic_shapes=dynamic_shapes)
    ).exported_program()


def _matches(program):
    """The softmax nodes of a program that the fusion recognises."""
    return [
        node.name
        for node in program.graph_module.graph.nodes
        if vision_attention_pattern(node, program) is not None
    ]


def _delegates(program):
    """The delegate submodules of a lowered graph, in node order."""
    graph_module = program.graph_module
    return [
        graph_module.get_submodule(str(node.args[0].target))
        for node in graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]


def _lowered(model, inputs, dynamic_shapes=None, fuse=True):
    """The lowered program and the blob of every delegate it produced."""
    program = export(model.eval(), tuple(inputs), dynamic_shapes=dynamic_shapes)
    lowered = to_edge_transform_and_lower(
        program,
        transform_passes=[FuseVisionAttention()] if fuse else None,
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    delegates = _delegates(lowered)
    for delegate in delegates:
        assert delegate.backend_id == BACKEND_ID
    return lowered, [bytes(delegate.processed_bytes) for delegate in delegates]


def _only_attention(blobs):
    """The one blob carrying a VISION_ATTENTION_FP16 command, and that command."""
    found = [
        (blob, command)
        for blob in blobs
        for command in read_blob(blob)[1]
        if command.type == VISION_ATTENTION_FP16
    ]
    assert len(found) == 1, f"{len(found)} vision attention commands, not one"
    return found[0]


def _scale_bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


@pytest.mark.parametrize(
    "batch, tokens, heads, head_dim", [(2, 5, 3, 7), (1, 6, 4, 8), (3, 1, 1, 1)]
)
def test_the_token_major_index_is_the_same_written_two_ways(
    batch, tokens, heads, head_dim
):
    """The kernel's expression and the products of the extents agree.

    The first is transcribed from `attention_entry.cc:36,41,57,61`; the second
    is what a row-major `[batch, tokens, heads, headDim]` tensor implies. They
    share no term, so agreeing over every element is the layout claim checked
    rather than restated.
    """
    for b in range(batch):
        for token in range(tokens):
            for head in range(heads):
                for dimension in range(head_dim):
                    assert vision_attention_kernel_offset(
                        b, token, head, dimension, tokens, heads, head_dim
                    ) == vision_attention_row_major_offset(
                        b, token, head, dimension, tokens, heads, head_dim
                    )


def test_the_head_major_read_would_be_a_different_tensor():
    """The layout test above is not vacuous: the other reading disagrees.

    A `[batch, heads, tokens, headDim]` tensor -- what a batched matmul in torch
    already holds -- puts the same elements in another order, so a kernel read
    that got it backwards would produce other numbers, and this is the shape
    that would show it.
    """
    batch, tokens, heads, head_dim = 2, 5, 3, 7
    differences = [
        (b, token, head, dimension)
        for b in range(batch)
        for token in range(tokens)
        for head in range(heads)
        for dimension in range(head_dim)
        if vision_attention_kernel_offset(
            b, token, head, dimension, tokens, heads, head_dim
        )
        != ((b * heads + head) * tokens + token) * head_dim + dimension
    ]
    assert len(differences) > batch * tokens * heads * head_dim // 2


def test_the_command_type_is_the_one_the_c_enum_names():
    """The number the emitter writes, re-read from the vendored header.

    Every param is positional and unchecked, so a type that drifted by one would
    run a different kernel with these numbers rather than fail.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "third-party/mnn-htp-ops"
    values = {}
    next_value = 0
    for line in (root / "include/htp_command.h").read_text().splitlines():
        line = line.strip()
        if not line.startswith("DSP_OP_") or "," not in line:
            continue
        name, _, _ = line.partition(",")
        if "=" in name:
            name, _, explicit = name.partition("=")
            next_value = int(explicit.strip())
        values[next_value] = name.strip()
    assert values[VISION_ATTENTION_FP16] == "DSP_OP_VISION_ATTENTION_FP16"
    assert VISION_ATTENTION_FP16 == 43


def test_an_attention_comes_out_as_one_command_and_the_whole_block_delegates():
    """The positive assertions, before any number is compared.

    A lowering that left the attention on the portable kernels would still run
    and still produce the right answer, so the delegate and the command are
    checked for first: one delegate holding the whole block, and in it one
    command of the vision type with the geometry the graph had, three operands
    of the right size, and the two outputs the kernel requires.
    """
    torch.manual_seed(0)
    model = _Attention().eval().half()
    x = torch.randn(BATCH, TOKENS, EMBED, dtype=torch.float16)

    lowered, blobs = _lowered(model, (x,))
    assert len(blobs) == 1, "the fused block is one partition"
    header, commands = read_blob(blobs[0])
    assert [command.type for command in commands].count(VISION_ATTENTION_FP16) == 1

    _blob, command = _only_attention(blobs)
    values = BATCH * TOKENS * HEADS * HEAD_DIM
    # batch, tokens, heads, headDim, scale, maskStride, workspaceBytes.
    assert command.params == [
        BATCH,
        TOKENS,
        HEADS,
        HEAD_DIM,
        _scale_bits(model.scale),
        0,
        TOKENS * 4 + 128,
    ]
    # The operands are the tensors under the head transposes, so they are the
    # `[batch, tokens, heads, headDim]` views of the three projections.
    assert len(command.inputs) == 3
    for operand in command.inputs:
        assert operand.space is blob_interpreter.B.TensorSpace.ACTIVATION
        assert operand.size == values * 2
    assert len({operand.offset for operand in command.inputs}) == 3
    # The kernel fails the command without a workspace, and wants `tokens` fp32
    # scores plus the 127 bytes it aligns its own pointer up by.
    result, workspace = command.outputs
    assert result.size == values * 2
    assert workspace.size >= TOKENS * 4 + 127
    assert header.n_outputs == 1


def test_the_numbers_are_the_ones_attention_computes():
    """The command run on the host, against the module torch runs.

    The interpreter walks the kernel's own addressing, and the reference is the
    model's own forward, so the two do not share an implementation. The fused
    program is compared as well, which is the `vision_attention` op against the
    decomposition it replaces.
    """
    torch.manual_seed(1)
    model = _Attention().eval().half()
    x = torch.randn(BATCH, TOKENS, EMBED, dtype=torch.float16)
    with torch.no_grad():
        reference = model(x).numpy()

    lowered, blobs = _lowered(model, (x,))
    blob, _command = _only_attention(blobs)
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16).reshape(
        reference.shape
    )
    assert np.allclose(got, reference, atol=2e-3)

    # The fused graph run eagerly, which is the `vision_attention` op's own
    # implementation against the decomposition it replaced.
    fused = FuseVisionAttention()(_edge(model, (x,))).exported_program.module()
    with torch.no_grad():
        assert np.allclose(fused(x).numpy(), reference, atol=1e-3)


def _attention_numpy(query, key, value, batch, tokens, heads, head_dim, scale):
    """Attention over `[batch, tokens, heads, headDim]` operands, in numpy.

    The same function the command computes, written from the shape rather than
    from the kernel, so the two can be asked for different answers. The
    operands arrive as flat buffers, which is all the command ever sees, and the
    cast to a contiguous fp32 array first is what makes a transposed argument
    mean the bytes it would have on the wire rather than a view of them.
    """
    wide = lambda tensor: tensor.astype(np.float32).reshape(  # noqa: E731
        batch, tokens, heads, head_dim
    )
    query, key, value = wide(query), wide(key), wide(value)
    scores = np.einsum("bthd,bshd->bhts", query, key) * scale
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhts,bshd->bthd", weights, value).reshape(batch, tokens, -1)


def test_the_operand_layout_the_command_assumes_is_the_one_that_matches():
    """The reading the command does is the one that reproduces the model.

    The operands are the tensors under the head transposes, so their bytes are
    token-major; read head-major -- which is what the graph's own `[batch,
    heads, tokens, dim]` tensors would be -- the same bytes name a different
    function. With all four extents different, that function is a different set
    of numbers, so the comparison below distinguishes the two readings instead
    of accepting whichever one the shape happens to make equal.
    """
    torch.manual_seed(2)
    model = _Attention().eval().half()
    x = torch.randn(BATCH, TOKENS, EMBED, dtype=torch.float16)
    with torch.no_grad():
        reference = model(x)
        split = lambda tensor: tensor.view(BATCH, TOKENS, HEADS, HEAD_DIM)  # noqa: E731
        head = lambda tensor: tensor.transpose(1, 2)  # noqa: E731
        query, key, value = (
            split(model.query(x)),
            split(model.key(x)),
            split(model.value(x)),
        )
        # The attention itself, before the output projection: what the command
        # computes, in torch's own terms.
        weights = ((head(query) @ head(key).transpose(-1, -2)) * model.scale).softmax(
            -1
        )
        attention = (
            (weights @ head(value))
            .transpose(1, 2)
            .reshape(BATCH, TOKENS, EMBED)
            .numpy()
        )
        query, key, value = query.numpy(), key.numpy(), value.numpy()

    token_major = _attention_numpy(
        query, key, value, BATCH, TOKENS, HEADS, HEAD_DIM, model.scale
    )
    # The graph's own `[batch, heads, tokens, dim]` values, laid out as the bytes
    # a command bound to them would read: the same elements in another order.
    head_major = _attention_numpy(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        BATCH,
        TOKENS,
        HEADS,
        HEAD_DIM,
        model.scale,
    )
    assert not np.allclose(token_major, head_major, atol=1e-2)
    assert np.allclose(token_major, attention, atol=2e-3)

    _program, blobs = _lowered(model, (x,))
    blob, _command = _only_attention(blobs)
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16).reshape(
        BATCH, TOKENS, EMBED
    )
    assert np.allclose(got, reference.numpy(), atol=2e-3)
    assert not np.allclose(got, head_major.astype(np.float16), atol=1e-2)


def test_a_masked_attention_is_left_alone():
    """A mask added to the logits is not this command and must not match.

    The kernel takes a mask operand and reads it only when it is given a
    positive stride (attention_entry.cc:49), which is the silent wrong answer
    the FLASH_ATTN emitter already refuses. Nothing here binds one, so a masked
    node has to stay where it is.
    """
    model = _Attention(masked=True).eval()
    x = torch.randn(BATCH, TOKENS, EMBED)
    assert _matches(_edge(model, (x,))) == []


def test_a_causal_attention_is_left_alone():
    """`is_causal` is the one thing bidirectional attention is not."""
    model = _Attention(causal=True).eval()
    x = torch.randn(BATCH, TOKENS, EMBED)
    assert _matches(_edge(model, (x,))) == []


def test_a_cross_attention_is_left_alone():
    """One `tokens` serves both sides, so query and key runs have to be equal.

    A square cross-attention would match, because the command cannot tell where
    its operands came from and the function is the same; an unequal one is a
    different function and gets no command.
    """
    model = _CrossAttention().eval()
    long_x = torch.randn(BATCH, TOKENS, EMBED)
    short_x = torch.randn(BATCH, TOKENS - 2, EMBED)
    assert _matches(_edge(model, (long_x, short_x))) == []


def test_head_major_operands_are_left_alone():
    """A graph that already holds `[batch, heads, tokens, dim]` is another function.

    There are no head splits to consume here, so the operands the emitter would
    bind are head-major bytes the kernel would read as token-major ones.
    """
    model = _HeadMajor().eval()
    tensors = [
        torch.randn(BATCH, HEADS, TOKENS, HEAD_DIM, dtype=torch.float16)
        for _ in range(3)
    ]
    assert _matches(_edge(model, tuple(tensors))) == []


def test_a_scale_that_is_not_a_constant_is_left_alone():
    """The scale is a param, so a value only the caller knows has no command."""
    model = _RuntimeScale().eval()
    x = torch.randn(BATCH, TOKENS, EMBED)
    assert _matches(_edge(model, (x, torch.tensor(0.25)))) == []


def test_a_control_flow_free_sdpa_decomposition_is_left_alone():
    """`scaled_dot_product_attention` decomposes to the same matmuls plus a guard.

    The eager decomposition wraps the softmax in a masked-row guard, and that
    guard is not the identity on a row of minus infinities -- it zeroes the row
    where the kernel would return NaN. So the pattern refuses it rather than
    fusing a function it cannot express.
    """

    class WithSDPA(_Attention):
        def forward(self, x):
            batch, tokens, _ = x.shape
            head = lambda tensor: tensor.view(  # noqa: E731
                batch, tokens, self.heads, self.head_dim
            ).transpose(1, 2)
            return torch.nn.functional.scaled_dot_product_attention(
                head(self.query(x)), head(self.key(x)), head(self.value(x))
            )

    model = WithSDPA().eval()
    x = torch.randn(BATCH, TOKENS, EMBED)
    assert _matches(_edge(model, (x,))) == []


def test_a_dynamic_patch_count_patches_the_token_param():
    """The token count is the run of patches this call was handed.

    The blob carries the geometry of the longest export plus a trailer that
    overwrites param 1 with the run-time length, which is the same mechanism the
    row gather uses for its row count. The interpreter here does not model the
    trailer, so this is the assertion that the count is patched at all.
    """
    torch.manual_seed(3)
    model = _Attention().eval().half()
    x = torch.randn(BATCH, TOKENS, EMBED, dtype=torch.float16)
    _, blobs = _lowered(
        model,
        (x,),
        dynamic_shapes={"x": {1: Dim("tokens", min=1, max=16)}},
    )
    blob, command = _only_attention(blobs)
    assert command.params[1] == 16
    assert command.patch_param == 0xFFFFFFFF
    # The longest run pays for the whole workspace, since the command's own
    # params cannot grow at run time.
    assert command.params[6] == 16 * 4 + 128
    assert command.outputs[1].size == 16 * 4 + 128

    trailer_magic = (0x44594E48).to_bytes(4, "little")
    at = blob.find(trailer_magic)
    assert at >= 0
    magic, version, _input, axis, max_length, patches, _example = struct.unpack_from(
        "<7I", blob, at
    )
    assert (magic, axis, max_length) == (0x44594E48, 1, 16)
    assert version >= 2
    assert patches >= 1
    # (op_index, param_index, scale, add): the token count is the run length.
    records = [
        struct.unpack_from("<4i", blob, at + 28 + 16 * index)
        for index in range(patches)
    ]
    assert any(
        record[1] == 1 and record[2] == 1 and record[3] == 0 for record in records
    )


def test_a_dynamic_batch_keeps_the_node_off_the_dsp():
    """Three of the four numbers are params the run-time length cannot refresh.

    Only the token count has a patch slot, so a symbolic batch, head count or
    head width would emit the traced example and compute a different function at
    run time -- a wrong answer rather than a failure. The pass still fuses, so
    the graph stays correct; what must not happen is a command.
    """
    torch.manual_seed(4)
    model = _Attention().eval().half()
    x = torch.randn(BATCH, TOKENS, EMBED, dtype=torch.float16)
    edge = _edge(model, (x,), dynamic_shapes={"x": {0: Dim("batch", min=1, max=4)}})
    assert len(_matches(edge)) == 1

    lowered = to_edge_transform_and_lower(
        export(
            model,
            (x,),
            dynamic_shapes={"x": {0: Dim("batch", min=1, max=4)}},
        ),
        transform_passes=[FuseVisionAttention()],
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    for blob in (bytes(delegate.processed_bytes) for delegate in _delegates(lowered)):
        assert VISION_ATTENTION_FP16 not in {
            command.type for command in read_blob(blob)[1]
        }, "a symbolic batch reached the DSP"


def test_the_supported_shapes_are_the_ones_the_command_can_carry():
    """The support check, on the shapes the pass can hand it.

    The batch, head count and head width are params, so a node carrying them
    symbolically is not one the command can express; the token count is the one
    axis a patch refreshes.
    """
    from executorch.backends.hexagon.hexagon_ops import vision_attention_is_emittable
    from executorch.backends.hexagon.vision_attention import VISION_ATTENTION

    def call(*shapes, scale=0.5):
        graph = torch.fx.Graph()
        operands = []
        for name, shape in zip("qkv", shapes):
            operand = graph.placeholder(name)
            operand.meta["val"] = torch.empty(shape, dtype=torch.float16)
            operands.append(operand)
        node = graph.call_function(VISION_ATTENTION, tuple(operands) + (scale,))
        node.meta["val"] = torch.empty(shapes[0], dtype=torch.float16)
        return node

    square = (BATCH, TOKENS, HEADS, HEAD_DIM)
    assert vision_attention_is_emittable(call(square, square, square))
    # A query run and a key run that differ are another function.
    assert not vision_attention_is_emittable(
        call(square, (BATCH, TOKENS + 1, HEADS, HEAD_DIM), square)
    )
    # A scale the graph only learns at run time has nothing to put in the param.
    assert not vision_attention_is_emittable(call(square, square, square, scale="x"))
    # A 3-D "batch, tokens, dim" operand has no head axis to stride by.
    assert not vision_attention_is_emittable(
        call((BATCH, TOKENS, EMBED), (BATCH, TOKENS, EMBED), (BATCH, TOKENS, EMBED))
    )
