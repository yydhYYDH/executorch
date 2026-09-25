# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An SD cross-attention, and the two properties the fused paths both refuse.

Stable Diffusion 1.5's cross-attention is a 320-wide query run over a 768-wide
context in 8 heads of 40, so its query run is 256-to-4096 long against 77 keys
and its head width is neither 32 nor 64. The fused `VISION_ATTENTION_FP16` walks
one `tokens` extent for both sides, and `DSP_OP_FLASH_ATTN` stores 32-wide
columns per head, so each refuses one of those properties in the kernel. What
carries the geometry is the chain the export already writes -- three projections,
`q @ k^T`, the scale, a softmax, `p @ v`, an output projection -- and it reaches
the DSP as matmuls and blits once `permute_region` stops counting the batch axis
of a head split as a loop: `[1, tokens, heads, dim]` to `[1, heads, tokens, dim]`
is four groups of axes and three loops.

**What this file asserts is the host tier only**: one delegate, nothing refused,
and the command stream that delegate carries. The DSP tier for the same blob is a
separate measurement and is *not* green: on the phone the whole chain disagrees
with torch by 3.2e-02, and a stage-by-stage bisect puts the damage in the first
`q @ k^T` -- rows 1..19 of every 32-row tile wrong and 20..31 zero, at 77 and at
32 keys, for any head width. A host model that agreed with torch would say
nothing about that, which is why the assertions below are about which command
exists and not about what it computes.
"""



import torch


from blob_interpreter import read_blob

from executorch.backends.hexagon.hexagon_ops import (
    DSP_OP_BATCH_MATMUL,
    DSP_OP_BINARY_ELEMENTWISE,
    DSP_OP_RASTER_BLIT,
    DSP_OP_REDUCTION,
    DSP_OP_UNARY,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

#: One head of 40 is 1/sqrt(40) of a unit-scaled product, the factor the export bakes
#: into the graph as a scalar multiply between the two matmuls.
SCALE = 40**-0.5

#: The chain as one stream: q, k and v projections, the three head splits, the key
#: transpose, the scale, the softmax, `p @ v`, the output merge, the output
#: projection, and the copy into the output slot.
#:
#: The softmax here is over the 77 context tokens, i.e. one row past the width the
#: standalone softmax command is right for, so it is the five-command shifted sum
#: of exponentials the emitter uses above that width: the maximum over the row,
#: the shift, the exponential, the sum of at most ones, and the division by it.
#:
#: The second blit after `p @ v` is the broadcast's `clone_dim_order`, which
#: arrives with `dim_order = [0, 1, 2, 3]`, so the blit copies the bytes unchanged.
#: It is in the stream because the partitioner learned to delegate that node; the
#: base tree had no `CLONE_DIM_ORDER` case and left it portable.
COMMANDS = [
    DSP_OP_BATCH_MATMUL,
    DSP_OP_RASTER_BLIT,
    DSP_OP_BATCH_MATMUL,
    DSP_OP_RASTER_BLIT,
    DSP_OP_BATCH_MATMUL,
    DSP_OP_RASTER_BLIT,
    DSP_OP_RASTER_BLIT,
    DSP_OP_BATCH_MATMUL,
    DSP_OP_BINARY_ELEMENTWISE,
    DSP_OP_REDUCTION,
    DSP_OP_BINARY_ELEMENTWISE,
    DSP_OP_UNARY,
    DSP_OP_REDUCTION,
    DSP_OP_BINARY_ELEMENTWISE,
    DSP_OP_BATCH_MATMUL,
    DSP_OP_RASTER_BLIT,
    DSP_OP_RASTER_BLIT,
    DSP_OP_BATCH_MATMUL,
    DSP_OP_RASTER_BLIT,
]


class _CrossAttention(torch.nn.Module):
    """SD1.5's cross-attention: 320 queries, a 768-wide context, 8 heads of 40."""

    def __init__(self, dim=320, context_dim=768, heads=8, head_dim=40):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = torch.nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, heads * head_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, heads * head_dim, bias=False)
        self.to_out = torch.nn.Linear(heads * head_dim, dim, bias=False)

    def _split(self, x, tokens):
        return (
            x.reshape(1, tokens, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(self.heads, tokens, self.head_dim)
        )

    def forward(self, x, context):
        q = self._split(self.to_q(x), x.shape[1])
        k = self._split(self.to_k(context), context.shape[1])
        v = self._split(self.to_v(context), context.shape[1])
        scores = torch.bmm(q, k.transpose(-1, -2)) * SCALE
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.bmm(probabilities, v)
        merged = (
            attended.reshape(1, self.heads, -1, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(1, -1, self.heads * self.head_dim)
        )
        return self.to_out(merged)


def _case_inputs():
    torch.manual_seed(0)
    model = _CrossAttention().eval().half()
    x = torch.randn(1, 256, 320, dtype=torch.float16) * 0.3
    context = torch.randn(1, 77, 768, dtype=torch.float16) * 0.3
    return model, x, context


def _lowered(module, args):
    return to_edge_transform_and_lower(
        export(module, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _commands(program):
    """Every delegate's command types, in order."""
    out = []
    for node in program.graph_module.graph.nodes:
        if node.target is not torch.ops.higher_order.executorch_call_delegate:
            continue
        module = program.graph_module.get_submodule(node.args[0].target)
        _, commands = read_blob(bytes(module._processed_bytes))
        out.append([command.type for command in commands])
    return out


def test_the_whole_cross_attention_is_one_delegate():
    """With the batch group not counted, nothing in the module is left behind.

    The four `permute_copy` nodes are the head splits and the output merge, each
    one a region of three levels; while the batch axis counted as a fourth group
    all four were refused and the graph fell into three delegates around three
    portable permutes. The command stream is the claim, and the phone counted the
    same one out of the blob it ran.
    """
    model, x, context = _case_inputs()
    program = _lowered(model, (x, context))

    assert _commands(program) == [COMMANDS]


def test_the_only_node_left_outside_the_delegate_is_a_getitem():
    """A delegate boundary is a copy, so what stays out is worth pinning.

    The output projection used to be a delegate of its own; the head splits that
    sat beside it now travel inside the same one, and the single `getitem` is the
    shape plumbing of the emitter's own tuple, which carries no command.
    """
    model, x, context = _case_inputs()
    program = _lowered(model, (x, context))

    portable = [
        getattr(node.target, "__name__", str(node.target))
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is not torch.ops.higher_order.executorch_call_delegate
    ]
    assert portable == ["getitem"], portable


def test_the_same_split_at_a_batch_of_two_is_still_refused():
    """Two rows of batch make the batch group a loop, and the refusal stays.

    This is the control for the two tests above: the acceptance is about an extent
    of one, so the identical permutation with a batch of two keeps its fourth loop
    and keeps its refusal -- no delegate at all, and the move runs on the CPU. A
    graph-level check, because a predicate that said no to everything would pass
    the unit test next door and fail here.
    """

    class Two(torch.nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1, 3)

    x = torch.randn(2, 16, 8, 40, dtype=torch.float16)
    program = _lowered(Two().eval(), (x,))

    assert _commands(program) == [], "a batch of two delegated the head split"
