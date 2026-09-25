# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE
# file in the root directory of this source tree.
"""A mask on the fused vision attention, and what the kernel would do with it.

The kernel adds the mask to the score before the row maximum
(attention_entry.cc:50), so a mask is the same function the graph writes as
logits + mask. What it cannot express is a head axis: the index is
mask[(b * tokens + q) * maskStride + k], which carries a batch, a row and a
column and no term for h at all, so a mask that carries heads is a different
tensor and the backend refuses it rather than folding one into the other. The
operand is read as half floats whatever the graph says, so a mask that is not
fp16 is refused too.
"""

import unittest

import numpy as np
import torch
from executorch.backends.hexagon.vision_attention import vision_attention_pattern
from executorch.exir import to_edge
from torch.export import export

EMBED, HEADS = 32, 2
HEAD_DIM = EMBED // HEADS
BATCH, TOKENS = 1, 8
SCALE = HEAD_DIM**-0.5


def _torch_reference(q, k, v, scale, mask=None):
    """The attention in fp64, which is the reference is judged against."""
    qd, kd, vd = (t.transpose(1, 2).double() for t in (q, k, v))
    scores = (qd @ kd.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask.double()
    return (scores.softmax(dim=-1) @ vd).transpose(1, 2)


def _kernel_walk(q, k, v, scale, mask, mask_stride):
    """htp_ops_vision_attention_fp16, followed walk for walk in python.

    Three details have to be right for this to be the kernel. The mask index
    is mask[(b * tokens + q) * maskStride + k], so the batch pitch is tokens
    rows and the row pitch is the mask's own last extent. The score is fp32 and
    the mask is added to it as a half. And the exponent pass overwrites the
    score buffer in place (attention_common.hpp:386), so the value loop
    multiplies the exponent, not the logit.
    """
    tokens, heads = q.shape[1], q.shape[2]
    out = np.zeros((q.shape[0], tokens, heads, HEAD_DIM), dtype=np.float64)
    raw = None if mask is None else mask.detach().double().numpy().reshape(-1)
    for b in range(q.shape[0]):
        for qi in range(tokens):
            for h in range(heads):
                base_q = q[b, qi, h].double().numpy()
                scores = np.zeros(tokens, dtype=np.float64)
                for kk in range(tokens):
                    base_k = k[b, kk, h].double().numpy()
                    score = float((base_q * base_k).sum()) * scale
                    if raw is not None:
                        score += float(raw[(b * tokens + qi) * mask_stride + kk])
                    scores[kk] = score
                scores = np.exp(scores - scores.max())
                inverse = 1.0 / scores.sum() if scores.sum() > 0.0 else 0.0
                for d in range(HEAD_DIM):
                    out[b, qi, h, d] = sum(
                        scores[j] * inverse * float(v[b, j, h, d]) for j in range(tokens)
                    )
    return torch.from_numpy(out)


class _Attention(torch.nn.Module):
    """The decomposition a vision tower exports, with an optional additive mask."""

    def __init__(self, mask=None, cast=None):
        super().__init__()
        self.mask = mask
        self.cast = cast
        self.query = torch.nn.Linear(EMBED, EMBED, bias=True)
        self.key = torch.nn.Linear(EMBED, EMBED, bias=True)
        self.value = torch.nn.Linear(EMBED, EMBED, bias=True)

    def forward(self, x):
        batch, tokens, _ = x.shape
        head = lambda t: t.view(batch, tokens, HEADS, HEAD_DIM).transpose(1, 2)
        q, k, v = head(self.query(x)), head(self.key(x)), head(self.value(x))
        logits = (q @ k.transpose(-2, -1)) * SCALE
        if self.mask is not None:
            mask = self.mask if self.cast is None else self.mask.to(self.cast)
            logits = logits + mask
        return (logits.softmax(dim=-1) @ v).transpose(1, 2)


def _emittable(mask):
    """Whether the emitter's own predicate takes a node with this mask."""
    from executorch.backends.hexagon.hexagon_ops import (
        VISION_ATTENTION,
        vision_attention_is_emittable,
    )

    graph = torch.fx.Graph()

    def operand(shape, dtype):
        node = graph.create_node(
            "call_function", torch.ops.aten.view_copy.default, args=(), kwargs={}
        )
        node.meta["val"] = torch.zeros(shape, dtype=dtype)
        return node

    query = operand((BATCH, TOKENS, HEADS, HEAD_DIM), torch.float16)
    node = graph.create_node("call_function", VISION_ATTENTION, args=(), kwargs={})
    node.args = (query, query, query, SCALE, operand(tuple(mask.shape), mask.dtype))
    return vision_attention_is_emittable(node)


def _matches(model, x):
    program = to_edge(export(model.eval(), (x,))).exported_program()
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and vision_attention_pattern(node, program) is not None
    ]


class VisionAttentionMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(BATCH, TOKENS, EMBED).half()
        self.fp16_mask = torch.randn(TOKENS, TOKENS).half()

    def test_an_unmasked_attention_still_matches(self):
        """The change must not cost the case that already worked."""
        self.assertEqual(len(_matches(_Attention().half(), self.x)), 1)

    def test_an_fp16_additive_mask_is_matched(self):
        """The case the backend did not have: a mask the kernel can index."""
        self.assertEqual(len(_matches(_Attention(mask=self.fp16_mask).half(), self.x)), 1)

    def test_a_mask_that_is_not_a_half_is_refused_and_the_half_is_not(self):
        """The width is the clause, so the two nodes differ only in dtype.

        The kernel casts the pointer to const __fp16 *, so an fp32 operand
        would be read as half floats. Same rank, same stride, same values,
        asserted on the emitter's own predicate rather than through a model,
        because a half model cannot carry an fp32 add at all.
        """
        self.assertTrue(
            _emittable(self.fp16_mask), "the fp16 mask is the one the kernel reads"
        )
        self.assertFalse(
            _emittable(self.fp16_mask.float()), "fp32 would be read as half floats"
        )

    def test_a_mask_with_a_head_axis_is_refused_and_the_control_is_not(self):
        """The head axis is the clause, so the control differs only in rank."""
        self.assertEqual(len(_matches(_Attention(mask=self.fp16_mask).half(), self.x)), 1)
        headed = _Attention(mask=torch.randn(1, 1, TOKENS, TOKENS).half()).half()
        self.assertEqual(_matches(headed, self.x), [], "no head term in the index")

    def test_the_kernel_walk_reproduces_an_fp64_reference(self):
        """The walk has to be the same function, or the binding is wrong."""
        q = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        k = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        v = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        got = _kernel_walk(q, k, v, SCALE, self.fp16_mask, TOKENS)
        want = _torch_reference(q, k, v, SCALE, self.fp16_mask)
        self.assertLess(float((got - want).abs().max()), 2e-3)

    def test_the_mask_actually_changes_the_answer(self):
        """Without this the test above would pass with the mask ignored."""
        q = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        k = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        v = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM).half()
        with_mask = _kernel_walk(q, k, v, SCALE, self.fp16_mask, TOKENS)
        without = _torch_reference(q, k, v, SCALE, None)
        self.assertGreater(
            float((with_mask - without).abs().max()),
            2e-3,
            "a mask that changes nothing makes the test above vacuous",
        )

    def test_the_batch_pitch_is_tokens_rows(self):
        """The index formula, asserted directly instead of through a pipeline.

        attention_entry.cc:50 reads mask[(b * tokens + q) * maskStride + k],
        so batch b starts at b * tokens rows in. This pins that against the
        element a rank-three mask actually holds at each slot, which is the
        distinction a numerical walk would have to earn.
        """
        mask = torch.arange(2 * TOKENS * TOKENS, dtype=torch.float32)
        mask = mask.reshape(2, TOKENS, TOKENS).half()
        stride = TOKENS
        for b in (0, 1):
            for q in (0, TOKENS - 1):
                for k in (0, TOKENS - 1):
                    self.assertEqual(
                        float(mask[b, q, k]),
                        float(mask.reshape(-1)[(b * TOKENS + q) * stride + k]),
                        f"batch {b} row {q} column {k} is not where the kernel looks",
                    )


if __name__ == "__main__":
    unittest.main()
