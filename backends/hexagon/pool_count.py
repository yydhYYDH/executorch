# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes an average whose overhang is padded, so the kernel's divisor is the right one.

Ceil mode adds an output position whose window runs past the end of the input,
and torch's divisor for that position is the part of the window still inside.
`pool_fp16.c`'s countType 1 divides by `kY * kX` whatever the window does, so a
node that has one is refused rather than answered at half the right value.

The refusal is right, and it is still a break in the delegate chain: a portable
average in the middle of a squeeze-and-excitation block costs a round trip per
inference, and CAMPPlus has 52 of them at
`avg_pool2d(x, (1, 100), (1, 100), (0, 0), ceil_mode=True)` over 150 frames.
At `padding=0` the two divisor rules are the same average over the same
elements -- a window clipped to the padded input and a window clipped to the
unpadded one are the same window when there is no padding -- so writing
`count_include_pad=False` is not an approximation of the model. It is the same
tensor, spelled the way countType 0 computes it.

That equivalence is what makes the rewrite sound and it is only true at
`padding=0`, which is the whole of the precondition. `padding=1` already
splits the two, and a node with one is left alone.

The rewrite belongs here rather than in `pool_spec` because it is a decision
about the graph, not a command: the node the emitters already accept is a node
the partitioner has to be told about.
"""

from typing import List, Optional, Tuple

import torch

from executorch.backends.hexagon.hexagon_ops import AVG_POOL2D
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
)

_CEIL_MODE = 4
_COUNT_INCLUDE_PAD = 5


def _int_pair(value, default) -> Optional[Tuple[int, int]]:
    if value is None:
        value = default
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _window_hangs(extent: int, window: int, stride: int, out: int) -> bool:
    """Whether the last output position's window ends past the input.

    With no padding the window at `o` is `[o * stride, o * stride + window)`, so
    the tightest position is the last one and this is the comparison the
    emitter's own divisor test makes.
    """
    return (out - 1) * stride + window > extent


def avg_pool_rewrite_reason(
    node: torch.fx.Node, program: ExportedProgram
) -> Optional[str]:
    """Why this average cannot be respelled, or None when it can.

    Every clause is a precondition of the arithmetic rather than a preference:
    the equivalence is false without `padding=0`, the divisor is not the
    window's own when the node already fits, and `divisor_override` is an
    explicit divisor that survives the rewrite untouched.
    """
    if node.op != "call_function" or node.target is not AVG_POOL2D:
        return "not an avg_pool2d"
    args = node.args
    if not args or not isinstance(args[0], torch.fx.Node):
        return "no input node"
    if len(args) > 6 and args[6] is not None:
        return "divisor_override is explicit"

    padding = _int_pair(args[3] if len(args) > 3 else None, 0)
    if padding is None:
        return "padding is not a pair"
    if padding != (0, 0):
        return f"padding is {padding}, and the two divisors split there"

    if len(args) > _CEIL_MODE:
        if not args[_CEIL_MODE]:
            return "floor mode, so no window overhangs"
    if len(args) > _COUNT_INCLUDE_PAD:
        if not args[_COUNT_INCLUDE_PAD]:
            return "already the divisor the kernel can compute"
    if node.kwargs.get("ceil_mode") is False:
        return "floor mode, so no window overhangs"
    if node.kwargs.get("count_include_pad") is False:
        return "already the divisor the kernel can compute"

    kernel = _int_pair(args[1] if len(args) > 1 else None, None)
    if kernel is None:
        return "kernel is not a pair"
    stride = _int_pair(args[2] if len(args) > 2 else None, kernel)
    if stride is None:
        return "stride is not a pair"

    value = args[0].meta.get("val")
    result = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or not isinstance(result, torch.Tensor):
        return "no static shape to measure the window against"
    if len(value.shape) != len(result.shape):
        return "the pool changes the rank"

    axes = zip(value.shape[-2:], result.shape[-2:], kernel, stride)
    if not any(
        _window_hangs(extent, window, step, out)
        for extent, out, window, step in axes
    ):
        return "every window fits, so the divisor is already the kernel's"
    return None


class RewriteCeilPoolCountToValid(ExportedProgramPassBase):
    """Respells every padded-ceil average the kernel would divide wrongly.

    Leaves a node alone unless every precondition holds, and records why, so a
    model that keeps its averages portable is a list of reasons rather than a
    delegate count nobody can interpret.
    """

    def __init__(self) -> None:
        self.rewritten: List[str] = []
        self.refused: List[str] = []

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        self.rewritten = []
        self.refused = []
        for node in graph.nodes:
            reason = avg_pool_rewrite_reason(node, exported_program)
            if reason is not None:
                if node.op == "call_function" and node.target is AVG_POOL2D:
                    self.refused.append(f"{node.name}: {reason}")
                continue
            node.args = node.args[:_COUNT_INCLUDE_PAD] + (False,)
            self.rewritten.append(node.name)

        return ExportedProgramPassResult(exported_program, bool(self.rewritten))
