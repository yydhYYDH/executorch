# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What each op family's other overloads actually do, pinned row by row.

The backend's coverage is a set of edge-op targets, and an op family has more
than one: `torch.mean(x)` and `torch.mean(x, dim=1)` are different targets,
`torch.clamp` and `torch.clamp(tensor, tensor, tensor)` are different targets,
and `torch.max(x, dim=1)` becomes a two-output node whose values and indices are
read by a getitem. A target with no emitter is invisible -- the partitioner never
sees the node, no refusal is reported, and the model simply keeps that op on the
portable kernels. Two such absences were closed in this batch (`mean.default`
and the clamp-shaped family); the rest of the census is here so the next one
shows up as a failing test rather than as a delegate that never appears.

Every row names the targets `to_edge` produces for one source-level expression,
and gives each one a verdict:

``unwired``
    no entry in `EMITTERS`, so the node is never considered. Closing it needs a
    kernel or an emitter.
``refused``
    wired, but the predicate rejects this shape or this argument. Closing it
    needs different work -- a pass, a second command, a narrower kernel -- and
    the row says which by existing.
``wired``
    delegated. These rows are the baseline the others are measured against, and
    they fail if coverage shrinks.

The claims are about the targets the exporter produces and the predicate's
answer, which is the layer both kinds of gap live in. Nothing here runs on
hardware.
"""

import operator


import pytest
import torch


from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import export

F16 = torch.float16

#: The targets that reach `_emit_getitem`; it is one entry in EMITTERS serving
#: several producers, so a row's verdict for it is about the predicate.
_GETITEM = "operator.getitem"


class _M(torch.nn.Module):
    """A module whose forward is whatever it was handed."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _x(*shape):
    return torch.randn(*shape, dtype=F16)


class _PReLU(torch.nn.Module):
    """prelu takes a weight, so it has to be a constant rather than an argument."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((1,), 0.25, dtype=F16))

    def forward(self, x):
        return torch.nn.functional.prelu(x, self.w)


def _name(target) -> str:
    return _GETITEM if target is operator.getitem else target.__name__


def _has_emitter(name: str) -> bool:
    return any(_name(key) == name for key in hexagon_ops.EMITTERS)


def _edge_program(model, inputs):
    return to_edge(
        export(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _edge_nodes(model, inputs):
    """The edge graph's call_function nodes, before any partitioning."""
    return [
        node
        for node in _edge_program(model, inputs).graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _accepted(model, inputs):
    """The edge targets the partitioner takes, in graph order.

    This is the same predicate `HexagonPartitioner` calls, over the same support
    object it builds: `partition` passes the names of the program's own tensors,
    and a bare `HexagonOperatorSupport()` would have an empty set of them, which
    makes every gate that wants a constant weight stricter than the partitioner
    is. Only nodes this list contains are ever partitioned, so a target absent
    from it is one that produces no delegate command.
    """
    program = _edge_program(model, inputs)
    support = HexagonOperatorSupport(_data_placeholders(program))
    return [
        _name(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and support.is_node_supported({}, node)
    ]


def _edge_targets(model, inputs):
    return [_name(node.target) for node in _edge_nodes(model, inputs)]


def _delegate_count(model, inputs):
    program = to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    return len(
        [
            node
            for node in program.graph_module.graph.nodes
            if node.target is torch.ops.higher_order.executorch_call_delegate
        ]
    )


# --------------------------------------------------------------------------
# Rows: (label, forward, inputs, [(edge target, verdict), ...])
# --------------------------------------------------------------------------

_ROWS = [
    # --- mean / sum / amax: the reductions whose dim argument varies ---------
    (
        "mean over every dim delegates",
        lambda a: torch.mean(a),
        (_x(2, 3, 4),),
        [("aten.mean.default", "wired")],
    ),
    (
        "mean along one dim delegates",
        lambda a: torch.mean(a, dim=1),
        (_x(2, 3, 4),),
        [("aten.mean.dim", "wired")],
    ),
    (
        "mean over a dim set that is not one span",
        lambda a: torch.mean(a, dim=(0, 2)),
        (_x(2, 3, 4),),
        [("aten.mean.dim", "refused")],
    ),
    (
        "mean that names a wider accumulator",
        lambda a: torch.mean(a, dim=1, dtype=torch.float32),
        (_x(2, 3, 4),),
        [("aten.mean.dim", "refused")],
    ),
    (
        "sum over every dim delegates",
        lambda a: torch.sum(a),
        (_x(2, 3, 4),),
        [("aten.sum.dim_IntList", "wired")],
    ),
    (
        "sum over a dim set that is not one span",
        lambda a: torch.sum(a, dim=(0, 2)),
        (_x(2, 3, 4),),
        [("aten.sum.dim_IntList", "refused")],
    ),
    (
        "sum that names a wider accumulator",
        lambda a: torch.sum(a, dim=1, dtype=torch.float32),
        (_x(2, 3, 4),),
        [("aten.sum.dim_IntList", "refused")],
    ),
    (
        "amax over every dim delegates",
        lambda a: torch.amax(a),
        (_x(2, 3, 4),),
        [("aten.amax.default", "wired")],
    ),
    (
        "amax over a dim set that is not one span",
        lambda a: torch.amax(a, dim=(0, 2)),
        (_x(2, 3, 4),),
        [("aten.amax.default", "refused")],
    ),
    (
        "an empty dim set is every dim on sum",
        lambda a: torch.sum(a, dim=()),
        (_x(2, 3, 4),),
        [("aten.sum.dim_IntList", "wired")],
    ),
    (
        "and on mean",
        lambda a: torch.mean(a, dim=()),
        (_x(2, 3, 4),),
        [("aten.mean.dim", "wired")],
    ),
    (
        "and on amax",
        lambda a: torch.amax(a, dim=()),
        (_x(2, 3, 4),),
        [("aten.amax.default", "wired")],
    ),
    # --- amax's siblings: the ones with no op type in the reduction enum -----
    (
        "amin selects the minimum reduction kind",
        lambda a: torch.amin(a),
        (_x(2, 3, 4),),
        [("aten.amin.default", "wired")],
    ),
    (
        "amin along one dim uses the same reduction kind",
        lambda a: torch.amin(a, dim=1),
        (_x(2, 3, 4),),
        [("aten.amin.default", "wired")],
    ),
    # --- the whole-tensor reduce-all overloads -------------------------------
    (
        "torch.max(x) reduces every dim",
        lambda a: torch.max(a),
        (_x(2, 3, 4),),
        [("aten.max.default", "wired")],
    ),
    (
        "torch.min(x) uses the minimum reduction kernel",
        lambda a: torch.min(a),
        (_x(2, 3, 4),),
        [("aten.min.default", "wired")],
    ),
    (
        "torch.max(x, dim) is two outputs, so it is a getitem rule",
        lambda a: torch.max(a, dim=1).values,
        (_x(2, 3, 4),),
        [("aten.max.dim", "wired"), (_GETITEM, "wired")],
    ),
    (
        "torch.min(x, dim) values use the reduction kernel",
        lambda a: torch.min(a, dim=1).values,
        (_x(2, 3, 4),),
        [("aten.min.dim", "wired"), (_GETITEM, "wired")],
    ),
    (
        "torch.max(x, dim).indices has no producer",
        lambda a: torch.max(a, dim=1).indices,
        (_x(2, 3, 4),),
        [("aten.max.dim", "refused"), (_GETITEM, "refused")],
    ),
    (
        "torch.maximum / minimum are the elementwise pair",
        lambda a, b: torch.maximum(a, b) + torch.minimum(a, b),
        (_x(8), _x(8)),
        [
            ("aten.maximum.default", "wired"),
            ("aten.minimum.default", "wired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    # --- argmax / argmin / topk / sort: index-producing reductions -----------
    (
        "argmax",
        lambda a: torch.argmax(a, dim=1),
        (_x(2, 3),),
        [("aten.argmax.default", "unwired")],
    ),
    (
        "topk",
        lambda a: torch.topk(a, 2, dim=-1).values,
        (_x(2, 3),),
        # Wired and refused: the kernel holds one element per row, so k == 2 is
        # an argument it has no slot for rather than an emitter this table
        # lacks. The values getitem is still accepted on its own, because a
        # getitem is judged without looking at its producer -- the same pair the
        # dilated pool's row below carries -- and that verdict is not a
        # partition: nothing forms one across a tuple, so the reader stays on
        # the host with the node it reads.
        [("aten.topk.default", "refused"), (_GETITEM, "wired")],
    ),
    (
        "sort",
        lambda a: torch.sort(a, dim=-1).values,
        (_x(2, 3),),
        [("aten.sort.default", "unwired"), (_GETITEM, "refused")],
    ),
    # --- other reductions ---------------------------------------------------
    (
        "prod",
        lambda a: torch.prod(a),
        (_x(2, 3),),
        [("aten.prod.default", "unwired")],
    ),
    (
        "var",
        lambda a: torch.var(a),
        (_x(2, 3),),
        [("aten.var.correction", "unwired")],
    ),
    (
        # A prefix scan reaches the DSP only as et_hexagon.cumsum.default, which
        # FuseCumsumPass writes and this row cannot see: the census measures the
        # bare edge graph, ahead of the opt-in pass. So aten.cumsum.default has
        # no emitter of its own and is unwired here, and the fused op has no
        # row in either table because no graph produces it without the pass.
        # test_cumsum.py is where both spellings are pinned.
        "cumsum",
        lambda a: torch.cumsum(a, dim=-1),
        (_x(2, 3),),
        [("aten.cumsum.default", "unwired")],
    ),
    # --- the unary enum: which entries have an emitter -----------------------
    (
        "sigmoid",
        lambda a: torch.sigmoid(a),
        (_x(8),),
        [("aten.sigmoid.default", "wired")],
    ),
    (
        "tanh",
        lambda a: torch.tanh(a),
        (_x(8),),
        [("aten.tanh.default", "wired")],
    ),
    (
        "erf has no entry in the unary table",
        lambda a: torch.erf(a),
        (_x(8),),
        [("aten.erf.default", "unwired")],
    ),
    (
        "sin has an entry, and now an emitter",
        lambda a: torch.sin(a),
        (_x(8),),
        [("aten.sin.default", "wired")],
    ),
    (
        "cos the same",
        lambda a: torch.cos(a),
        (_x(8),),
        [("aten.cos.default", "wired")],
    ),
    (
        "expm1 the same, and the measurement is why it stays unwired",
        lambda a: torch.expm1(a),
        (_x(8),),
        [("aten.expm1.default", "unwired")],
    ),
    (
        "torch.square arrives as pow.Tensor_Scalar",
        lambda a: torch.square(a),
        (_x(8),),
        [("aten.pow.Tensor_Scalar", "wired")],
    ),
    (
        "x ** 2 the same",
        lambda a: a**2,
        (_x(8),),
        [("aten.pow.Tensor_Scalar", "wired")],
    ),
    (
        "x ** 3 has no unary entry point",
        lambda a: a**3,
        (_x(8),),
        [("aten.pow.Tensor_Scalar", "refused")],
    ),
    (
        "pow with a run-time tensor exponent has no uniform exponent to gate on",
        lambda a, b: torch.pow(a, b),
        (_x(8), _x(8)),
        [("aten.pow.Tensor_Tensor", "refused")],
    ),
    # --- the clamp-shaped family -------------------------------------------
    (
        "clamp with scalar bounds",
        lambda a: torch.clamp(a, -1.0, 1.0),
        (_x(8),),
        [("aten.clamp.default", "wired")],
    ),
    (
        "clamp_min / clamp_max are the one-bound clamp",
        lambda a: torch.clamp_min(a, 0.0) + torch.clamp_max(a, 1.0),
        (_x(8),),
        [
            ("aten.clamp.default", "wired"),
            ("aten.clamp.default", "wired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    (
        "relu is clamp between 0 and infinity",
        lambda a: torch.relu(a),
        (_x(8),),
        [("aten.relu.default", "wired")],
    ),
    (
        "hardtanh shares clamp's argument slots",
        lambda a: torch.nn.functional.hardtanh(a),
        (_x(8),),
        [("aten.hardtanh.default", "wired")],
    ),
    (
        "relu6 is hardtanh(0, 6)",
        lambda a: torch.nn.functional.relu6(a),
        (_x(8),),
        [("aten.hardtanh.default", "wired")],
    ),
    (
        "clamp with tensor bounds is another overload",
        lambda a: torch.clamp(
            a, torch.tensor(0.0, dtype=F16), torch.tensor(1.0, dtype=F16)
        ),
        (_x(8),),
        [
            ("dim_order_ops._clone_dim_order.default", "wired"),
            ("dim_order_ops._clone_dim_order.default", "wired"),
            ("aten.clamp.Tensor", "unwired"),
        ],
    ),
    (
        "relu with inplace=True functionalizes to relu",
        lambda a: torch.nn.functional.relu(a, inplace=True),
        (_x(8),),
        [("aten.relu.default", "wired")],
    ),
    (
        "leaky_relu",
        lambda a: torch.nn.functional.leaky_relu(a),
        (_x(8),),
        [("aten.leaky_relu.default", "wired")],
    ),
    (
        "elu",
        lambda a: torch.nn.functional.elu(a),
        (_x(8),),
        [("aten.elu.default", "unwired")],
    ),
    (
        "prelu decomposes, and both halves of it are wired",
        _PReLU(),
        (_x(2, 4),),
        [
            ("aten.view_copy.default", "wired"),
            ("aten.gt.Scalar", "unwired"),
            ("aten.mul.Tensor", "wired"),
            ("aten.where.self", "wired"),
        ],
    ),
    # --- softmax family ----------------------------------------------------
    (
        "softmax over the last axis",
        lambda a: torch.nn.functional.softmax(a, dim=-1),
        (_x(2, 8),),
        [("aten._softmax.default", "wired")],
    ),
    (
        "softmax over another axis delegates through blits",
        lambda a: torch.nn.functional.softmax(a, dim=0),
        (_x(2, 8),),
        # The kernel is last-axis, so this axis is moved last and back: one
        # delegate carrying a blit, the softmax and a blit. The row is two wide,
        # under one 64-lane vector, so it is the SOFTMAX command rather than the
        # shifted sum. A source that cannot be permuted this way stays portable.
        [("aten._softmax.default", "wired")],
    ),
    (
        "log_softmax over the last axis",
        lambda a: torch.nn.functional.log_softmax(a, dim=-1),
        (_x(2, 8),),
        # No command describes a log softmax and the unary table has no such
        # entry, so this one is six commands: the row's maximum, the shift, the
        # exponentials, their sum, the log of it, and the subtraction that puts
        # the shift back. The two-command log(softmax(x)) is the obvious
        # composition and it is not the one emitted -- the softmax stores its
        # small probabilities as fp16 zeroes and the log of one is the kernel's
        # -65504, where the shifted form keeps the answer at the logit's own size.
        [("aten._log_softmax.default", "wired")],
    ),
    (
        "log_softmax over another axis is refused",
        lambda a: torch.nn.functional.log_softmax(a, dim=0),
        (_x(2, 8),),
        # The composition is written for a span with nothing inside it: every
        # command reduces the row and broadcasts back over it.
        [("aten._log_softmax.default", "refused")],
    ),
    # --- shape: what cat/stack/split/slice/select/permute/view each become ---
    (
        "cat of two",
        lambda a, b: torch.cat([a, b], dim=0),
        (_x(2, 3), _x(2, 3)),
        [("aten.cat.default", "wired")],
    ),
    (
        # The row used to be refused: a blit command holds three regions, and
        # four inputs needed one more than that. cat_region now splits the inputs
        # over as many commands as the region budget needs, so four is two blits
        # and seven is three, every input still writing its own disjoint slice.
        "cat of four spans two blit commands",
        lambda a, b, c, d: torch.cat([a, b, c, d], dim=0),
        (_x(2), _x(2), _x(2), _x(2)),
        [("aten.cat.default", "wired")],
    ),
    (
        "stack is a cat plus a view",
        lambda a, b: torch.stack([a, b], dim=0),
        (_x(2, 3), _x(2, 3)),
        [("aten.cat.default", "wired"), ("aten.view_copy.default", "wired")],
    ),
    (
        "split is one blit per piece",
        lambda a: torch.split(a, 2, dim=0)[0],
        (_x(6, 3),),
        [("aten.split_with_sizes_copy.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "chunk is the same op",
        lambda a: torch.chunk(a, 2, dim=0)[0],
        (_x(6, 3),),
        [("aten.split_with_sizes_copy.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "slice along an outer dim",
        lambda a: a[:, 1:3],
        (_x(4, 6),),
        [("aten.slice_copy.Tensor", "wired")],
    ),
    (
        "slice with a step",
        lambda a: a[:, ::2],
        (_x(4, 6),),
        [("aten.slice_copy.Tensor", "refused")],
    ),
    (
        "select",
        lambda a: a[1],
        (_x(4, 6),),
        [("aten.select_copy.int", "wired")],
    ),
    (
        "permute",
        lambda a: a.permute(1, 0),
        (_x(4, 6),),
        [("aten.permute_copy.default", "wired")],
    ),
    (
        "transpose is a permute",
        lambda a: a.transpose(-1, -2),
        (_x(4, 6),),
        [("aten.permute_copy.default", "wired")],
    ),
    (
        "reshape / view / flatten are view_copy",
        lambda a: a.reshape(-1) + a.view(-1).flatten(),
        (_x(4, 6),),
        [
            ("aten.view_copy.default", "wired"),
            ("aten.view_copy.default", "wired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    (
        "unsqueeze",
        lambda a: a.unsqueeze(0),
        (_x(4, 6),),
        [("aten.unsqueeze_copy.default", "wired")],
    ),
    (
        "squeeze",
        lambda a: a.squeeze(0),
        (_x(1, 4, 6),),
        [("aten.squeeze_copy.dims", "wired")],
    ),
    (
        "expand that is a real broadcast",
        lambda a: a.expand(2, 3),
        (_x(1, 3),),
        [("aten.expand_copy.default", "refused")],
    ),
    (
        # A repeat of one axis is cat([x] * factor, dim=axis), which is a region
        # walk over the operand's own bytes: one RASTER_BLIT, and the factor
        # rides on a level's extent rather than on a region per phase. The
        # shapes the gate still turns away -- two repeated axes, an
        # empty operand, a symbolic extent -- are pinned in test_repeat_flip.py.
        "repeat",
        lambda a: a.repeat(1, 2),
        (_x(2, 3),),
        [("aten.repeat.default", "wired")],
    ),
    (
        # A flip is a negative source stride, which the region's int32 stride
        # and the kernel's signed walk already carry. A flip of unit axes is a
        # view and emits no command at all.
        "flip",
        lambda a: torch.flip(a, [0]),
        (_x(4, 6),),
        [("aten.flip.default", "wired")],
    ),
    (
        # The zero-filling pad was the one entry in the "no kernel at all" list
        # that was not of that kind: it is a memset plus one region, and the
        # library has both. What moved is the table entry; the shapes the gate
        # still turns away are pinned in test_pad.py.
        "pad",
        lambda a: torch.nn.functional.pad(a, (1, 1)),
        (_x(4, 6),),
        [("aten.constant_pad_nd.default", "wired")],
    ),
    # --- pooling -----------------------------------------------------------
    (
        "max_pool2d is decomposed into with_indices plus a getitem",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2),
        (_x(1, 64, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "max_pool2d reading the indices",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2, return_indices=True)[1],
        (_x(1, 64, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "refused"), (_GETITEM, "refused")],
    ),
    (
        "max_pool2d with a dilated window",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2, dilation=2),
        (_x(1, 64, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "refused"), (_GETITEM, "wired")],
    ),
    (
        # 32 channels is one block with a 32-wide tail, which is a narrower blit
        # region and a memset rather than a refusal; 192 is three whole blocks.
        # The row is here as a census of what the command now takes, and the
        # refusal the pool still has is the dilated window above.
        "max_pool2d over a half block and over three blocks",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2),
        (_x(1, 32, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "accepted"), (_GETITEM, "wired")],
    ),
    (
        "max_pool2d over three whole blocks",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2),
        (_x(1, 192, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "accepted"), (_GETITEM, "wired")],
    ),
    (
        "avg_pool2d",
        lambda a: torch.nn.functional.avg_pool2d(a, 2, 2),
        (_x(1, 64, 8, 8),),
        [("aten.avg_pool2d.default", "wired")],
    ),
    (
        "avg_pool2d with a divisor_override",
        lambda a: torch.nn.functional.avg_pool2d(a, 2, 2, divisor_override=3),
        (_x(1, 64, 8, 8),),
        [("aten.avg_pool2d.default", "refused")],
    ),
    (
        "adaptive_avg_pool2d",
        lambda a: torch.nn.functional.adaptive_avg_pool2d(a, 2),
        (_x(1, 64, 8, 8),),
        [("aten._adaptive_avg_pool2d.default", "wired")],
    ),
    # --- elementwise arithmetic: Tensor vs Scalar ---------------------------
    (
        "add of two tensors",
        lambda a, b: a + b,
        (_x(8), _x(8)),
        [("aten.add.Tensor", "wired")],
    ),
    (
        "add of a python float is lowered to a tensor constant",
        lambda a: a + 1.5,
        (_x(8),),
        [
            ("dim_order_ops._to_dim_order_copy.default", "wired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    (
        "mul of a python float the same",
        lambda a: a * 2.0,
        (_x(8),),
        [
            ("dim_order_ops._to_dim_order_copy.default", "wired"),
            ("aten.mul.Tensor", "wired"),
        ],
    ),
    (
        "sub / div / fmod have the same split",
        lambda a, b: (a - b) / b,
        (_x(8), _x(8)),
        [("aten.sub.Tensor", "wired"), ("aten.div.Tensor", "wired")],
    ),
    (
        "add alpha other than 1 folds into the constant",
        lambda a, b: torch.add(a, b, alpha=2.0),
        (_x(8), _x(8)),
        [("aten.add.Tensor", "wired")],
    ),
    (
        "add over a broadcast pair",
        lambda a, b: a + b,
        (_x(2, 3), _x(3)),
        [("aten.add.Tensor", "wired")],
    ),
    # --- matmul ------------------------------------------------------------
    (
        "mm",
        lambda a, b: torch.mm(a, b),
        (_x(4, 8), _x(8, 5)),
        [("aten.mm.default", "wired")],
    ),
    (
        "2-D @, the spelling mm is a rewrite of",
        lambda a, b: a @ b,
        (_x(4, 8), _x(8, 5)),
        [("aten.mm.default", "wired")],
    ),
    (
        "batched @ against a 2-D weight",
        lambda a, b: a @ b,
        (_x(2, 4, 8), _x(8, 5)),
        [
            ("aten.view_copy.default", "wired"),
            ("aten.mm.default", "wired"),
            ("aten.view_copy.default", "wired"),
        ],
    ),
    (
        "bmm",
        lambda a, b: torch.bmm(a, b),
        (_x(2, 4, 8), (_x(2, 8, 5))),
        [("aten.bmm.default", "wired")],
    ),
    (
        "addmm with alpha 1",
        lambda b, a, c: torch.addmm(b, a, c),
        (_x(5), _x(4, 8), _x(8, 5)),
        [("aten.addmm.default", "wired")],
    ),
    (
        "addmm with alpha 2",
        lambda b, a, c: torch.addmm(b, a, c, alpha=2.0),
        (_x(5), _x(4, 8), _x(8, 5)),
        [("aten.addmm.default", "refused")],
    ),
    # --- constant constructors ---------------------------------------------
    (
        "full",
        lambda a: torch.full((2, 3), 1.0, dtype=F16),
        (_x(1),),
        [("aten.full.default", "unwired")],
    ),
    (
        "zeros / ones are a full",
        lambda a: torch.zeros(2, 3, dtype=F16) + torch.ones(2, 3, dtype=F16),
        (_x(1),),
        [
            ("aten.full.default", "unwired"),
            ("aten.full.default", "unwired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    (
        "zeros_like / ones_like / full_like",
        lambda a: torch.zeros_like(a) + torch.ones_like(a) + torch.full_like(a, 2.0),
        (_x(8),),
        [
            ("aten.full_like.default", "unwired"),
            ("aten.full_like.default", "unwired"),
            ("aten.add.Tensor", "wired"),
            ("aten.full_like.default", "unwired"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    (
        "arange",
        lambda a: torch.arange(4, dtype=F16),
        (_x(1),),
        [("aten.arange.start_step", "unwired")],
    ),
    (
        "a constant tensor times the activation uses the binary kernel",
        lambda a: a * torch.full((3,), 2.0, dtype=F16),
        (_x(2, 3),),
        [("aten.full.default", "unwired"), ("aten.mul.Tensor", "wired")],
    ),
    # --- comparisons, where and masked_fill --------------------------------
    (
        "eq.Tensor",
        lambda a, b: a == b,
        (_x(8), _x(8)),
        [("aten.eq.Tensor", "unwired")],
    ),
    (
        "ne.Tensor",
        lambda a, b: a != b,
        (_x(8), _x(8)),
        [("aten.ne.Tensor", "unwired")],
    ),
    (
        "gt.Tensor",
        lambda a, b: a > b,
        (_x(8), _x(8)),
        [("aten.gt.Tensor", "unwired")],
    ),
    (
        "lt.Tensor",
        lambda a, b: a < b,
        (_x(8), _x(8)),
        [("aten.lt.Tensor", "unwired")],
    ),
    (
        "ge.Tensor",
        lambda a, b: a >= b,
        (_x(8), _x(8)),
        [("aten.ge.Tensor", "unwired")],
    ),
    (
        "le.Tensor",
        lambda a, b: a <= b,
        (_x(8), _x(8)),
        [("aten.le.Tensor", "unwired")],
    ),
    (
        "where over a computed condition",
        lambda a, b: torch.where(a > 0, a, b),
        (_x(8), _x(8)),
        [("aten.gt.Scalar", "unwired"), ("aten.where.self", "wired")],
    ),
    (
        "masked_fill is a where with a lifted constant",
        lambda a, m: a.masked_fill(m, 0.0),
        (_x(8), torch.zeros(8, dtype=torch.bool)),
        [("scalar_tensor.default", "unwired"), ("aten.where.self", "wired")],
    ),
    (
        "a bool operand is refused rather than read two bytes per element",
        lambda a, b: a + (b > 0).to(torch.float16),
        (_x(8), _x(8)),
        [
            ("aten.gt.Scalar", "unwired"),
            ("dim_order_ops._to_dim_order_copy.default", "refused"),
            ("aten.add.Tensor", "wired"),
        ],
    ),
    # --- attention / norms --------------------------------------------------
    (
        "layer_norm over the trailing dims",
        lambda a: torch.nn.functional.layer_norm(a, [6]),
        (_x(4, 6),),
        [("aten.native_layer_norm.default", "wired"), (_GETITEM, "wired")],
    ),
    # Both of these are refused for the table, which is a method input here and
    # so has no bytes to rearrange at export -- not for the index width, which the
    # host narrows either way.
    (
        "embedding with int64 indices",
        lambda t, i: torch.nn.functional.embedding(i, t),
        (torch.randn(4, 8, dtype=F16), torch.tensor([0, 2], dtype=torch.int64)),
        [("aten.embedding.default", "refused")],
    ),
    (
        "index.Tensor with an int64 index",
        lambda a, i: a[i],
        (_x(8), torch.tensor([0, 2], dtype=torch.int64)),
        [("aten.index.Tensor", "refused")],
    ),
]


@pytest.mark.parametrize(
    "label,forward,inputs,rows", _ROWS, ids=[row[0] for row in _ROWS]
)
def test_each_overload_row_matches_its_recorded_verdict(label, forward, inputs, rows):
    """The census, one row at a time.

    Each row states what `to_edge` produces and what the partitioner does with
    it. A row that fails is a change in coverage, a change in what the exporter
    emits, or a new emitter -- all three worth stopping for, because the failure
    mode this file guards against is coverage that moves without a test moving
    with it.
    """
    model = _M(forward)
    assert _edge_targets(model, inputs) == [
        name for name, _ in rows
    ], "the graph this expression produces has changed"
    accepted = _accepted(model, inputs)
    for name, verdict in rows:
        if verdict == "unwired":
            assert not _has_emitter(name), f"{name} now has an emitter"
            assert name not in accepted, f"{name} is being delegated now"
        elif verdict == "refused":
            assert _has_emitter(name), f"{name} has no emitter, so it is unwired"
            assert name not in accepted, f"{name} started accepting this shape"
        else:
            assert _has_emitter(name), f"{name} lost its emitter"
            assert name in accepted, f"{name} is no longer delegated"


@pytest.mark.parametrize(
    "label,forward,inputs",
    [
        ("mean over every dim", lambda a: torch.mean(a), (_x(2, 3, 4),)),
        ("max over every dim", lambda a: torch.max(a), (_x(2, 3, 4),)),
        ("relu", lambda a: torch.relu(a), (_x(8),)),
        ("relu6", lambda a: torch.nn.functional.relu6(a), (_x(8),)),
        ("clamp", lambda a: torch.clamp(a, -1.0, 1.0), (_x(8),)),
        ("x squared", lambda a: a**2, (_x(8),)),
        (
            "where with a computed condition",
            lambda a, b: torch.where(a > 0, a, b),
            (_x(8), _x(8)),
        ),
        # max.dim moved here in the census batch after this one: its values are
        # the amax the reduction kernel already ran, and the rule that places
        # them is the pool's own.
        ("max.dim values", lambda a: torch.max(a, dim=1).values, (_x(2, 3, 4),)),
        # split moved here when its pieces landed: the same row that was
        # unwired plus a refused getitem is now one delegate, and it is the
        # positive half of that move rather than a deletion.
        ("split's first piece", lambda a: torch.split(a, 2, dim=0)[0], (_x(6, 3),)),
    ],
)
def test_the_closed_gaps_are_one_delegate_end_to_end(label, forward, inputs):
    """The positive half, through a real lowering.

    Every one of these produced no delegate before this batch, for the same
    reason as the unwired rows above: the target was not in the emitter table,
    so the node was never considered. They are here so the census is a rate and
    not just a list of absences.
    """
    assert _delegate_count(_M(forward), inputs) == 1


@pytest.mark.parametrize(
    "label,forward,inputs",
    [
        # `split` stood here while it was an unwired multi-output op; it is now
        # the first row of the closed-gap list below, and `sort` takes its place
        # here because it is the same shape of gap with no command behind it.
        ("sort's values", lambda a: torch.sort(a, dim=0)[0], (_x(6, 3),)),
        ("erf", lambda a: torch.erf(a), (_x(8),)),
        ("zeros_like", lambda a: torch.zeros_like(a), (_x(8),)),
    ],
)
def test_the_uncovered_overloads_leave_the_whole_graph_on_the_host(
    label, forward, inputs
):
    """The same rows through a real lowering, for the ones worth the extra time.

    The per-target verdicts above imply this, but the delegate count is what a
    reader of a model report sees, and the partitioner could in principle
    separate the two.
    """
    assert _delegate_count(_M(forward), inputs) == 0
