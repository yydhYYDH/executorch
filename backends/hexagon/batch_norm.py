# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrites a batch norm with no convolution in front of it as a broadcast affine.

`FoldBatchNormIntoConv` applies a batch norm to the convolution before it and
drops the node, which is exact and free. It reaches only that shape, and a real
network has batch norms whose predecessor is not a convolution: the residual sum
in a residual block and the concatenation in a densely connected one. Those nodes
have no convolution to fold into, so they fall back to a portable kernel and split
the delegate chain around them, one CPU round trip per inference.

A batch norm at inference is `y = x * s + b` with a per-channel `s` and `b`, and
that needs no new kernel. A `[C, 1, 1]` constant against an `[N, C, ...]` activation
is a stride-0 broadcast on every spatial axis with a real extent on the channel
axis, which is what the binary elementwise broadcast descriptor already
describes: the operand is right-aligned against the output, so its own row-major
strides come out `[0, 1, 0, 0]` -- zero on the axes where the operand's extent is
one, and one step into the constant for each channel. The whole node is then one
`MUL` and one `ADD` against two constants, and this backend already emits both
for every other binary elementwise op it supports. Measured over 105 elements at
`[1, 3, 5, 7]` with a `[3, 1, 1]` scale of 1, 8 and 64, the answer is bit-identical
to torch's, and the values are powers of two so a wrong element would differ by
eight times rather than by a rounding step.

What this deliberately does not do is fold. Folding a batch norm through a `cat`
means splitting the affine across the concatenation's slices and writing it into
each producer, and pushing that back through a dense block means rewriting every
layer in the block; folding it through a residual `add` means writing the same
affine into both branches of the sum. Both are exact, and both are larger than the
two commands they save, so the arithmetic is executed here instead.

Three things this refuses, each with the reason it is refused rather than a
fallback nobody can distinguish from success.

**Training mode.** `aten.native_batch_norm`'s sixth argument is `training`, and a
node set on it means the batch statistics. The affine this writes is built from
the running ones, so a training-mode node would compute the wrong function with no
error anywhere -- on the single-block graph in `test_fold_batch_norm.py` the two
readings of that node are 4.49 apart. The name `legit_no_training` is not evidence:
the overload that carries the flag is matched too, and read.

**A statistic the export cannot see.** `s` and `b` are built from `running_mean`
and `running_var`, so a graph whose statistics are method inputs has no constant
affine to write, and those nodes keep their own.

**A reader of the other two outputs.** The op returns `(output, save_mean,
save_invstd)`, and only element zero is a value this can compute, so a graph
holding a `getitem` of either statistic is left whole rather than given a
stand-in that means something else.
"""
import operator
from typing import List, NamedTuple, Optional

import torch
from torch._export.utils import get_buffer, get_lifted_tensor_constant, get_param

from executorch.backends.hexagon.fold_batch_norm import repoint_outputs
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram

NO_TRAINING = exir_ops.edge.aten._native_batch_norm_legit_no_training.default
WITH_TRAINING = exir_ops.edge.aten.native_batch_norm.default

#: The two overloads an eval export reaches the edge program as. The second is
#: the only one that carries a training flag, at this index.
BATCH_NORM_TARGETS = (NO_TRAINING, WITH_TRAINING)
_TRAINING_ARG = 5

_MUL = exir_ops.edge.aten.mul.Tensor
_ADD = exir_ops.edge.aten.add.Tensor
_GETITEM = operator.getitem

#: The channel axis of a `[N, C, ...]` activation, which is the axis the
#: statistics are per and the one the `[C, 1, 1]` constant lines up with.
_CHANNEL_AXIS = 1

#: A node below this many dimensions has no channel axis distinct from its batch,
#: and torch's own reduction for it is over axis 0, so a `[C, 1, 1]` constant is
#: not the right shape for it.
_MIN_DIMS = 2


class AffineMatch(NamedTuple):
    """A batch norm this pass can write as two broadcast commands."""

    node: torch.fx.Node
    #: The value being normalized, and the channel count its statistics are per.
    source: torch.fx.Node
    channels: int
    #: `[C]` in fp64, the numbers `s` and `b` are built from.
    scale: torch.Tensor
    shift: torch.Tensor


def _tensor_of(program: ExportedProgram, node) -> Optional[torch.Tensor]:
    """The real tensor behind a constant, or None if the export cannot see it.

    The same three sources the backend reads its own weights from, in the same
    order, because the question is the same one: is this operand's value a fact
    about the program rather than about the call. A placeholder is a parameter, a
    buffer or a lifted constant, and each of those is resolvable; a method input
    is none of them and has no value here at all.
    """
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor) and type(value).__name__ != "FakeTensor":
        return value.detach()
    for fetch in (get_param, get_buffer, get_lifted_tensor_constant):
        tensor = fetch(program, node)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
    if node.op == "get_attr":
        tensor = getattr(program.graph_module, node.target, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
        if isinstance(tensor, torch.nn.Parameter):
            return tensor.detach()
    return None


def _eps_of(node: torch.fx.Node) -> Optional[float]:
    """The node's own epsilon, or None when it is not a number.

    It is the last argument of both overloads, and a fold or an affine that drops
    it is a different function: the scale is `1/sqrt(var + eps)`, so an eps of
    zero against a variance of 1e-12 is a difference of six orders of magnitude.
    """
    for arg in reversed(node.args):
        if isinstance(arg, bool):
            continue
        if isinstance(arg, (int, float)):
            return float(arg)
    return None


def trains_on_batch_statistics(node: torch.fx.Node) -> bool:
    """Whether this node means the batch statistics rather than the running ones.

    Only the overload that carries the flag can be set, so a node of the other
    form is a `False` here whatever its name says.
    """
    if node.target is not WITH_TRAINING or len(node.args) <= _TRAINING_ARG:
        return False
    training = node.args[_TRAINING_ARG]
    return training is True or training == 1


def batch_norm_is_rewritable(
    node: torch.fx.Node, program: Optional[ExportedProgram] = None
) -> Optional[str]:
    """Why this node cannot be written as a broadcast affine, or None if it can.

    The reason is returned rather than a bare bool because a refusal that does not
    say what refused it is a test that can be satisfied by some other clause.
    """
    if node.op != "call_function" or node.target not in BATCH_NORM_TARGETS:
        return "not a batch norm"
    if trains_on_batch_statistics(node):
        return "training mode means the batch statistics"
    reason = _value_is_writable(node.args[0])
    if reason is not None:
        return reason
    if _eps_of(node) is None:
        return "epsilon is not a number this can read"
    reason = _statistics_are_readable(node, program)
    if reason is not None:
        return reason
    for user in node.users:
        if user.op != "call_function" or user.target is not _GETITEM or user.args[1] != 0:
            return "a reader of save_mean or save_invstd"
    return None


def _value_is_writable(source) -> Optional[str]:
    """Why the value a batch norm reads cannot carry a broadcast constant."""
    if not isinstance(source, torch.fx.Node):
        return "the value being normalized is not a node"
    val = source.meta.get("val")
    if val is None or not isinstance(val, torch.Tensor):
        return "the value being normalized has no shape"
    if val.dim() < _MIN_DIMS:
        return f"a {val.dim()}-D value has no channel axis"
    if not val.is_contiguous(memory_format=torch.contiguous_format):
        # The broadcast tail carries strides, and the emitter writes the row-major
        # ones of the operand's own shape. A channels_last value is not stored in
        # that order, so the strides the descriptor would carry do not place it.
        return "the value being normalized is not contiguous in the default order"
    return None


def _statistics_are_readable(
    node: torch.fx.Node, program: Optional[ExportedProgram]
) -> Optional[str]:
    """Why the constants the affine is built from are not all there at export."""
    for index, what in ((1, "weight"), (2, "bias"), (3, "running_mean"), (4, "running_var")):
        operand = node.args[index]
        # affine=False leaves the weight and the bias as None, which is the
        # identity map rather than a missing constant.
        if operand is None and index in (1, 2):
            continue
        if not isinstance(operand, torch.fx.Node):
            return f"{what} is not a node"
        if _tensor_of(program, operand) is None:
            return f"{what} is not a constant at export"
    return None


def affine_of(node: torch.fx.Node, program: ExportedProgram) -> Optional[AffineMatch]:
    """The per-channel affine this batch norm is, read at export.

    Both halves are built in fp64 and rounded once to fp16, which is the
    correctly rounded value of each coefficient; building them in fp16 would round
    `sqrt` and the division separately and land a step away.
    """
    if batch_norm_is_rewritable(node, program) is not None:
        return None
    source = node.args[0]
    eps = _eps_of(node)
    weight = _tensor_of(program, node.args[1])
    bias = _tensor_of(program, node.args[2])
    mean = _tensor_of(program, node.args[3]).to(torch.float64)
    var = _tensor_of(program, node.args[4]).to(torch.float64)

    channels = int(source.meta["val"].shape[_CHANNEL_AXIS])
    scale = torch.rsqrt(var + eps)
    if weight is not None:
        scale = scale * weight.to(torch.float64)
    shift = (bias.to(torch.float64) if bias is not None else torch.zeros_like(mean)) - mean * scale
    if int(scale.numel()) != channels:
        return None
    return AffineMatch(node, source, channels, scale, shift)


def _val_of(source: torch.fx.Node, tensor: torch.Tensor) -> torch.Tensor:
    """The meta a node needs downstream: a shape and a dtype, nothing else.

    A fake tensor of the right shape is what the callers read, and making one
    costs a shape walk; the real tensor is never large enough to matter here.
    """
    return source.meta["val"].new_empty(tensor.shape)


def _rewrite(program: ExportedProgram, match: AffineMatch) -> None:
    graph = program.graph_module.graph
    node = match.node
    # Right-aligned against the activation, the channel is the constant's axis 0
    # and every axis after it is a broadcast axis: a `[C, 1, 1]` against
    # `[N, C, H, W]` and a `[C, 1]` against `[N, C, L]`. One more trailing 1 than
    # that would put the channel on the wrong axis and widen the result, which is
    # a shape error rather than a wrong number.
    rank = len(node.args[0].meta["val"].shape)
    shape = (match.channels,) + (1,) * (rank - 2)
    # The arena is fp16 for every kernel, and the constant is read out of the
    # weights section as raw bytes, so this is the width it has to be stored at.
    scale = match.scale.to(torch.float16).reshape(shape).contiguous()
    shift = match.shift.to(torch.float16).reshape(shape).contiguous()

    scale_name = f"_hexagon_bn_scale_{node.name}"
    shift_name = f"_hexagon_bn_shift_{node.name}"
    # A `get_attr` whose value is neither a parameter in the state_dict nor a
    # buffer in the signature is a constant the verifier has never heard of, and
    # a pass run through a pass manager is verified after it. `program.constants`
    # is where a value like this belongs, and the backend already looks there
    # for a name it cannot read any other way.
    program.graph_module.register_parameter(scale_name, torch.nn.Parameter(scale))
    program.graph_module.register_parameter(shift_name, torch.nn.Parameter(shift))
    state = program.state_dict
    state[scale_name] = program.graph_module.get_parameter(scale_name)
    state[shift_name] = program.graph_module.get_parameter(shift_name)

    with graph.inserting_before(node):
        scale_node = graph.create_node("get_attr", scale_name)
        scale_node.meta["val"] = _val_of(match.source, scale)
        shift_node = graph.create_node("get_attr", shift_name)
        shift_node.meta["val"] = _val_of(match.source, shift)
        scaled = graph.call_function(_MUL, (match.source, scale_node))
        scaled.meta["val"] = match.source.meta["val"]
        result = graph.call_function(_ADD, (scaled, shift_node))
        result.meta["val"] = match.source.meta["val"]

    # The reads go first: each one is a getitem of element zero, and re-pointing
    # its own users at the affine is what leaves the batch norm with none, so the
    # node can be erased. Doing it the other way round leaves a graph output
    # holding a getitem of a node that is being deleted.
    reads = list(node.users)
    for read in reads:
        read.replace_all_uses_with(result)
    for read in reads:
        graph.erase_node(read)
    graph.erase_node(node)


class RewriteBatchNormToAffine(ExportedProgramPassBase):
    """Writes every batch norm that cannot be folded as a broadcast affine.

    The rewrite belongs here rather than in the emitters because it is a decision
    about the graph, not a command: a node the emitters can run is a node the
    partitioner has to be told about, and a `mul` and an `add` it already
    accepts. `FoldBatchNormIntoConv` runs first in a pass list, so the batch
    norms a convolution precedes are folded and this sees only the rest.
    """

    def __init__(self) -> None:
        #: The names of the nodes this rewrote, and why it left the rest alone. A
        #: caller that wants to know what stayed portable reads these rather than
        #: re-deriving the reasons, and a test that means to prove a refusal is
        #: the one it thinks it is can check the sentence.
        self.rewritten: List[str] = []
        self.refused: List[str] = []

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        matches: List[AffineMatch] = []
        refused: List[str] = []
        for node in exported_program.graph_module.graph.nodes:
            reason = batch_norm_is_rewritable(node, exported_program)
            if reason is not None:
                if node.op == "call_function" and node.target in BATCH_NORM_TARGETS:
                    refused.append(f"{node.name}: {reason}")
                continue
            match = affine_of(node, exported_program)
            if match is None:
                refused.append(f"{node.name}: the statistics are not per channel")
                continue
            matches.append(match)
        if not matches:
            self.refused = refused
            return ExportedProgramPassResult(exported_program, False)

        for match in matches:
            _rewrite(exported_program, match)
        # A rewritten node that was the graph's output is now named by the add
        # that replaced it, and the signature still names the node that is gone:
        # a pass manager verifies after every pass and would fail on it.
        repoint_outputs(exported_program)
        exported_program.graph_module.recompile()
        self.refused = refused
        self.rewritten = [match.node.name for match in matches]
        return ExportedProgramPassResult(exported_program, True)