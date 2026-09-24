# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folds the weight preparations the DSP would otherwise run every inference.

Two shapes live here, both of which end the same way: a weight becomes a
constant the emitters can see, instead of a run of pure data movement the DSP
walks on every inference.

The first is a matmul's `permute_copy(weight, [1, 0])`, which is a transpose of
the weight and nothing else. The second is the preparation a *transposed*
convolution is usually written with: a diffuse model flips and permutes its
convolution weight before handing it to `conv_transpose2d`, which leaves the
weight a graph computation rather than a constant at export. The emitters
require a constant weight -- `conv_spec` reads it to pack the command's weight
section -- so without this fold a transposed convolution written that way stays
on a portable kernel and never reaches the convolution walk that would carry it.

Every op folded here is a permutation of the same bytes: no arithmetic, no
rounding, no dtype change. That is what makes folding it safe and what makes the
folded tensor comparable to the chain it replaces by exact equality.
"""

import torch

from executorch.backends.hexagon.conv_patch_embed import _val_of_shape
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.remove_unused_parameters_pass import (
    remove_unused_parameters_pass,
)
from executorch.exir.program._program import lift_constant_tensor_pass
from torch.export import ExportedProgram

#: The transpose a matmul's weight is written with, in either dialect's name.
_PERMUTE_NAMES = {"aten::permute", "aten::permute_copy"}

#: The ops a weight's preparation may consist of, by schema name so that the
#: same set matches whichever dialect the pass is run against. Every one is a
#: pure permutation of the operand's bytes, so a chain of them computes exactly
#: the tensor this pass stores. A chain that contains anything else is left
#: alone: an op with arithmetic in it would be a fold this pass cannot justify
#: by exactness, and an emission that reached for one would be guessing.
_WEIGHT_PREP = {
    # The names an ATen program uses, which is the stage EXIR calls the
    # partitioner's pre-decomposition transform at.
    "aten::permute",
    "aten::view",
    "aten::reshape",
    "aten::flip",
    "aten::transpose",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::t",
    "aten::contiguous",
    # And their functionalised edge spellings, for the same pass run through a
    # caller's `transform_passes` instead.
    "aten::permute_copy",
    "aten::view_copy",
    "aten::reshape_copy",
    "aten::transpose_copy",
    "aten::squeeze_copy",
    "aten::unsqueeze_copy",
    "aten::t_copy",
}


def _constant_of(program: ExportedProgram, node):
    """The tensor behind a node, whether an attribute or a lifted weight.

    torch.export lifts a graph's parameters into placeholders named after them,
    so a weight arrives as an input. None means this program does not hold the
    value, and then the transpose has to stay on the DSP.
    """
    if node.op == "get_attr":
        return program.state_dict.get(str(node.target))
    if node.op != "placeholder":
        return None
    matches = [
        k for k in program.state_dict if "p_" + k.replace(".", "_") == str(node.target)
    ]
    if len(matches) != 1:
        return None
    return program.state_dict[matches[0]]


def _schema_name(node) -> str:
    return getattr(getattr(node.target, "_schema", None), "name", "")


def _is_convolution(node) -> bool:
    """Whether this node consumes a convolution weight, in either dialect.

    The ATen program spells the op `aten::convolution`,
    `aten::conv_transpose2d.input` and the like; the edge program spells it
    `aten::convolution` for every one of them. All of them take the weight as
    their second argument, which is the one thing the walk needs to know.
    """
    return _schema_name(node).startswith("aten::conv")


def _prepared_weight(program: ExportedProgram, node):
    """The tensor a weight's preparation computes, and the chain that made it.

    Walks back from a weight over the pure data-movement ops an exported
    preparation consists of, then evaluates the chain by calling the very ops it
    names, in order, on the constant it reaches. Running the graph's own targets
    rather than a hand-written table of equivalents is what keeps this from
    folding a different axis than the graph did.

    None means the chain reached something this pass does not hold. A weight
    that is a run-time input is the case that matters: folding it would bake one
    call's data into the program, which is worse than leaving the graph alone.
    Returns an empty chain when the operand is already a constant.
    """
    chain = []
    while True:
        tensor = _constant_of(program, node)
        if tensor is not None:
            break
        if not node.args or _schema_name(node) not in _WEIGHT_PREP:
            return None
        chain.append(node)
        node = node.args[0]
        if not isinstance(node, torch.fx.Node):
            return None

    value = tensor
    for step in reversed(chain):
        try:
            value = step.target(value, *step.args[1:])
        except Exception:
            # An argument shape this pass cannot replay is a chain it does not
            # understand, and an unfolded weight is a portable kernel: correct,
            # and the pipeline's business rather than an exception here.
            return None
    return value, chain


class FoldConstantTransposes(ExportedProgramPassBase):
    """Turns a weight's preparation into the prepared weight itself.

    Every chain this folds is a run of pure permutations over a constant, so the
    folded tensor is the same bytes in another order and the graph is unchanged
    in value. What changes is where the work happens: once at export, into the
    blob's weight section, instead of a pass over the weight on each inference
    and -- for a transposed convolution -- instead of a portable kernel.

    A chain whose root is not a constant is left alone, and so is one whose
    intermediate results are read by anything else, since folding that would
    duplicate the preparation rather than move it.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        matches = []
        for node in graph.nodes:
            if _schema_name(node) not in _PERMUTE_NAMES or len(node.args) < 2:
                continue
            if tuple(node.args[1]) != (1, 0):
                continue
            tensor = _constant_of(exported_program, node.args[0])
            if tensor is None or tensor.dim() != 2 or tensor.dtype != torch.float16:
                continue
            matches.append((node, tensor))

        folded = []
        for node in graph.nodes:
            if not _is_convolution(node) or len(node.args) < 2:
                continue
            weight = node.args[1]
            if not isinstance(weight, torch.fx.Node):
                continue
            found = _prepared_weight(exported_program, weight)
            if found is None:
                continue
            value, chain = found
            if not chain:
                continue
            if any(len(step.users) != 1 for step in chain):
                continue
            folded.append((weight, chain, value))

        if not matches and not folded:
            return ExportedProgramPassResult(exported_program, False)

        for node, tensor in matches:
            transposed = tensor.detach().t().contiguous()
            self._replace(
                exported_program, node, transposed, "%s_transposed" % node.name, [node]
            )

        for weight, chain, value in folded:
            prepared = value.detach().to(torch.float16).contiguous()
            self._replace(
                exported_program,
                weight,
                prepared,
                "%s_prepared" % chain[0].name,
                chain,
            )

        remove_unused_parameters_pass(exported_program)
        lift_constant_tensor_pass(exported_program)
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)

    @staticmethod
    def _replace(program, node, tensor, name, chain) -> None:
        graph = program.graph_module.graph
        program.graph_module.register_parameter(name, torch.nn.Parameter(tensor))
        constant = graph.create_node("get_attr", name)
        constant.meta["val"] = _val_of_shape(node.meta["val"], tensor.shape)
        node.replace_all_uses_with(constant)
        for step in chain:
            if not step.users:
                graph.erase_node(step)
