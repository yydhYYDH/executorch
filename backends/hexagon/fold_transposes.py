# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folds the weight transposes the DSP would otherwise run every inference."""

import torch
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
)
from executorch.exir.passes.remove_unused_parameters_pass import (
    remove_unused_parameters_pass,
)
from executorch.exir.program._program import lift_constant_tensor_pass
from torch.export import ExportedProgram

from executorch.backends.hexagon.conv_patch_embed import _val_of_shape

_PERMUTE_COPY = {torch.ops.aten.permute_copy.default, torch.ops.aten.permute_copy.out}


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
    matches = [k for k in program.state_dict if "p_" + k.replace(".", "_") == str(node.target)]
    if len(matches) != 1:
        return None
    return program.state_dict[matches[0]]


class FoldConstantTransposes(ExportedProgramPassBase):
    """Turns permute_copy(weight, [1, 0]) into the transposed weight itself.

    Every one of these turns a weight round for the matmul that consumes it.
    Left to the DSP it costs a pass over the weight on each and every
    inference; done here it costs once, at export, and the emitter then stores
    the result in the blob's weight section.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        matches = []
        for node in graph.nodes:
            if node.target not in _PERMUTE_COPY or len(node.args) < 2:
                continue
            if tuple(node.args[1]) != (1, 0):
                continue
            tensor = _constant_of(exported_program, node.args[0])
            if tensor is None or tensor.dim() != 2 or tensor.dtype != torch.float16:
                continue
            matches.append((node, tensor))
        if not matches:
            return ExportedProgramPassResult(exported_program, False)

        for node, tensor in matches:
            transposed = tensor.detach().t().contiguous()
            name = "%s_transposed" % node.name
            exported_program.graph_module.register_parameter(
                name, torch.nn.Parameter(transposed)
            )
            weight = graph.create_node("get_attr", name)
            weight.meta["val"] = _val_of_shape(node.meta["val"], transposed.shape)
            node.replace_all_uses_with(weight)
            graph.erase_node(node)

        remove_unused_parameters_pass(exported_program)
        lift_constant_tensor_pass(exported_program)
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
