# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
)

NAMESPACE = "et_hexagon"
_library = torch.library.Library(NAMESPACE, "FRAGMENT")
_library.define("reflect_pad(Tensor x, int[] pad) -> Tensor")


def _reflect_pad(x, pad):
    return torch.nn.functional.pad(x, tuple(pad), mode="reflect")


_library.impl("reflect_pad", _reflect_pad, "CompositeExplicitAutograd")
REFLECT_PAD = exir_ops.edge.et_hexagon.reflect_pad.default


class PreserveReflectPad(ExportedProgramPassBase):
    def call(self, exported_program):
        changed = False
        graph = exported_program.graph_module.graph
        for node in graph.nodes:
            schema = getattr(node.target, "_schema", None)
            if schema is None or schema.name != "aten::pad":
                continue
            if len(node.args) < 3 or node.args[2] != "reflect":
                continue
            with graph.inserting_before(node):
                replacement = graph.create_node(
                    "call_function",
                    REFLECT_PAD,
                    args=(node.args[0], tuple(node.args[1])),
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            changed = True
        if changed:
            graph.eliminate_dead_code()
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, changed)
