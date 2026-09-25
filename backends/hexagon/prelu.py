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
_library.define("prelu(Tensor x, Tensor weight) -> Tensor")


def _prelu(x, weight):
    return torch.nn.functional.prelu(x, weight)


_library.impl("prelu", _prelu, "CompositeExplicitAutograd")
PRELU = exir_ops.edge.et_hexagon.prelu.default


class PreservePRelu(ExportedProgramPassBase):
    def call(self, exported_program):
        changed = False
        graph = exported_program.graph_module.graph
        for node in graph.nodes:
            if getattr(node.target, "_schema", None) is None:
                continue
            if node.target._schema.name != "aten::prelu":
                continue
            with graph.inserting_before(node):
                replacement = graph.create_node("call_function", PRELU, args=node.args)
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            changed = True
        if changed:
            graph.eliminate_dead_code()
            exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, changed)
