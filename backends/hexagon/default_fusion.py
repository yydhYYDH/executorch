# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The graph rewrites to run before partitioning, in one list.

Every pass in this directory is a good rewrite on its own and none of them is
mandatory, so nothing puts them together and every caller assembles the set it
remembers. That is a losing trade, because the passes compose and a caller
who leaves one out does not get a slightly worse graph so much as a *split*
one. The split is invisible in the artifact: both programs load and both run,
and the only difference is how many times the model crosses the line between
the host and the DSP in one forward pass.

A batch norm is the clearest case. `FoldBatchNormIntoConv` applies one to the
convolution in front of it and drops the node, which is exact and free -- but
it reaches only that shape, and a real network has batch norms whose
predecessor is a residual sum or a concatenation instead. Those have no
convolution to fold into, and `RewriteBatchNormToAffine` is the pass that
takes them. Running the first without the second leaves 56 of them on portable
kernels, and every one is a break in the delegate chain around it.

On CAMPPlus (`iic/speech_campplus_zh-cn_common`, exported at (1, 300, 80))
the difference is:

    [FoldBatchNormIntoConv()]                              435 nodes, 109 subgraphs, 217 crossings
    [FoldBatchNormIntoConv(), RewriteBatchNormToAffine()]  265 nodes,  54 subgraphs, 107 crossings

Same model, same partitioner, same blob format -- half as many crossings.

    to_edge_transform_and_lower(
        ep,
        transform_passes=default_fusion_passes(),
        partitioner=[HexagonPartitioner()],
    )

The order is the one the rewrites need. The `Preserve*` passes go first
because they exist to stop a pattern being decomposed before anything else
looks at it. Constants are folded next, so a later match sees a canonical
transpose. The convolutions that have no kernel are rewritten into ones that
do before any pass tries to match a convolution. The two batch-norm passes
are last, the fold that needs a convolution ahead of it and then the rewrite
that does not.
"""

from typing import List

from executorch.backends.hexagon.batch_norm import RewriteBatchNormToAffine
from executorch.backends.hexagon.conv_patch_embed import DecomposePatchEmbed
from executorch.backends.hexagon.decompose_conv3d import DecomposeFrameConv3d
from executorch.backends.hexagon.fold_batch_norm import FoldBatchNormIntoConv
from executorch.backends.hexagon.fold_transposes import FoldConstantTransposes
from executorch.backends.hexagon.prelu import PreservePRelu
from executorch.backends.hexagon.reflect_pad import PreserveReflectPad
from executorch.backends.hexagon.row_guard import FuseMaskedRowGuard
from executorch.backends.hexagon.vision_attention import FuseVisionAttention
from executorch.exir.pass_base import ExportedProgramPassBase


def default_fusion_passes() -> List[ExportedProgramPassBase]:
    """The rewrites a caller should run before it hands a graph to the partitioner.

    Each pass here leaves the program alone when its pattern is not in the
    graph, so the list is safe to hand whole: a model that matches none of
    them pays the traversal and nothing else.

    The passes that work on the graph module rather than the program --
    `FuseAddReluPass`, `FuseAddRmsNormPass`, `FuseMulSiluPass`,
    `FuseRmsNormPass`, `FuseRopePass`, `FuseCumsumPass`,
    `FuseKvCachePass` -- are not in this list. They take the graph module, not
    the program, so a caller that wants them applies them first and hands the
    result back:

        program = export(model, inputs)
        for fuse in (FuseAddReluPass(), FuseMulSiluPass()):
            program.graph_module = fuse(program.graph_module)
        to_edge_transform_and_lower(
            program,
            transform_passes=default_fusion_passes(),
            partitioner=[HexagonPartitioner()],
        )

    They compose the same way this list does, which is the reason to name them
    here rather than leave them to be found: a caller reading this has been
    told which ones exist and that they are the other kind.
    """
    return [
        PreservePRelu(),
        PreserveReflectPad(),
        FoldConstantTransposes(),
        DecomposeFrameConv3d(),
        DecomposePatchEmbed(),
        FoldBatchNormIntoConv(),
        RewriteBatchNormToAffine(),
        FuseMaskedRowGuard(),
        FuseVisionAttention(),
    ]
