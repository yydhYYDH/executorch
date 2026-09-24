# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folds a batch norm that follows a convolution into that convolution's weights.

The DSP has no batch-norm command -- none of the `DSP_OP_*` entries the runtime
knows is a normalization -- so a batch norm that reaches the partitioner can only
fall back to a portable kernel. Because it is not a Hexagon op it also splits the
delegate chain around it: a stack of conv/BN/relu blocks lowers to one delegate
per convolution plus one for the trailing rectifier, where without the batch
norms the whole stack is a single delegate. Every split in between is a round
trip through the CPU on every inference.

A batch norm over a convolution's output is an affine map per channel, so it can
be applied to the convolution's own weight and bias instead and dropped from the
graph. `FuseBatchNormWithConvPass` in `backends/transforms` is that rewrite, and
is imported rather than reimplemented; this module is the shape a caller can put
in `to_edge_transform_and_lower`'s `transform_passes`:

    to_edge_transform_and_lower(
        program,
        transform_passes=[FoldBatchNormIntoConv()],
        partitioner=[HexagonPartitioner()],
    )

or, without a pass manager, as `FoldBatchNormIntoConv()(program).exported_program`.

Measured over four conv/BN/relu blocks whose batch norms are not the identity --
a fresh `BatchNorm2d` scales by `1/sqrt(1+eps)`, 1-5e-6, and folding that is a
no-op no tolerance can tell from a broken fold: 4 batch-norm nodes and 5 delegates
become 0 and 1, carrying the same four `IM2COL_CONVOLUTION_FP16`, eight
`RASTER_BLIT`, four `ZERO` and four `UNARY` commands in one blob; the phone's
runner reports `enter d0: ops=20` where it reported four `ops` lines around one
`ops=1`; the `.pte` shrinks from 90,756 to 86,308 bytes; and the delegate bytes
are identical to those of the same weights fused by hand and written as a model
with no batch norm at all. On the phone the folded chain is 8.92e-4 from torch,
which is 0.91 of one fp16 step at that output's own peak magnitude.

Two things the upstream pass leaves to its caller, both of which cost something
measurable when they are skipped.

**The unfused weight is not pruned.** Folding points the convolution at a new
fused attribute, which leaves the convolution's original weight and bias
placeholders unused -- and an unused parameter placeholder is still in the
program's signature, so it is serialized. Without the prune the `.pte` carries a
second, dead fp32 copy of every convolution weight: on the four-block stack above
the program grows from 86,308 to 117,028 bytes, and on a real residual network it
would carry the whole weight payload twice. `remove_unused_parameters_pass` is
that prune, and it takes the program being transformed -- which is why this is an
`ExportedProgramPassBase` and not the graph-module pass upstream is. The pass
manager hands a graph-module pass only the graph module, and the program its
caller built is not the one under transformation, so a constructor argument
cannot reach it. The 544 bytes of batch-norm buffers the prune leaves behind
(it handles `InputKind.PARAMETER` and not `InputKind.BUFFER`) are the remainder.

**A fold that lands on the graph's output renames the output node.** When the
batch norm is the last node its `getitem` was the output, and the fold replaces it
with the convolution, so `graph_signature`'s output specs name a node that no
longer exists and `exported_program.validate()` fails at the end of the transform
with `User output aten_convolution_default is not in the correct order or is not
found in the exported program's user_output list`. The pass manager rebuilds the
signature itself for graph-module passes and not for these, so the re-pointing
below is what keeps a model that ends in a batch norm lowerable.

The precondition upstream does not check is the training flag. It matches
`aten.native_batch_norm.default`, whose fifth argument is `training`, as well as
the `_native_batch_norm_legit_no_training` form an eval export produces, and it
reads `eps` from the last argument of both. Folding a node whose training flag is
set would apply the running statistics to a graph that means the batch
statistics -- on the single-block graph in `test_fold_batch_norm.py` the two
readings of that node are 4.49 apart -- which is a wrong answer with no error
anywhere. A graph holding such a node is left whole.

Only the pattern the upstream pass matches is rewritten: a convolution with a
single user, that user a batch norm whose every use is element 0 of its
three-element tuple, with both weights constant. A convolution a residual branch
also reads, a batch norm with `affine=False` (its weight operand is absent, which
the upstream pass refuses even though the fold is still defined for it), a batch
norm after a linear or after another batch norm, and a batch norm whose running
statistics are method inputs all keep their node. Folding those is not reachable
through this pass.
"""

from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.remove_unused_parameters_pass import (
    remove_unused_parameters_pass,
)
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    ExportGraphSignature,
    OutputSpec,
)

#: The batch-norm overloads the upstream pass matches. An eval export reaches the
#: edge program as the second; the first is what a graph from elsewhere carries.
_FOLDABLE = (
    exir_ops.edge.aten.native_batch_norm.default,
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
)

#: `aten.native_batch_norm`'s arguments: input, weight, bias, running_mean,
#: running_var, training, momentum, eps. The no-training overload has no entry at
#: this index, so a node shorter than this cannot be training on batch statistics.
_TRAINING_ARG = 5


def _batch_norm_count(graph_module) -> int:
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target in _FOLDABLE
    )


def _trains_on_batch_statistics(graph_module) -> bool:
    """Whether the graph holds a batch norm that means the batch statistics."""
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != _FOLDABLE[0]:
            continue
        if len(node.args) > _TRAINING_ARG:
            training = node.args[_TRAINING_ARG]
            if isinstance(training, bool) and training:
                return True
    return False


def _repoint_outputs(program: ExportedProgram) -> None:
    """Name the output specs after the nodes the output node actually holds.

    Only the fold's own rename can need this here, and a graph whose outputs it
    did not touch is left exactly as it was. A constant output keeps its own
    argument, the way `exir._program_utils._get_updated_graph_signature` does it.
    """
    graph = program.graph_module.graph
    specs = program.graph_signature.output_specs
    outputs = graph.output_node().args[0]
    if len(specs) != len(outputs):
        return
    repointed = [
        OutputSpec(
            spec.kind,
            spec.arg
            if isinstance(spec.arg, ConstantArgument)
            else type(spec.arg)(node.name),
            spec.target,
        )
        for spec, node in zip(specs, outputs)
    ]
    if repointed == specs:
        return
    program._graph_signature = ExportGraphSignature(
        input_specs=program.graph_signature.input_specs,
        output_specs=repointed,
    )


class FoldBatchNormIntoConv(ExportedProgramPassBase):
    """Applies a batch norm to the convolution before it, and drops the node.

    The rewrite itself belongs to `backends/transforms/fuse_batch_norm_with_conv.py`
    and is imported unchanged. What this adds is the precondition and the two
    cleanups described in the module docstring, and a `modified` flag that is the
    truth: upstream returns `True` unconditionally, so a caller cannot tell
    "folded nothing" from "folded something", and a pass that claims a change it
    did not make gets a verifier run for nothing.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph_module = exported_program.graph_module
        before = _batch_norm_count(graph_module)
        if not before or _trains_on_batch_statistics(graph_module):
            return ExportedProgramPassResult(exported_program, False)

        upstream = FuseBatchNormWithConvPass(exported_program)
        upstream(graph_module)
        if _batch_norm_count(graph_module) == before:
            return ExportedProgramPassResult(exported_program, False)

        remove_unused_parameters_pass(exported_program)
        _repoint_outputs(exported_program)
        graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
