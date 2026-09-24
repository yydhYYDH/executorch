# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The weight fold, and the two ways it could be wrong.

A diffuse model's FIR upsampler writes its transposed convolution's weight as a
chain -- reshape, flip, permute, reshape -- over a parameter, so at export the
weight is a graph computation rather than a constant. The emitters require a
constant weight, so without a fold that node never reaches the convolution walk
and the whole transposed convolution stays on a portable kernel.

The fold is a permutation of the same bytes, which makes it safe by exactness
and also makes it easy to get wrong in a way that still lowers: flip the wrong
axis and the graph delegates exactly as before, with every output value wrong.
So the tests below pin the folded tensor against the chain's own arithmetic,
with a fixture that says which wrong answer would have been accepted instead,
and they pin the other direction too -- a weight that is a run-time input must
not be folded, because baking one call's data into the program is a much worse
failure than not folding.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.fold_transposes import (  # noqa: E402
    FoldConstantTransposes,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_IM2COL = 12
_BLIT = 3


def _weight_of(shape, seed) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-1, 2, shape, generator=generator).to(torch.float16)


def _prepare(weight, groups):
    """The FIR upsampler's weight preparation, with the flip's dims as a knob.

    `diffusers/models/upsampling.py`'s `FirUpsample2D._upsample_2d` writes
    exactly this: reshape to a group view, flip the two spatial axes, permute
    the group and input-channel axes together, reshape back.
    """
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    layered = weight.reshape(groups, -1, weight.shape[1], kernel_h, kernel_w)
    flipped = torch.flip(layered, dims=[3, 4])
    return flipped.permute(0, 2, 1, 3, 4).reshape(
        groups * weight.shape[1], -1, kernel_h, kernel_w
    )


def _deconv(x, prepared, groups):
    """`conv_transpose2d(x, prepared, stride=2, padding=0, output_padding=op)`.

    `output_padding` is what the upsampler computes: the shape that makes the
    output `(H - 1) * 2 + K` exactly, which is zero for this kernel.
    """
    height, width = x.shape[2], x.shape[3]
    out_h = (height - 1) * 2 + prepared.shape[2]
    out_w = (width - 1) * 2 + prepared.shape[3]
    output_padding = (
        out_h - (height - 1) * 2 - prepared.shape[2],
        out_w - (width - 1) * 2 - prepared.shape[3],
    )
    return torch.nn.functional.conv_transpose2d(
        x,
        prepared,
        stride=(2, 2),
        padding=0,
        output_padding=output_padding,
        groups=groups,
    )


class _FirUpsample(torch.nn.Module):
    """diffusers' `FirUpsample2D`, conv_transpose half, weight as a parameter.

    `grouped` is the real one: SDXL's FIR upsampler passes `groups=in_channels`,
    so its weight is `[C, 1, K, K]`. The ungrouped variant is here because its
    preparation actually moves a non-trivial axis, which is the shape that would
    expose a fold permuting two axes the wrong way round.
    """

    def __init__(self, channels, grouped=True, kernel=3) -> None:
        super().__init__()
        self.grouped = grouped
        groups = channels if grouped else 1
        self.conv = torch.nn.Conv2d(
            channels, channels, kernel, stride=1, padding=1, groups=groups
        )

    def forward(self, x):
        groups = x.shape[1] if self.grouped else 1
        return _deconv(x, _prepare(self.conv.weight, groups), groups)


class _RuntimeWeight(torch.nn.Module):
    """The same convolution with its weight handed in at call time.

    Nothing about this shape is refused by geometry -- it is the constant that is
    missing, and that has to be what keeps it portable.
    """

    def __init__(self, grouped=True) -> None:
        super().__init__()
        self.grouped = grouped

    def forward(self, x, weight):
        groups = x.shape[1] if self.grouped else 1
        return _deconv(x, _prepare(weight, groups), groups)


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _lower(module, inputs):
    return to_edge_transform_and_lower(
        export(module.eval(), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _commands(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"the model did not reach one delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    data = bytes(lowered._processed_bytes)
    _, commands = read_blob(data)
    return data, commands


def _portable(program):
    return [
        str(getattr(node.target, "_schema", node.target))
        .split("schema = ")[-1]
        .split("(")[0]
        .split("::")[-1]
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is not torch.ops.higher_order.executorch_call_delegate
    ]


@pytest.mark.parametrize(
    "channels,grouped,kernel",
    [(8, True, 3), (16, True, 3), (8, False, 3), (4, True, 2), (12, False, 3)],
)
def test_the_folded_weight_is_the_chain_it_replaces(channels, grouped, kernel):
    """The constant the pass stores equals the chain's own arithmetic.

    Nothing here goes near the partitioner: the pass is run on the program and
    the tensor it registered is compared, byte for byte, against the same ops
    the graph named, evaluated here. A fold that flipped the wrong axis, or that
    permuted two axes the wrong way round, produces a different tensor and is
    caught here rather than downstream, where it would look like a lowering that
    succeeded and numbers that are wrong.
    """
    module = _FirUpsample(channels, grouped, kernel)
    groups = channels if grouped else 1
    module.conv.weight.data = _weight_of(module.conv.weight.shape, 21)
    exported = export(
        module.eval(), (torch.randint(-1, 2, (1, channels, 4, 4)).half(),)
    )

    folded = FoldConstantTransposes()(exported)
    assert folded.modified, "the chain was not folded at all"

    expected = _prepare(module.conv.weight, groups).detach()
    # The fold lifts the tensor it registered into a constant of its own, so the
    # stored weight is found by shape rather than by the name it was made with.
    stored = [
        tensor
        for name, tensor in folded.exported_program.state_dict.items()
        if name != "conv.weight" and tuple(tensor.shape) == tuple(expected.shape)
    ]
    assert len(stored) == 1, sorted(folded.exported_program.state_dict)
    got = stored[0].detach()
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert got.contiguous().numpy().tobytes() == expected.contiguous().numpy().tobytes()

    # The comparison discriminates: the same chain with the flip on the wrong
    # axes -- the mistake that would still lower and put every value elsewhere --
    # is not this tensor.
    layered = module.conv.weight.reshape(
        groups, -1, module.conv.weight.shape[1], kernel, kernel
    )
    wrong = (
        torch.flip(layered, dims=[2, 3])
        .permute(0, 2, 1, 3, 4)
        .reshape(groups * module.conv.weight.shape[1], -1, kernel, kernel)
        .detach()
        .contiguous()
    )
    assert wrong.shape == expected.shape
    assert got.contiguous().numpy().tobytes() != wrong.numpy().tobytes()


def test_a_run_time_weight_is_not_folded():
    """The other direction, and the one that would be an accident.

    A weight that arrives as an input is not a value this pass holds. Folding it
    would bake one call's data into the program, so the node has to stay on a
    portable kernel -- and the test asserts that from the lowering, not from the
    pass's return value, because what matters is where the op ends up.
    """
    module = _RuntimeWeight().half()
    x = torch.randint(-1, 2, (1, 8, 4, 4)).half()
    weight = _weight_of((8, 1, 3, 3), 22)

    program = _lower(module, (x, weight))
    portable = _portable(program)
    # The op is spelled `convolution` with transposed=True once EXIR has
    # decomposed it, so the test asks whether a convolution is left portable
    # rather than which spelling it has.
    assert any(
        "conv" in name for name in portable
    ), f"a run-time weight was folded: {portable}"
    for delegate in _delegates(program):
        lowered = program.graph_module.get_submodule(delegate.args[0].target)
        _, commands = read_blob(bytes(lowered._processed_bytes))
        assert all(
            command.type != _IM2COL for command in commands
        ), "the convolution reached the DSP with a weight this pass cannot hold"


def test_the_plan_the_pass_leaves_alone_is_still_correct_without_it():
    """The control for the negative above: the same graph, weight in hand.

    If the run-time-weight case stayed portable for some unrelated reason -- the
    geometry, an unsupported dtype -- then the test above would pass while saying
    nothing. This is the same module with a constant weight, and it has to reach
    the DSP, which makes the pair a comparison rather than an assertion about one
    graph.
    """

    module = _FirUpsample(8, grouped=False).half()
    module.conv.weight.data = _weight_of(module.conv.weight.shape, 23)
    x = torch.randint(-1, 2, (1, 8, 4, 4)).half()

    program = _lower(module, (x,))
    _, commands = _commands(program)
    assert _IM2COL in [
        command.type for command in commands
    ], "the fold did not reach the DSP"


def test_a_grouped_fir_weight_is_folded_but_its_convolution_stays_portable():
    """The boundary next to the case above, and not the shape SDXL exports.

    `FirUpsample2D._upsample_2d` passes no `groups` to `conv_transpose2d`, so its
    weight is `[C, C, 3, 3]` and the convolution is ungrouped -- that is the case
    the test below measures. A depthwise variant (`groups == C`, weight
    `[C, 1, 3, 3]`) prepares its weight the same way and the fold still runs, but
    the transposed path admits only `groups == 1`: its weight is a dense in/out
    pair even at one channel each, so the depthwise walk is not the right one for
    it. The convolution therefore stays portable while the fold itself does not,
    which is the pair worth pinning -- it separates "the fold did not fire" from
    "the geometry is not this walk's".
    """
    module = _FirUpsample(8, grouped=True).half()
    module.conv.weight.data = _weight_of(module.conv.weight.shape, 26)
    x = torch.randint(-1, 2, (1, 8, 4, 4)).half()

    exported = export(module.eval(), (x,))
    assert FoldConstantTransposes()(exported).modified, "the chain was not folded"

    program = _lower(module, (x,))
    assert any("conv" in name for name in _portable(program)), _portable(program)


def test_the_fir_upsampler_reaches_the_dsp_and_keeps_its_numbers():
    """End to end: the fold's whole purpose, and its arithmetic.

    The convolution walk is the only thing that changed -- it now carries a
    transposed convolution whose weight was written as a chain -- so the numbers
    have to come out of the DSP as the same bytes. Integer weights and inputs
    keep every partial sum exact in fp16, which is what makes agreement a
    statement about the mapping rather than about rounding.
    """
    module = _FirUpsample(64, grouped=False).half()
    module.conv.weight.data = _weight_of(module.conv.weight.shape, 24)
    x = torch.randint(-1, 2, (1, 64, 8, 8)).half()

    data, commands = _commands(_lower(module, (x,)))
    types = [command.type for command in commands]
    assert types.count(_BLIT) == 3, types
    assert types.count(_IM2COL) == 1, types

    got = np.frombuffer(execute(data, [x.numpy()])[0], dtype=np.float16)
    with torch.no_grad():
        expected = module(x).half().numpy().reshape(-1)
    assert got.tobytes() == expected.tobytes()
