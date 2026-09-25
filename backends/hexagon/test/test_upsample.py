# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nearest upsampling on the DSP, as raster regions.

Stable Diffusion's UNet up-blocks and its VAE decoder both upsample with
`F.interpolate(scale_factor=2.0, mode="nearest")` before a 3x3 convolution
(diffusers `Upsample2D`, `src/diffusers/models/upsampling.py`), so this is the
resize an SD export actually contains. It reaches the DSP without a sampling
kernel and without arithmetic: a replication is many-to-one, so one affine blit
region cannot describe it, but a *set* of regions can -- one per phase of the
destination index. For an exact integer multiple `s`, destination index
`k * s + t` reads source index `k`, so each phase is an affine map and the whole
op is `s * s` regions that write the input's own bytes.

That is why the cases here compare **bytes** rather than tolerances: nothing on
this path computes, so there is no rounding to excuse.

The boundary is an exact integer multiple, and the tests below pin it from both
sides -- a fractional ratio, a shrinking ratio and `upsample_bilinear2d` all
have to stay on the portable kernels.
"""



import numpy as np
import pytest
import torch


from blob_interpreter import execute, read_blob
from executorch.backends.hexagon.hexagon_ops import (
    BLIT_BLOCKS_PER_COMMAND,
    BLIT_REGION_INTS,
    upsample_regions,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

_BLIT = 3

#: A raster blit spends three of its forty parameter ints on a header and twelve
#: per region, so a command carries three regions and a factor of `s` takes
#: `ceil(s * s / 3)` of them.
_HEADER = 3


class _Nearest(torch.nn.Module):
    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        import torch.nn.functional as F

        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")


class _NearestSize(torch.nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def forward(self, x):
        import torch.nn.functional as F

        return F.interpolate(x, size=self.size, mode="nearest")


class _Bilinear(torch.nn.Module):
    def __init__(self, scale_factor, align_corners) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        import torch.nn.functional as F

        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
        )


class _SdUpsampleBlock(torch.nn.Module):
    """diffusers' `Upsample2D(channels, use_conv=True)`: nearest 2x, then a conv."""

    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        import torch.nn.functional as F

        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


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


def _blob(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"the model did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    data = bytes(lowered._processed_bytes)
    _, commands = read_blob(data)
    return data, commands


def _run(data, inputs):
    return np.frombuffer(execute(data, inputs)[0], dtype=np.float16)


def _exact(shape, seed):
    """Integer values, so an upsample's bytes are the input's bytes."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-2000, 2000, shape, generator=generator).half()


def _expected(module, x):
    return module(x).detach().half().numpy().reshape(-1)


def _upsample_regions(module, x):
    """The region plan the emitter would build, from the graph before lowering.

    After lowering the node itself is inside the delegate's module, so this goes
    to the plain export and reads the plan off the same node the partitioner
    gates and the emitter emits from.
    """
    for node in export(module.eval(), (x,)).graph_module.graph.nodes:
        if node.target is torch.ops.aten.upsample_nearest2d.vec:
            return upsample_regions(node)
    raise AssertionError("the export contains no nearest upsample node")


def _apply(regions, source, destination):
    """The blit kernel's own region walk, so a plan can be checked on the host."""
    written = np.zeros(destination.shape, dtype=np.int32)
    for start in range(0, len(regions), BLIT_REGION_INTS):
        _, src_offset, dst_offset, *rest = regions[start : start + BLIT_REGION_INTS]
        size, src_stride, dst_stride = rest[:3], rest[3:6], rest[6:9]
        for z in range(size[0]):
            for y in range(size[1]):
                for xx in range(size[2]):
                    destination[
                        dst_offset
                        + z * dst_stride[0]
                        + y * dst_stride[1]
                        + xx * dst_stride[2]
                    ] = source[
                        src_offset
                        + z * src_stride[0]
                        + y * src_stride[1]
                        + xx * src_stride[2]
                    ]
                    written[
                        dst_offset
                        + z * dst_stride[0]
                        + y * dst_stride[1]
                        + xx * dst_stride[2]
                    ] += 1
    return written


def test_a_nearest_upsample_is_regions_and_nothing_else():
    """The whole command stream for a 2x upsample: four regions, two commands.

    Nothing here multiplies, adds or rounds -- the output is the input's own
    bytes, copied four times into the four phases of the destination.
    """
    module = _Nearest(2.0)
    x = _exact((2, 4, 4, 6), 11)
    data, commands = _blob(_lower(module, (x,)))

    assert len(commands) == 2
    for command in commands:
        assert command.type == _BLIT

    regions = _upsample_regions(module, x)
    # The first command carries three phases and the second the fourth.
    assert BLIT_BLOCKS_PER_COMMAND == 3
    assert len(regions) == 4 * BLIT_REGION_INTS

    first, second = commands[0], commands[1]
    assert list(first.params[:3]) == [3, 2, 1]
    assert list(second.params[:3]) == [1, 2, 1]
    assert first.params[_HEADER:] + second.params[_HEADER:] == regions

    # Every phase reads the whole source plane at its own strides and differs
    # only in where in the destination it starts.
    phases = [
        regions[start : start + BLIT_REGION_INTS]
        for start in range(0, len(regions), 12)
    ]
    assert sorted(phase[2] for phase in phases) == [0, 1, 12, 13]
    for phase in phases:
        assert (
            phase[0] == 0 and phase[1] == 0
        ), "no source offset: each phase reads it all"
        assert phase[3:6] == [2 * 4, 4, 6], "plane, rows and columns of the source"
        assert phase[6:9] == [4 * 6, 6, 1], "source strides: plane, row, element"
        assert phase[9:12] == [8 * 12, 2 * 12, 2], "destination: plane, row, interleave"

    assert _run(data, [x.numpy()]).tobytes() == _expected(module, x).tobytes()


@pytest.mark.parametrize("scale", [2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 1), (1, 4, 3, 5), (2, 8, 4, 4), (1, 16, 7, 3), (2, 64, 8, 9)],
)
def test_the_numbers_are_torch_bit_for_bit(scale, shape):
    """A pure copy differs from torch in no bit at all, so compare bytes."""
    module = _Nearest(scale)
    x = _exact(shape, 12)
    data, _ = _blob(_lower(module, (x,)))
    assert _run(data, [x.numpy()]).tobytes() == _expected(module, x).tobytes()


def test_the_size_spelling_lands_on_the_same_commands():
    """`size=` and `scale_factor=` are the same op when the ratio is integral.

    The two spellings reach the same ATen node with different arguments, and
    torch reads the input at `floor(index * scale)` in both -- with `scale` the
    reciprocal of the factor, or the shapes' ratio. Both have to produce the
    same commands, and the same bytes.
    """
    x = _exact((1, 8, 5, 7), 13)
    by_factor, _ = _blob(_lower(_Nearest(2.0), (x,)))
    by_size, _ = _blob(_lower(_NearestSize((10, 14)), (x,)))
    assert by_factor == by_size, "the two spellings should lower identically"
    assert (
        _run(by_factor, [x.numpy()]).tobytes() == _expected(_Nearest(2.0), x).tobytes()
    )


def test_the_regions_are_the_phase_maps_and_cover_the_output():
    """Every destination cell is written exactly once, by the phase that owns it.

    The property the replication rests on, checked against an independent
    expectation over a swept set of shapes: applying the regions by hand must
    reproduce `np.repeat(x, s, axis=1)` and `axis=2`, and must touch each
    destination cell exactly once.
    """
    checked = 0
    for scale in (2, 3, 4):
        for batch, channels, height, width in (
            (1, 1, 3, 5),
            (2, 4, 7, 3),
            (1, 64, 4, 4),
        ):
            x = _exact((batch, channels, height, width), 14 + scale)
            source = x.numpy().reshape(-1).astype(np.int32)
            regions = _upsample_regions(_Nearest(float(scale)), x)
            assert regions is not None
            destination = np.zeros(
                (batch * channels * height * scale * width * scale,), dtype=np.int32
            )
            written = _apply(regions, source, destination)
            assert written.min() == 1, "a destination cell was never written"
            assert written.max() == 1, "a destination cell was written twice"
            expected = np.repeat(
                np.repeat(
                    source.reshape(batch * channels, height, width), scale, axis=1
                ),
                scale,
                axis=2,
            )
            assert destination.tolist() == expected.reshape(-1).tolist()
            checked += 1
    assert checked == 9


def test_a_region_that_is_wrong_produces_a_wrong_answer():
    """The comparison above can fail, so its passing means something.

    The destination interleave stride is what makes each phase land in its own
    cells. Dense it -- the stride a plain copy would use -- and the destination
    is a different tensor, while the command count and the command types are
    unchanged, which is exactly why the byte comparison and not a command count
    is what these tests assert.
    """
    x = _exact((1, 4, 4, 6), 15)
    module = _Nearest(2.0)
    regions = list(_upsample_regions(module, x))
    for start in range(0, len(regions), BLIT_REGION_INTS):
        assert regions[start + 11] == 2
        regions[start + 11] = 1
    source = x.numpy().reshape(-1).astype(np.float16)
    destination = np.zeros((1 * 4 * 8 * 12,), dtype=np.float16)
    _apply(regions, source, destination)
    assert destination.tobytes() != _expected(module, x).tobytes()


@pytest.mark.parametrize(
    "module,shape",
    [
        (_Nearest(1.5), (1, 8, 8, 8)),
        (_Nearest(2.5), (1, 8, 4, 4)),
        (_Nearest(0.5), (1, 8, 8, 8)),
        (_Bilinear(2.0, False), (1, 8, 8, 8)),
        (_Bilinear(2.0, True), (1, 8, 8, 8)),
        (_Nearest(2.0), (1, 8, 8, 8)),
    ],
)
def test_only_an_integer_multiple_has_a_region_set(module, shape):
    """The boundary, from both sides.

    A fractional ratio repeats source elements in runs of unequal length, so the
    phases are no longer `s` of them; a shrinking ratio reads several sources per
    destination cell. Both are refused. `upsample_bilinear2d` is refused for a
    different reason -- its taps depend on the output's parity -- and the last
    case is the control: an integer multiple must still reach the delegate, or
    these refusals would be indistinguishable from an op that never delegates.
    """
    x = _exact(shape, 16)
    delegates = len(_delegates(_lower(module, (x,))))
    if module.scale_factor == 2.0 and not isinstance(module, _Bilinear):
        assert delegates == 1, "an integer multiple must reach the DSP"
    else:
        assert delegates == 0, "this ratio has no region set and must stay portable"


def test_the_sd_upsample_block_reaches_the_dsp_whole():
    """diffusers' `Upsample2D`: nearest 2x then a 3x3 convolution, one delegate.

    The block an SD UNet up-stage and the VAE decoder both contain. Before the
    upsample had an emitter this whole pair stayed portable; the phase regions
    put the first half on the DSP and the convolution was already there.
    """
    module = _SdUpsampleBlock(64).half()
    generator = torch.Generator().manual_seed(18)
    for parameter in module.parameters():
        parameter.data = torch.randint(-1, 2, parameter.shape, generator=generator).to(
            parameter.dtype
        )
    # +-1 in and +-1 weights, so a 64-channel 3x3 window sums to at most 576 and
    # every partial sum stays an exact fp16 integer. Anything larger would make
    # this a comparison about rounding instead of about the mapping.
    x = torch.randint(
        -1, 2, (1, 64, 16, 16), generator=torch.Generator().manual_seed(19)
    ).half()
    data, commands = _blob(_lower(module, (x,)))
    types = [command.type for command in commands]
    # Two commands for the upsample's four phases, then the convolution's own
    # pack and unpack.
    assert types.count(_BLIT) == 4, types
    assert 12 in types, "the im2col convolution"
    assert _run(data, [x.numpy()]).tobytes() == _expected(module, x).tobytes()
