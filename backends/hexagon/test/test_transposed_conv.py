# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transposed convolution on the DSP, through the convolution walk.

There is no transposed-convolution kernel here and this backend adds none. What
it does instead is use the identity

    conv_transpose(x, w, s, p, op) ==
        conv2d(zero_insert(x, s, op), flip(w).transpose(ic, oc),
                d * (K - 1) - p, dilation=d)

so that the two kernels that already carry convolution carry this too: the
weight is transposed and flipped at export time, the transposed window's padding
is remapped onto the convolution's, and the input is interleaved with zeros by
two commands the backend already emits (`DSP_OP_ZERO` and a raster region) rather
than by a kernel that does not exist.

The identity is exact in exact arithmetic, which is what the integer-valued
cases below measure: with integer inputs and weights every partial sum stays
inside fp16's exact range, so agreement with torch is a statement about the
mapping rather than about rounding. A stride of 1 needs no interleave at all and
is the same three commands a plain convolution is.

Also here: why 3-D convolutions are a different question with the same answer
they always had, and why the interleave region is left whole -- the blit
kernel's interleave fast path guards on strides that make it write the bytes the
region walk would, so a region it claims is a region it carries.
"""

import os
import pathlib
import sys

import blob_interpreter
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
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    conv_spec,
    ConvSpec,
    deconv_weight_as_conv,
    pack_conv_weight,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: The four commands a transposed convolution lowers to, before the channel
#: blocking adds a pair of its own: DSP_OP_ZERO, DSP_OP_RASTER_BLIT, the im2col
#: convolution and the unpack blit.
_ZERO = 24
_BLIT = 3
_IM2COL = 12

_PACK = 64


class _Transposed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, **kwargs) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel, bias=True, **kwargs
        )

    def forward(self, x):
        return self.conv(x)


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


def _whole(model, seed):
    """Integer weights and bias, so every product in these tests is exact."""
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.data = torch.randint(-1, 2, parameter.shape, generator=generator).to(
            parameter.dtype
        )
    return model


def _exact(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-2, 3, shape, generator=generator).half()


def _transposed(model, channels, kernel, stride, padding, output_padding):
    """Whether this node and no other reached the delegate."""
    return len(
        _delegates(
            _lower(model, (torch.randn(1, channels, 8, 8),))
        )
    ) == 1


def test_transposed_convolution_of_stride_one_is_three_commands():
    """No interleave: the transposed window's padding is the whole difference.

    With a stride of 1 the interleaved input *is* the input, so the identity
    costs one weight transform and nothing on the DSP: the blit in, the im2col
    convolution and the blit out are the commands a plain convolution of the
    same geometry emits.
    """
    model = _Transposed(8, 8, 3, stride=1, padding=1).half()
    _whole(model, 5)
    x = _exact((1, 8, 8, 8), 6)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_ZERO, _BLIT, _IM2COL, _BLIT]
    # The convolution walks the input as it stands and pads it by
    # d * (K - 1) - p,
    # which for k=3, p=1 is 1: params[0:2] are padX and padY.
    assert list(commands[2].params[:8]) == [
        1,
        1,  # the remapped padding, which here is the transposed padding itself
        1,
        1,  # dilation
        1,
        1,  # stride, which the identity always reduces to 1
        3,
        3,  # kernel
    ]
    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_transposed_convolution_of_stride_two_interleaves_first():
    """The four commands, the interleave region's exact fields, and the numbers.

    The region is one strided box: the source reads a row at stride 1 and the
    destination writes it at the interleave factor, with each layout's own row
    and plane strides. `output_padding` does not appear in it -- it only makes
    the destination plane taller, which is the `plane` stride here.
    """
    model = _Transposed(4, 6, 3, stride=2, padding=1, output_padding=1).half()
    _whole(model, 7)
    x = _exact((1, 4, 8, 8), 8)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [
        _ZERO,
        _BLIT,
        _ZERO,
        _BLIT,
        _IM2COL,
        _BLIT,
    ]

    # (8 - 1) * 2 + 1 + 1 = 16: the transposed window reaches 15 rows and
    # output_padding adds the sixteenth, which the region leaves clear.
    interleave = commands[1]
    assert list(interleave.params) == [
        1,  # one region
        2,  # fp16
        1,  # one source
        0,
        0,
        0,  # source index and both offsets
        4,
        8,
        8,  # size: batch * channels planes, then the source's height and width
        64,
        8,
        1,  # source strides: plane, row, element
        256,
        32,
        2,  # destination strides: plane, row, interleave
    ]
    assert interleave.inputs[0].size == 4 * 8 * 8 * 2
    assert interleave.outputs[0].size == 4 * 16 * 16 * 2

    # The convolution then walks the interleaved plane -- 16x16 -- as an
    # ordinary convolution with the remapped padding and a stride of 1.
    walk = commands[4]
    assert list(walk.params[:8]) == [1, 1, 1, 1, 1, 1, 3, 3]
    assert list(walk.params[11:13]) == [16, 16]

    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_dilated_transposed_convolution_keeps_dilation_on_the_kernel():
    """The remapped padding changes, while the DSP walk remains dilated."""
    model = _Transposed(4, 6, 3, stride=2, padding=2, dilation=2, output_padding=1).half()
    _whole(model, 17)
    x = _exact((1, 4, 8, 8), 18)
    data, commands = _blob(_lower(model, (x,)))
    walk = next(command for command in commands if command.type == _IM2COL)
    assert list(walk.params[:8]) == [2, 2, 2, 2, 1, 1, 3, 3]
    assert list(walk.params[10:14]) == [16, 16, 16, 16]
    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_grouped_transposed_convolution_partitions_channels_before_each_walk():
    """Each group gets its own input slice, weight tile, and output slice."""
    model = _Transposed(8, 12, 3, stride=2, padding=1, groups=4).half()
    _whole(model, 19)
    x = _exact((1, 8, 8, 8), 20)
    data, commands = _blob(_lower(model, (x,)))
    walks = [command for command in commands if command.type == _IM2COL]
    assert len(walks) == 4
    assert [command.params[18] for command in walks] == [2, 2, 2, 2]
    assert [command.params[20] for command in walks] == [3, 3, 3, 3]
    assert [command.type for command in commands] == [_ZERO, _BLIT] + [
        _BLIT, _ZERO, _BLIT, _IM2COL, _BLIT, _BLIT
    ] * 4
    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


@pytest.mark.parametrize("channels_per_group", [63, 64, 65])
def test_grouped_transposed_convolution_preserves_the_64_channel_boundary(
    channels_per_group,
):
    """Each group packs its own tail, rather than inheriting the whole tensor's."""
    groups = 3
    channels = groups * channels_per_group
    model = _Transposed(
        channels, channels, 3, stride=1, padding=1, groups=groups
    ).half()
    _whole(model, 30 + channels_per_group)
    x = _exact((1, channels, 4, 4), 40 + channels_per_group)
    data, commands = _blob(_lower(model, (x,)))
    walks = [command for command in commands if command.type == _IM2COL]
    assert [command.params[18] for command in walks] == [channels_per_group] * groups
    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "channels,kernel,stride,padding,output_padding,side",
    [
        (4, 3, 1, 0, 0, 8),
        (8, 3, 1, 1, 0, 8),
        (4, 3, 2, 0, 0, 8),
        (4, 3, 2, 1, 0, 8),
        (4, 3, 2, 1, 1, 8),
        (8, 4, 2, 1, 0, 8),
        (4, 3, 3, 1, 0, 7),
        (4, 2, 2, 0, 0, 9),
        (4, 5, 2, 3, 0, 10),
        (4, 3, 1, 1, 0, 16),
        (4, 3, 2, 1, 0, 16),
        (4, 3, 4, 0, 0, 8),
        (4, 4, 4, 1, 0, 16),
    ],
)
def test_the_numbers_are_torch_bit_for_bit(
    channels, kernel, stride, padding, output_padding, side
):
    """Integer-valued fp16 over a geometry sweep, compared as bytes.

    A different window origin, a missing interleave or a weight that was flipped
    the wrong way all move a number here rather than changing it in the last
    place, because every product is a product of small integers.
    """
    model = _Transposed(
        channels,
        channels + 2,
        kernel,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    ).half()
    _whole(model, 11)
    x = _exact((1, channels, side, side), 12)
    data, _ = _blob(_lower(model, (x,)))
    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_the_weight_is_the_convolution_weight_transposed_and_flipped():
    """The transform, and that the emitter's packed bytes are of that transform.

    `pack_conv_weight` reads an `(oc, ic, ky, kx)` tensor and the transposed
    convolution's own weight is `(ic, oc, ky, kx)` wound the other way, so the
    two axes swap and both spatial axes reverse. Comparing the blob's operand
    against `pack_conv_weight` of the transformed tensor is what makes this a
    statement about what the DSP is handed rather than about the helper.
    """
    weight = torch.arange(2 * 3 * 3 * 3, dtype=torch.float16).reshape(2, 3, 3, 3)
    transformed = deconv_weight_as_conv(weight.float().numpy())
    assert transformed.shape == (3, 2, 3, 3)
    for ic in range(2):
        for oc in range(3):
            assert transformed[oc, ic].tolist() == weight[ic, oc].flip(0, 1).tolist()

    model = _Transposed(2, 3, 3, stride=1, padding=1).half()
    model.conv.weight.data = weight.clone()
    model.conv.bias.data = torch.zeros(3, dtype=torch.float16)
    x = _exact((1, 2, 6, 6), 13)
    data, commands = _blob(_lower(model, (x,)))
    walk = commands[2]
    header, _ = read_blob(data)
    ref = walk.inputs[1]
    start = blob_interpreter.B.HEADER_SIZE + header.n_ops * blob_interpreter.B.OP_SIZE
    found = data[start + ref.offset : start + ref.offset + ref.size]
    assert found == pack_conv_weight(transformed, conv_spec_of(walk, model, x))


def conv_spec_of(walk, model, x):
    """The spec the emitter used, rebuilt from the node's own geometry."""
    return ConvSpec(
        batch=1,
        in_channels=2,
        in_h=6,
        in_w=6,
        out_channels=3,
        out_h=6,
        out_w=6,
        kernel_y=3,
        kernel_x=3,
        stride_y=1,
        stride_x=1,
        pad_y=1,
        pad_x=1,
        dilate_y=1,
        dilate_x=1,
        depthwise=False,
        transposed=True,
    )


def _apply_region(region, source, destination):
    """The blit kernel's own region walk, and the fast path that may claim it.

    `region` is the twelve ints the command carries, `source` is the flat
    `[planes][H][W]` buffer and `destination` the flat `[planes][uh][uw]` one.
    Returns the two results separately so a test can compare them: the walk
    applies the region's strides verbatim (blit_ops.cc:1817), and
    `htp_ops_try_interleave_c64_single_blit` (:1555) is the one fast path that
    can claim this shape, with its sixteen lane offsets written out by hand.
    """
    src_index, src_offset, dst_offset, *rest = region
    size, src_stride, dst_stride = rest[:3], rest[3:6], rest[6:9]
    assert src_index == 0
    walked = destination.copy()
    for z in range(size[0]):
        for y in range(size[1]):
            for x in range(size[2]):
                walked[
                    dst_offset
                    + z * dst_stride[0]
                    + y * dst_stride[1]
                    + x * dst_stride[2]
                ] = source[
                    src_offset
                    + z * src_stride[0]
                    + y * src_stride[1]
                    + x * src_stride[2]
                ]

    claimed = None
    if (
        size[2] == 16
        and src_stride[2] == 1
        and dst_stride[2] == 4
        and size[0] > 0
        and size[1] > 0
    ):
        claimed = destination.copy()
        for z in range(size[0]):
            for y in range(size[1]):
                base = dst_offset + z * dst_stride[0] + y * dst_stride[1]
                read = src_offset + z * src_stride[0] + y * src_stride[1]
                for i in range(16):
                    claimed[base + 4 * i] = source[read + i]
    return walked, claimed, size, dst_stride


def test_a_region_the_blit_kernels_fast_path_may_claim_is_the_same_mapping():
    """The one geometry a fast path accepts, and why it is not a hazard.

    A width of 16 interleaved by 4 satisfies `htp_ops_try_interleave_c64_single_blit`'s
    guard exactly (`size[2] == 16`, `srcStride[2] == 1`, `dstStride[2] == 4`), so
    the kernel does take that path rather than the walk. It writes the same
    bytes: the guard fixes those three numbers, and the body is `dst[4 * i] =
    src[i]` over the same row and plane grid, which is what the walk does with
    the strides it matched on. Measured on the simulator in
    test_zero_insert_sim.py, including a mutant region that comes out wrong.
    """
    model = _Transposed(4, 4, 3, stride=4, padding=1).half()
    _whole(model, 17)
    x = _exact((1, 4, 16, 16), 18)
    data, commands = _blob(_lower(model, (x,)))

    interleave = commands[1]
    assert interleave.type == _BLIT
    params = list(interleave.params)
    assert params[0] == 1, "one region carries the whole scatter"
    region = params[3:15]
    assert region[4:6] == [16, 16], "the source row and its 16 columns"
    assert region[9:12] == [61 * 61, 4 * 61, 4], "plane, row and interleave strides"

    planes, source_h, source_w = 4, 16, 16
    up_h = up_w = (16 - 1) * 4 + 1
    plane = up_h * up_w
    source = np.arange(planes * source_h * source_w, dtype=np.float16)
    destination = np.zeros(planes * plane, dtype=np.float16)
    walked, claimed, size, dst_stride = _apply_region(region, source, destination)
    assert claimed is not None, "the guard should accept this region"
    assert claimed.tobytes() == walked.tobytes()

    # And the destination is the zero-insert the emitter intended, read back at
    # the region's own plane and row strides.
    for p in range(planes):
        for y in range(source_h):
            for xx in range(source_w):
                assert (
                    walked[p * plane + y * 4 * up_w + xx * 4]
                    == source[p * source_h * source_w + y * source_w + xx]
                )

    expected = model(x).detach().half().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_every_interleave_region_is_the_zero_insert_it_claims_to_be():
    """The region encoding itself, swept, checked against numpy on the host.

    This is about the region the emitter builds rather than about what the
    interpreter does with it: the strides are applied here the way the kernel's
    walk applies them, and the result has to be the zero-insert. It also runs
    the fast path's own mapping wherever its guard matches, so a geometry whose
    two readings differ would fail here rather than on a phone.
    """
    geometries = [
        (batch, channels, kernel, stride, padding)
        for batch, channels, kernel, stride, padding in [
            (1, 1, 2, 2, 0),
            (1, 4, 3, 1, 1),
            (1, 4, 3, 2, 0),
            (1, 4, 3, 2, 1),
            (1, 4, 3, 3, 1),
            (1, 4, 3, 4, 0),
            (1, 4, 3, 4, 1),
            (1, 4, 4, 2, 1),
            (1, 4, 2, 3, 1),
            (2, 4, 3, 2, 1),
            (1, 16, 3, 4, 1),
            (1, 16, 2, 4, 0),
            (1, 4, 1, 2, 0),
            (1, 4, 3, 8, 2),
        ]
    ]
    seen = 0
    for batch, channels, kernel, stride, padding in geometries:
        if padding > kernel - 1:
            continue
        for side in (1, 2, 5, 16, 17):
            if (side - 1) * stride - 2 * padding + kernel <= 0:
                continue
            model = _Transposed(
                channels, channels, kernel, stride=stride, padding=padding
            ).half()
            _whole(model, 23)
            x = _exact((batch, channels, side, side), 24)
            try:
                _, commands = _blob(_lower(model, (x,)))
            except (AssertionError, RuntimeError):
                continue
            blits = [c for c in commands if c.type == _BLIT]
            # The first blit of a transposed model over a plane that has to grow
            # is the interleave; the channel-block pair follows it.
            if stride == 1 or side == 1:
                continue
            params = list(commands[1].params)
            assert params[0] == 1 and params[2] == 1, params
            region = params[3:15]
            up_h = (side - 1) * stride + 1
            plane = up_h * ((side - 1) * stride + 1)
            source = np.arange(batch * channels * side * side, dtype=np.float16)
            destination = np.zeros(batch * channels * plane, dtype=np.float16)
            walked, claimed, size, dst_stride = _apply_region(
                region, source, destination
            )
            if claimed is not None:
                assert claimed.tobytes() == walked.tobytes(), region
                assert region[5] == 16 and region[8] == 1 and region[11] == 4
            for p in range(batch * channels):
                for y in range(side):
                    for xx in range(side):
                        assert (
                            walked[p * plane + y * stride * up_h + xx * stride]
                            == source[p * side * side + y * side + xx]
                        )
            seen += 1
    assert seen > 0, "the sweep found no interleave region to check"


def test_conv_spec_describes_the_convolution_the_commands_carry():
    """The spec is the convolution's geometry, with the interleave alongside it."""
    model = _Transposed(4, 6, 3, stride=2, padding=1, output_padding=1).half()
    x = _exact((1, 4, 8, 8), 25)
    calls = _delegates(_lower(model, (x,)))
    inner = calls[0].args[0].target
    program = _lower(model, (x,))
    lowered = program.graph_module.get_submodule(inner)
    node = next(
        candidate
        for candidate in lowered.original_module.graph_module.graph.nodes
        if candidate.op == "call_function"
    )
    spec = conv_spec(node, lambda _node: True)
    assert spec is not None
    assert (spec.in_h, spec.in_w) == (16, 16), "the interleaved extent"
    assert (spec.stride_y, spec.stride_x) == (1, 1)
    assert (spec.pad_y, spec.pad_x) == (1, 1), "d * (K - 1) - p"
    assert (spec.out_h, spec.out_w) == (16, 16)
    assert spec.transposed
    assert (spec.upsample_y, spec.upsample_x) == (2, 2)
    assert (spec.tail_y, spec.tail_x) == (1, 1)


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel,kwargs,why",
    [
        (4, 6, 3, dict(stride=1, padding=5, dilation=2), "padding wider than the effective kernel"),
    ],
)
def test_what_the_identity_cannot_carry_stays_portable(
    in_channels, out_channels, kernel, kwargs, why
):
    """Each refusal named, because each is a different reason.

    Dilation and groups are host lowering concerns now. A padding wider than
    the effective dilated kernel still leaves the remapped convolution a
    negative padding to walk, so that nearest refusal remains pinned.
    """
    assert not _transposed(
        _Transposed(in_channels, out_channels, kernel, **kwargs),
        in_channels,
        kernel,
        kwargs.get("stride", 1),
        kwargs.get("padding", 0),
        kwargs.get("output_padding", 0),
    ), why


def test_a_weight_the_exporter_cannot_read_stays_portable():
    """The weight is transformed at export, so a run-time one has no transform."""

    class Runtime(torch.nn.Module):
        def forward(self, x, weight):
            return torch.nn.functional.conv_transpose2d(x, weight)

    program = _lower(Runtime(), (torch.randn(1, 4, 8, 8), torch.randn(4, 6, 3, 3)))
    assert _delegates(program) == []


def test_three_dimensional_convolution_is_a_different_target():
    """3-D is not this identity short of a kernel -- it is a different operator.

    The two-dimensional rule above cannot be stretched to three: `Im2ColParameter`
    carries `kernelX`, `kernelY`, `iw`, `ih`, `ow` and `oh` and no depth axis at
    all (third-party/mnn-htp-ops/include/dsp/ops.h:45), so no kernel here walks a
    volume. EXIR agrees by not asking: `nn.Conv3d` exports to
    `aten.conv3d.default` and `nn.ConvTranspose3d` to
    `aten.conv_transpose3d.input`, neither of which is in `CONV_TARGETS`, so
    both stay portable for a reason that predates this change.
    """
    for module in (
        torch.nn.Conv3d(4, 6, 3, padding=1),
        torch.nn.ConvTranspose3d(4, 6, 3, stride=2, padding=1),
    ):

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)

        program = _lower(Wrapper(module), (torch.randn(1, 4, 4, 8, 8),))
        assert _delegates(program) == [], type(module).__name__


@pytest.mark.parametrize("mode", ["bilinear"])
def test_resize_has_no_emitter_and_stays_portable(mode):
    """Bilinear resize is refused at the operator, not at a geometry.

    There is no sampling or scaling entry point in the vendored kernels --
    `DSPOpType` (third-party/mnn-htp-ops/include/htp_command.h:32) has no
    upsample case -- and bilinear is arithmetic rather than an index map: its
    taps alternate with the output row's parity, an even row reading the row
    below it and an odd row the row above, so it is neither a shift-invariant
    filter nor the constant-kernel transposed convolution that would let the
    convolution walk carry it.

    Nearest is the other case and is no longer here: at an integer multiple its
    replication is a set of affine phase regions, which `test_upsample.py`
    covers. What stays portable is the ratio that is not an integer multiple,
    and that case is below.
    """

    class Upsample(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.up = torch.nn.Upsample(scale_factor=2, mode=mode)

        def forward(self, x):
            return self.up(x)

    program = _lower(Upsample(mode), (torch.randn(1, 64, 8, 8),))
    assert _delegates(program) == []


def test_a_fractional_nearest_resize_stays_portable():
    """The half of the resize question that is a boundary, not an emitter.

    A non-integer ratio repeats source rows in runs of unequal length, so the
    destination phases are no longer a constant stride apart and the region set
    that carries the integer case does not exist. That is a statement about the
    geometry rather than about the op.
    """

    class Upsample(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=1.5, mode="nearest")

    program = _lower(Upsample(), (torch.randn(1, 64, 8, 8),))
    assert _delegates(program) == []
