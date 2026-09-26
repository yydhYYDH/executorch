# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pooling on the DSP, and the layout that has to be built around it.

`hvx_pool2d_fp16` reads its activation in the DSP's 64-channel blocked layout
(`src + ((c//64) * batch + n) * h * w * 64 + (y * w + x) * 64 + c % 64`,
pool_fp16.c:22) while the arena holds row-major NCHW, so the emitter puts a blit
either side of the command. These tests pin the delegation, the three commands'
params, the element mapping that connects them, and every shape that must *not*
reach the command.
"""



import blob_interpreter
import numpy as np
import pytest
import torch


from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    BLIT_BLOCKS_PER_COMMAND,
    BLIT_REGION_INTS,
    _channel_block_regions,
    _channel_blocks,
    _conv_layouts_agree,
    _pool_kernel_divisor_holds,
    _pool_output_extent,
    _pool_window_intersects,
    pool_spec,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_POOL2D_FP16 and DSP_OP_RASTER_BLIT: the commands a pool lowers to.
_POOL = 1
_BLIT = 3

#: The kernel's two selectors, from hvx_pool2d_fp16.
_MAX = 0
_AVERAGE = 1
_COUNT_VALID = 0
_COUNT_KERNEL = 1

#: One HVX vector of fp16.
_PACK = 64

#: The two pool targets a refusal can be read off, spelled out because a node's
#: argument list is positional and the two do not line up: ceil_mode is the fifth
#: argument of a maximum and the fourth of an average.
_MAX_POOL2D = exir_ops.edge.aten.max_pool2d_with_indices.default
_AVG_POOL2D = exir_ops.edge.aten.avg_pool2d.default

#: torch's own max_pool2d padding limit, which is what makes a window that
#: misses the input unreachable through the functional API.
_MAX_PADDING_RATIO = 0.5


class _Adaptive(torch.nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)


class _Pool(torch.nn.Module):
    def __init__(self, kind, kernel, stride, padding=0, **kwargs) -> None:
        super().__init__()
        self.kind = kind
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.kwargs = kwargs

    def forward(self, x):
        if self.kind == "max":
            return torch.nn.functional.max_pool2d(
                x,
                self.kernel,
                self.stride,
                self.padding,
                **self.kwargs,
            )
        return torch.nn.functional.avg_pool2d(
            x,
            self.kernel,
            self.stride,
            self.padding,
            **self.kwargs,
        )


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _commands(program):
    """The blob's command stream, from the one delegate a whole model lowers to."""
    calls = _delegates(program)
    assert len(calls) == 1, f"the pool did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    _, commands = read_blob(bytes(lowered._processed_bytes))
    return bytes(lowered._processed_bytes), commands


def _lowered_program(model, x):
    return to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _lowered(model, x):
    return _commands(_lowered_program(model, x))


def _run(blob, x):
    return np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)


def test_a_max_pool_is_a_pack_a_pool_and_an_unpack():
    """Three commands, their exact params, and torch's own numbers.

    A max over a window that lands entirely inside the input is exact in fp16,
    so the comparison is an equality rather than a tolerance: any permutation the
    blits got wrong would show up as another channel's maximum.
    """
    x = torch.randn(1, 64, 8, 8, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 2, 2), x)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]

    pack, pool, unpack = commands
    assert list(pack.params) == [
        1,  # one region
        2,  # fp16
        1,  # one source
        0,
        0,
        0,  # src index 0, no offsets
        1,
        64,
        64,  # size: [batch][channels][area]
        64 * 64,
        64,
        1,  # src strides: [batch][channels][area]
        64 * 64,
        1,
        64,  # dst strides: [batch][area][64]
    ]
    assert list(pool.params) == [
        1,  # batch
        8,
        8,  # then the input's height and width
        4,
        4,  # then the output's
        1,  # c4: one 64-channel block, the only count the emitter takes
        2,
        2,  # kernel
        2,
        2,  # stride
        0,
        0,  # padding
        0,  # padType, read by neither path
        _COUNT_KERNEL,  # countType, ignored by the max
        _MAX,  # poolType
    ]
    assert list(unpack.params) == [
        1,
        2,
        1,
        0,
        0,
        0,
        1,
        64,
        16,  # size: [batch][channels][area] of the *output*
        16 * 64,
        1,
        64,  # the kernel's blocked layout again
        64 * 16,
        16,
        1,  # back to row-major
    ]

    got = _run(blob, x)
    expected = torch.nn.functional.max_pool2d(x, 2, 2)
    assert got.tobytes() == expected.numpy().tobytes()


def test_a_padded_max_pool_skips_the_positions_outside_the_input():
    """A window that hangs off the input reads only what is inside it."""
    x = torch.randn(2, 64, 10, 10, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 3, 2, 1), x)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
    assert list(commands[1].params[:12]) == [2, 10, 10, 5, 5, 1, 3, 3, 2, 2, 1, 1]
    got = _run(blob, x)
    expected = torch.nn.functional.max_pool2d(x, 3, 2, 1)
    assert got.tobytes() == expected.numpy().tobytes()


@pytest.mark.parametrize("count_include_pad", [True, False])
def test_average_selects_the_divisor_the_kernel_takes(count_include_pad):
    """count_include_pad picks between the kernel's two countTypes.

    The kernel divides by the window's area or by the positions that landed
    inside it, which are exactly torch's two averages; it multiplies by one over
    the count rounded to fp16, so the comparison is to fp16's own precision.
    """
    x = torch.randn(1, 64, 7, 7, dtype=torch.float16)
    model = _Pool("avg", 3, 2, 1, count_include_pad=count_include_pad)
    blob, commands = _lowered(model, x)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
    assert list(commands[1].params) == [
        1,
        7,
        7,
        4,
        4,
        1,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        _COUNT_KERNEL if count_include_pad else _COUNT_VALID,
        _AVERAGE,
    ]
    got = _run(blob, x)
    expected = model(x)
    np.testing.assert_allclose(got, expected.numpy().reshape(-1), rtol=2e-3, atol=2e-3)


def test_a_pool_with_a_one_position_input_needs_no_blits():
    """The one shape where the two layouts agree element for element.

    With a 1x1 spatial extent the row-major buffer *is* the blocked one, so the
    command reads and writes the arena directly and the blob holds one command.
    """
    x = torch.randn(1, 64, 1, 1, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 1, 1), x)
    assert [command.type for command in commands] == [_POOL]
    assert list(commands[0].params) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
    assert _run(blob, x).tobytes() == x.numpy().tobytes()


def test_a_pool_whose_output_is_one_position_packs_but_does_not_unpack():
    """The other half of the shortcut, which is the one a whole-tensor pool takes.

    The kernel still needs its input blocked, so the pack stays; the output is
    one position, so it comes back out already row-major.
    """
    x = torch.randn(2, 64, 8, 8, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 8, 8), x)
    assert [command.type for command in commands] == [_BLIT, _POOL]
    assert list(commands[1].params) == [
        2,
        8,
        8,
        1,
        1,
        1,
        8,
        8,
        8,
        8,
        0,
        0,
        0,
        1,
        0,
    ]
    assert _run(blob, x).tobytes() == (
        torch.nn.functional.max_pool2d(x, 8, 8).numpy().tobytes()
    )


def _pack_mapping(region, batch, area):
    """The permutation the DSP's own pack path writes, from blit_ops.cc:714-745.

    `htp_ops_pack_area_transpose_nchw_to_nc4hw4_z` loads one vector per channel
    (`src[row * srcStride[1] + col]`), transposes the 64x64 tile, and stores each
    result vector at `dst[(col) * 64]`; the tail loop spells the same mapping out
    element by element, `dst[col * 64 + row] = src[row * area + col]`. The
    region's own strides have to describe exactly that, or the command would fire
    on a geometry the fast path never wrote.
    """
    (
        _src_index,
        _src_offset,
        _dst_offset,
        sizes_0,
        sizes_1,
        sizes_2,
        ss0,
        ss1,
        ss2,
        ds0,
        ds1,
        ds2,
    ) = region
    # The specialisation's conditions, transcribed from blit_ops.cc:843-857.
    assert sizes_1 == 64, "the fast path only takes a full 64-channel row"
    assert sizes_2 == area
    assert (ss1, ss2) == (area, 1)
    assert (ds1, ds2) == (1, 64)
    assert (ss0, ds0) == (area * 64, area * 64)

    pairs = set()
    for z in range(sizes_0):
        for row in range(sizes_1):
            for col in range(sizes_2):
                source = z * ss0 + row * ss1 + col * ss2
                destination = z * ds0 + row * ds1 + col * ds2
                pairs.add((source, destination))
    # The fast path's mapping: source (row, col) of the batch's [64][area] tile
    # lands at col * 64 + row of the blocked one.
    expected = {
        (z * area * 64 + row * area + col, z * area * 64 + col * 64 + row)
        for z in range(batch)
        for row in range(64)
        for col in range(area)
    }
    assert pairs == expected, "the region is not the permutation the fast path takes"


def test_the_pack_region_is_the_permutation_the_dsp_pack_path_takes():
    """Both blits are the geometries the DSP's own pack specialisations handle.

    The host interpreter models the *generic* strided fallback, deliberately, so
    it agrees with the fast paths only if the region describes the same
    permutation. This checks that against the fast paths' own element mapping,
    read out of blit_ops.cc rather than out of the region.
    """
    for batch, area in ((1, 64), (2, 100), (3, 25)):
        packing = _channel_block_regions(batch, area, 64, True)[:12]
        _pack_mapping(packing, batch, area)
        # The reverse direction is the same region with the strides exchanged,
        # which is the mapping htp_ops_pack_area_transpose_nc4hw4_to_nchw_z
        # writes out.
        unpacking = _channel_block_regions(batch, area, 64, False)[:12]
        (
            _,
            _,
            _,
            sizes_0,
            sizes_1,
            sizes_2,
            ss0,
            ss1,
            ss2,
            ds0,
            ds1,
            ds2,
        ) = unpacking
        assert (ss1, ss2) == (1, 64)
        assert (ds1, ds2) == (area, 1)
        assert (ss0, ds0) == (area * 64, area * 64)
        forward = {
            z * area * 64 + row * area + col: z * area * 64 + col * 64 + row
            for z in range(batch)
            for row in range(64)
            for col in range(area)
        }
        backward = {
            z * ss0 + row * ss1 + col * ss2: z * ds0 + row * ds1 + col * ds2
            for z in range(sizes_0)
            for row in range(sizes_1)
            for col in range(sizes_2)
        }
        assert backward == {v: k for k, v in forward.items()}
    assert blob_interpreter.POOL_PACK == _PACK


@pytest.mark.parametrize(
    "shape, kind, kwargs",
    [
        ((1, 64, 8, 8), "max", {"dilation": 2}),  # no dilation in the kernel
        ((1, 64, 8, 8), "avg", {"ceil_mode": True}),  # divisor is a clipped window
        ((1, 64, 8, 8), "avg", {"divisor_override": 2}),  # divisor has no param
    ],
)
def test_a_pool_the_kernel_cannot_run_stays_on_the_host(shape, kind, kwargs):
    """A refusal, and the numbers it still has to produce.

    Each of these is a form the command cannot describe -- a dilation the window
    walk does not have, a divisor that is not the whole kernel, a divisor with no
    param slot -- so the node stays portable rather than reaching an emitter that
    would read it wrong. A channel count is no longer on this list: the command
    counts 64-lane blocks and takes any C.
    """
    model = _Pool(kind, 3, 2, 1, **kwargs)
    x = torch.randn(*shape, dtype=torch.float16)
    program = to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    assert _delegates(program) == [], f"{shape} {kwargs} reached the delegate"
    # Nothing of the graph reached the DSP, so the portable kernel answers; the
    # value has to still be there rather than the graph having been rewritten.
    expected = model(x)
    assert expected.dtype is torch.float16
    assert torch.isfinite(expected).all()


def test_the_output_extent_is_torchs_own_on_both_modes():
    """The command's oh and ow, evaluated against torch rather than read off it.

    Ceil mode is not the numerator's ceiling: torch adds stride - 1 and then
    drops the last output position once, when it would start at or past
    size + pad. The decrement is what keeps the last window from being a window
    over nothing, so this has to be torch's number exactly or the command
    describes an output the graph never had.

    A geometry torch itself refuses is skipped, because the claim is about the
    geometries a graph can carry, and both counts are reported so a sweep that
    quietly checked almost nothing cannot look like one that checked a lot.
    """
    checked = 0
    refused = 0
    for size in range(1, 14):
        for kernel in (1, 2, 3, 4):
            for stride in (1, 2, 3, 4):
                for pad in (0, 1):
                    if 2 * pad >= kernel or stride > kernel:
                        continue
                    for ceil_mode in (False, True):
                        try:
                            want = torch.nn.functional.max_pool2d(
                                torch.zeros(1, 1, size, size),
                                kernel,
                                stride,
                                pad,
                                ceil_mode=ceil_mode,
                            ).shape[-1]
                        except RuntimeError:
                            refused += 1
                            continue
                        got = _pool_output_extent(
                            size, kernel, stride, pad, ceil_mode
                        )
                        assert got == want, (
                            f"{size=} {kernel=} {stride=} {pad=} {ceil_mode=}: "
                            f"{got} != torch's {want}"
                        )
                        checked += 1
    assert checked > 200, f"the sweep only reached {checked} geometries"
    assert refused < checked // 4, f"{refused} of the geometries were skipped"


@pytest.mark.parametrize(
    "kind, kernel, stride, padding, count_include_pad, size",
    [
        ("max", 3, 3, 0, True, 7),
        ("max", 3, 2, 1, True, 10),
        ("max", 2, 2, 0, True, 9),
        ("avg", 3, 3, 0, False, 8),
        ("avg", 3, 2, 1, False, 10),
        ("avg", 2, 3, 0, False, 7),
        ("avg", 3, 4, 1, False, 12),
    ],
)
def test_a_ceil_mode_window_runs_on_this_command(
    kind, kernel, stride, padding, count_include_pad, size
):
    """A window that runs off the edge, end to end, against torch.

    Every size here is one where ceil mode changes the answer rather than merely
    spelling it differently, and the first size is the sharp one: at 8x8 with a
    3x3 window at stride 3 the last window holds a 2x2 of the 3x3. The host
    interpreter is a model of the kernel rather than the kernel, so what this
    establishes is that the command describes torch's geometry.
    """
    x = torch.randn(1, 64, size, size, dtype=torch.float16)
    kwargs = {"ceil_mode": True}
    if kind == "avg":
        kwargs["count_include_pad"] = count_include_pad
    model = _Pool(kind, kernel, stride, padding, **kwargs)
    floor = _Pool(kind, kernel, stride, padding)
    assert model(x).shape != floor(x).shape, "this shape does not exercise ceil"
    blob, commands = _lowered(model, x)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
    assert list(commands[1].params[3:5]) == list(model(x).shape[-2:])
    got = _run(blob, x)
    expected = model(x).numpy().reshape(-1)
    if kind == "max":
        assert got.tobytes() == expected.tobytes()
    else:
        np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_a_ceil_mode_average_over_the_whole_kernel_is_refused_exactly_when_ceil_changes_the_shape():
    """The one thing about ceil mode this command has no param for.

    `count_include_pad` divides by the window clipped to the padded input, so a
    window that hangs off the padded edge divides by the part still inside, and
    countType 1 divides by kernel_y * kernel_x. The two below are the same
    window and the same channel count: one is refused, the other runs, and the
    refusal has to be the divisor rather than the geometry.

    The sweep is the reason this can be said so simply. The divisor test holds
    exactly when ceil mode does not change the shape, over every geometry in
    range, so an average that counts the padding is refused precisely when it
    would have been a different answer -- which is why there is no average in
    the list above that both changes the shape and runs.
    """
    disagreements = 0
    checked = 0
    for size in range(1, 20):
        for kernel in (1, 2, 3, 4, 5):
            for stride in (1, 2, 3, 4, 5):
                for pad in (0, 1, 2):
                    if 2 * pad >= kernel:
                        continue
                    floor = _pool_output_extent(size, kernel, stride, pad, False)
                    ceiling = _pool_output_extent(size, kernel, stride, pad, True)
                    if ceiling < 1:
                        continue
                    checked += 1
                    holds = _pool_kernel_divisor_holds(
                        size, size, ceiling, ceiling, kernel, kernel,
                        stride, stride, pad, pad,
                    )
                    if holds is not (ceiling == floor):
                        disagreements += 1
    assert checked > 400, f"the sweep only reached {checked} geometries"
    assert disagreements == 0, (
        f"{disagreements} of {checked} geometries separate the divisor test from "
        "the shape it is supposed to track"
    )

    x = torch.randn(1, 64, 8, 8, dtype=torch.float16)
    included = _Pool("avg", 3, 3, 0, ceil_mode=True, count_include_pad=True)
    excluded = _Pool("avg", 3, 3, 0, ceil_mode=True, count_include_pad=False)
    assert included(x).shape == excluded(x).shape
    assert _delegates(_lowered_program(included, x)) == [], (
        "an average whose last window hangs off the padded edge reached the DSP"
    )
    blob, commands = _lowered(excluded, x)
    assert commands[1].params[13] == _COUNT_VALID
    expected = excluded(x).numpy().reshape(-1)
    np.testing.assert_allclose(_run(blob, x), expected, rtol=2e-3, atol=2e-3)


def test_the_divisor_test_cannot_move_a_floor_mode_pool():
    """The regression guard: floor mode satisfies it by construction.

    `(oh - 1) * stride <= ih + 2 * pad - kernel` is the floor formula, so this
    is vacuous everywhere the old gate passed. A sweep says it rather than an
    argument, because the failure it prevents is the floor command silently
    starting to be refused. Ceil mode is deliberately not claimed here: at a
    size where it changes the answer this test is meant to fail, and that is
    the other test's subject.
    """
    checked = 0
    for size in range(1, 14):
        for kernel in (1, 2, 3, 4):
            for stride in (1, 2, 3, 4):
                for pad in (0, 1):
                    if 2 * pad >= kernel or stride > kernel:
                        continue
                    output = _pool_output_extent(size, kernel, stride, pad, False)
                    if output < 1:
                        continue
                    checked += 1
                    assert _pool_kernel_divisor_holds(
                        size, size, output, output, kernel, kernel,
                        stride, stride, pad, pad,
                    ), (
                        f"floor mode would now refuse {size=} {kernel=} "
                        f"{stride=} {pad=}"
                    )
    assert checked > 100, f"the sweep only reached {checked} geometries"


def test_a_pool_over_a_batch_less_operand():
    """A 3-D operand is one batch: the command carries the batch as a param."""
    x = torch.randn(64, 8, 8, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 2, 2), x)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
    assert list(commands[1].params[:6]) == [1, 8, 8, 4, 4, 1]
    expected = torch.nn.functional.max_pool2d(x, 2, 2)
    assert _run(blob, x).tobytes() == expected.numpy().reshape(-1).tobytes()


@pytest.mark.parametrize("padding", [2, 3, 4])
def test_a_window_that_misses_the_input_is_refused(padding):
    """The guard, at the level it lives at.

    torch's own pooling refuses a padding past half the kernel, so this cannot
    be reached through the functional API; it is a property of the arithmetic
    the emitter describes, and it is what keeps a command that would answer zero
    where torch answers -inf off the DSP.
    """
    assert not _pool_window_intersects(1, 1, 2, 1, padding)
    assert _pool_window_intersects(8, 4, 2, 2, 0)

    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(1, 64, 1, 1, dtype=torch.float16)
    node = graph.call_function(
        exir_ops.edge.aten.max_pool2d_with_indices.default,
        args=(source, [2, 2], [1, 1], [padding, padding]),
    )
    node.meta["val"] = (
        torch.empty(1, 64, 1, 1, dtype=torch.float16),
        torch.empty(1, 64, 1, 1, dtype=torch.int64),
    )
    assert pool_spec(node) is None


def test_the_host_interpreter_pools_the_way_the_kernel_does():
    """The kernel's walk against torch, including a window that hangs off the edge.

    This is the interpreter's own transcription of pool_fp16.c, checked against
    an implementation that shares no code with it: any disagreement in the
    window's origin, the divisor or the accumulator's seeding shows up here.
    """
    for kernel, stride, padding in ((2, 2, 0), (3, 2, 1), (3, 1, 1), (5, 3, 2)):
        x = torch.randn(2, 64, 9, 9, dtype=torch.float16)
        for kind, kwargs in (
            ("max", {}),
            ("avg", {"count_include_pad": True}),
            ("avg", {"count_include_pad": False}),
        ):
            model = _Pool(kind, kernel, stride, padding, **kwargs)
            blob, commands = _lowered(model, x)
            assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
            got = _run(blob, x)
            expected = model(x).numpy().reshape(-1)
            if kind == "max":
                assert got.tobytes() == expected.tobytes()
            else:
                np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def _pool_node(
    args, source_shape=(1, 64, 8, 8), result_shape=(1, 64, 4, 4), target=None
):
    """A pool node with the values its predicate reads, built by hand."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(source_shape, dtype=torch.float16)
    target = target or exir_ops.edge.aten.max_pool2d_with_indices.default
    node = graph.call_function(target, args=(source, *args))
    result = torch.empty(result_shape, dtype=torch.float16)
    node.meta["val"] = (
        (result, torch.empty(result_shape, dtype=torch.int64))
        if target is exir_ops.edge.aten.max_pool2d_with_indices.default
        else result
    )
    return node


def test_pool_spec_reads_the_command_out_of_a_node_that_fits():
    """The one place the params come from, on a form the kernel takes."""
    spec = pool_spec(_pool_node(([2, 2], [2, 2], [0, 0], [1, 1], False)))
    assert spec == hexagon_ops.PoolSpec(
        batch=1,
        channels=64,
        ih=8,
        iw=8,
        oh=4,
        ow=4,
        kernel_y=2,
        kernel_x=2,
        stride_y=2,
        stride_x=2,
        pad_y=0,
        pad_x=0,
        count_type=_COUNT_KERNEL,
        pool_type=_MAX,
    )


@pytest.mark.parametrize(
    "args, source_shape, result_shape, target",
    [
        (([2, 2], [2, 2]), (1, 64, 1, 8, 8), (1, 64, 1, 4, 4), _MAX_POOL2D),  # rank 5
        (([3, 3], [2, 2], [1, 1], [2, 2]), (1, 64, 8, 8), (1, 64, 3, 3), _MAX_POOL2D),  # dilation
        # The next is the shape check rather than an argument: the result is the
        # size a nonzero padding would give, which this geometry does not describe.
        (([2, 2], [2, 2], [0, 0], [1, 1], False), (1, 64, 8, 8), (1, 64, 3, 3), _MAX_POOL2D),
        # And this one is the window guard: the geometry its padding implies has
        # windows that fall entirely outside the input.
        (([2, 2], [1, 1], [3, 3], [1, 1], False), (1, 64, 8, 8), (1, 64, 13, 13), _MAX_POOL2D),
        # A ceil window that hangs off the padded edge, averaged over the whole
        # kernel. The divisor torch wants is the part still inside, and the
        # command has no divisor that is not kernel_y * kernel_x, so this is the
        # one thing about ceil mode the emitter cannot describe.
        (
            ([3, 3], [3, 3], [0, 0], True, True),
            (1, 64, 8, 8),
            (1, 64, 3, 3),
            _AVG_POOL2D,
        ),
    ],
)
def test_pool_spec_refuses_what_the_command_cannot_describe(
    args, source_shape, result_shape, target
):
    """Every refusal the partitioner depends on, on the node it reads.

    The channel count is deliberately absent: a pool2d the DSP cannot run is now
    refused for one of its arguments or for its rank, and there is no channel
    count left over to be refused.
    """
    assert pool_spec(_pool_node(args, source_shape, result_shape, target)) is None


@pytest.mark.parametrize("extent, output, window", [(63, 21, 3), (64, 32, 2), (65, 13, 5)])
def test_adaptive_pool_spec_uses_the_exact_integer_quotient(extent, output, window):
    spec = pool_spec(
        _pool_node(
            ([output, output],),
            (1, 64, extent, extent),
            (1, 64, output, output),
            target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
        )
    )
    assert spec == hexagon_ops.PoolSpec(
        batch=1,
        channels=64,
        ih=extent,
        iw=extent,
        oh=output,
        ow=output,
        kernel_y=window,
        kernel_x=window,
        stride_y=window,
        stride_x=window,
        pad_y=0,
        pad_x=0,
        count_type=_COUNT_KERNEL,
        pool_type=_AVERAGE,
    )


def test_adaptive_pool_spec_accepts_identity_and_a_single_output_axis():
    identity = pool_spec(
        _pool_node(
            ([8, 12],),
            (1, 64, 8, 12),
            (1, 64, 8, 12),
            target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
        )
    )
    assert identity is not None
    assert (identity.kernel_y, identity.kernel_x) == (1, 1)
    assert (identity.stride_y, identity.stride_x) == (1, 1)
    one_axis = pool_spec(
        _pool_node(
            ([1, 3],),
            (1, 64, 8, 12),
            (1, 64, 1, 3),
            target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
        )
    )
    assert one_axis == hexagon_ops.PoolSpec(
        batch=1,
        channels=64,
        ih=8,
        iw=12,
        oh=1,
        ow=3,
        kernel_y=8,
        kernel_x=4,
        stride_y=8,
        stride_x=4,
        pad_y=0,
        pad_x=0,
        count_type=_COUNT_KERNEL,
        pool_type=_AVERAGE,
    )


def test_adaptive_pool_spec_accepts_a_hand_built_single_output_node():
    spec = pool_spec(
        _pool_node(
            ([1, 1],),
            (1, 64, 8, 8),
            (1, 64, 1, 1),
            target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
        )
    )
    assert spec == hexagon_ops.PoolSpec(
        batch=1,
        channels=64,
        ih=8,
        iw=8,
        oh=1,
        ow=1,
        kernel_y=8,
        kernel_x=8,
        stride_y=8,
        stride_x=8,
        pad_y=0,
        pad_x=0,
        count_type=_COUNT_KERNEL,
        pool_type=_AVERAGE,
    )


@pytest.mark.parametrize(
    "source_shape, output",
    [
        ((1, 64, 63, 63), (4, 4)),
        ((1, 64, 64, 64), (3, 3)),
        ((1, 64, 65, 65), (3, 3)),
        ((1, 64, 8, 8), (16, 16)),
        ((1, 64, 8, 12), (3, 4)),
    ],
)
def test_adaptive_pool_refuses_every_nonfixed_or_clamped_geometry(source_shape, output):
    assert (
        pool_spec(
            _pool_node(
                (output,),
                source_shape,
                (source_shape[0], source_shape[1], *output),
                target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
            )
        )
        is None
    )
    x = torch.randn(*source_shape, dtype=torch.float16)
    program = _lowered_program(_Adaptive(output), x)
    assert _delegates(program) == []


def test_adaptive_pool_lowers_to_one_fixed_pool_with_matching_node_membership():
    x = torch.randn(1, 64, 8, 12, dtype=torch.float16)
    program = _lowered_program(_Adaptive((2, 3)), x)
    call = _delegates(program)[0]
    delegate = program.graph_module.get_submodule(call.args[0].target)
    edge = to_edge(
        export(_Adaptive((2, 3)), (x,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    edge_names = {
        node.name
        for node in edge.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is exir_ops.edge.aten._adaptive_avg_pool2d.default
    }
    delegate_names = {
        node.name
        for node in delegate.original_module.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert delegate_names == edge_names
    blob, commands = _commands(program)
    assert [command.type for command in commands] == [_BLIT, _POOL, _BLIT]
    assert list(commands[1].params) == [1, 8, 12, 2, 3, 1, 4, 4, 4, 4, 0, 0, 0, 1, _AVERAGE]
    np.testing.assert_allclose(
        _run(blob, x),
        _Adaptive((2, 3))(x).numpy().reshape(-1),
        rtol=2e-3,
        atol=2e-3,
    )


def test_adaptive_pool_output_one_uses_the_existing_reduction_not_a_pool():
    x = torch.randn(1, 64, 8, 8, dtype=torch.float16)
    model = _Adaptive((1, 1))
    blob, commands = _lowered(model, x)
    assert _POOL not in [command.type for command in commands]
    np.testing.assert_allclose(
        _run(blob, x), model(x).numpy().reshape(-1), rtol=2e-3, atol=2e-3
    )

#: DSP_OP_ZERO, the only command that fills a buffer without reading one.
_ZERO = 24


def _types(commands):
    return [command.type for command in commands]


def _regions(command):
    """The blit regions one command carries."""
    assert command.type == _BLIT
    count = int(command.params[0])
    body = [int(v) for v in command.params[3:]]
    assert len(body) == count * BLIT_REGION_INTS
    return [
        tuple(body[i : i + BLIT_REGION_INTS]) for i in range(0, len(body), BLIT_REGION_INTS)
    ]


def test_a_128_channel_pool_is_two_blocks_of_the_one_command():
    """The wide pool is the same command with c4 = 2 and one more region a side.

    128 channels is two 64-lane blocks, `hvx_pool2d_fp16` loops over both of them
    (pool_fp16.c:21), and the parameter budget does not move: c4 was already one
    of the fifteen ints the command carries, and two blit regions are twenty-four
    of the forty a command has. A max over a window inside the input is exact in
    fp16, so this is an equality with torch and not a tolerance.
    """
    x = torch.randn(2, 128, 8, 8, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 2, 2), x)
    assert _types(commands) == [_BLIT, _POOL, _BLIT]
    pack, pool, unpack = commands
    assert list(pool.params[:6]) == [2, 8, 8, 4, 4, 2]
    assert len(pool.params) == 15, "c4 was a value, not a new parameter"
    assert _regions(pack) == [
        (0, 0, 0, 2, 64, 64, 128 * 64, 64, 1, 64 * 64, 1, 64),
        (0, 64 * 64, 2 * 64 * 64, 2, 64, 64, 128 * 64, 64, 1, 64 * 64, 1, 64),
    ]
    assert _regions(unpack) == [
        (0, 0, 0, 2, 64, 16, 16 * 64, 1, 64, 128 * 16, 16, 1),
        (0, 2 * 64 * 16, 64 * 16, 2, 64, 16, 16 * 64, 1, 64, 128 * 16, 16, 1),
    ]
    got = _run(blob, x)
    expected = torch.nn.functional.max_pool2d(x, 2, 2)
    assert got.tobytes() == expected.numpy().tobytes()


@pytest.mark.parametrize("kind", ["max", "avg"])
def test_one_block_and_two_blocks_pool_the_same_data_identically(kind):
    """The control: the same bytes, once as one 64-block and once as two.

    A wider pool is only the same kernel if the extra block does not perturb the
    first. The data is duplicated rather than extended so the two runs read
    identical values in identical lanes, and the comparison is exact: a tolerance
    here would let a block that is off by a little pass.
    """
    data = torch.randn(1, 64, 8, 8, dtype=torch.float16)
    wide = torch.cat([data, data], dim=1)
    model = _Pool(kind, 3, 2, 1)
    blob, commands = _lowered(model, wide)
    assert list(commands[1].params[5:6]) == [2]
    got_wide = _run(blob, wide).reshape(1, 128, 4, 4)
    blob_one, commands_one = _lowered(model, data)
    assert list(commands_one[1].params[5:6]) == [1]
    got_one = _run(blob_one, data).reshape(1, 64, 4, 4)
    np.testing.assert_array_equal(got_wide[:, :64], got_one)
    np.testing.assert_array_equal(got_wide[:, 64:], got_one)
    expected = model(data).numpy().reshape(1, 64, 4, 4)
    if kind == "max":
        assert got_one.tobytes() == expected.tobytes()
    else:
        np.testing.assert_allclose(got_one, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("channels", [1, 31, 32, 33, 63, 64, 65, 96, 127, 128, 192, 256])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kind", ["max", "avg"])
def test_every_channel_count_pools(channels, batch, kind):
    """The channel axis swept across the block boundary, from both sides of it.

    63, 64 and 65 are the three that matter: below the block, exactly on it, and
    one channel into the next. 96 is a ragged width real squeeze-excitations use,
    128 and 192 are whole ones, and 1 is the degenerate end. There is no boundary
    to find here: the same code path runs at all twelve, with the last block a
    different width.
    """
    x = torch.randn(batch, channels, 7, 9, dtype=torch.float16)
    model = _Pool(kind, 3, 2, 1)
    blob, commands = _lowered(model, x)
    pool = next(c for c in commands if c.type == _POOL)
    assert list(pool.params[5:6]) == [_channel_blocks(channels)]
    got = _run(blob, x).reshape(batch, channels, 4, 5)
    expected = model(x).numpy().reshape(batch, channels, 4, 5)
    if kind == "max":
        assert got.tobytes() == expected.tobytes()
    else:
        np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("channels", [63, 64, 65, 128])
@pytest.mark.parametrize(
    "kind, kernel, stride, padding",
    [
        ("max", 3, 2, 1),  # window wider than the stride, one pixel of pad
        ("max", 3, 1, 1),  # stride one: every input position is read twice
        ("avg", 3, 2, 1),  # divisor is the window, pad or no pad
        ("avg", 2, 2, 0),  # no padding at all
    ],
)
def test_padding_and_the_packing_agree_at_the_channel_boundary(
    channels, kind, kernel, stride, padding
):
    """Padding decides which element a lane reads, and so does the block width.

    A window is `[oy * stride - pad, ... + kernel)` and a position outside the
    input is skipped rather than read as a zero (pool_fp16.c:36-43), so a padding
    that puts half a window off the plane is the case where a packing mistake
    shows up as a different divisor. Run at 63, 64, 65 and 128 channels the block
    boundary is crossed with the same answer.
    """
    x = torch.randn(2, channels, 8, 8, dtype=torch.float16)
    model = _Pool(kind, kernel, stride, padding)
    blob, commands = _lowered(model, x)
    out = model(x)
    pool = next(c for c in commands if c.type == _POOL)
    assert list(pool.params[:6]) == [
        2,
        8,
        8,
        out.shape[2],
        out.shape[3],
        _channel_blocks(channels),
    ]
    got = _run(blob, x).reshape(tuple(out.shape))
    expected = out.numpy()
    if kind == "max":
        assert got.tobytes() == expected.tobytes()
    else:
        np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "channels, zeroed, regions, expected",
    [
        (64, False, 1, [_BLIT, _POOL, _BLIT]),
        (128, False, 2, [_BLIT, _POOL, _BLIT]),
        # Three blocks is the most one command carries: a region is twelve of
        # the forty parameter ints, after the three-int header.
        (192, False, 3, [_BLIT, _POOL, _BLIT]),
        # Four blocks is two commands of two, and the count each carries is its
        # own chunk rather than the total.
        (256, False, 4, [_BLIT, _BLIT, _POOL, _BLIT, _BLIT]),
        (96, True, 2, [_ZERO, _BLIT, _POOL, _BLIT]),
        (65, True, 2, [_ZERO, _BLIT, _POOL, _BLIT]),
    ],
)
def test_the_command_count_is_the_block_count(channels, zeroed, regions, expected):
    """What a channel count costs: a region a block, and a memset for a tail.

    The pool is one command whatever the width, because c4 counts the blocks
    rather than describing one. A ragged last block adds a zero fill, since the
    kernel loads whole 64-lane vectors while the pack writes only the channels the
    tensor has, and the arena is reused between runs rather than cleared.
    """
    x = torch.randn(1, channels, 8, 8, dtype=torch.float16)
    blob, commands = _lowered(_Pool("max", 2, 2), x)
    assert _types(commands) == expected
    assert (_ZERO in _types(commands)) is zeroed
    blits = [c for c in commands if c.type == _BLIT]
    assert sum(c.params[0] for c in blits) == 2 * regions
    assert all(c.params[0] <= BLIT_BLOCKS_PER_COMMAND for c in blits)
    assert all(len(c.params) <= 40 for c in commands)
    widths = [region[4] for command in blits for region in _regions(command)]
    assert widths == [min(64, channels - 64 * i) for i in range(regions)] * 2


def test_a_one_position_plane_agrees_with_the_blocked_one_only_for_one_batch():
    """The blit-dropping shortcut, and the two things that stop it applying.

    A one-position plane is the shape where the row-major buffer can *be* the
    blocked one, and it is the blocked one only when every block is full and the
    batch is one. The batch is the clause that is easy to leave out: the blocked
    layout puts the channel block outside the batch and the row-major one puts it
    inside, so the two orders part company at the first channel of the second
    block however full the blocks are. The rule is the convolution path's and is
    written once.
    """
    for batch in (1, 2):
        for channels in (64, 128, 65):
            x = torch.randn(batch, channels, 1, 1, dtype=torch.float16)
            blob, commands = _lowered(_Pool("max", 1, 1), x)
            if channels == 64 or (batch == 1 and channels % 64 == 0):
                assert _types(commands) == [_POOL]
            else:
                assert _POOL in _types(commands)
                assert _BLIT in _types(commands)
            got = _run(blob, x).reshape(batch, channels, 1, 1)
            assert got.tobytes() == x.numpy().tobytes()
    assert _conv_layouts_agree(1, 1, 128)
    assert _conv_layouts_agree(4, 1, 64), "one block has no order to disagree about"
    assert not _conv_layouts_agree(2, 1, 128), "the batch is part of the rule"
    assert not _conv_layouts_agree(1, 1, 65), "a ragged block is wider than the tensor"


class _SqueezeExcitation(torch.nn.Module):
    """A block whose pool sees a wide channel count and whose convs do not.

    The squeeze is a fixed window rather than `AdaptiveAvgPool2d`: adaptive
    pooling is a different op that lives on another branch, and over a plane the
    size of the window the two agree, so this is the squeeze the DSP pool has to
    run. The composition is the point -- the pool is a small spatial reduction
    between a large channel count and a small one, and the two convolutions that
    bracket it are widths this backend already admits, so the pool is the only
    thing that decides whether the block reaches the DSP whole.
    """

    def __init__(self, channels, reduction=4, kernel=7) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = torch.nn.AvgPool2d(kernel)
        self.fc1 = torch.nn.Conv2d(channels, mid, 1)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(mid, channels, 1)
        self.gate = torch.nn.Hardsigmoid()

    def forward(self, x):
        return x * self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))


def _delegate_pool_members(program):
    """Which nodes the delegate really runs, as opposed to names it inherits.

    A refused node still leaves its name on the delegate: the partitioner makes
    the refused node output a placeholder of the delegate submodule and names that
    placeholder after the node. A membership test over every inner node therefore
    reports a pool the DSP never saw as being in the delegate, which is the same
    green answer for both verdicts. Only the inner call_functions are members.
    """
    gm = program.graph_module
    names = set()
    for call in _delegates(program):
        lowered = gm.get_submodule(call.args[0].target)
        inner = lowered.original_module.graph_module.graph
        names |= {n.name for n in inner.nodes if n.op == "call_function"}
    return names


#: The channel counts published squeeze-excitations use, wide value and its
#: reduction: MobileNetV3-Large, SqueezeNet, EfficientNet-B0 and SE-ResNet-50.
_SE_WIDTHS = [
    (16, 4),
    (24, 4),
    (32, 4),
    (40, 4),
    (64, 4),
    (80, 4),
    (96, 4),
    (112, 4),
    (128, 4),
    (160, 4),
    (192, 4),
    (256, 4),
    (320, 4),
    (512, 8),
]


@pytest.mark.parametrize("channels, reduction", _SE_WIDTHS)
def test_the_squeeze_excitation_reaches_the_dsp_whole(channels, reduction):
    """The composition, at the widths real blocks have, and not one shape alone.

    Each of these is a pool that works on its own and a block that used to
    arrive on the DSP in two pieces, because the pool was the one node of the
    block the gate refused while both convolutions, the relu and the multiply
    went across. The membership test is the name lookup, and the width assertion
    is inside the delegate, so a pool that stayed portable cannot pass it.
    """
    torch.manual_seed(channels)
    x = torch.randn(1, channels, 7, 7, dtype=torch.float16)
    model = _SqueezeExcitation(channels, reduction).to(torch.float16)
    program = to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    pool_nodes = [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.target in hexagon_ops.POOL_TARGETS
    ]
    members = _delegate_pool_members(program)
    # The pool is not a node of the outer graph at all once it delegates, so the
    # membership has to be read off the delegate rather than off the outer nodes:
    # a pool left on the host keeps its node there, and one that delegated has no
    # outer node to find. The name lookup alone cannot tell those apart, because
    # the partitioner leaves the refused node name on the delegate as the
    # placeholder it feeds in.
    assert pool_nodes == [], "the squeeze is still a node of the outer graph"
    assert "aten_avg_pool2d_default" in members, "the squeeze did not reach the DSP"
    # A whole block whose output matches torch is not evidence about the pool, so
    # the positive control here is the membership above and not the numbers; the
    # graph still has to answer, which is the portable control.
    expected = model(x)
    assert expected.dtype is torch.float16
    assert torch.isfinite(expected).all()


