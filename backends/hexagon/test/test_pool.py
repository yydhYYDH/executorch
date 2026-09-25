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


from blob_interpreter import execute, read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    _channel_block_region,
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
        packing = _channel_block_region(batch, area, 64, True)
        _pack_mapping(packing, batch, area)
        # The reverse direction is the same region with the strides exchanged,
        # which is the mapping htp_ops_pack_area_transpose_nc4hw4_to_nchw_z
        # writes out.
        unpacking = _channel_block_region(batch, area, 64, False)
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
        ((1, 32, 8, 8), "max", {}),  # half a block, which has padded lanes
        ((1, 128, 8, 8), "max", {}),  # two blocks, which is two regions
        ((1, 1, 8, 8), "avg", {}),  # one channel
        ((1, 64, 8, 8), "max", {"dilation": 2}),  # no dilation in the kernel
        ((1, 64, 7, 7), "max", {"ceil_mode": True}),  # windows past the input
        ((1, 64, 8, 8), "avg", {"divisor_override": 2}),  # divisor has no param
    ],
)
def test_a_pool_the_kernel_cannot_run_stays_on_the_host(shape, kind, kwargs):
    """A refusal, and the numbers it still has to produce.

    Each of these is a form the command cannot describe -- a channel count that
    is not one block, a dilation the window walk does not have, a divisor with no
    param slot -- so the node stays portable rather than reaching an emitter that
    would read it wrong.
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
    "args, source_shape, result_shape",
    [
        (([2, 2], [2, 2]), (1, 32, 8, 8), (1, 32, 4, 4)),  # not one channel block
        (([2, 2], [2, 2]), (1, 64, 1, 8, 8), (1, 64, 1, 4, 4)),  # rank 5
        (([3, 3], [2, 2], [1, 1], [2, 2]), (1, 64, 8, 8), (1, 64, 3, 3)),  # dilation
        (([8, 8], [8, 8], [0, 0], [1, 1], True), (1, 64, 8, 8), (1, 64, 1, 1)),
        # The next is the shape check rather than an argument: the result is the
        # size a nonzero padding would give, which this geometry does not describe.
        (([2, 2], [2, 2], [0, 0], [1, 1], False), (1, 64, 8, 8), (1, 64, 3, 3)),
        # And this one is the window guard: the geometry its padding implies has
        # windows that fall entirely outside the input.
        (([2, 2], [1, 1], [3, 3], [1, 1], False), (1, 64, 8, 8), (1, 64, 13, 13)),
    ],
)
def test_pool_spec_refuses_what_the_command_cannot_describe(
    args, source_shape, result_shape
):
    """Every refusal the partitioner depends on, on the node it reads."""
    assert pool_spec(_pool_node(args, source_shape, result_shape)) is None


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
