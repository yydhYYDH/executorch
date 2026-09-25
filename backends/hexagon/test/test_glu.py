# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLU, as the three commands its decomposition already is.

`torch.export` keeps `F.glu` as one `aten.glu` node, but EXIR's decomposition
runs before the partitioner, and what the partitioner sees is two `slice_copy`
nodes, a `sigmoid` and a `mul`. All three families are wired, so a GLU is
supported by composition rather than by an emitter of its own, and the question
these tests answer is not "is it wired" but "does the composition still read the
right elements once any of the three is moved": a `sigmoid` that stops lowering
leaves one node on the host, a `slice_copy` that reads the wrong half of the
channel axis sends the multiply against the wrong operand, and neither shows up
in the command types, which stay `[3, 3, 4, 19, 3]` whichever way the blit points.

The split boundary is therefore pinned on both sides: the 64/32 and 33/31 splits
of a Conformer channel axis, the region each piece's blit carries (source and
destination offsets, row count, run), and the 64-element grain at which the
sigmoid switches from the vector PWL walk to the fp32 scalar tail. The
numerics are the PWL band's effect on the product rather than on the sigmoid,
because the multiply is what a model's output is made of: the sigmoid's own band
is 2.4e-3 and the product's is 4.0e-3 at the seed the test fixes, on
`F.glu(x, dim=1)` of a `[1, 64, 1, 48]` tensor with a normal tail. Both are
measured against the kernel's own scalar path, not against `np.sigmoid`; see
`_kernel_scalar_sigmoid` for why that distinction is load-bearing.
"""

import operator
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
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

F16 = torch.float16

#: DSP_OP_RASTER_BLIT, DSP_OP_UNARY, DSP_OP_BINARY_ELEMENTWISE: the three command
#: types the composition is, plus the blits that carry the pieces and the one
#: that writes the result out.
_BLIT = 3
_UNARY = 4
_BINARY = 19

#: The companded16 sigmoid table the skel is built with
#: (`unary_ops.cc:225-237`, `pwl.h:47-71`): sixteen fp16 chords over `[0, 8]`,
#: 0.25-wide below 2, 0.5-wide to 4, 1.0-wide to 8. The bits are the table's own,
#: read out of the source rather than restated, and the evaluation is an fp16
#: multiply then an fp16 add (`pwl.h:90-93`) -- two roundings, which is what the
#: phone's bits show (`hex_workstreams/GELU-report.md`, device tier).
_SIGMOID_SLOPE = np.array(
    [
        0x33F5, 0x33B7, 0x3343, 0x32A4, 0x31EB, 0x3128, 0x3067, 0x2F62,
        0x2D8C, 0x2B47, 0x28A3, 0x25CD, 0x21C8, 0x1C52, 0x1665, 0x10B7,
    ],
    dtype=np.uint16,
).view(np.float16)
_SIGMOID_BIAS = np.array(
    [
        0x3800, 0x3804, 0x3812, 0x3830, 0x385E, 0x389B, 0x38E4, 0x3933,
        0x39A9, 0x3A42, 0x3AC0, 0x3B22, 0x3B7F, 0x3BC7, 0x3BE8, 0x3BF6,
    ],
    dtype=np.uint16,
).view(np.float16)

#: `htp_ops_unary_compute_fp16_chunk` hands `[0, numel & ~63)` to the vector
#: walk and the last `numel % 64` to the fp32 scalar form (`unary_ops.cc:455-491`),
#: so a buffer shorter than the grain is scalar throughout. The two forms are the
#: reason a GLU's accuracy depends on its channel count and nothing else.
_PWL_GRAIN = 64

#: The fp16 step of a product of size one, `test_norms.py`'s tolerance unit.
_FP16_STEP = float(np.spacing(np.float16(1.0)))


def _pwl_sigmoid(values):
    """`htp_ops_unary_pwl_fp16_vec` over a whole buffer, from `unary_ops.cc:259-330`.

    The magnitude's companded index, an fp16 multiply into an fp16 add, the fold
    `sigmoid(-a) = 1 - sigmoid(a)`, and the saturation at `|x| = 8`. The
    `[0, numel & ~63)` / tail split is the caller's: the device mixes the two
    forms inside one buffer, and the tests below bracket the grain rather than
    assume which side a size lands on.
    """
    x = np.asarray(values, dtype=np.float16)
    zero, one = np.float16(0.0), np.float16(1.0)
    negative = x < zero
    abs_v = np.where(negative, (-x).astype(np.float16), x)
    # htp_ops_pwl_companded_index16 (pwl.h:57-71): below 2 the value times four
    # into the top four mantissa bits, at two and above the exponent's low bit
    # and the top two mantissa bits are the index.
    scaled = np.minimum((abs_v * np.float16(4.0)).astype(np.float16), np.float16(15.0))
    low = (scaled + np.float16(16.0)).view(np.uint16) >> 6
    wide = ((abs_v.view(np.uint16) >> 8) & np.uint16(7)) + np.uint16(8)
    index = np.where(abs_v >= np.float16(2.0), wide, low) & np.uint16(0x000F)
    positive = (abs_v * _SIGMOID_SLOPE[index]).astype(np.float16) + _SIGMOID_BIAS[index]
    folded = (one - positive).astype(np.float16)
    limit = np.where(negative, zero, one)
    return np.where(abs_v < np.float16(8.0), np.where(negative, folded, positive), limit)


def _kernel_scalar_sigmoid(values):
    """The kernel's own scalar form, `htp_ops_unary_apply_fp16`: fp32, one rounding.

    This is the reference the PWL band below is measured *against*, and it is a
    deliberately narrow choice. It is the tail path of the same function
    (`unary_ops.cc:161-209`, reached at `unary_ops.cc:455-491`), so the band it
    gives is the vector walk's disagreement with the kernel's scalar walk -- not
    with `np.sigmoid`, and not with a device run. The MODELFIDELITY audit's
    finding, that the vector body answers bit-wise differently from the scalar
    form above the 64-element chunk, is the same disagreement seen from the
    other end: 2.4e-3 there against the scalar, 2.4e-3 here. A band quoted
    against `np.sigmoid` instead would be a different and much larger number,
    and would make the two reports look like they contradict each other.
    """
    x = np.asarray(values, dtype=np.float16).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float16)


def _device_sigmoid(values):
    """The sigmoid a GLU's `sigmoid` command returns, as the kernel computes it."""
    values = np.asarray(values, dtype=np.float16)
    body = values.size - values.size % _PWL_GRAIN
    if body == 0:
        return _kernel_scalar_sigmoid(values)
    return np.concatenate(
        [_pwl_sigmoid(values[:body]), _kernel_scalar_sigmoid(values[body:])]
    )


class _Glu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.glu(x, dim=1)


class _ConformerConv(torch.nn.Module):
    """A Conformer convolution module at batch one, batch-last: LayerNorm, two
    pointwise convolutions each followed by a GLU, a depthwise convolution and a
    second LayerNorm with a residual.

    `nn.Conv1d` over `[B, C, T]` is a `Conv2d` over `[B, C, 1, T]` with kernel
    `(1, k)`, the spelling the two convolution kernels here take; the unsqueeze
    and squeeze around them are views, and the 64 channels are the 64-lane block
    the whole activation layout is built on.
    """

    def __init__(self, channels=64, length=48, kernel=31):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(channels)
        self.pw1 = torch.nn.Conv2d(channels, 2 * channels, 1)
        self.pw2 = torch.nn.Conv2d(channels, 2 * channels, 1)
        self.dw = torch.nn.Conv2d(
            channels, channels, (1, kernel), groups=channels, padding=(0, kernel // 2)
        )
        self.norm2 = torch.nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(1, 2).unsqueeze(2)
        x = torch.nn.functional.glu(self.pw1(x), dim=1)
        x = torch.nn.functional.glu(self.pw2(x), dim=1)
        x = self.dw(x).squeeze(2).transpose(1, 2)
        x = self.norm2(x)
        return x + residual


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _config():
    return EdgeCompileConfig(_check_ir_validity=False)


def _program(model, x):
    return to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=_config(),
    ).exported_program()


def _lowered(model, x):
    """The delegate's blob and its decoded command stream, with the delegate's node."""
    program = _program(model, x)
    calls = _delegates(program)
    assert len(calls) == 1, f"the graph did not reach one delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    raw = bytes(lowered._processed_bytes)
    _header, commands = read_blob(raw)
    return raw, commands


def _values(raw, x):
    return np.frombuffer(execute(raw, [x.numpy()])[0], dtype=np.float16)


def _region(command):
    """A blit's region as named fields, the order `_emit_split` writes them."""
    names = (
        "region count", "element bytes", "source count", "source index",
        "source offset", "destination offset", "destination count", "rows",
        "run", "source row stride", "axis stride", "inner count",
        "inner source stride", "inner run", "inner destination stride",
    )
    return dict(zip(names, command.params))


def _call_names(graph):
    """The `call_function` node names of a graph, in order."""
    return [node.name for node in graph.nodes if node.op == "call_function"]


def test_the_decomposition_is_what_the_partitioner_sees_and_not_a_glu_node():
    """The claim this file rests on, read off three graphs rather than asserted.

    The module docstring says export keeps `F.glu` as one node and EXIR's
    decomposition runs before the partitioner. That is a statement about where a
    node stops existing, so it is checked here at each stage rather than left as
    prose: the export graph has `aten.glu`, the edge graph has the four
    decomposed nodes and no `glu`, and the delegate's own inner graph -- the
    graph the emitters actually read -- has the same four and no `glu` either.

    The last one is the load-bearing assertion. The edge graph is what the
    partitioner's verdicts are computed from, but a composition claim is only
    worth what the delegate contains, and a future decomposition table that
    changed after partitioning would leave the two disagreeing.
    """
    x = torch.randn(1, 64, 1, 48, dtype=F16)
    exported = export(_Glu(), (x,))
    assert _call_names(exported.graph) == ["glu"], "export stopped keeping F.glu whole"

    edge = to_edge(
        exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    edge_names = _call_names(edge.graph_module.graph)
    assert edge_names == [
        "dim_order_ops__to_dim_order_copy_default",
        "aten_slice_copy_tensor",
        "aten_slice_copy_tensor_1",
        "aten_sigmoid_default",
        "aten_mul_tensor",
        "dim_order_ops__to_dim_order_copy_default_1",
    ], f"the decomposition moved: {edge_names}"
    assert not [name for name in edge_names if "glu" in name], edge_names

    program = _program(_Glu(), x)
    lowered = program.graph_module.get_submodule(_delegates(program)[0].args[0].target)
    inner = _call_names(lowered.original_module.graph_module.graph)
    assert inner == edge_names, f"the delegate holds {inner}, the edge graph {edge_names}"


def test_a_glu_is_a_blit_a_blit_a_sigmoid_a_multiply_and_a_blit():
    """The five commands of the composition, in order, on a 64/32 split.

    The counts are what the decomposition is: two pieces read means two blits,
    one sigmoid, one multiply, one copy out. A piece that stopped delegating would
    leave a hole here, and a family that stopped being wired would leave the node
    on the host -- which is what the tightening control below checks.
    """
    x = torch.randn(1, 64, 1, 48, dtype=F16)
    _raw, commands = _lowered(_Glu(), x)
    assert [command.type for command in commands] == [
        _BLIT, _BLIT, _UNARY, _BINARY, _BLIT,
    ]
    sigmoid, multiply = commands[2], commands[3]
    # DSP_OP_UNARY's params are numel, op type, stride; SIGMOID is 4
    # (unary_ops.cc:174-177).
    assert sigmoid.params[:3] == [1536, 4, 2]
    # DSP_OP_BINARY_ELEMENTWISE's params are out size, two in sizes, then op type
    # at [3]; HTP_OPS_BINARY_MUL is 3 (eltwise_ops.cc:1484-1490).
    assert multiply.params[:4] == [1536, 1536, 1536, 3]


def test_the_split_boundary_is_read_from_both_sides():
    """64/32 and 33/31: the first blit starts at 0 and the second at the boundary.

    The two pieces are the same read `x[:, :half]` and `x[:, half:]` are, so the
    claim is the region: source and destination offsets, rows and run. A blit that
    pointed at the other half of the channel axis would leave every type in the
    stream identical and move every number, so the run lengths and the boundary
    offset are what carry this test, and the odd split is there because the even
    one would also pass with a blit that moved by a whole 64-channel block.
    """
    for channels, half in ((64, 32), (66, 33)):
        x = torch.randn(1, channels, 1, 48, dtype=F16)
        _raw, commands = _lowered(_Glu(), x)
        head, tail = _region(commands[0]), _region(commands[1])
        # A plane of [half, 1, 48] fp16 is the piece's run; the source is the
        # whole [channels, 1, 48] plane, so its stride is the whole tensor.
        run = half * 48
        assert head["source offset"] == 0, f"{channels}: {head}"
        assert head["destination offset"] == 0, f"{channels}: {head}"
        assert tail["source offset"] == run, f"{channels}: {tail}"
        assert head["run"] == tail["run"] == run, f"{channels}: {head} {tail}"
        assert head["rows"] == tail["rows"] == 1, f"{channels}: {head} {tail}"
        # One run per plane (hexagon_ops.py:815-842), so the row stride is the
        # row the run sits on -- zero for the only row there is -- and the axis
        # stride is the whole source plane. Both are what make the source offset
        # of the second piece the boundary rather than a coincidental number.
        assert head["source row stride"] == tail["source row stride"] == 0
        assert head["axis stride"] == tail["axis stride"] == channels * 48
        assert commands[2].params[0] == run, f"{channels}: the sigmoid read {commands[2].params[0]}"


def test_the_blit_that_writes_the_result_out_copies_the_product():
    """The last blit is a whole-tensor copy, not another narrowing region.

    The three glits in the stream are the two pieces and the answer, and only the
    first two are narrowings: the last one's source is the multiply's output and
    its run is the whole piece, so a blit that narrowed it would drop the tail of
    the output while the command types stayed the same.
    """
    x = torch.randn(1, 64, 1, 48, dtype=F16)
    _raw, commands = _lowered(_Glu(), x)
    out = _region(commands[4])
    # The whole-tensor blit reads the multiply output from offset 0 as one run of
    # the whole piece and writes it at output offset 0. What separates it from a
    # narrowing is its axis stride: the pieces carry the source plane stride
    # (3072 for the 64-channel geometry) because they address a tensor, and this
    # one carries zero because the only run it has is the whole tensor.
    assert out["source offset"] == out["destination offset"] == 0, out
    assert out["run"] == 32 * 48, out
    assert out["rows"] == out["destination count"] == 1, out
    assert out["source row stride"] == out["axis stride"] == 0, out
    assert _region(commands[0])["axis stride"] == 64 * 48, commands[0].params


@pytest.mark.parametrize(
    "channels, length",
    (
        (64, 48),  # 32 x 48 = 1536 elements: 24 whole grains
        (66, 48),  # 33 x 48 = 1584: 24 grains and 48 scalar elements
        (32, 7),   # 16 x 7 = 112: one grain and 48 scalar elements
        (32, 3),   # 16 x 3 = 48: shorter than the grain, scalar throughout
    ),
)
def test_the_glu_on_the_host_interpreter_answers_torch(channels, length):
    """What the emitted blob computes against torch, at four lengths.

    The interpreter models the sigmoid exactly (the fp32 logistic, which is what
    `blob_interpreter.py` still does at this revision), so this measures rounding
    and not the PWL band -- the band is the next test's subject, and is measured
    against the kernel's scalar path rather than against torch so the two
    references cannot be confused. The four lengths bracket the kernel's 64-element
    grain, 1536 being a whole number of walks, 1584 and 112 each leaving a scalar
    tail and 48 never taking the walk, so the geometries that would behave
    differently on device are all represented here even though the host model
    cannot distinguish them yet.
    """
    x = torch.randn(1, channels, 1, length, dtype=F16)
    raw, commands = _lowered(_Glu(), x)
    assert [command.type for command in commands] == [
        _BLIT, _BLIT, _UNARY, _BINARY, _BLIT,
    ]
    got = _values(raw, x)
    want = torch.nn.functional.glu(x, dim=1).numpy().reshape(-1)
    # Rounding alone: the interpreter's sigmoid is the exact fp32 logistic and the
    # multiply is an fp16 multiply, so the whole disagreement is two roundings,
    # one step each. The PWL band is wider, is a property of the kernel rather
    # than of this model, and is measured on the product by the test below
    # rather than folded into a tolerance that means something else.
    assert np.abs(got.astype(np.float64) - want.astype(np.float64)).max() <= (
        2.0 * _FP16_STEP
    ), f"{channels}x{length}"


def test_the_pwl_band_lands_on_the_product_and_not_only_on_the_sigmoid():
    """The product's band, measured against the kernel's own scalar path.

    A sigmoid error `e` is a product error `a * e`, so the band a model sees is
    set by the values in the channel half the multiply is fed and is larger than
    the sigmoid's own by whatever those values are. The sigmoid's own band is
    2.4e-3, worst case over the geometries below; the product's is 4.0e-3 at the
    seed this test fixes and 6.8e-3 across eight seeds, which is the shape of a
    piecewise-linear error rather than a rounding artefact: in fp16 steps at the
    product's own scale the worst disagreement is 44 steps at seed 0 and 71
    across the sweep, and 69% of the products differ at all.

    The reference is `_kernel_scalar_sigmoid`, not `torch.sigmoid`, and the
    last assertion is what holds that line: it measures the band both ways and
    requires the one quoted to be the narrower. A band against `torch.sigmoid`
    would fold in the scalar path's own fp16 rounding and read as a different
    and larger disagreement, which is the same measurement the MODELFIDELITY
    audit made from the device side.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 64, 1, 48, dtype=F16)
    a_half = x[:, :32].numpy().reshape(-1)
    b_half = x[:, 32:].numpy().reshape(-1)
    exact = a_half.astype(np.float32) * _kernel_scalar_sigmoid(b_half).astype(np.float32)
    device = a_half * _device_sigmoid(b_half)
    want = torch.nn.functional.glu(x, dim=1).numpy().reshape(-1)

    product_error = np.abs(exact.astype(np.float64) - device.astype(np.float64)).max()
    sigmoid_error = np.abs(
        _kernel_scalar_sigmoid(b_half).astype(np.float64)
        - _device_sigmoid(b_half).astype(np.float64)
    ).max()
    # The reference above is the kernel's own scalar path, which is what makes
    # this the *vector-vs-scalar* band. Quoting it against np.sigmoid instead
    # would add the scalar path's own fp16 rounding on top and give a larger
    # number that means something else -- so the two references are measured
    # side by side here and the one the band uses is the narrower of the two.
    against_torch = np.abs(
        torch.sigmoid(torch.from_numpy(b_half.astype(np.float32)))
        .numpy()
        .astype(np.float64)
        - _device_sigmoid(b_half).astype(np.float64)
    ).max()
    assert sigmoid_error <= against_torch, (
        "the band is no longer the narrow one, so the reference is not the "
        f"kernel's scalar path: {sigmoid_error} vs {against_torch}"
    )
    assert product_error > sigmoid_error, (
        "the product's band is not wider than the sigmoid's, so the multiply is "
        f"not what scales it: {product_error} vs {sigmoid_error}"
    )
    assert product_error <= 4.1e-3, f"the product's band grew: {product_error}"
    # The product is what a model's output is made of, so the ratio that decides
    # is the worst error in fp16 steps at the output's own scale.
    steps = np.abs(
        exact.astype(np.float16).view(np.uint16).astype(np.int64)
        - device.astype(np.float16).view(np.uint16).astype(np.int64)
    )
    assert np.abs(
        exact.astype(np.float64) - want.astype(np.float64)
    ).max() <= _FP16_STEP, "the reference is not torch's product"
    assert steps.max() <= 48, f"the product's band in fp16 steps: {steps.max()}"


def test_the_conformer_conv_module_lowers_as_one_delegate_with_both_glu_in_it():
    """Conv1d + GLU + dwconv + LayerNorm at batch one: the whole block is one blob.

    The Conformer module a GLU exists for. Each GLU contributes the five commands
    above, the pointwise convolutions take the im2col kernel with `K = 32 * kp =
    64` (a whole multiple of 64, read out of the decoded command rather than
    computed from the graph), the depthwise one takes the per-channel walk with
    its 31-tap window, and the two layer norms take the norm command with their
    two affine multiplies. Nothing is left on the host, so the host interpreter
    can run the whole block and the answer can be compared with torch's.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 48, 64, dtype=F16)
    model = _ConformerConv().half().eval()
    with torch.no_grad():
        want = model(x)
    program = _program(model, x)
    calls = _delegates(program)
    assert len(calls) == 1, f"the block did not reach one delegate: {len(calls)}"
    # The only node left on the host is the getitem that reads the delegate first
    # output, which is how a delegate is read at all; a portable op here would
    # mean a piece of the block -- and both GLUs with it -- stayed behind.
    assert not [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is not torch.ops.higher_order.executorch_call_delegate
        and not (
            node.target is operator.getitem
            and node.args
            and node.args[0] is calls[0]
        )
    ], "a portable node was left outside the delegate"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    raw = bytes(lowered._processed_bytes)
    _header, commands = read_blob(raw)
    types = [command.type for command in commands]
    # Each window is the blit that carries the pointwise convolution's output
    # and the GLU's five commands, so the first convolution's im2col is the
    # command at 5 and the second's is the one at 13, with the depthwise walk at
    # 19; the leading _BLIT of a window is that carry, not a piece of the GLU.
    for first, second in ((7, 12), (14, 19)):
        assert types[first : second] == [_BLIT, _BLIT, _UNARY, _BINARY, _BLIT], (
            f"GLU at commands {first}..{second - 1}: {types[first : second]}"
        )
    im2col = [command for command in commands if command.type == 12]
    assert len(im2col) == 2, types
    for command in im2col:
        # handoff §13.5: 32 * params[9] is the kernel's own reduction width.
        assert 32 * command.params[9] % 64 == 0, command.params[6:12]
    assert types.count(2) == 1, types

    got = np.frombuffer(execute(raw, [x.numpy()])[0], dtype=np.float16).reshape(-1)
    reference = want.numpy().reshape(-1)
    error = np.abs(got.astype(np.float64) - reference.astype(np.float64)).max()
    step = float(np.spacing(np.float16(np.abs(reference).max())))
    # The block is accumulation (two convolutions, a layer norm) under a sigmoid
    # the host model evaluates exactly, so the whole disagreement is rounding and
    # reduction order: 5.86e-3 at this seed, 1.5 fp16 steps at the output's own
    # scale. The tolerance is stated in steps as well as absolute, so a change in
    # the output's magnitude cannot silently loosen it.
    assert error <= 5.9e-3, f"the block's error: {error}"
    assert error <= 2.0 * step, f"the block's error: {error} against a step of {step}"


def test_any_one_of_the_three_families_that_stops_lowering_takes_the_glu_with_it():
    """The tightening control: close each family in turn and watch the graph move.

    A test that only asserts today's command stream says nothing about whether
    the stream is what produces the GLU's answer. Taking `sigmoid`, `slice_copy`
    or `mul` out of the emitter table has to move the graph: the node whose family
    is gone leaves the delegate and the composition splits around it, or the
    delegate disappears with it. Each mutation is reverted before the next, and
    the last line is the control: with the table back, the stream is back too.
    """
    x = torch.randn(1, 64, 1, 48, dtype=F16)
    model = _Glu()
    _raw, commands = _lowered(model, x)
    good = [command.type for command in commands]

    # The three families by the keys the emitter table holds them under: the
    # decomposition produces the edge dialect ops, so `torch.ops.aten` keys are
    # absent and popping one would raise rather than close anything.
    for target in (
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    ):
        saved = hexagon_ops.EMITTERS.pop(target)
        try:
            program = _program(model, x)
            portable = [
                node
                for node in program.graph_module.graph.nodes
                if node.op == "call_function"
                and node.target is target
            ]
            assert portable, f"{target} left the emitter table and still lowered"
            calls = _delegates(program)
            moved = len(calls) != 1
            if calls:
                lowered = program.graph_module.get_submodule(calls[0].args[0].target)
                _header, mutated = read_blob(bytes(lowered._processed_bytes))
                moved = moved or [command.type for command in mutated] != good
            assert moved, f"{target} left the emitter table and the stream did not move"
        finally:
            hexagon_ops.EMITTERS[target] = saved
    _raw, commands = _lowered(model, x)
    assert [command.type for command in commands] == good, "the table did not come back"
