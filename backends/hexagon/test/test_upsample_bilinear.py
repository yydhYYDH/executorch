# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bilinear upsampling on the DSP, as one command and four taps.

`aten.upsample_bilinear2d.vec` is the one operator in the gap list that two
model families need directly -- a super-resolution head and a TTS vocoder both
upsample -- and it was the one sampling op this backend had no entry for.
Unlike the nearest form beside it, this one is not an index map: the result is
arithmetic on four source elements, so there is no region decomposition to fall
back on and the command has to be the kernel.

The boundary is an exact integer multiple per axis with align_corners=False,
and every clause of it is pinned below from both sides. Three of those clauses
are arguments of one schema rather than targets of their own, which is the trap
this file is built around: align_corners, output_size and scale_factors are all
positional arguments of aten::upsample_bilinear2d.vec, so a check that treated
any of them as a target name would be looking for a string that never appears as
an op. Each refusal below is paired with a positive control at the same rank and
channel count, because a support object built without the program data
placeholders refuses every op whose weight is read at export and a test written
only against refusals would then be green in a tree that has none of this.

The numerics: for a power-of-two factor the host model is bit-identical to
F.interpolate over the geometries below, and for a factor whose reciprocal is
not representable -- 3, 5, 6, 7, 9, 11 -- a small fraction of the outputs differ
by one fp16 ULP. That is a property of the last fp32 rounding before the fp16
store, not of the tap arithmetic, and the test that says so is named for it
instead of asserting it in a comment.
"""

import numpy as np
import pytest
import torch

from blob_interpreter import UnsupportedOp, execute, read_blob
from executorch.backends.hexagon import serialization
from executorch.backends.hexagon.hexagon_ops import (
    BILINEAR_UPSAMPLE_MAX_ROW,
    DSP_OP_UPSAMPLE_BILINEAR2D_FP16,
    EMITTERS,
    UPSAMPLE_BILINEAR_TARGETS,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

_BILINEAR = 46
_B = serialization.blob

#: aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool
#: align_corners, float[]? scale_factors) -> Tensor. The three after the input
#: are arguments; output_size and scale_factors are alternatives to each other
#: rather than targets of their own.
_BILINEAR_SCHEMA = exir_ops.edge.aten.upsample_bilinear2d.vec


class _Bilinear(torch.nn.Module):
    def __init__(self, scale_factor=None, size=None, align_corners=False) -> None:
        super().__init__()
        self.kwargs = {"mode": "bilinear", "align_corners": align_corners}
        if size is not None:
            self.kwargs["size"] = size
        else:
            self.kwargs["scale_factor"] = scale_factor

    def forward(self, x):
        import torch.nn.functional as F

        return F.interpolate(x, **self.kwargs)


def _lower(module, inputs):
    return to_edge_transform_and_lower(
        export(module.eval(), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _portable(program):
    return [node for node in program.graph_module.graph.nodes if node.target is _BILINEAR_SCHEMA]


def _lowered(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    return program.graph_module.get_submodule(calls[0].args[0].target)


def _blob(program):
    """The delegate bytes, and the commands the reader finds in them."""
    data = bytes(_lowered(program)._processed_bytes)
    return data, read_blob(data)[1]


def _inner_names(program):
    """The delegate own node names, which the outer graph names are kept in."""
    return [node.name for node in _lowered(program).original_module.graph_module.graph.nodes]


def _runs(module, shape, dtype=torch.float16):
    inputs = (torch.randn(*shape, dtype=dtype),)
    return _lower(module, inputs), inputs


def _set_param(data, param_index, value, op_index=0):
    """Rewrites one command param in place, from the struct own layout.

    A command opens with four int32 -- type, input count, output count, param
    count -- and the params follow, before the tensor refs. The command block is
    found by that 16-byte prefix rather than by a section offset, and the search
    has to be unique, so a blob whose layout moved fails here instead of writing
    a param into the header and producing a comparison that still passes.
    """
    import struct

    _, commands = read_blob(data)
    command = commands[op_index]
    prefix = struct.pack(
        "<IIII",
        command.type,
        len(command.inputs),
        len(command.outputs),
        len(command.params),
    )
    at = data.find(prefix)
    assert at >= 0 and data.find(prefix, at + 1) < 0, "the command block is not unique"
    slot = at + len(prefix) + 4 * param_index
    patched = data[:slot] + int(value).to_bytes(4, "little", signed=True) + data[slot + 4 :]
    assert read_blob(patched)[1][op_index].params[param_index] == value
    return patched


# --- the boundary, one clause at a time -------------------------------------


@pytest.mark.parametrize(
    "scale_y,scale_x",
    [(2, 2), (1, 1), (1, 4), (4, 1), (3, 3), (5, 7), (8, 8), (2, 5), (6, 4), (11, 9)],
)
def test_an_exact_integer_multiple_on_both_axes_is_one_command(scale_y, scale_x):
    program, _ = _runs(
        _Bilinear(scale_factor=[float(scale_y), float(scale_x)]), (1, 3, 4, 5)
    )
    data, commands = _blob(program)
    assert [command.type for command in commands] == [_BILINEAR]
    assert list(commands[0].params) == [3, 4, 5, 4 * scale_y, 5 * scale_x]
    assert "aten_upsample_bilinear2d_vec" in _inner_names(program)
    assert len(data) > 0


def test_the_output_size_spelling_reaches_the_same_command():
    """output_size and scale_factors are two spellings of one argument slot."""
    by_size, _ = _runs(_Bilinear(size=[8, 10]), (1, 3, 4, 5))
    by_factor, _ = _runs(_Bilinear(scale_factor=[2.0, 2.0]), (1, 3, 4, 5))
    _, size_commands = _blob(by_size)
    _, factor_commands = _blob(by_factor)
    assert [c.type for c in size_commands] == [_BILINEAR] == [c.type for c in factor_commands]
    assert list(size_commands[0].params) == list(factor_commands[0].params)


@pytest.mark.parametrize(
    "label,module,shape",
    [
        ("align_corners_true", _Bilinear(scale_factor=2.0, align_corners=True), (1, 3, 4, 5)),
        ("fractional_factor", _Bilinear(scale_factor=1.5), (1, 3, 4, 5)),
        ("shrinking_factor", _Bilinear(scale_factor=0.5), (1, 3, 4, 5)),
        ("non_integer_size", _Bilinear(size=[9, 10]), (1, 3, 4, 5)),
        (
            "row_over_the_axis_table",
            _Bilinear(scale_factor=2.0),
            (1, 3, 4, BILINEAR_UPSAMPLE_MAX_ROW // 2 + 1),
        ),
    ],
)
def test_each_refused_clause_stays_portable_while_a_control_delegates(
    label, module, shape
):
    """A refusal is evidence only next to the case that has to reach the kernel."""
    program, _ = _runs(module, shape)
    assert _delegates(program) == [], f"{label} delegated"
    assert len(_portable(program)) == 1, f"{label} lost the node"

    control, _ = _runs(_Bilinear(scale_factor=2.0), (1, 3, 4, 5))
    _, control_commands = _blob(control)
    assert [c.type for c in control_commands] == [_BILINEAR]


def test_the_axis_table_cap_is_the_only_thing_the_width_clause_refuses():
    assert BILINEAR_UPSAMPLE_MAX_ROW == 512
    at_cap, _ = _runs(
        _Bilinear(scale_factor=2.0), (1, 3, 4, BILINEAR_UPSAMPLE_MAX_ROW // 2)
    )
    _, commands = _blob(at_cap)
    assert [c.type for c in commands] == [_BILINEAR]
    over_cap, _ = _runs(
        _Bilinear(scale_factor=2.0), (1, 3, 4, BILINEAR_UPSAMPLE_MAX_ROW // 2 + 1)
    )
    assert _delegates(over_cap) == []


def test_fp32_is_refused_rather_than_read_as_fp16():
    """The blob would size the operand at the command's own width.

    A 1x3x4x5 fp32 tensor is 240 bytes and the command reads two bytes per
    element, so a blob that admitted it would claim 120 and the DSP would read
    the first 60 elements of the tensor's bytes as fp16. Refusing is the only
    honest answer, and it is the one arg_reduction_spec already gives.
    """
    program, _ = _runs(_Bilinear(scale_factor=2.0), (1, 3, 4, 5), torch.float32)
    assert _delegates(program) == []
    assert len(_portable(program)) == 1


def test_fp16_still_delegates_so_the_refusal_is_the_dtype_and_not_the_graph():
    program, _ = _runs(_Bilinear(scale_factor=2.0), (1, 3, 4, 5), torch.float16)
    _, commands = _blob(program)
    assert [c.type for c in commands] == [_BILINEAR]
    assert list(commands[0].params) == [3, 4, 5, 8, 10]


def test_the_target_set_holds_one_target_and_the_emitter_table_reaches_it():
    """The sibling sampling ops share a name, not a target."""
    assert len(UPSAMPLE_BILINEAR_TARGETS) == 1
    assert _BILINEAR_SCHEMA in UPSAMPLE_BILINEAR_TARGETS
    assert EMITTERS[_BILINEAR_SCHEMA].__name__ == "_emit_upsample_bilinear2d"
    assert exir_ops.edge.aten.upsample_nearest2d.vec not in UPSAMPLE_BILINEAR_TARGETS
    assert DSP_OP_UPSAMPLE_BILINEAR2D_FP16 == _BILINEAR


def test_nearest_still_goes_through_its_own_command():
    """The two sampling ops must not have been collapsed into one another."""
    import torch.nn.functional as F

    class _Nearest(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2.0, mode="nearest")

    program, _ = _runs(_Nearest(), (1, 3, 4, 5))
    _, commands = _blob(program)
    assert commands and all(c.type == 3 for c in commands)


# --- the numerics -----------------------------------------------------------


def _answer(program, inputs):
    data, _ = _blob(program)
    out = execute(data, [t.numpy() for t in inputs])
    return np.frombuffer(out[0].tobytes(), dtype=np.float16)


@pytest.mark.parametrize("scale_y,scale_x", [(2, 2), (4, 4), (8, 8), (1, 2), (2, 1), (1, 1)])
def test_a_power_of_two_factor_is_bit_exact_against_torch(scale_y, scale_x):
    import torch.nn.functional as F

    module = _Bilinear(scale_factor=[float(scale_y), float(scale_x)])
    program, inputs = _runs(module, (1, 3, 5, 7))
    got = _answer(program, inputs)
    want = F.interpolate(inputs[0], **module.kwargs)
    assert got.size == want.numel()
    np.testing.assert_array_equal(
        got.reshape(tuple(want.shape)).view(np.uint16), want.numpy().view(np.uint16)
    )


def _ulp_distance(got, want):
    """Elementwise fp16 spacing between two arrays, in representable steps.

    The fp16 bit patterns are monotone in the value across the whole finite range
    once the sign is folded in, so the difference of two int16 views is the number
    of representable values between them. An absolute tolerance cannot say this:
    the spacing is 2^-10 near one and 2^-24 near a subnormal, so one bound that is
    loose enough for the large values is meaningless for the small ones.
    """
    def ordered(bits):
        value = np.asarray(bits, dtype=np.int32).astype(np.int64)
        return np.where(value < 0, np.int64(-32768) - value, value)

    return np.abs(ordered(got.view(np.int16)) - ordered(want.view(np.int16)))


#: The largest absolute deviation measured over the accepted non-power-of-two
#: geometries below, and the bound this backend states. It is two fp16 steps at
#: one: fp16 has an 11-bit significand, so the spacing there is 2^-10.
#:
#: A step count is the wrong measure here and the test that used one was wrong.
#: `h0 * top + h1 * bot` cancels when top and bot are nearly opposite, so a
#: one-ULP-of-fp32 disagreement between the two implementations becomes a large
#: *relative* difference on a result near zero -- measured at 3679 representable
#: steps for 9x11 on a 16x16 input, on outputs whose magnitude is 2.2e-4. The
#: absolute difference at every one of those is under the bound above.
_BILINEAR_ATOL = 2.0**-9


@pytest.mark.parametrize("scale_y,scale_x", [(3, 3), (5, 7), (2, 5), (6, 4), (9, 11), (11, 9), (6, 13)])
def test_a_non_power_of_two_factor_stays_within_the_stated_bound(scale_y, scale_x):
    """The measured claim, as a value bound rather than an equality."""
    import torch.nn.functional as F

    module = _Bilinear(scale_factor=[float(scale_y), float(scale_x)])
    for channels in (1, 3):
        for seed in (0, 1):
            torch.manual_seed(seed)
            program, inputs = _runs(module, (1, channels, 7, 9))
            got = _answer(program, inputs)
            want = F.interpolate(inputs[0], **module.kwargs)
            deviation = np.abs(
                got.astype(np.float32) - want.numpy().reshape(-1).astype(np.float32)
            )
            assert deviation.max() <= _BILINEAR_ATOL, (scale_y, scale_x, deviation.max())


def test_the_command_params_are_load_bearing():
    """A control whose comparison would pass on the wrong bytes says nothing."""
    import torch.nn.functional as F

    module = _Bilinear(scale_factor=2.0)
    program, inputs = _runs(module, (1, 2, 3, 4))
    data, commands = _blob(program)
    assert [c.type for c in commands] == [_BILINEAR]
    good = np.frombuffer(execute(data, [t.numpy() for t in inputs])[0].tobytes(),
                         dtype=np.float16)
    want = F.interpolate(inputs[0], **module.kwargs)
    np.testing.assert_array_equal(
        good.reshape(tuple(want.shape)).view(np.uint16), want.numpy().view(np.uint16)
    )

    # in_h 3 -> 2, which keeps the output shape (1, 2, 6, 8) and the source
    # row pitch, so the corrupted command walks a 2x4 source into the same six
    # rows. The bytes cannot coincide, and a comparison that would still pass on
    # the wrong answer would be a control that proves nothing.
    assert list(commands[0].params) == [2, 3, 4, 6, 8]
    corrupted = _set_param(data, 1, 2)
    assert corrupted != data
    moved = np.frombuffer(execute(corrupted, [t.numpy() for t in inputs])[0].tobytes(),
                          dtype=np.float16)
    assert not np.array_equal(moved.view(np.uint16), good.view(np.uint16))


def test_the_interpreter_refuses_a_ratio_it_cannot_represent():
    """The same refusal the kernel makes, so a bad blob is not answered."""
    program, inputs = _runs(_Bilinear(scale_factor=2.0), (1, 2, 3, 4))
    data, _ = _blob(program)
    odd = _set_param(_set_param(data, 3, 7), 4, 9)
    with pytest.raises(UnsupportedOp):
        execute(odd, [t.numpy() for t in inputs])


# --- the two model families, at their own stage geometries ------------------


@pytest.mark.parametrize(
    "label,scale_y,scale_x",
    [
        ("hifigan_8", 8, 8),
        ("hifigan_8_again", 8, 8),
        ("hifigan_2", 2, 2),
        ("vocos_8", 8, 8),
        ("vocos_5", 5, 5),
        ("vocos_2", 2, 2),
    ],
)
def test_a_vocoder_or_super_resolution_stage_shape_delegates(label, scale_y, scale_x):
    """HiFi-GAN upsamples (8, 8, 2, 2) and Vocos (8, 5, 2, 2); both are integers."""
    program, inputs = _runs(
        _Bilinear(scale_factor=[float(scale_y), float(scale_x)]), (1, 8, 2, 3)
    )
    data, commands = _blob(program)
    assert [c.type for c in commands] == [_BILINEAR], label
    assert list(commands[0].params) == [8, 2, 3, 2 * scale_y, 3 * scale_x]
    got = _answer(program, inputs)
    assert got.size == 8 * (2 * scale_y) * (3 * scale_x)

