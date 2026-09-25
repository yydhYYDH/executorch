# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the DSP's gelu, sigmoid and tanh compute, held to a device run.

All three take `htp_ops_unary_pwl_fp16_vec` (unary_ops.cc:259-330) inside
`htp_ops_unary_compute_fp16_chunk` (unary_ops.cc:455-491): a companded index into
a table of fp16 slopes and biases, one fp16 multiply and one fp16 add, a fold
through the odd or the shifted identity, and a saturation at the magnitude the
table stops at. The walk covers `[0, numel & ~63)` and nothing else -- the last
`numel % 64` elements, and every element of a buffer shorter than 64, are what
`htp_ops_unary_apply_fp16` computes, which for gelu is the fp32 tanh form. So the
same input has two answers and the length of the tensor is what picks one.

None of the three is torch's function, and none of them is exact: measured
against the definition over a dense sweep in fp64, gelu is off by up to 6.1e-3,
sigmoid by 2.4e-3 and tanh by 6.3e-3, worst where a transformer's activations
live (|x| < 0.5, where the first chord of the table is fitted). That is tens of
fp16 steps at the scale of a layer's output, so it is a property to know about
rather than a rounding error to ignore, and this file is where the numbers and
the device run they came from live. The tables are the `HTP_OPS_PWL_COMPANDED16`
branch the skel is built with (skel/CMakeLists.txt:133).

The reference answers below are one run of five one-op subgraphs on a OnePlus
PJZ110 (SM8750, Android 15) over `ssh oneplus13-reverse`, through the shared
runner and skel at `/data/data/com.termux/files/home/csm/execuTorch-ds`:
`executor_runner` md5 `0a06a48acfb541bb85cb043a3efc4a77` and
`skel/libhex-htp-skel-v79.so` md5 `db74410ba0f9a44ad8e6ed2d2d4421ec`. The blobs
came from this tree, each one `DSP_OP_UNARY` over fp16 with `HTP_OPS_PWL_*` built
into the skel. The binary is older than the tree and its revision is not attested
beyond this: for these three op types its answers match the in-tree vendored
tables and index arithmetic bit for bit, which is what the cases below assert.
Re-measuring them is one device run of the same five cases; a table that was
refitted upstream would land here as a red test rather than as a silent drift.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on the
# path makes `import executorch` resolve to this tree. Without it the editable
# install wins, and in this environment that points at a different checkout.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import UNARY_OP_TYPES  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from torch.export import export  # noqa: E402

#: The vector every case carries, cycled to the case's length: one value from each
#: kind of chord the tables have, both ends of the saturation, and the negative of
#: each. Values sit on the boundaries of the quarter-wide chords so a table read
#: at the wrong index cannot pass by being nearly right.
_HEAD = [
    -9.0,
    -4.0,
    -3.5,
    -2.0,
    -1.0,
    -0.5,
    -0.125,
    -0.03125,
    0.0,
    0.03125,
    0.125,
    0.5,
    1.0,
    2.0,
    3.5,
    4.0,
    9.0,
]

#: (index, input, the phone's output bits) for each op and buffer length. The
#: cases at 63 and 64 elements carry the same values and are the boundary itself:
#: below it the buffer is the scalar form, at it the buffer is the table, and the
#: answers differ. The 100-element case puts the same values in the tail, where
#: they come back with the scalar form's bits again.
_GOLDEN = {
    "gelu": {
        64: [
            (0, -9.0, 0x0000),
            (1, -4.0, 0x0000),
            (2, -3.5, 0x8000),
            (3, -2.0, 0xA9C0),
            (4, -1.0, 0xB114),
            (5, -0.5, 0xB0F0),
            (6, -0.125, 0xAA6C),
            (7, -0.03125, 0xA26C),
            (8, 0.0, 0x0000),
            (9, 0.03125, 0x24CA),
            (10, 0.125, 0x2CCA),
            (11, 0.5, 0x3588),
            (12, 1.0, 0x3ABB),
            (13, 2.0, 0x3FD2),
            (14, 3.5, 0x4300),
            (15, 4.0, 0x4400),
            (16, 9.0, 0x4880),
        ],
        63: [
            (0, -9.0, 0x8000),
            (1, -4.0, 0x849A),
            (2, -3.5, 0x910C),
            (3, -2.0, 0xA9D0),
            (4, -1.0, 0xB115),
            (5, -0.5, 0xB0F0),
            (6, -0.125, 0xAB34),
            (7, -0.03125, 0xA3CD),
            (8, 0.0, 0x0000),
            (9, 0.03125, 0x241A),
            (10, 0.125, 0x2C66),
            (11, 0.5, 0x3588),
            (12, 1.0, 0x3ABB),
            (13, 2.0, 0x3FD2),
            (14, 3.5, 0x4300),
            (15, 4.0, 0x4400),
            (16, 9.0, 0x4880),
        ],
        100: [
            (63, 1.0, 0x3ABB),
            (64, 2.0, 0x3FD2),
            (72, -1.0, 0xB115),
            (74, -0.125, 0xAB34),
            (75, -0.03125, 0xA3CD),
            (77, 0.03125, 0x241A),
            (78, 0.125, 0x2C66),
            (99, 3.5, 0x4300),
        ],
    },
    "sigmoid": {
        64: [
            (0, -9.0, 0x0000),
            (1, -4.0, 0x2480),
            (2, -3.5, 0x2780),
            (3, -2.0, 0x2FA0),
            (4, -1.0, 0x344E),
            (5, -0.5, 0x360C),
            (6, -0.125, 0x3780),
            (7, -0.03125, 0x37E0),
            (8, 0.0, 0x3800),
            (9, 0.03125, 0x3810),
            (10, 0.125, 0x3840),
            (11, 0.5, 0x38FA),
            (12, 1.0, 0x39D9),
            (13, 2.0, 0x3B0C),
            (14, 3.5, 0x3BC4),
            (15, 4.0, 0x3BDC),
            (16, 9.0, 0x3C00),
        ],
    },
    "tanh": {
        64: [
            (0, -9.0, 0xBC00),
            (1, -4.0, 0xBC00),
            (2, -3.5, 0xBBFC),
            (3, -2.0, 0xBBB6),
            (4, -1.0, 0xBA18),
            (5, -0.5, 0xB765),
            (6, -0.125, 0xAFD6),
            (7, -0.03125, 0xA7D6),
            (8, 0.0, 0x0000),
            (9, 0.03125, 0x27D6),
            (10, 0.125, 0x2FD6),
            (11, 0.5, 0x3765),
            (12, 1.0, 0x3A18),
            (13, 2.0, 0x3BB6),
            (14, 3.5, 0x3BFC),
            (15, 4.0, 0x3C00),
            (16, 9.0, 0x3C00),
        ],
    },
}

#: The worst absolute distance from the definition each of the three may have, at
#: the buffer length the walk covers the whole of. The measured values are 6.1e-3
#: for gelu, 2.4e-3 for sigmoid and 6.3e-3 for tanh, and the ceilings sit just
#: above them: at a rounder number a refitted table could drift without saying so.
_BOUND = {"gelu": 6.2e-3, "sigmoid": 2.5e-3, "tanh": 6.4e-3}

#: The index the walk switches forms at, below which it is the table.
_GRAIN = 64


class _Activation(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.fn = getattr(torch.nn.functional, name)

    def forward(self, x):
        return self.fn(x)


def _input(length: int) -> torch.Tensor:
    values = (_HEAD * (length // len(_HEAD) + 1))[:length]
    return torch.tensor([values], dtype=torch.float16)


def _blob(name: str, length: int) -> bytes:
    """The delegate blob one activation over `length` elements lowers to."""
    x = _input(length)
    program = to_edge(export(_Activation(name), (x,))).exported_program()
    return HexagonBackend.preprocess(program, []).processed_bytes


def _dsp_answer(name: str, length: int) -> np.ndarray:
    """What the command stream computes, through the host model of it."""
    blob = _blob(name, length)
    _header, commands = read_blob(blob)
    assert len(commands) == 1, f"{name}: {len(commands)} commands"
    assert commands[0].params[0] == length, (
        f"{name}: the command walks {commands[0].params[0]} of {length} elements"
    )
    outputs = execute(blob, [_input(length).numpy()])
    return np.frombuffer(outputs[0], dtype=np.float16)


def _fold(pattern: int) -> int:
    """A negative zero folded into a positive one.

    The phone returns `0x8000` for gelu where the table's last chord cancels the
    magnitude exactly (x in [-4, -3.5), where the fold is `table(a) - a`) and the
    model's fp16 subtraction gives `+0.0` there. No comparison of values sees the
    difference and it is the only point band below the two disagree on, so it is
    folded rather than asserted: a test that pinned the sign of a zero would be
    pinning a bit the vendored source does not explain.
    """
    return 0 if pattern == 0x8000 else pattern


def _bits(values: np.ndarray) -> np.ndarray:
    patterns = np.asarray(values, dtype=np.float16).view(np.uint16)
    return np.array([_fold(int(p)) for p in patterns], dtype=np.uint16)


@pytest.mark.parametrize(
    "name,length",
    [(name, length) for name, cases in _GOLDEN.items() for length in cases],
)
def test_the_host_model_reproduces_the_phone(name, length):
    """Every recorded answer, at the index it was recorded at.

    The interpreter is what the rest of the suite uses to check a delegated
    subgraph, so what is asserted here is that its model of the walk is the
    device's: table lookups, the two fp16 roundings, the fold and the saturation,
    at the length that picks the form.
    """
    got = _dsp_answer(name, length)
    inputs = _input(length)
    for index, value, bits in _GOLDEN[name][length]:
        assert inputs[0, index].item() == value, (
            f"{name}: index {index} of the vector is {inputs[0, index].item()}, "
            f"not the {value} the answer was measured at"
        )
        assert _bits(got[index : index + 1])[0] == _fold(bits), (
            f"{name}({value}) at index {index} of {length} elements: the model "
            f"says {float(got[index])!r} and the phone said "
            f"{float(np.array([bits], dtype=np.uint16).view(np.float16)[0])!r}"
        )


def test_the_position_of_an_element_picks_the_form():
    """The same input, two answers, decided by where it sits and not by its value.

    -0.125 comes back as 0xAA6C in a 64-element buffer and as 0xAB34 in the tail
    of a 100-element one, and 0xAB34 is also what a 63-element buffer answers --
    the scalar form throughout. That is the whole of the length dependence: the
    walk covers [0, numel & ~63) and the rest of the buffer is the scalar path,
    so a tensor one element shorter than the grain is not approximated at all.
    """
    def answer(length, value):
        return {
            index: bits for index, at, bits in _GOLDEN["gelu"][length] if at == value
        }

    body = _fold(answer(64, -0.125)[6])
    tail = _fold(answer(100, -0.125)[74])
    short = _fold(answer(63, -0.125)[6])
    assert body != tail, (
        "the table and the scalar form agree at -0.125, so this case cannot tell "
        "which one an element took"
    )
    assert tail == short, (
        "the tail of a 100-element buffer and a 63-element buffer disagree, so "
        "the walk is not the only difference between them"
    )
    # And the model follows the switch, not one form everywhere.
    for length, expected in ((64, body), (100, tail)):
        got = _dsp_answer("gelu", length)
        index = 74 if length > _GRAIN else 6
        assert _input(length)[0, index].item() == -0.125, (
            f"index {index} of the {length}-element vector is not -0.125"
        )
        assert _bits(got[index : index + 1])[0] == expected, (
            f"gelu at {length} elements answered "
            f"0x{int(_bits(got[index : index + 1])[0]):04x} at -0.125, not "
            f"0x{expected:04x}"
        )


@pytest.mark.parametrize("name", ["gelu", "sigmoid", "tanh"])
def test_the_approximation_is_within_the_band_and_is_not_exact(name):
    """The distance from the definition, against the bound the device measured.

    The reference is the value in fp64 at the fp16 input, so what is measured is
    the kernel's error and not torch's. Both halves of the assertion matter: a
    ceiling alone would pass a table that had been refitted to the exact function
    -- which would no longer be what the DSP computes -- and no ceiling at all
    would be the state this file exists to end.
    """
    length = 4096
    got = _dsp_answer(name, length).astype(np.float64).reshape(-1)
    x = _input(length).numpy().astype(np.float64).reshape(-1)
    if name == "gelu":
        import math

        want = x * 0.5 * (
            1.0 + np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)))(x)
        )
    elif name == "sigmoid":
        want = 1.0 / (1.0 + np.exp(-x))
    else:
        want = np.tanh(x)
    worst = float(np.abs(got - want).max())
    at = float(x[int(np.argmax(np.abs(got - want)))])
    assert worst <= _BOUND[name], (
        f"the DSP's {name} is off by {worst:.3e} at x = {at:.6g}, past the "
        f"{_BOUND[name]:.3e} this file records for it"
    )
    assert worst > _BOUND[name] / 64, (
        f"the DSP's {name} is exact to {worst:.3e}, which is not what this "
        "kernel computes -- the tables or the length are wrong"
    )


@pytest.mark.parametrize("name", ["gelu", "sigmoid", "tanh"])
def test_the_three_are_wired_as_the_approximation(name):
    """The band above is about a delegated op and not about a portable one.

    These three are in the emitter table with no geometry gate, so a model that
    uses them is running the table on the DSP. If one is ever taken off the
    delegate, the measurement above stops being about a model's arithmetic and
    this is the test to move with it.
    """
    _header, commands = read_blob(_blob(name, 128))
    assert commands[0].type == 4, f"{name}: command type {commands[0].type}"
    assert commands[0].params[1] == UNARY_OP_TYPES[name], (
        f"{name}: the command carries subtype {commands[0].params[1]}"
    )
