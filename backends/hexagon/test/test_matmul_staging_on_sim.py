# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A matmul whose K is not a multiple of 64, through the staging tile it reads.

`htp_ops_loop_matmul_hmx_general` stages a contiguous activation tile in VTCM
before handing it to the matrix unit. The tile is read a row at a time at
`up_div(K, 64) * 64` elements -- the width the packed weight tiles use -- so the
copy that fills it has to put each row at that stride. Copying `validRows * K`
elements straight through leaves every row but the first short of where the
readers look for it: for K = 40 each row's window lands 1.6 rows further along
than it should (`up_div(40, 64) * 64 == 64`), and from row 20 of each 32-row tile
the window starts past the copied bytes and reads the zero fill. Measured on the
simulator before the copy was padded, that is a max difference of 12.05 against
torch with 7392 elements exactly zero, and the three rows that do land on a row
boundary -- 5, 10 and 15 -- are exactly reference rows 8, 16 and 24, the 64/40
ratio. `K = 64` and `K = 128` are unaffected either way, because for them the
staging stride and K are the same number; they are here as the controls that say
so, and their answers are bit-identical before and after.

The shapes are the ones that take this path at all: the kernel only chooses it
when every operand's inner stride is two bytes, both outer strides are known,
and `E * K * N >= 32768`. `test_blob_on_sim.py`'s own matmul cases are `4 x 8`
by `8 x 3`, which is 96, so nothing there reaches a staging tile -- which is why
a K that is not a multiple of 64 could be wrong for as long as it was.

Operands are random rather than the periodic small integers the other cases
prefer: a row read from the wrong offset is *another row of the same matmul*
rather than noise, and periodic data can make a shifted read agree by accident,
which would be a case that decides nothing.
"""



import numpy as np
import pytest
import torch


import hexagon_sim
import test_blob_on_sim as blob_sim

#: (tag, M, K, N). 40 is Stable Diffusion 1.5's head width, 32 is a K that is a
#: multiple of 32 and not of 64, 77 is SD's context length -- which is the K of
#: the second product in its attention -- and 64 and 128 are the controls.
_CONFIGS = (
    ("S40", 256, 40, 77),
    ("S32", 256, 32, 77),
    ("S77", 256, 77, 40),
    ("S64", 256, 64, 77),
    ("S128", 256, 128, 77),
)

#: The repository's fp16 matmul allowance: these are products over K, so the
#: comparison is a tolerance and not a bit pattern.
TOLERANCE = 1e-2


class _Matmul(torch.nn.Module):
    def forward(self, left, right):
        return torch.mm(left, right)


def _cases():
    out = []
    for tag, rows, k, columns in _CONFIGS:
        generator = torch.Generator().manual_seed(11)
        left = (torch.rand((rows, k), generator=generator) * 2 - 1).half()
        right = (torch.rand((k, columns), generator=generator) * 2 - 1).half()
        expected = (left.float() @ right.float()).half()
        out.append(
            blob_sim._case(
                tag,
                _Matmul(),
                (left, right),
                expected,
                kind="close",
                tolerance=TOLERANCE,
            )
        )
    return out


@pytest.fixture(scope="module")
def cases():
    """Built lazily, so a blob that cannot be produced fails a test, not collection."""
    return _cases()


@pytest.fixture(scope="module")
def simulated(cases):
    try:
        return hexagon_sim.run(
            blob_sim._RUNNER,
            blob_sim._SOURCES,
            headers={
                "blob_fixture.h": blob_sim._fixture_header(cases),
                "htp_ops.h": blob_sim._HT_P_OP_SHIM,
            },
            includes_more=[str(blob_sim._SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


def _dsp(case, simulated):
    bits = simulated[f"{case.tag}0"]
    return np.asarray(blob_sim._from_bits(bits), dtype=np.float16).reshape(
        case.expected.shape
    )


def test_every_case_reaches_the_staging_tile(cases):
    """A shape below the entry gate tests the other kernel, not this one.

    The gate is `E * K * N >= 32768` and two-byte inner strides; a case that
    missed it would pass on any copy at all, and the file would be about
    nothing.
    """
    for (_tag, rows, k, columns), case in zip(_CONFIGS, cases):
        assert rows * k * columns >= 32768, (rows, k, columns)
        assert [command.type for command in case.commands] == [38], case.tag


def _host(case):
    return np.asarray(
        blob_sim._from_bits(blob_sim._host_bits(case.host[0])), dtype=np.float16
    ).reshape(case.expected.shape)


def test_every_matmul_agrees_with_torch(cases, simulated):
    """The product itself, with the DSP and the host model held to the same bar.

    Two of the five K values here are multiples of 64, where the staging stride
    and K are the same number and the copy is a straight one; they are the
    controls for the three that are not.
    """
    for case in cases:
        expected = np.asarray(case.expected, dtype=np.float16).astype(np.float64)
        for name, got in (("host model", _host(case)), ("DSP", _dsp(case, simulated))):
            worst = float(np.max(np.abs(got.astype(np.float64) - expected)))
            assert worst < TOLERANCE, f"{case.tag}: {name} differs by {worst}"


def test_a_staging_tile_short_of_its_stride_leaves_no_row_zero(cases, simulated):
    """The staging bug's own signature, pinned separately from the tolerance.

    A read that runs off the copied bytes answers zero for the rest of the tile
    -- twelve rows of every thirty-two at K = 40 -- so a nonzero row is what
    separates "computed from the wrong row" from "computed from no row", and it
    is the half that a tolerance on a small reference could miss.
    """
    for case in cases:
        rows = np.asarray(_dsp(case, simulated), dtype=np.float64)
        empty = [index for index in range(rows.shape[0]) if not rows[index].any()]
        assert not empty, f"{case.tag}: rows {empty[:4]} came back entirely zero"
