"""What the M <= 32 prefill kernel costs in stack, which no other suite sees.

`hmx_matmulq4fp16_mle32_part` builds one DMA descriptor per K/32 step, and at 32
bytes each that array is exactly K bytes. While it lived on the stack this
branch's frame grew with K, and the phone aborts on it -- 0x8000040d at K =
12736, while the M > 32 branch is fine at K = 25216. No host-side test can see
that, and neither can an ordinary run on the simulator: on its main thread
K = 12800 returns the right answer whether the array is on the stack or not,
because that stack is far larger than the phone's.

So this suite supplies the stack. `sim/prefill_stack_runner.cpp` runs the kernel
as a job on a one-worker pool built with 8 KiB, while the kernel's own internal
submits go to the global pool and keep the normal worker stacks -- the phone's
arrangement, a small stack at the top of the call and ordinary stacks
underneath. It then runs K = 12800 on both branches: the M > 32 branch, which
heap-allocates the same array, is the control that shows 8 KiB is a workable
stack for this path, and the M <= 32 branch is the claim.

Measured before the move and after it: with the array on the stack the
simulator reports `QuRT error code 0x2701` and a call trace through
`hmx_matmulq4fp16_mle32` at K = 12672 on this stack, and the process is gone;
with the array on the heap every shape here returns 0 with every output element
exactly K.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

import pytest  # noqa: E402

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from test_prefill_on_sim import _SOURCES  # noqa: E402

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/prefill_stack_runner.cpp"

#: tag, M, K, N, in the order the runner runs them: the case expected to overrun
#: is last, because a stack fault takes the whole process with it.
_CASES = [
    ("SMALLK64", 4, 64, 32),
    ("SMALLK512", 4, 512, 32),
    ("OTHERBRANCH", 33, 12800, 32),
    ("MLE32K12800", 4, 12800, 32),
]


@pytest.fixture(scope="module")
def measured():
    """The runner's results, or a failure naming the case that killed it.

    `hexagon_sim.run` raises `Unavailable` both when there is no toolchain and
    when the simulator did not run the runner. Here the second is the thing being
    measured -- an overrun is exactly a runner that does not finish -- so the two
    are separated on purpose rather than turned into one skip.
    """
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    try:
        return hexagon_sim.run(_RUNNER, _SOURCES)
    except hexagon_sim.Unavailable as error:
        started = [
            line.split()[0]
            for line in hexagon_sim.LAST_STDOUT.splitlines()
            if line.endswith("START")
        ]
        pytest.fail(
            "the simulator did not run the runner; the last case to start was "
            f"{started[-1] if started else '(none)'}, which is an overrun and not "
            f"an environment problem: {error}"
        )


def _assert_case(measured, tag, m, k, n):
    assert measured[f"{tag}RET"][0] == 0, f"{tag} (M={m}, K={k}) returned nonzero"
    assert measured[f"{tag}M"][0] == m, f"{tag} ran with the wrong M"
    assert measured[f"{tag}K"][0] == k, f"{tag} ran with the wrong K"
    # Every one of the M*N outputs should be exactly K, which is exact in fp16
    # here. Counting rather than indexing keeps the check clear of the pack
    # layout, which `test_prefill_on_sim.py` is the suite that pins down.
    expected = int(np.float16(k).view(np.uint16))
    assert measured[f"{tag}VALUE"][0] == expected, (
        f"{tag} (M={m}, K={k}) wrote {measured[f'{tag}VALUE'][0]:#06x} in the "
        f"first element, expected {expected:#06x}"
    )
    assert measured[f"{tag}CORRECT"][0] == m * n, (
        f"{tag} (M={m}, K={k}) produced {measured[f'{tag}CORRECT'][0]} of the "
        f"{m * n} outputs it was asked for"
    )


def test_the_m_over_32_branch_runs_on_a_stack_of_8_kib(measured):
    """The control: this path works on 8 KiB when its frame ignores K.

    Without it, `test_the_m_le_32_branch...` would only be saying that 8 KiB is
    too small for this kernel, which is not a claim about K.
    """
    for tag, m, k, n in _CASES:
        if tag == "OTHERBRANCH":
            _assert_case(measured, tag, m, k, n)


def test_the_m_le_32_branch_will_not_build_a_frame_that_grows_with_k(measured):
    """The claim: K-sized stack objects are not affordable on this branch.

    K = 12800 is the shape the phone aborts on. The two small shapes run first
    for the same reason as the control above -- they are what a frame that does
    not grow with K looks like -- and they are also where a regression that
    shrank the stack budget in general would show up first.
    """
    for tag, m, k, n in _CASES:
        if tag == "OTHERBRANCH":
            continue
        _assert_case(measured, tag, m, k, n)
