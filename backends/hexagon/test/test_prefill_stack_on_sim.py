"""What the M <= 32 prefill kernel costs in stack, which no other suite sees.

The checked-out Q4 small-M source heap-allocates one DMA descriptor per K/32
step. The historical VLA implementation put the same K-sized array on the
command thread's stack, where the phone reported a stack failure at larger K.
This suite compiles the current heap-backed source and supplies an 8 KiB worker
stack; it verifies that source on a constrained stack, but it does not compile
or compare a VLA source variant. The M > 32 branch is the control that confirms
the runner and the stack budget are viable.

`sim/prefill_stack_runner.cpp` runs the kernel as a job on a one-worker pool
built with 8 KiB, while the kernel's internal submits go to the global pool and
keep the normal worker stacks. The cases cover small K, the other prefill branch,
and the Q4 M <= 32 branch at K = 12800. Every current-source case must return 0
with every output element exactly K.
"""

from __future__ import annotations

import pathlib

import numpy as np


import pytest


import hexagon_sim
from test_prefill_on_sim import _SOURCES

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/prefill_stack_runner.cpp"

#: tag, M, K, N, in the order the runner runs them: the largest case is last
#: so a failure identifies the shape that ran last.
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
