# Copyright (c) Meta Platforms, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The trust probes run, so their controls cannot rot.

`scripts/trust_probe.py` exists because a detector that reports nothing and a
broken detector report the same thing, so each probe carries a control that says
it can still see what it looks for. A control that is only run by hand is a
control that goes stale quietly, which is the failure it was written to catch,
so the suite runs them.

The probes are read-only and need no device and no simulator. The two that lower
a graph ask for `PYTHONPATH` to include `src`, which the suite already sets.
"""

import pathlib
import sys

import pytest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import trust_probe  # noqa: E402


@pytest.mark.parametrize("probe", trust_probe.PROBES, ids=lambda p: p.__name__)
def test_every_trust_probe_control_is_live(probe):
    """A probe that fails here is a broken probe, not a clean tree.

    The probes return ok for the hazards they are about, so a red line here
    means the detector stopped seeing rather than that the backend changed. The
    detail is in the failure message because the two readings are different
    problems with different next steps.
    """
    result = probe()
    assert result.ok, f"{result.name}: {result.detail}"


def test_the_probe_file_carries_a_control_for_every_probe():
    """Each probe must mention a control, or it is a scan with no positive case."""
    for probe in trust_probe.PROBES:
        source = pathlib.Path(inspect_file(probe)).read_text()
        assert "control" in source, f"{probe.__name__} has no control in its body"


def inspect_file(func):
    return func.__code__.co_filename

