# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The hexagon runtime options: what a load-time spec may change, and what it may not.

These are the knobs that do not change a byte of the blob -- a trace, a tile cap,
a diagnostic split -- so a runner may set them per device and per load. What they
may not do is change behaviour when nothing is set: every one of them has an
environment variable it has always had, and a process that sets neither the
option nor the variable has to behave exactly as it did before options existed.
That is what most of this checks.

The resolver is C++, and the part of it that runs on the host is the part that
can be checked without a device, so the probe below calls the same
`HexagonBackendOptions::resolve` that init() calls, against a real
`BackendInitContext` built from real `BackendOption`s.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

_HEXAGON_DIR = pathlib.Path(__file__).resolve().parents[1]
_PROBE_SRC = pathlib.Path(__file__).resolve().parent / "runtime_options_probe.cpp"

#: The platform layer, which the runtime links and the probe has to bring along:
#: it is where ET_LOG and the allocator's checks end up. Spelled through the
#: checkout's `executorch` entry, which is how the tree is reachable from the
#: directory the compiler is given as its include root.
_RUNTIME_SOURCES = (
    "runtime/platform/log.cpp",
    "runtime/platform/platform.cpp",
    "runtime/platform/default/posix.cpp",
    "runtime/platform/abort.cpp",
)

#: What a process that sets nothing gets, spelled out once so a change to any of
#: them has to be a change here too.
_DEFAULTS = {
    "trace": 0,
    "delegate": -1,
    "cmd_start": 0,
    "cmd_limit": 0,
    "stop_after": -1,
    "fake_cache": 0,
    "phase": 0,
    "kv_state": 1,
    "watchdog_seconds": 15,
    "tile_budget": -1,
    "acct": 0,
}

#: Each field, the kind of value it takes, and a value that proves it moved.
_SETTINGS = {
    "trace": ("b", 1),
    "delegate": ("i", 3),
    "cmd_start": ("i", 2),
    "cmd_limit": ("i", 5),
    "stop_after": ("i", 0),
    "fake_cache": ("b", 1),
    "phase": ("b", 1),
    "kv_state": ("b", 0),
    "watchdog_seconds": ("i", 0),
    "tile_budget": ("i", 4096),
    "acct": ("b", 1),
}


def _arguments(values=None):
    return [
        f"{name}={kind}:{value}"
        for name, (kind, value) in (values or _SETTINGS).items()
    ]


_PROBE_BINARY = None


def _checkout_root():
    """The directory the compiler should resolve `executorch/...` through.

    A source checkout is imported through a shim directory -- `src` here, the
    checkout's parent upstream -- and the headers are reached by the same
    `executorch/...` spelling, so the compiler is given whichever root the
    interpreter loaded this backend from.
    """
    for entry in sys.path:
        if not entry:
            continue
        probe = pathlib.Path(entry) / "executorch" / "backends" / "hexagon"
        if (probe / "runtime" / "HexagonBackendOptions.h").is_file():
            return entry
    raise AssertionError("no include root in sys.path holds this backend")


def _probe():
    """The host compiler builds the probe; failing to build is a failure."""
    global _PROBE_BINARY
    if _PROBE_BINARY is None:
        cxx = os.environ.get("CXX", "c++")
        if shutil.which(cxx) is None:
            pytest.skip(f"no host C++ compiler ({cxx})")
        root = _checkout_root()
        binary = pathlib.Path(tempfile.mkdtemp(prefix="hexagon_options_")) / "probe"
        built = subprocess.run(
            [
                cxx,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",  # BackendInitContext's unused event_tracer
                f"-I{root}",
                f"-I{pathlib.Path(torch.__file__).parent / 'include'}",
                "-I",
                os.fspath(_HEXAGON_DIR / "runtime"),
                os.fspath(_PROBE_SRC),
                os.fspath(_HEXAGON_DIR / "runtime" / "HexagonBackendOptions.cpp"),
                *[
                    os.fspath(pathlib.Path(root) / "executorch" / source)
                    for source in _RUNTIME_SOURCES
                ],
                "-o",
                os.fspath(binary),
            ],
            capture_output=True,
            text=True,
        )
        assert (
            built.returncode == 0
        ), f"the options probe failed to build:\n{built.stderr}"
        _PROBE_BINARY = binary
    return _PROBE_BINARY


def _resolve(options=(), environment=None):
    """Runs the probe, returning (fields, stderr).

    The environment is whatever a caller asks for and nothing else, so a knob's
    fallback is checked against a process that really does not have it.
    """
    environment = {} if environment is None else environment
    done = subprocess.run(
        [os.fspath(_probe()), *options],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), **environment},
    )
    assert done.returncode == 0, f"the probe failed:\n{done.stderr}"
    fields = {}
    for line in done.stdout.splitlines():
        name, _, value = line.partition(" ")
        fields[name] = value
    return fields, done.stderr


def test_a_process_that_sets_nothing_gets_the_defaults():
    fields, _ = _resolve()
    assert fields.get("ok") == ""
    for name, expected in _DEFAULTS.items():
        assert fields[name] == str(expected), f"{name} is {fields[name]}"


def test_every_option_reaches_its_own_field():
    """All at once, because the point is that each key names one field."""
    fields, _ = _resolve(_arguments())
    assert "error" not in fields, fields
    for field, (_, value) in _SETTINGS.items():
        assert fields[field] == str(
            value
        ), f"{field} is {fields[field]}, wanted {value}"


def test_the_environment_is_the_fallback_the_options_replace():
    """The variables these knobs have always had still work, and a spec wins."""
    environment = {
        "HEXAGON_TRACE": "1",
        "HEXAGON_DELEGATE": "2",
        "HEXAGON_CMD_START": "1",
        "HEXAGON_CMD_LIMIT": "3",
        "HEXAGON_STOP_AFTER": "4",
        "HEXAGON_FAKE_CACHE": "1",
        "HEXAGON_PHASE": "1",
        "HEXAGON_KV_STATE": "0",
        "HEXAGON_WATCHDOG_SECONDS": "30",
        "HEXAGON_HMX_TILE_BUDGET": "2048",
        "HEXAGON_ACCT": "1",
    }
    # The tile budget is the one whose variable is not its key in upper case.
    from_environment = {
        "trace": 1,
        "delegate": 2,
        "cmd_start": 1,
        "cmd_limit": 3,
        "stop_after": 4,
        "fake_cache": 1,
        "phase": 1,
        "kv_state": 0,
        "watchdog_seconds": 30,
        "tile_budget": 2048,
        "acct": 1,
    }
    fields, _ = _resolve(environment=environment)
    for field, value in from_environment.items():
        assert fields[field] == str(value), f"{field} is {fields[field]}"

    # And an option of the same name is what is used, not the variable.
    overridden, _ = _resolve(["trace=b:0", "watchdog_seconds=i:5"], environment)
    assert overridden["trace"] == "0"
    assert overridden["watchdog_seconds"] == "5"
    assert (
        overridden["delegate"] == "2"
    ), "the variable still applies where no spec does"


def test_a_knob_that_changes_the_blob_is_not_read_here_at_all():
    """HEXAGON_HMX_PREPACK moves the blob, so it is a compile spec now.

    A runtime that still honoured the variable would be claiming a layout the
    .pte was not built with; the field it used to fill is not in these options.
    """
    fields, _ = _resolve(
        environment={"HEXAGON_HMX_PREPACK": "0", "HEXAGON_ATTN_PAGED": "1"}
    )
    for name, expected in _DEFAULTS.items():
        assert fields[name] == str(expected), f"{name} moved to {fields[name]}"


def test_a_variable_that_is_not_a_number_keeps_the_old_reading():
    """atoi semantics, which is what these reads always had: the default."""
    fields, _ = _resolve(environment={"HEXAGON_WATCHDOG_SECONDS": "abc"})
    # atoi("abc") is 0, not "not a number", so this is the documented reading and
    # not a silent fallback to 15.
    assert fields["watchdog_seconds"] == "0"


def test_an_option_of_the_wrong_type_is_refused_by_name():
    """Not a fallback: a caller that asked for a trace has to get one or an error."""
    fields, stderr = _resolve(["trace=i:1"])
    assert fields["error"] == "invalid_argument 18", fields
    assert "backend option trace must be a bool" in stderr

    for key in ("delegate", "watchdog_seconds", "tile_budget"):
        fields, stderr = _resolve([f"{key}=b:1"])
        assert fields["error"] == "invalid_argument 18", (key, fields)
        assert f"backend option {key} must be an int" in stderr


@pytest.mark.parametrize(
    "option,expected",
    [
        ("delegate=i:-2", "delegate must be -1"),
        ("cmd_start=i:-1", "cannot be negative"),
        ("cmd_limit=i:-1", "cannot be negative"),
        ("stop_after=i:-2", "stop_after must be -1"),
        ("watchdog_seconds=i:-1", "watchdog_seconds must be 0"),
        ("tile_budget=i:-2", "tile_budget must be -1"),
    ],
)
def test_an_option_out_of_range_is_refused_with_a_sentence(option, expected):
    fields, stderr = _resolve([option])
    assert fields["error"] == "invalid_argument 18", fields
    assert expected in stderr, stderr
