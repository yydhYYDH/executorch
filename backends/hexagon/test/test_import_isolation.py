# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Assert that this session is testing this checkout, not another one."""

import pathlib

_CHECKOUT = pathlib.Path(__file__).resolve().parents[3]


def test_backend_and_helpers_are_from_this_checkout():
    import blob_interpreter
    from executorch.backends.hexagon import hexagon_backend, hexagon_ops
    from executorch.backends.hexagon.serialization import blob

    for module in (hexagon_ops, hexagon_backend, blob, blob_interpreter):
        resolved = pathlib.Path(module.__file__).resolve()
        assert resolved.is_relative_to(_CHECKOUT), (
            f"{module.__name__} was imported from {resolved}, outside {_CHECKOUT}"
        )


def test_every_test_file_pins_a_directory_inside_this_checkout():
    """The insert above is only as good as the level it walks up.

    Thirty-four files each said parents[4] where the comment above them said the
    point was to resolve executorch to this checkout. parents[4] from backends/
    hexagon/test is the worktree's parent, which has no src/ in it at all, so the
    insert pinned nothing and the import fell through to whatever the environment
    had -- in this one, an editable install of a different checkout. Each file
    runs in its own process, so the test above, which pins itself correctly, said
    nothing about any of them.

    What is checked here is static, so it holds in every process: a file can only
    pin this checkout by naming a directory inside it. Some inserts are not that --
    one file puts backends/hexagon on the path so hexagon_ops imports under its own
    name -- and a directory inside the checkout is fine for those too. A directory
    outside it is not fine for anything, because nothing under it can be this tree.
    """
    import glob
    import os
    import re

    here = pathlib.Path(__file__).resolve().parent
    pattern = re.compile(r"sys\.path\.insert\(0,.*?parents\[(\d+)\]")
    outside = []
    for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
        with open(path) as handle:
            for line in handle:
                found = pattern.search(line)
                if not found:
                    continue
                target = pathlib.Path(path).resolve().parents[int(found.group(1))]
                if not target.is_relative_to(_CHECKOUT):
                    outside.append(f"{pathlib.Path(path).name} -> {target}")
    assert not outside, "these pin a directory outside the checkout: " + "; ".join(
        outside
    )
