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
