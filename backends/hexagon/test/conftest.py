# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Make every test module in this directory import this checkout.

The checkout directory is itself named `executorch`, so its parent is the
entry that makes the namespace resolve here. The test directory is not a package
and contains the bare-import helpers used by the suite, so it is placed first.
Pytest loads this conftest before importing any test module in the directory.
"""

import pathlib
import sys

_TEST_DIR = pathlib.Path(__file__).resolve().parent
_CHECKOUT = _TEST_DIR.parents[2]

for _path in (str(_CHECKOUT.parent), str(_TEST_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
