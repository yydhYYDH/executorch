# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guard the generated op support table against drifting from the emitters.

``scripts/gen_op_support.py --check`` is the same assertion, but nothing in the
build or the CI suite runs a script, so the table would go stale the first time
an emitter was added without regenerating it. Collected by ``pytest.ini``'s
``backends/hexagon/test`` entry, which runs in the generic Linux unittest job.
"""

from pathlib import Path

from executorch.backends.hexagon.scripts import gen_op_support

# Anchored on this file rather than on the generator's repository root, which is
# inferred by walking three parents up from the script.
TABLE = Path(__file__).resolve().parents[1] / "OP_SUPPORT.md"


def _generated() -> str:
    try:
        return gen_op_support.render()
    except SystemExit as error:
        # render() is what reports an emitter with no row, and an uncaught
        # SystemExit would take the whole pytest session down with it.
        raise AssertionError(f"the op support table cannot be generated: {error}")


def test_op_support_table_is_current():
    assert TABLE.exists(), (
        f"{TABLE} is missing; generate it with "
        "PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py"
    )
    assert TABLE.read_text() == _generated(), (
        f"{TABLE} is stale; regenerate it with "
        "PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py"
    )
