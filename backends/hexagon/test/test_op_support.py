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


#: The quantized matmul's two prefill commands, as the generator names them:
#: an int4 weight reaches the first and an int8 weight the second. Both are
#: measured on the command stream -- the W8A16 one by a real partitioned
#: `w8a16` graph at M = 8, K = 64, N = 128, whose delegate decodes as
#: `DSP_OP_MATMUL_W8A16_BLOCK_FP16` (42) followed by one `DSP_OP_RASTER_BLIT`
#: (3), through `_quantized_prefill_fits` -> `weight_only_matmul_fits`.
#: Agreement between the table and the generator is not the point: a table that
#: says a command does not exist is wrong in the direction no test here can see
#: unless the rendered output itself is asked.
PREFILL_COMMANDS = (
    "DSP_OP_MATMUL_Q4A16_FP16",
    "DSP_OP_MATMUL_W8A16_BLOCK_FP16",
)


def _row(rendered: str, op: str) -> str:
    prefix = f"| `{op}` | "
    rows = [line for line in rendered.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, f"expected one row for {op}, found {len(rows)}"
    return rows[0]


def test_the_generated_table_names_the_int8_prefill_command():
    """Both prefill commands belong on the mm and addmm rows, and nowhere
    in the exclusion list: real lowering through HexagonPartitioner decodes
    [42, 3] for a w8a16 matmul with M > 1 (M = 8, K = 64, N = 128), so a row
    saying int8 prefill is unsupported is false whether the generator and the
    file agree with each other or not.
    """
    rendered = _generated()
    for op in ("aten.mm.default", "aten.addmm.default"):
        row = _row(rendered, op)
        for name in PREFILL_COMMANDS:
            assert f"`{name}` (" in row, f"{op} does not name {name}: {row}"
    assert "M > 1 the int4 prefill entry" not in rendered
    assert "prefill (M > 1) with a quantized int8 weight" not in rendered
