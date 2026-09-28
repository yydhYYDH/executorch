# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hexagon flow for the backend test suite.

Registered unconditionally, unlike the QNN and Cortex-M flows, because nothing a
case needs to lower is gated on the SDK or on a device: the partitioner, the blob
serialization and the ``.pte`` all run on the collecting host. A flow that is left
unregistered is invisible rather than failed -- ``pytest -m flow_hexagon`` collects
zero tests and reports success -- which is why the guard those two flows use is the
wrong shape here.

Execution is the part this backend cannot do off-device. The delegate runs HVX/HMX
kernels built for one Hexagon arch behind a FastRPC session, so the suite's
in-process portable pybindings cannot load a serialized program and a host CPU
cannot execute one either. ``HexagonTester`` therefore skips the run stage with its
own reason; every stage before it still runs, so a collected case covers lowering
and serialization. See the execution policy note in
``backends/hexagon/test/tester/tester.py`` for what a device-capable runner would
have to provide.

``quantize`` is left off, and the reason is what the quantizer annotates rather than
what the kernels accept. ``backends/hexagon/quantizer.py`` is a weight-only PT2E
scheme: only the matmul's weight is annotated, so no observer goes on an activation
and a case without a weight-only matmul gains nothing from being quantized. The
emitters themselves are not the limit -- there is a GEMV entry for M == 1 and a
prefill entry for M > 1 at each of the two widths -- so a quantized flow would lower
the matmul cases and leave the rest portable, which measures the suite's case mix
rather than this backend. The quantized path has its own coverage in
``backends/hexagon/test/``.
"""

from executorch.backends.hexagon.test.tester import HexagonTester
from executorch.backends.test.suite.flow import TestFlow


def _create_hexagon_flow(name: str) -> TestFlow:
    return TestFlow(
        name,
        backend="hexagon",
        tester_factory=HexagonTester,
    )


HEXAGON_TEST_FLOW = _create_hexagon_flow("hexagon")
