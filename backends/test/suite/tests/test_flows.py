# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guard the flow registry itself.

A flow that fails to register is swallowed by ``_register_flow`` and shows up as
``pytest -m flow_<name>`` collecting nothing while reporting success, so the
condition has to be asserted somewhere. The Hexagon flow is the one that is not
gated on an SDK or a tool being present, so it is the one that can be asserted
unconditionally; QNN and Cortex-M are skipped by design and asserting them would
fail on a machine that is missing exactly what they need.
"""

from executorch.backends.test.suite.flow import all_flows


def test_hexagon_flow_is_registered():
    flows = all_flows()
    assert "hexagon" in flows, sorted(flows)
    assert flows["hexagon"].backend == "hexagon"
