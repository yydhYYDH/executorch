# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Hexagon backend's entry point into the shared backend test suite.

This mirrors ``backends/xnnpack/test/tester`` and ``backends/qualcomm/tests/tester``:
it specializes the lowering stages so the shared suite lowers through
``HexagonPartitioner``, and it imports without a Hexagon SDK or a device.

Execution policy
----------------
The suite's run stage loads the serialized program into the in-process portable
pybindings, which is a path this backend does not have. A Hexagon program calls
into the delegate, whose kernels are HVX/HMX code built for one Hexagon arch and
whose host side opens a FastRPC session through ``libcdsprpc``; both live on the
device. Neither the SDK nor a device is needed to *lower* a case, so a case
reaches the run stage and skips there with the reason below: everything up to and
including serialization still runs, which is what this backend can assert
without a DSP.

A backend is handed cases by a flow, which is registered in the suite's own
directory rather than here, so nothing in ``backends/hexagon`` instantiates this
class. It is the backend half: the stages below are the ones a flow has to lower
through.

Executing a case instead of skipping it needs a runner that can load the ``.pte``
with the delegate registered and the matching skel on the device, which nothing
in this repository builds or drives. ``Serialize.run_artifact`` is the one place
that would hand the program over.
"""

from typing import Any, List, Optional, Tuple

import executorch.backends.test.harness.stages as BaseStages
import pytest
import torch
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import Partitioner

EXECUTION_SKIP_REASON = (
    "Hexagon programs run on the DSP, so a case cannot be executed here: the "
    "kernels are HVX/HMX code built for one Hexagon arch, and the host side "
    "opens a FastRPC session through libcdsprpc, which the suite's in-process "
    "portable pybindings and a host CPU each lack. The lowering stages ran. "
    "Running a case needs a device with a matching skel and a runner built "
    "with EXECUTORCH_BUILD_HEXAGON=ON."
)


class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(partitioner=partitioner or HexagonPartitioner())


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        # default_partitioner_cls is what registers the partitioner at all: the
        # base stage only reads its partitioners list when that argument is set,
        # so passing `partitioners=[HexagonPartitioner()]` alone leaves the stage
        # with none and every case lowers undelegated without saying so.
        super().__init__(
            default_partitioner_cls=HexagonPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config,
        )


class Serialize(BaseStages.Serialize):
    """Serializes the program, then refuses to run it in this process.

    ``BaseStages.Serialize.run`` stores the buffer and ``dump_artifact`` writes
    it, so the serialization itself is real work that a case can fail on. Only
    ``run_artifact``, the part that loads the program into the running process,
    is not available here (see the module docstring). Skipping rather than
    returning keeps a case that was never executed out of the pass column.
    """

    def run_artifact(self, inputs):
        pytest.skip(EXECUTION_SKIP_REASON)


class HexagonTester(TesterBase):
    __test__ = False

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        **kwargs,
    ):
        stage_classes = TesterBase.default_stage_classes() | {
            StageType.PARTITION: Partition,
            StageType.SERIALIZE: Serialize,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
        }

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
            **kwargs,
        )
