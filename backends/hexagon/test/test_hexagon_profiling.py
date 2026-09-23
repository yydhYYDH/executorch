# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the runtime needs to turn a DSP command into a graph node.

A profile of a delegated subgraph is a list of commands, and the only thing that
makes it readable is the map from command index to debug handle that preprocess
hands to EXIR. These tests pin that map, that producing it does not change the
blob, that a dump of one resolves in the code that reads dumps, and the probe
layout the two C++ sides have to agree on for the per-command times to arrive at
all.
"""

import json
import os
import pathlib
import re
import struct
import sys

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.devtools.etdump.schema_flatcc import (  # noqa: E402
    ETDumpFlatCC,
    Event as FlatEvent,
    ProfileEvent,
    RunData,
)
from executorch.devtools.inspector import EventBlock  # noqa: E402
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.backend.backend_api import LoweredBackendModule  # noqa: E402
from torch.export import export  # noqa: E402

_HEXAGON = pathlib.Path(__file__).resolve().parents[1]
_RUNTIME_SOURCE = _HEXAGON / "runtime" / "hexagon_backend.cpp"
_DSP_SOURCE = _HEXAGON / "third-party" / "mnn-htp-ops" / "src" / "dsp" / "execute_command.cc"


class _Mm(torch.nn.Module):
    """One command per node: a matmul then a bias add."""

    def forward(self, x, w, b):
        return torch.mm(x, w) + b


class _LayerNorm(torch.nn.Module):
    """A node the emitter lowers to more than one command."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (8,), self.weight, self.bias, 1e-5)


def _mm_args():
    return (
        torch.randn(4, 8, dtype=torch.float16),
        torch.randn(8, 6, dtype=torch.float16),
        torch.randn(6, dtype=torch.float16),
    )


def _command_count(blob: bytes) -> int:
    # n_ops is the third int of HexagonBlobHeader.
    return struct.unpack_from("<9I", blob, 0)[2]


def _handles(program) -> list:
    """The subgraph's debug handles, in graph order, one per node."""
    return [
        node.meta["debug_handle"]
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.meta.get("debug_handle") is not None
    ]


def test_a_command_is_numbered_by_the_handle_of_the_node_that_asked_for_it():
    """The runtime logs command i as delegate debug identifier i.

    So command i has to be the one node i's emitter added, in the order the
    emitters ran: that index is the whole linkage between a per-op event and the
    node it belongs to.
    """
    program = to_edge(export(_Mm(), _mm_args())).exported_program()
    result = HexagonBackend.preprocess(program, [])
    assert result.debug_handle_map == {
        0: (_handles(program)[0],),
        1: (_handles(program)[1],),
    }
    assert _command_count(result.processed_bytes) == 2


def test_a_node_that_emits_several_commands_appears_once_per_command():
    """Two commands from one node are two identifiers pointing at one handle.

    Which is what the handle means: the commands a single aten op decomposes
    into are all that op's time.
    """
    program = to_edge(export(_LayerNorm(), (torch.randn(4, 8, dtype=torch.float16),))).exported_program()
    result = HexagonBackend.preprocess(program, [])
    commands = _command_count(result.processed_bytes)
    assert commands > 1
    assert sorted(result.debug_handle_map) == list(range(commands))
    handle = _handles(program)[0]
    assert set(result.debug_handle_map.values()) == {(handle,)}


def test_the_map_does_not_reach_the_blob():
    """The mapping travels in the ETRecord, not in the .pte.

    A blob is what the device reads, and a command index is stable without any
    handle in it, so carrying handles in the blob would buy nothing and change
    the wire format. The strongest way to say that is that the same graph
    lowered with handles and without produces the same bytes.
    """
    program = to_edge(export(_Mm(), _mm_args())).exported_program()
    with_handles = HexagonBackend.preprocess(program, [])
    assert with_handles.debug_handle_map != {}
    for node in program.graph_module.graph.nodes:
        node.meta.pop("debug_handle", None)
    without_handles = HexagonBackend.preprocess(program, [])
    assert without_handles.debug_handle_map == {}
    assert with_handles.processed_bytes == without_handles.processed_bytes


def test_a_partitioned_subgraph_is_profiled_by_the_map_it_records():
    """The whole AOT path: the partitioner's subgraph, lowered and inspected.

    What matters here is that the map survives into the lowered module's meta,
    which is where EXIR reads it from when it assembles the ETRecord.
    """
    lowered = to_edge_transform_and_lower(
        export(_Mm(), _mm_args()),
        partitioner=[HexagonPartitioner()],
    )
    modules = [
        module
        for _, module in lowered.exported_program().graph_module.named_modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(modules) == 1
    mapping = modules[0].meta["debug_handle_map"]
    assert sorted(mapping) == list(range(_command_count(modules[0].processed_bytes)))
    assert all(isinstance(handle, int) for (handle,) in mapping.values())


def _constants(source: pathlib.Path) -> dict:
    """Every `constexpr int NAME = EXPR;` in a C++ file, evaluated.

    The expressions are plain arithmetic over ints and earlier names, which is
    all this needs: the point is to compare two files, not to parse C++.
    """
    values: dict = {}
    text = re.sub(r"//[^\n]*", "", source.read_text())
    for name, expression in re.findall(
        r"constexpr\s+(?:int|uint32_t|int32_t|int64_t|size_t)\s+(\w+)\s*=\s*([^;]+);",
        text,
    ):
        try:
            values[name] = int(eval(expression, {}, values))
        except (NameError, TypeError, ValueError):
            # An initializer over something this cannot see, such as a sizeof or
            # another file's constant. Nothing under test needs it.
            continue
    return values


def test_the_host_and_the_dsp_agree_on_the_probe_layout():
    """The two sides write and read the same buffer with no shared header.

    Nothing but these numbers ties them together, so a change to one that is not
    made to the other is either a record read at the wrong offset or, worse, a
    duration read out of the wrong record.
    """
    host = _constants(_RUNTIME_SOURCE)
    dsp = _constants(_DSP_SOURCE)

    # The host calls the base offset kProbeBaseInts; execute_command.cc calls it
    # kProbeBase. Everything else shares a name.
    assert host["kProbeBaseInts"] == dsp["kProbeBase"]
    assert host["kProbeMagic"] == dsp["kProbeMagic"]
    for name in (
        "kProbeHeaderInts",
        "kProbeRecordInts",
        "kProbeStages",
        "kProbeRecordTimeInt",
        "kProbeHeaderVersionInt",
        "kProbeVersionCommandTime",
    ):
        assert host[name] == dsp[name], name

    # The host's buffer has to be big enough for every region the DSP writes
    # into, or the DSP silently drops the ones that do not fit: records are
    # capped at kProbeMaxRecords and the stage region starts after them.
    stage_offset = (
        dsp["kProbeHeaderInts"] + dsp["kProbeMaxRecords"] * dsp["kProbeRecordInts"]
    )
    dsp_words = dsp["kProbeBase"] + stage_offset + dsp["kProbeStages"] * 4
    assert host["kProbeRecords"] >= dsp["kProbeMaxRecords"]
    assert dsp_words * 4 <= host["kProbeBytes"]

    # The written record has to be one the host reads whole.
    assert dsp["kProbeRecordTimeInt"] < host["kProbeRecordInts"]


#: The instruction id the delegate's node ends up with in the lowered program.
#: Any id works here; what matters is that both maps key on it.
_DELEGATE_INSTRUCTION_ID = 3

#: What ETDump leaves an unset identifier at, and what it leaves an unset
#: string at. A profile event carries one or the other, never both.
_UNSET_ID = -1
_UNSET_NAME = ""


def _lowered_module(model, args):
    lowered = to_edge_transform_and_lower(
        export(model, args),
        partitioner=[HexagonPartitioner()],
    )
    modules = [
        module
        for _, module in lowered.exported_program().graph_module.named_modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(modules) == 1
    return modules[0]


def _profile_event(
    instruction_id: int,
    start_time: int,
    end_time: int,
    delegate_index: int = _UNSET_ID,
    delegate_name: str = _UNSET_NAME,
    metadata: bytes = b"",
) -> FlatEvent:
    """One event, shaped the way ETDumpGen writes it.

    Neither kind of delegate event carries a name of its own: ETDumpGen adds the
    identifier instead, and it is the Inspector that calls a delegated event by
    it. So an event for one command has an integer identifier and no name, and
    an event for a phase has a string one.
    """
    return FlatEvent(
        profile_event=ProfileEvent(
            name=_UNSET_NAME,
            chain_index=0,
            instruction_id=instruction_id,
            delegate_debug_id_int=delegate_index,
            delegate_debug_id_str=delegate_name,
            delegate_debug_metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        ),
        allocation_event=None,
        debug_event=None,
    )


def _etdump(events) -> ETDumpFlatCC:
    return ETDumpFlatCC(
        version=1,
        run_data=[
            RunData(
                name="Execute",
                bundled_input_index=0,
                allocators=[],
                events=events,
            )
        ],
    )


def _maplike_exir_writes(module) -> dict:
    """The delegate map the way it comes back out of an ETRecord.

    EXIR writes debug_handle_map into the record and the Inspector reads it back
    as JSON, which stringifies the identifiers and turns the handles into lists.
    Both are visible to the lookup, so both are what these tests assert on.
    """
    return json.loads(json.dumps(module.meta["debug_handle_map"]))


def _delegate_map(module) -> dict:
    return {
        str(_DELEGATE_INSTRUCTION_ID): {
            "name": module.backend_id,
            "delegate_map": _maplike_exir_writes(module),
        }
    }


def _handle_map(module) -> dict:
    return {
        str(_DELEGATE_INSTRUCTION_ID): sorted(
            {
                handle
                for handles in module.meta["debug_handle_map"].values()
                for handle in handles
            }
        )
    }


def _resolved_events(etdump, module) -> dict:
    blocks = EventBlock._gen_from_etdump(etdump)
    assert len(blocks) == 1
    blocks[0]._gen_resolve_debug_handles(_handle_map(module), _delegate_map(module))
    return {event.name: event for event in blocks[0].events}


def test_a_per_op_event_resolves_to_the_node_the_map_names():
    """The events the runtime logs, read back by the code that reads dumps.

    An event for one command carries an identifier and nothing else, so the node
    comes back only through the delegate map -- as the string the ETRecord
    round-tripped it through. The duration is in nanoseconds, the unit the event
    is logged in.
    """
    module = _lowered_module(_Mm(), _mm_args())
    command_map = _maplike_exir_writes(module)
    microseconds = {index: 11 - index for index in range(len(command_map))}
    events = [
        _profile_event(
            _DELEGATE_INSTRUCTION_ID,
            start_time=1_000_000 + 1000 * index,
            end_time=1_000_000 + 1000 * index + microseconds[index] * 1000,
            delegate_index=index,
            metadata=struct.pack("<iii", 38, microseconds[index], 0),
        )
        for index in microseconds
    ]
    resolved = _resolved_events(_etdump(events), module)

    for index in microseconds:
        event = resolved[str(index)]
        handles = command_map[str(index)]
        assert event.is_delegated_op
        assert event.debug_handles == handles
        assert event.delegate_backend_name == "HexagonBackend"
        assert event.perf_data.avg == pytest.approx(microseconds[index] * 1000)
        assert event.raw_delegate_debug_metadatas == [
            struct.pack("<iii", 38, microseconds[index], 0)
        ]


def test_a_phase_event_is_named_but_belongs_to_no_node():
    """The phase events carry a string identifier, and nothing resolves it.

    They describe the delegate call rather than a node, so they must read as
    named rows with no handles rather than as rows attributed to the whole
    subgraph.
    """
    module = _lowered_module(_Mm(), _mm_args())
    command_map = module.meta["debug_handle_map"]
    per_op_events = [
        _profile_event(
            _DELEGATE_INSTRUCTION_ID,
            start_time=1000 * index,
            end_time=1000 * index + 7000,
            delegate_index=index,
        )
        for index in range(len(command_map))
    ]
    call = _profile_event(
        _DELEGATE_INSTRUCTION_ID,
        start_time=0,
        end_time=18_000,
        delegate_name="HEXAGON_DSP_CALL",
        metadata=struct.pack("<ii", 38, 18),
    )
    resolved = _resolved_events(_etdump(per_op_events + [call]), module)
    assert set(resolved) == {
        *[str(index) for index in range(len(command_map))],
        "HEXAGON_DSP_CALL",
    }

    event = resolved["HEXAGON_DSP_CALL"]
    assert event.is_delegated_op
    assert event.debug_handles is None
    assert event.perf_data.avg == pytest.approx(18_000)
    assert event.raw_delegate_debug_metadatas == [struct.pack("<ii", 38, 18)]
