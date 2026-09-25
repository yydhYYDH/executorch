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
import subprocess

import tempfile

import executorch

import pytest
import torch


from executorch.backends.hexagon.hexagon_backend import HexagonBackend
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.devtools.etdump.schema_flatcc import (
    ETDumpFlatCC,
    Event as FlatEvent,
    ProfileEvent,
    RunData,
)
from executorch.devtools.etrecord import parse_etrecord
from executorch.devtools.inspector import EventBlock, Inspector
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import LoweredBackendModule
from torch.export import export

_HEXAGON = pathlib.Path(__file__).resolve().parents[1]
_RUNTIME_SOURCE = _HEXAGON / "runtime" / "hexagon_backend.cpp"
_DSP_SOURCE = (
    _HEXAGON / "third-party" / "mnn-htp-ops" / "src" / "dsp" / "execute_command.cc"
)


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
    program = to_edge(
        export(_LayerNorm(), (torch.randn(4, 8, dtype=torch.float16),))
    ).exported_program()
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


def _require_a_serializable_dump():
    """Skip when this tree cannot write a dump, which needs two things.

    The etdump schema is compiled by flatc, and the two schemas under
    exir/_serialize are generated by the build. An installed tree has all three;
    a fresh worktree has none of them, and no dump can be produced there.
    """
    from executorch.devtools.etdump.serialize import serialize_to_etdump_flatcc

    schemas = pathlib.Path(executorch.exir.__file__).parent / "_serialize"
    for name in ("program.fbs", "scalar_type.fbs"):
        if not (schemas / name).is_file():
            pytest.skip(f"{name} is written by the build; this needs an install")
    # No flatc, or one that cannot build this schema, says nothing about the
    # backend, so both are skips rather than failures.
    try:
        serialize_to_etdump_flatcc(ETDumpFlatCC(version=0, run_data=[]))
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"flatc cannot build the etdump schema here: {error}")
    return serialize_to_etdump_flatcc


def test_the_offline_pipeline_reports_a_node_per_command():
    """The AOT half of a profiling run, minus the device.

    A partitioned model is lowered to a .pte and an ETRecord, the events the
    runtime would log are serialized into a real dump, and the Inspector reads
    the two back together. What comes out is the point of the whole thing: one
    row per DSP command, each attributed to the node it came from, with the
    phases around them.
    """
    serialize = _require_a_serializable_dump()
    lowered = to_edge_transform_and_lower(
        export(_Mm(), _mm_args()),
        partitioner=[HexagonPartitioner()],
        generate_etrecord=True,
    )
    program = lowered.to_executorch()
    assert program.buffer  # the .pte the device would be handed
    etrecord = program.get_etrecord()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.etrecord")
        etrecord.save(path)

        # The expectations come from the record as the Inspector reads it, not
        # from the object in memory: the maps are keyed by integer until they go
        # through the record's JSON, and the resolution under test is on the
        # parsed keys. Right here is the round trip this test exists for.
        reloaded = parse_etrecord(path)
        graph = next(iter(reloaded._delegate_map))
        instruction_id = int(next(iter(reloaded._delegate_map[graph])))
        command_map = reloaded._delegate_map[graph][str(instruction_id)]["delegate_map"]
        assert command_map

        durations = {index: 11 - index for index in range(len(command_map))}
        events = [
            _profile_event(
                instruction_id,
                start_time=1_000_000,
                end_time=1_030_000,
                delegate_name="HEXAGON_DSP_CALL",
                metadata=struct.pack("<ii", 38, 30),
            )
        ]
        for index, microseconds in durations.items():
            events.append(
                _profile_event(
                    instruction_id,
                    start_time=1_000_000 + 1000 * index,
                    end_time=1_000_000 + 1000 * index + microseconds * 1000,
                    delegate_index=index,
                    metadata=struct.pack("<iii", 38, microseconds, 0),
                )
            )
        # A dump is size prefixed -- the length of what follows, which is what
        # the C++ writer emits and the only thing the Inspector reads.
        data = serialize(_etdump(events))
        dump = struct.pack("<I", len(data)) + data
        inspector = Inspector(etdump_data=dump, etrecord=path)

    by_name = {
        event.name: event for block in inspector.event_blocks for event in block.events
    }
    # The rows the runtime logs, named by the model's own graph: each command is
    # a row of its own, resolved to the node that asked for it.
    assert set(by_name) == {*[str(index) for index in durations], "HEXAGON_DSP_CALL"}
    for index, microseconds in durations.items():
        event = by_name[str(index)]
        assert event.debug_handles == command_map[str(index)]
        assert event.delegate_backend_name == "HexagonBackend"
        assert event.perf_data.avg == pytest.approx(microseconds / 1000)

    call = by_name["HEXAGON_DSP_CALL"]
    assert call.is_delegated_op
    assert call.op_types == []
    assert call.perf_data.avg == pytest.approx(0.03)
    assert call.raw_delegate_debug_metadatas == [struct.pack("<ii", 38, 30)]
    # And the rows are attributed to the ops the emitters ran.
    assert sorted(
        op_type
        for name, event in by_name.items()
        if name != "HEXAGON_DSP_CALL"
        for op_type in event.op_types
    ) == ["aten.add.Tensor", "aten.mm.default"]
