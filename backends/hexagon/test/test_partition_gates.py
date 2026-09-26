# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner's refusal gates: one row per clause, and the reason each one gave.

A node the partitioner turns away is a node on the portable kernels, which is a
correct program and a quiet one -- nothing prints, and the only symptom is a
delegate one op shorter than the graph reads. PARTITION_GATES.md is the map of
those clauses; this file is the part of the map that cannot go stale quietly.

Two properties are pinned. First, the table names every ``return False`` in
``HexagonOperatorSupport._verdict``, by set equality, so a clause the tree grows
fails here instead of going unlisted and a line that moved or vanished fails too.
Second, the reason a node was refused is read out of the decision the partitioner
made rather than out of a re-implementation of it: every refusal in ``_verdict``
is rewritten, in the source the module itself carries, so that it records the line
it is on and the node it is deciding. That instrument is what
PARTITION_GATES.md's node counts are a census of, and the controls below are what
say the instrument is reading a gate rather than something that looks like one.

Host tier: export, ``to_edge`` and the support predicate. No kernel runs, and a
passing run says nothing about a phone's DSP.
"""

import inspect
import textwrap
import weakref

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export

F16 = torch.float16
CFG = EdgeCompileConfig(_check_ir_validity=False)

#: (line in hexagon_partitioner.py, short name), in _verdict's own order. The
#: lines are this checkout's; PARTITION_GATES.md is the prose.
GATES = [
    (674, "NOT_CALL_FUNCTION"),
    (680, "TARGET_NOT_IN_TABLE"),
    # Added by the arg-reduction merge. It has to run before
    # RESULT_DTYPE below, which would otherwise reject every
    # int64 index before it reached the row-geometry predicate.
    (688, "ARG_REDUCTION_GEOMETRY"),
    (695, "RESULT_DTYPE"),
    (700, "OPERAND_DTYPES_READABLE"),
    (706, "WHERE_EMITTABLE"),
    (713, "SDPA_FITS_DSP_LIMITS"),
    (715, "BINARY_BROADCAST_FITS"),
    (717, "MM_OPERANDS_FLAT"),
    (719, "BMM_OPERANDS_FLAT"),
    (721, "ADDMM_FITS_FLAT_PATH"),
    (728, "QUANTIZED_MATMUL_REFUSED"),
    (731, "MEAN_REDUCES_ONE_SPAN"),
    (738, "MEAN_RESULT_WIDTH"),
    (742, "POW_IS_SQUARE"),
    (745, "POW_TENSOR_TENSOR_NO_PROGRAM"),
    (753, "POW_TENSOR_TENSOR_EMITTABLE"),
    (759, "REDUCTION_DIMS"),
    (761, "SUM_DIM_EMITTABLE"),
    (763, "MAX_DIM_EMITTABLE"),
    (765, "MIN_DIM_EMITTABLE"),
    (769, "MAX_POOL_EMITTABLE"),
    (776, "TOPK_EMITTABLE"),
    (786, "REPEAT_FLIP_EMITTABLE"),
    (796, "NEAREST_UPSAMPLE_EMITTABLE"),
    (805, "SPLIT_EMITTABLE"),
    (812, "POOL_SPEC"),
    (827, "CONV_SPEC"),
    (836, "GATHER_TABLE"),
    (842, "LAYER_NORM_EPS_CONST"),
    (844, "LAYER_NORM_TRAILING_DIMS"),
    (846, "NATIVE_LAYER_NORM_EMITTABLE"),
    (848, "ADD_RMS_NORM_EMITTABLE"),
    (856, "VISION_ATTENTION_EMITTABLE"),
    (858, "SOFTMAX_INNER_AXIS"),
    (868, "LOG_SOFTMAX_WITHIN_ARENA"),
    (876, "GROUP_NORM_ONE_GROUP_PER_ROW"),
    (882, "BATCH_NORM_ONE_SPAN"),
    (896, "GETITEM_NORM_PRODUCER"),
    (918, "GETITEM_PRODUCER"),
    (923, "ALIAS_BYTES_OR_SELECT"),
    (925, "SLICE_REGION"),
    (927, "CAT_PLAN"),
    (929, "PERMUTE_REGION"),
    (932, "LEAKY_RELU_SLOPE_CONST"),
    (938, "PRELU_SOURCE"),
    (940, "PRELU_SLOPE_RANK"),
    (943, "PRELU_SLOPE_NUMEL"),
    (945, "REFLECT_PAD_REGIONS"),
    (953, "CONSTANT_PAD_REGION"),
    (964, "CLONE_DIM_ORDER_CONTIGUOUS"),
    (968, "DIM_ORDER_KEEPS_BYTES"),
    (970, "UPDATE_CACHE_LAYOUT"),
    (978, "CUMSUM_EMITTABLE"),
    (985, "ARGUMENT_NOT_A_NODE"),
    (997, "GET_ATTR_NOT_FP16"),
]

#: The three the inventory inherits rather than measures, and why. A target
#: outside the emitter table and a target with no emitter are one fact, because
#: SUPPORTED_TARGETS is EMITTERS. The result-dtype clause and the get_attr-width
#: clause are one width rule in two places, because _require_arena_dtype raises
#: where the partitioner falls back.
MEASURED_ELSEWHERE = {
    "TARGET_NOT_IN_TABLE",
    "RESULT_DTYPE",
    "GET_ATTR_NOT_FP16",
    "ARG_REDUCTION_GEOMETRY",
}

_REFUSED = weakref.WeakKeyDictionary()


def _hit(line, node):
    _REFUSED[node] = line


class _Instrumented:
    """_verdict with every refusal tagged by the line it happens on."""

    def __init__(self):
        self.original = hpart.HexagonOperatorSupport._verdict
        self.namespace = None

    def install(self):
        start = self.original.__code__.co_firstlineno
        source = textwrap.dedent(inspect.getsource(self.original))
        rewritten = []
        for offset, line in enumerate(source.splitlines()):
            if line.strip() == "return False":
                line = line.replace(
                    "return False",
                    f"_hit({start + offset}, node); return False",
                )
            rewritten.append(line)
        namespace = dict(vars(hpart))
        namespace["_hit"] = _hit
        exec(compile("\n".join(rewritten), "<gate-instrumented>", "exec"), namespace)
        self.namespace = namespace
        hpart.HexagonOperatorSupport._verdict = namespace["_verdict"]

    def refresh(self):
        # The exec'd body reads module globals through the dict captured at
        # install time, so a caller that patched a gate afterwards would be
        # read past. Re-syncing keeps the instrument a view of the tree.
        self.namespace.update(vars(hpart))

    def remove(self):
        hpart.HexagonOperatorSupport._verdict = self.original


@pytest.fixture
def instrumented():
    tool = _Instrumented()
    tool.install()
    tool.refresh()
    try:
        yield tool
    finally:
        tool.remove()


class _M(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _x(*shape):
    return torch.randn(*shape, dtype=F16)


def _edge(module, inputs):
    program = HexagonPartitioner().transform_for_pre_decomposition(
        export(module, inputs)
    )
    return to_edge(program, compile_config=CFG).exported_program()


def _refusals(module, inputs):
    """{target name: the gate line that refused it} for one exported graph."""
    _REFUSED.clear()
    program = _edge(module, inputs)
    support = HexagonOperatorSupport(_data_placeholders(program), program)
    out = {}
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if support.is_node_supported({}, node):
            continue
        out[getattr(node.target, "__name__", str(node.target))] = _REFUSED.get(node)
    return out


#: One designed case per clause, each with the target it has to land on. Every
#: row here was measured to refuse at the named line before it was written; a
#: clause the tree has since widened is a failing row rather than a silent one.
CASES = [
    (680, "aten.prod.default", _M(lambda a: torch.prod(a)), (_x(2, 3),)),
    (
        695,
        "dim_order_ops._to_dim_order_copy.default",
        _M(lambda a: a.to(torch.int32)),
        (_x(2, 3),),
    ),
    (
        700,
        "dim_order_ops._to_dim_order_copy.default",
        _M(lambda a: (a > 0) + a),
        (_x(2, 3),),
    ),
    (
        706,
        "aten.where.self",
        _M(lambda a: F.scaled_dot_product_attention(a, a, a)),
        (_x(1, 2, 4, 8),),
    ),
    (
        715,
        "aten.add.Tensor",
        _M(lambda a: a + a[:, :1]),
        (_x(1, 2, 3, 4, 5, 6, 7, 8, 9),),
    ),
    (731, "aten.mean.dim", _M(lambda a: torch.mean(a, dim=(0, 2))), (_x(2, 3, 4),)),
    (
        738,
        "aten.mean.dim",
        _M(lambda a: torch.mean(a, dim=1, dtype=torch.float32)),
        (_x(2, 3, 4),),
    ),
    (742, "aten.pow.Tensor_Scalar", _M(lambda a: torch.pow(a, 3)), (_x(2, 3),)),
    (753, "aten.pow.Tensor_Tensor", _M(lambda a: torch.pow(a, _k(2, 3))), (_x(2, 3),)),
    (759, "aten.sum.dim_IntList", _M(lambda a: torch.sum(a, dim=(0, 2))), (_x(2, 3, 4),)),
    (
        761,
        "aten.sum.dim_IntList",
        _M(lambda a: torch.sum(a, dim=1, dtype=torch.float32)),
        (_x(2, 3, 4),),
    ),
    (
        763,
        "aten.max.dim",
        _M(lambda a: torch.max(a, dim=1, keepdim=True)),
        (_x(2, 3, 4),),
    ),
    (
        769,
        "aten.max_pool2d_with_indices.default",
        _M(lambda a: F.max_pool2d(a, 2, return_indices=True)),
        (_x(1, 64, 8, 8),),
    ),
    (776, "aten.topk.default", _M(lambda a: torch.topk(a, 2, dim=1)), (_x(2, 3, 4),)),
    (
        786,
        "aten.repeat.default",
        _M(lambda a: a.repeat(2, 3, 1, 1)),
        (_x(1, 3, 4, 5),),
    ),
    (
        796,
        "aten.upsample_nearest2d.vec",
        _M(lambda a: F.interpolate(a, scale_factor=1.5, mode="nearest")),
        (_x(1, 4, 8, 8),),
    ),
    (
        812,
        "aten.avg_pool2d.default",
        _M(lambda a: F.avg_pool2d(a, 2, ceil_mode=True)),
        (_x(1, 64, 8, 8),),
    ),
    (
        # The table is a graph input, not a parameter: GATHER_TABLE refuses a
        # table whose bytes this layer cannot see at export, and the index width
        # is not what it is refusing -- the host narrows that either way.
        836,
        "aten.embedding.default",
        _M(lambda table, index: F.embedding(index, table)),
        (torch.randn(4, 8, dtype=F16), torch.tensor([0, 2], dtype=torch.int64)),
    ),
    (
        868,
        "aten._log_softmax.default",
        _M(lambda a: torch.log_softmax(a, 1)),
        (_x(2, 70000, 3),),
    ),
    (918, "getitem", _M(lambda a: torch.sort(a)[0]), (_x(4, 3),)),
    (923, "aten.expand_copy.default", _M(lambda a: a.expand(2, 3, 4)), (_x(1, 3, 4),)),
    (925, "aten.slice_copy.Tensor", _M(lambda a: a[0, ::2]), (_x(4, 6),)),
    (
        953,
        "aten.constant_pad_nd.default",
        _M(lambda a: F.pad(a, (1, 1, 1, 1, 1, 1))),
        (_x(2, 3, 4),),
    ),
    (
        964,
        "dim_order_ops._clone_dim_order.default",
        _M(lambda a: a.clone(memory_format=torch.channels_last)),
        (_x(1, 4, 3, 3),),
    ),
    (
        968,
        "dim_order_ops._to_dim_order_copy.default",
        _M(lambda a: a.to(memory_format=torch.channels_last)),
        (_x(1, 4, 3, 3),),
    ),
]


def _k(*shape):
    return torch.randn(*shape, dtype=F16)


def _clause_lines_in_source():
    start = hpart.HexagonOperatorSupport._verdict.__code__.co_firstlineno
    return [
        start + offset
        for offset, line in enumerate(
            textwrap.dedent(inspect.getsource(hpart.HexagonOperatorSupport._verdict))
            .splitlines()
        )
        if line.strip() == "return False"
    ]


def test_the_table_covers_every_clause_in_verdict():
    """Set equality both ways: a clause added fails, and a clause moved fails."""
    in_source = _clause_lines_in_source()
    in_table = [line for line, _ in GATES]
    assert sorted(in_table) == sorted(in_source), (
        "the gate table and _verdict's clauses disagree: "
        f"unlisted {sorted(set(in_source) - set(in_table))}, "
        f"stale {sorted(set(in_table) - set(in_source))}"
    )


def test_the_table_names_each_clause_once_and_leaves_the_measured_three():
    names = [name for _, name in GATES]
    assert len(names) == len(set(names))
    assert MEASURED_ELSEWHERE <= set(names)
    assert len(GATES) - len(MEASURED_ELSEWHERE) == 52, (
        "four clauses are already measured elsewhere; the other 52 are the "
        "inventory"
    )


@pytest.mark.parametrize("line,target,module,inputs", CASES, ids=[str(c[0]) for c in CASES])
def test_a_clause_refuses_at_the_line_the_table_gives_it(instrumented, line, target, module, inputs):
    """Each row has to land on its own line, not merely on a refusal.

    A support object that refused everything would satisfy "the target is
    refused" for every row at once, so each row also carries the control that the
    same graph's ordinary fp16 elementwise chain is accepted -- otherwise a
    broken instrument and a correct one would both go green.
    """
    refusals = _refusals(module, inputs)
    assert target in refusals, f"line {line}: {target} was not refused at all"
    assert refusals[target] == line, (
        f"line {line}: {target} was refused at line {refusals[target]}"
    )


def test_the_control_chains_are_not_refused_by_anything(instrumented):
    """The negative control for every row above, in one place.

    A relu and a residual add at the same dtypes are the shapes the backend does
    run; if the instrument refused them it would be reporting a gate for a node
    the partitioner accepts, and every count in PARTITION_GATES.md would be a
    count of the instrument rather than of the tree.
    """
    for module, inputs in (
        (_M(lambda a: torch.relu(a)), (_x(2, 3),)),
        (_M(lambda a: a + a), (_x(2, 3),)),
        (_M(lambda a: torch.softmax(a, -1)), (_x(2, 3, 4),)),
        (_M(lambda a: a.transpose(0, 1)), (_x(2, 3),)),
    ):
        assert _refusals(module, inputs) == {}, module


def test_every_refused_node_carries_a_reason(instrumented):
    """A refusal with no line is a hole in the census, and a hole reads as a zero."""
    refusals = {}
    for module, inputs in (
        (_M(lambda a: F.max_pool2d(a, 2, return_indices=True)), (_x(1, 64, 8, 8),)),
        (_M(lambda a: torch.sort(a)[0]), (_x(4, 3),)),
        (_M(lambda a: torch.prod(a)), (_x(2, 3),)),
    ):
        refusals.update(_refusals(module, inputs))
    assert refusals, "the control graph refused nothing, so this proves nothing"
    missing = {target: line for target, line in refusals.items() if line is None}
    assert not missing, f"refused with no clause recorded: {missing}"
    assert set(refusals.values()) <= {line for line, _ in GATES}


def test_a_refused_getitem_names_the_producer_it_is_standing_behind(instrumented):
    """The two ends of one island, measured together.

    A sort's getitem is refused because the sort is, and the sort is refused
    because it is not in the table. Reporting the getitem as a gap in its own
    right would be a count of consequence dressed as a count of work.
    """
    program = _edge(_M(lambda a: torch.sort(a)[0]), (_x(4, 3),))
    support = HexagonOperatorSupport(_data_placeholders(program), program)
    by_name = {
        node.name: node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    }
    sort = by_name["aten_sort_default"]
    getitem = by_name["getitem"]
    assert not support.is_node_supported({}, sort)
    assert not support.is_node_supported({}, getitem)
    assert _REFUSED[getitem] == 918
    assert _REFUSED[sort] == 680
