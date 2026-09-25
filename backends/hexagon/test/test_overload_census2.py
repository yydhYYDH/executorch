# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The families the first overload census listed and did not check.

`test_overload_census.py` ended with a list of families it had not looked at:
the quantized path's overloads, `llama.*`, `et_hexagon.*`, the recurrent
ops, the other producers of a getitem, and `_to_copy` against
`_to_dim_order_copy`. This is that list, one row per claim, in the same shape:
what `to_edge` produces and what the partitioner does with it, with the two
failure directions -- an op that gains an emitter, and an op that starts or
stops being accepted -- both failing the row.

Three of the rows are also the reason the silent-gap diagnostic in
`test_unwired_targets.py` cannot be the whole answer: a per-tensor dequantize and
`aten.amin` are different schema names from the per-channel dequantize and
`aten.amax` the table does speak for, so the census is blind to them by
construction. They are pinned here instead.
"""

import operator


import pytest
import torch


from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.backends.hexagon.quantizer import get_hexagon_quantizer
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

F16 = torch.float16
_GETITEM = "operator.getitem"
_CONFIG = EdgeCompileConfig(_check_ir_validity=False)


class _M(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)


def _x(*shape, dtype=F16):
    return torch.randn(*shape, dtype=dtype)


def _name(target):
    """The name a row uses: the getitem entry serves several producers."""
    return _GETITEM if target is operator.getitem else target.__name__


def _has_emitter(name):
    return any(_name(key) == name for key in hexagon_ops.EMITTERS)


def _edge_targets(model, inputs):
    program = to_edge(
        export(_M(model), inputs), compile_config=_CONFIG
    ).exported_program()
    return [
        _name(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _accepted(model, inputs):
    """The edge targets the partitioner takes, from the support object it builds.

    A bare `HexagonOperatorSupport()` has an empty `data_names`, which makes every
    gate that wants a constant weight stricter than the partitioner is: it can
    report a refusal that does not happen, never an acceptance that does not. A
    weight that is a constant is a fact about the signature, not about the node.
    No row here is affected -- 122 rows of the two census files were compared both
    ways -- but the object is built the way `partition` builds it so that a row
    needing a constant weight cannot quietly measure the wrong thing.
    """
    program = to_edge(
        export(_M(model), inputs), compile_config=_CONFIG
    ).exported_program()
    support = HexagonOperatorSupport(_data_placeholders(program))
    return {
        _name(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and support.is_node_supported({}, node)
    }


def _delegate_count(model, inputs):
    program = to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=_CONFIG,
    ).exported_program()
    return len(
        [
            node
            for node in program.graph_module.graph.nodes
            if node.target is torch.ops.higher_order.executorch_call_delegate
        ]
    )


#: (label, forward, inputs, [(target, verdict)])
#:
#: A verdict is `unwired` (no emitter at all), `refused` (an emitter exists and
#: this node is turned away before it) or `wired` (delegated).
_ROWS = [
    # --- the producers of a getitem the partitioner has a branch for ---------
    (
        "a max pool's values",
        lambda a: torch.nn.functional.max_pool2d(a, 2, 2),
        (_x(1, 64, 8, 8),),
        [("aten.max_pool2d_with_indices.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "a layer norm's output 0",
        lambda a: torch.nn.functional.layer_norm(a, [6]),
        (_x(4, 6),),
        [("aten.native_layer_norm.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "a max over a dim, reading the values",
        lambda a: torch.max(a, dim=1).values,
        (_x(2, 3, 4),),
        [("aten.max.dim", "wired"), (_GETITEM, "wired")],
    ),
    (
        "a topk, reading the values",
        lambda a: torch.topk(a, 1).values,
        (_x(2, 8),),
        [("aten.topk.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "a topk, reading the positions",
        lambda a: torch.topk(a, 1).indices,
        (_x(2, 8),),
        [("aten.topk.default", "refused"), (_GETITEM, "refused")],
    ),
    # --- the producers it does not ------------------------------------------
    (
        "topk's values, for a k the kernel holds no second element for",
        lambda a: torch.topk(a, 2).values,
        (_x(8),),
        # The getitem is accepted on its own -- a getitem is judged without
        # looking at its producer, which is why the pool's dilated row reads the
        # same way -- and the partition is still not formed, because that
        # verdict does not cross a tuple.
        [("aten.topk.default", "refused"), (_GETITEM, "wired")],
    ),
    (
        "sort's values",
        lambda a: torch.sort(a).values,
        (_x(8),),
        [("aten.sort.default", "unwired"), (_GETITEM, "refused")],
    ),
    (
        "a split",
        lambda a: torch.split(a, 2, dim=0)[0],
        (_x(6, 3),),
        [("aten.split_with_sizes_copy.default", "wired"), (_GETITEM, "wired")],
    ),
    (
        "instance norm's two outputs",
        lambda a: torch.nn.functional.instance_norm(a),
        (_x(2, 3, 4),),
        # The exporter flattens the batch into the channel axis, so the op's
        # per-(n, c) statistics are one row of a [N*C][H*W] view -- the span the
        # norm kernel already reduces. The second view is the alias back to the
        # exported shape, and the batch norm's second and third outputs are the
        # op's own accumulators, which nothing here reads.
        [
            ("aten.view_copy.default", "wired"),
            ("aten._native_batch_norm_legit.no_stats", "wired"),
            (_GETITEM, "wired"),
            ("aten.view_copy.default", "wired"),
        ],
    ),
    (
        "group norm's two outputs",
        lambda a: torch.nn.functional.group_norm(
            a, 2, torch.ones(4, dtype=F16), torch.zeros(4, dtype=F16)
        ),
        (_x(2, 4, 3),),
        # One mean and one variance per group over that group's channels and the
        # whole spatial block is one row per (batch, group) of an
        # [N*G][(C/G)*H*W] view. The two fulls are the affinity the expression
        # builds in the graph rather than a parameter; they stay on the portable
        # kernels and reach the delegate as operands, one element per channel.
        [
            ("aten.full.default", "unwired"),
            ("aten.full.default", "unwired"),
            ("aten.native_group_norm.default", "wired"),
            (_GETITEM, "wired"),
        ],
    ),
    # --- the copies ---------------------------------------------------------
    (
        "a cast to fp32",
        lambda a: a.to(torch.float32),
        (_x(2, 3),),
        [("dim_order_ops._to_dim_order_copy.default", "wired")],
    ),
    (
        "aten._to_copy, which the exporter rewrites",
        lambda a: torch.ops.aten._to_copy(a, dtype=torch.float32),
        (_x(2, 3),),
        [("dim_order_ops._to_dim_order_copy.default", "wired")],
    ),
    (
        "a cast out and back",
        lambda a: a.float().half(),
        (_x(2, 3),),
        [
            ("dim_order_ops._to_dim_order_copy.default", "wired"),
            ("dim_order_ops._to_dim_order_copy.default", "wired"),
        ],
    ),
    (
        "a cast into another tensor's dtype",
        lambda a, b: a.to(b),
        (_x(2, 3), _x(2, 3, dtype=torch.float32)),
        [("dim_order_ops._to_dim_order_copy.default", "wired")],
    ),
    (
        "a cast to channels_last",
        lambda a: a.to(memory_format=torch.channels_last),
        (_x(1, 4, 3, 3),),
        [("dim_order_ops._to_dim_order_copy.default", "refused")],
    ),
    (
        "a clone",
        lambda a: a.clone(),
        (_x(2, 3),),
        [("dim_order_ops._clone_dim_order.default", "wired")],
    ),
    (
        "a clone into channels_last",
        lambda a: torch.clone(a, memory_format=torch.channels_last),
        (_x(1, 4, 3, 3),),
        [("dim_order_ops._clone_dim_order.default", "refused")],
    ),
]


def _assert_rows(model, inputs, rows):
    """The row's claims, in the two directions a coverage change can move."""
    assert _edge_targets(model, inputs) == [
        name for name, _ in rows
    ], "the graph this expression produces has changed"
    accepted = _accepted(model, inputs)
    for name, verdict in rows:
        if verdict == "unwired":
            assert not _has_emitter(name), f"{name} now has an emitter"
            assert name not in accepted, f"{name} is being delegated now"
        elif verdict == "refused":
            assert _has_emitter(name), f"{name} has no emitter, so it is unwired"
            assert name not in accepted, f"{name} started accepting this shape"
        else:
            assert _has_emitter(name), f"{name} lost its emitter"
            assert name in accepted, f"{name} is no longer delegated"


@pytest.mark.parametrize(
    "label,forward,inputs,rows", _ROWS, ids=[row[0] for row in _ROWS]
)
def test_each_row_matches_its_recorded_verdict(label, forward, inputs, rows):
    """One row at a time, so a failure names the family that moved."""
    _assert_rows(_M(forward), inputs, rows)


class _Mm(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n) * 0.3)

    def forward(self, x):
        return torch.mm(x, self.weight)


def _both_readers(a):
    """One max over a dim whose values and indices both leave the node."""
    values, indices = torch.max(a, dim=1)
    return values + indices.to(F16)


def test_a_two_output_max_with_both_readers_keeps_the_whole_node_on_the_host():
    """What the getitem rule accepts, and what EXIR then does with it.

    The predicate accepts a getitem 0 whether or not its producer was accepted:
    the producer's own check is what refuses a bad source, so the reader is judged
    the same either way. That is only a predicate verdict, and a getitem whose
    input is a multi-output value never becomes a partition of its own --
    `CapabilityBasedPartitioner` does not form one across a tuple -- so the values
    reader here stays on the host with the node it reads. Both halves are pinned,
    because the gap between them is where "the support check said yes" reasoning
    goes wrong.
    """
    inputs = (torch.randn(2, 3, 4, dtype=F16),)
    model = _M(_both_readers)
    support = HexagonOperatorSupport()
    program = to_edge(export(model, inputs), compile_config=_CONFIG).exported_program()
    calls = [
        node for node in program.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [_name(node.target) for node in calls] == [
        "aten.max.dim",
        _GETITEM,
        _GETITEM,
        "dim_order_ops._to_dim_order_copy.default",
        "aten.add.Tensor",
    ]
    # Two getitems, and only the values one is accepted: the indices are a second
    # output of the same node.
    assert [support.is_node_supported({}, node) for node in calls] == [
        False,
        True,
        False,
        False,
        True,
    ]
    # The one delegate left is the add, and the two tensors it takes arrive from
    # the host.
    assert _delegate_count(model, inputs) == 1


class _Linear(torch.nn.Module):
    def __init__(self, k, n, bias_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k) * 0.3)
        self.bias = torch.nn.Parameter(torch.randn(*bias_shape) * 0.1)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _quantized_model(module, scheme, inputs):
    """The module through the real quantizer, as `quantizer.py` is meant to be used."""
    prepared = prepare_pt2e(
        export(module, inputs).module(), get_hexagon_quantizer(scheme)
    )
    with torch.no_grad():
        prepared(*inputs)
    return convert_pt2e(prepared)


#: (label, scheme, m, k, n, [(target, verdict)])
#:
#: The weight-only GEMV kernels take rank-two operands with `K` a multiple of 64
#: and `N` a multiple of 32, and only a single-row activation: everything else
#: falls back with both halves of the pair, which is what makes one gate enough.
_QUANTIZED_ROWS = [
    (
        "q4a16 at the GEMV geometry",
        "q4a16",
        1,
        64,
        32,
        [
            ("quantized_decomposed.dequantize_per_channel.default", "wired"),
            ("aten.mm.default", "wired"),
        ],
    ),
    (
        "w8a16 at the GEMV geometry",
        "w8a16",
        1,
        64,
        32,
        [
            ("quantized_decomposed.dequantize_per_channel.default", "wired"),
            ("aten.mm.default", "wired"),
        ],
    ),
    (
        # This row used to be the pair's refused case: the weight-only path was
        # M == 1 only, so a prompt longer than one token left every matmul on the
        # portable kernels. The q4a16 and w8a16 prefill entries now cover the
        # admitted M > 1 geometry.
        "q4a16 with more than one row",
        "q4a16",
        2,
        64,
        32,
        [
            ("quantized_decomposed.dequantize_per_channel.default", "wired"),
            ("aten.mm.default", "wired"),
        ],
    ),
    (
        # The int8 prefill entry is what makes this row delegate, so the
        # quantizer now asks a question it can answer yes to and annotates the
        # dequantize. The row used to record the pair as refused, back when there
        # was no w8a16 prefill entry and this shape kept the fp16 matmul it had
        # rather than carry a dequantize the partitioner would refuse.
        "w8a16 with more than one row",
        "w8a16",
        2,
        64,
        32,
        [
            ("quantized_decomposed.dequantize_per_channel.default", "wired"),
            ("aten.mm.default", "wired"),
        ],
    ),
    (
        # Same reason: 32 is not a multiple of the 64-element block, so the
        # annotation is never made and the fp16 matmul is what the graph holds.
        "q4a16 with K not a multiple of 64",
        "q4a16",
        1,
        32,
        32,
        [
            ("aten.mm.default", "wired"),
        ],
    ),
    (
        # And 16 is not a multiple of the 32-channel tile, which is the shape a
        # classifier's `nn.Linear(k, 1000)` head has.
        "q4a16 with N not a multiple of 32",
        "q4a16",
        1,
        64,
        16,
        [
            ("aten.mm.default", "wired"),
        ],
    ),
]


@pytest.mark.parametrize(
    "label,scheme,m,k,n,rows", _QUANTIZED_ROWS, ids=[row[0] for row in _QUANTIZED_ROWS]
)
def test_each_quantized_row_matches_its_recorded_verdict(label, scheme, m, k, n, rows):
    """The weight-only path, through `prepare_pt2e`/`convert_pt2e`.

    The pair is judged together on purpose: the dequantize and the matmul are one
    pattern, and a row that accepted one and refused the other would be a graph
    where the weight is dequantized on the host and the matmul still runs on the
    DSP -- which is a different (and much slower) program than either verdict
    alone suggests.
    """
    model = _quantized_model(_Mm(k, n).eval(), scheme, (torch.randn(m, k),))
    inputs = (torch.randn(m, k),)
    _assert_rows(model, inputs, rows)


@pytest.mark.parametrize("bias_shape", [(32,), (1, 32)])
def test_the_quantized_linear_folds_its_weight_into_the_matmul(bias_shape):
    """A bias is fine on the `nn.Linear` path, and the permute is gone.

    `torch.nn.functional.linear` over a quantized weight used to reach `addmm`
    with the weight's projection permuted in: two targets, neither of them
    quantized, which is how this row passed while nothing was being quantized at
    all. The quantizer now rewrites the node into the `addmm` spelling over a
    `[k, n]` constant before the observers go in, so the dequantize lands on the
    matmul's weight operand and the permute is not a node here -- two targets,
    both of them the quantized pattern's. The dequantize emits nothing; it exists
    so the weight arrives as the pattern the emitter matches. Both bias shapes
    the kernel can repeat -- one per output channel, and a broadcast row -- are
    accepted.
    """
    k, n = 64, 32
    model = _quantized_model(
        _Linear(k, n, bias_shape).eval(), "q4a16", (torch.randn(1, k),)
    )
    inputs = (torch.randn(1, k),)
    assert _edge_targets(model, inputs) == [
        "quantized_decomposed.dequantize_per_channel.default",
        "aten.addmm.default",
    ]
    assert _accepted(model, inputs) == {
        "quantized_decomposed.dequantize_per_channel.default",
        "aten.addmm.default",
    }


class _PerTensorWeight(Quantizer):
    """The annotation a quantizer of one's own writes for a per-tensor weight.

    `quantizer.py` is the backend's setup, and it always annotates the weight per
    channel; a user running their own PT2E recipe reaches the same matmul with a
    per-tensor dequantize instead, which is a different op with a different name.
    """

    def annotate(self, model):
        for node in model.graph.nodes:
            if (
                node.op != "call_function"
                or node.target is not torch.ops.aten.mm.default
            ):
                continue
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={
                    node.args[1]: QuantizationSpec(
                        dtype=torch.int8,
                        quant_min=-128,
                        quant_max=127,
                        qscheme=torch.per_tensor_symmetric,
                        is_dynamic=False,
                        observer_or_fake_quant_ctr=MinMaxObserver,
                    )
                },
                output_qspec=None,
            )
        return model

    def validate(self, model):
        pass


def _per_tensor_model(module, inputs):
    """The same module through a quantizer that annotates the weight per tensor."""
    prepared = prepare_pt2e(export(module, inputs).module(), _PerTensorWeight())
    with torch.no_grad():
        prepared(*inputs)
    return convert_pt2e(prepared)


def test_the_per_tensor_dequantize_is_a_gap_the_census_cannot_see():
    """The blind spot the family rule has, pinned rather than left as a caveat.

    The table has `quantized_decomposed.dequantize_per_channel` and nothing else
    from that namespace, and the census groups by schema name -- so
    `dequantize_per_tensor` is a family the table does not speak for at all and is
    counted nowhere. What a user sees is a model that still delegates its matmul
    and no longer has a weight-only GEMV: the dequantize runs on the host, the
    weight reaches the DSP as an operand, and nothing says so.
    """
    inputs = (torch.randn(1, 64),)
    model = _per_tensor_model(_Mm(64, 32).eval(), inputs)
    assert _edge_targets(model, inputs) == [
        "quantized_decomposed.dequantize_per_tensor.default",
        "aten.mm.default",
    ]
    # The matmul is delegated and the dequantize is not: that difference is the
    # one the census would have to see to report this row.
    assert _accepted(model, inputs) == {"aten.mm.default"}
    assert not _has_emitter("quantized_decomposed.dequantize_per_tensor.default")
    assert _has_emitter("quantized_decomposed.dequantize_per_channel.default")

    reset_unwired_overload_census()
    assert _delegate_count(model, inputs) == 1
    assert unwired_overload_census() == {}


#: (label, module, the op the raw export carries, refused targets, delegates)
#:
#: The raw program's targets are torch ops rather than edge ops, so their names
#: are unqualified (`lstm.input`, not `aten.lstm.input`); `_name` reports them as
#: the graph holds them, which is what the row states.
_RECURRENT_ROWS = [
    (
        "LSTM",
        lambda: _Lstm(),
        "lstm.input",
        {
            "aten.cat.default": 1,
            "aten.expand_copy.default": 1,
            "aten.full.default": 2,
        },
        3,
    ),
    (
        "RNN",
        lambda: _Rnn(),
        "rnn_tanh.input",
        {"aten.cat.default": 1, "aten.expand_copy.default": 1, "aten.full.default": 1},
        3,
    ),
    (
        "GRU",
        lambda: _Gru(),
        "gru.input",
        {
            "aten.cat.default": 1,
            "aten.expand_copy.default": 1,
            "aten.full.default": 1,
        },
        3,
    ),
]


class _Lstm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(4, 5, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class _Rnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(4, 5, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class _Gru(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(4, 5, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


@pytest.mark.parametrize(
    "label,make,input_op,refused,delegates",
    _RECURRENT_ROWS,
    ids=[row[0] for row in _RECURRENT_ROWS],
)
def test_the_recurrent_ops_are_unrolled_before_the_partitioner(
    label, make, input_op, refused, delegates
):
    """`aten.lstm.input` never reaches the partitioner: it is unrolled first.

    What is delegated is the per-step arithmetic the unrolling leaves behind, and
    what stops it is the step boundary: the `cat` that gathers the steps, the
    `full` the initial state became, and the `expand` beside them. The split of
    the fused weight matrix into gates used to be a cut point as well -- it was
    the reader that made the getitem rule a rule, with twelve to eighteen refused
    getitems behind it -- and is now a blit per piece on the DSP instead, which is
    why the LSTM and GRU rows have half the delegates they had and no split or
    getitem left to refuse. The counts are here so a change in how much of a
    recurrent model is on the DSP is a failed row rather than a number nobody
    watches.
    """
    model = make().eval()
    inputs = (torch.randn(2, 3, 4),)
    raw = export(model, inputs)
    assert input_op in [
        _name(node.target) for node in raw.graph.nodes if node.op == "call_function"
    ]
    assert _delegate_count(model, inputs) == delegates

    support = HexagonOperatorSupport()
    program = to_edge(raw, compile_config=_CONFIG).exported_program()
    counts = {}
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if not support.is_node_supported({}, node):
            name = _name(node.target)
            counts[name] = counts.get(name, 0) + 1
    assert counts == refused


def _et_hexagon_ops():
    """Every `et_hexagon` op this checkout defines, by the name the table uses.

    Two sources, because neither is complete on its own: the namespace only
    carries the packets that have been looked up, and the module's constants only
    the ops that were named.
    """
    names = set()
    namespace = exir_ops.edge.et_hexagon
    for base in dir(namespace):
        if base.startswith("_"):
            continue
        packet = getattr(namespace, base)
        for overload in getattr(packet, "_overload_names", None) or ():
            names.add(getattr(packet, overload).__name__)
    for attribute in dir(hexagon_ops):
        name = getattr(getattr(hexagon_ops, attribute), "__name__", None)
        if isinstance(name, str) and name.startswith("et_hexagon."):
            names.add(name)
    return names


def test_every_et_hexagon_op_this_checkout_defines_has_an_emitter():
    """The family is this backend's own, so there is no reason for a gap in it.

    These eight ops exist to be fused into one DSP command each; a `et_hexagon`
    op without an emitter would be a fusion pass producing something the backend
    cannot run. The first census listed the family as unchecked; this is the
    check, and it is the kind that fails when a new fused op lands without one.
    """
    names = _et_hexagon_ops()
    assert len(names) == 10, f"the et_hexagon namespace changed: {sorted(names)}"
    assert not [name for name in sorted(names) if not _has_emitter(name)]


def test_attention_has_no_emitter_until_the_extension_registers_one():
    """Why the `llama.*` family cannot be measured here, and what is pinned.

    The LLM extension is a C++ library this checkout does not build, so
    `llama.custom_sdpa` does not exist in the dialect, `sdpa_targets()` is empty
    and the only `llama` op present is `llama.fallback` -- which is a marker the
    exporter writes for a node it deliberately kept on the host, not an op with
    a command. The mask decision that *is* checkable without hardware is pinned
    in `test_sdpa_mask.py`, which lowers graphs carrying a stand-in of the same
    schema through the emitter table and reads the command, the numbers and the
    refusals back out. That stand-in lives in a namespace of its own, so the
    census above does not move when that file has run.
    """
    assert hexagon_ops.sdpa_targets() == frozenset()
    assert [key for key in hexagon_ops.EMITTERS if "llama" in str(key)] == []
    packet = getattr(exir_ops.edge.llama, "fallback")
    assert sorted(getattr(packet, "_overload_names", ())) == ["", "out"]
    assert not _has_emitter("llama.fallback.default")
    assert not _has_emitter("llama.custom_sdpa.default")
    assert not _has_emitter("llama.custom_sdpa.out")
