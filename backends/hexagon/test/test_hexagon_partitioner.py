# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner's answers about a node: what it absorbs, and what it refuses."""



import torch
from torch.export import export
from torch.export.dynamic_shapes import Dim


from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    layer_norm_normalizes_the_trailing_dims,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _cast_stays_in_fp16,
    _emits_no_command,
    HexagonPartitioner,
)
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402


def _cast(source_dtype, result_dtype):
    """A _to_copy node with the two dtypes it is judged on."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(4, dtype=source_dtype)
    cast = graph.call_function(
        exir_ops.edge.aten._to_copy.default,
        args=(source,),
        kwargs={"dtype": result_dtype},
    )
    cast.meta["val"] = torch.empty(4, dtype=result_dtype)
    return cast


def test_a_cast_is_absorbed_only_between_the_two_widths_the_arena_holds():
    """The operand is what decides, because the result alone is not enough.

    A cast whose result is fp16 passes the dtype gate by itself, so an int64
    operand would be absorbed and the kernel would read eight-byte integers as
    half floats. The fp16 and fp32 pair is the one the runtime already converts
    at the boundary in both directions, so absorbing it changes no value.
    """
    assert _cast_stays_in_fp16(_cast(torch.float16, torch.float32))
    assert _cast_stays_in_fp16(_cast(torch.float32, torch.float16))
    assert not _cast_stays_in_fp16(_cast(torch.int64, torch.float16))
    assert not _cast_stays_in_fp16(_cast(torch.float16, torch.int64))
    assert _emits_no_command(_cast(torch.float16, torch.float32))
    assert not _emits_no_command(_cast(torch.int64, torch.float16))


def _dim_order(order, dtype=torch.float16, shape=(4, 3)):
    """A dim-order copy node, with the order it is judged on."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=dtype)
    copy = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        args=(source,),
        kwargs={"dim_order": order},
    )
    copy.meta["val"] = torch.empty(shape, dtype=dtype)
    return copy


def test_a_dim_order_copy_is_absorbed_only_in_the_order_the_arena_holds():
    """The order is the whole content of the op, so a named one is not a view.

    The arena holds row-major two-byte elements: an identity order re-reads the
    operand's bytes, with the runtime converting the width at the boundary in
    both directions, while any other order describes a layout those bytes are not
    in. Absorbing that one would hand the consumer the wrong numbers rather than
    fail, which is why the check is on the order and not on the shape.
    """
    assert _emits_no_command(_dim_order([0, 1]))
    assert _emits_no_command(_dim_order(None))
    assert not _emits_no_command(_dim_order([1, 0]))
    assert not _emits_no_command(_dim_order([0, 1], dtype=torch.int64))


def _norm_over_the_input_shape(shape=(2, 4, 8, 16)):
    """A norm whose normalized shape is read off the input it normalizes.

    `normalized_shape` is `SymInt[]`, so a model that writes the norm's shape as
    `x.shape[1:]` hands the argument the `sym_size` nodes the exporter kept for
    those axes rather than numbers.
    """
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=torch.float16)
    normalized = [
        graph.call_function(exir_ops.edge.aten.sym_size.int, args=(source, axis))
        for axis in range(1, len(shape))
    ]
    norm = graph.call_function(
        exir_ops.edge.aten.native_layer_norm.default,
        args=(source, normalized, None, None, 1e-5),
    )
    out = torch.empty(shape, dtype=torch.float16)
    norm.meta["val"] = (out, out[..., 0], out[..., 0])
    return norm


def test_a_shape_the_graph_left_symbolic_is_refused_rather_than_read():
    """The norm's shape can arrive as nodes, and reading one raises in here.

    `int()` over a `sym_size` node raises `TypeError` from inside the partitioner,
    so the node would take the whole lowering down rather than stay on a portable
    kernel the way every other shape this layer cannot see does. The assertion is
    on the argument's kind first, because a literal shape here would make the
    answer below mean nothing.
    """
    norm = _norm_over_the_input_shape()
    assert all(isinstance(size, torch.fx.Node) for size in norm.args[1])
    assert layer_norm_normalizes_the_trailing_dims(norm) is False


class _NormOverItsOwnShape(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), None, None)


def test_the_same_norm_through_a_real_export_keeps_the_lowering_alive():
    """The reachable form of it: mark the axis dynamic and the nodes appear.

    Nothing else about the graph is unusual -- one norm over fp16 -- and the
    export lowers, with the norm left to the portable kernels rather than failing
    the call. A literal shape cannot reach here: the exporter folds `x.shape[-1]`
    into a number unless the axis is dynamic, which is the whole reason this was
    easy to miss.
    """
    example = torch.empty(2, 4, 8, 16, dtype=torch.float16)
    program = export(
        _NormOverItsOwnShape().eval(),
        (example,),
        dynamic_shapes={"x": {3: Dim("d3")}},
    )
    norm = next(
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "layer_norm" in str(node.target)
    )
    assert all(isinstance(size, torch.fx.Node) for size in norm.args[1])
    lowered = to_edge_transform_and_lower(
        program, partitioner=[HexagonPartitioner()]
    ).exported_program()
    assert _delegates(lowered) == []


class _NormOverALiteralShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm((8, 16))

    def forward(self, x):
        return self.norm(x)


def test_a_literal_shape_still_passes_the_predicate():
    """The control: a literal trailing shape is still admitted.

    The lowering of a plain fp32 LayerNorm is refused later, inside the emitter,
    for a reason this predicate does not own -- the affine's broadcast -- so the
    control asserts the half this change is about: the literal shape is read as
    numbers and the answer is yes.
    """
    example = torch.empty(2, 4, 8, 16, dtype=torch.float32)
    program = export(_NormOverALiteralShape().eval(), (example,))
    norm = next(
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "layer_norm" in str(node.target)
    )
    assert all(isinstance(size, int) for size in norm.args[1])
    assert layer_norm_normalizes_the_trailing_dims(norm)


class _NormAndItsNeighbour(torch.nn.Module):
    def forward(self, x):
        return torch.relu(torch.nn.functional.layer_norm(x, (x.shape[-1],), None, None))


def test_the_refusal_leaves_the_rest_of_the_graph_delegating():
    """A refusal is about the node, not about the graph.

    The norm is still a portable call in the lowered graph, which is the proof it
    did not go inside a delegate, while the unary next to it does lower. Taking
    the whole graph portable would be a worse trade than the crash it replaced.
    """
    example = torch.empty(2, 4, 8, 16, dtype=torch.float16)
    program = export(
        _NormAndItsNeighbour().eval(),
        (example,),
        dynamic_shapes={"x": {3: Dim("d3")}},
    )
    lowered = to_edge_transform_and_lower(
        program, partitioner=[HexagonPartitioner()]
    ).exported_program()
    portable = [
        node
        for node in lowered.graph_module.graph.nodes
        if node.op == "call_function" and "layer_norm" in str(node.target)
    ]
    assert len(portable) == 1, "the norm is missing from the graph, so it was absorbed"
    assert len(_delegates(lowered)) == 1


def _delegates(program):
    """The backend's delegates in a lowered program, checked to be its own."""
    delegates = [
        program.graph_module.get_submodule(str(node.args[0].target))
        for node in program.graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]
    assert all(
        delegate.backend_id == "HexagonBackend" for delegate in delegates
    ), "a delegate from another backend would make this test say nothing"
    return delegates
