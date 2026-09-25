# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner's answers about a node whose emitter adds no command."""



import torch


from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _cast_stays_in_fp16,
    _emits_no_command,
)
from executorch.exir.dialects._ops import ops as exir_ops


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
