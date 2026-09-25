# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PT2E quantizer for the Hexagon backend's weight-only matmul path.

What the graph holds and what the arithmetic is are two different things here,
and the docs have to keep them apart.

The annotation is weight-only. Only the matmul's weight is annotated, so no
observer goes on an activation and the graph keeps its activations at the width
it was exported at -- the runtime narrows a fp32 operand to fp16 on the way
into the arena. Two schemes are exposed, named for the operands each command
takes:

    q4a16   int4 weight, fp16 activation   -> DSP_OP_MATMUL_Q4A16_GEMV_I8
    w8a16   int8 weight, fp16 activation   -> DSP_OP_MATMUL_W8A16_GEMV_I8

The arithmetic is dynamic int8, not fp16. Both kernels take that fp16
activation row and quantize it to symmetric per-token int8 themselves -- absmax
over the whole row, scale absmax/127, computed in the kernel -- then run an
integer dot product and apply the activation scale and the weight's
per-output-channel fp32 scale to the result. So the activation is quantized
without any calibration, per row, at every inference. The `a16` in the scheme
names is the width at the command boundary, not the width of the multiply.

That is not a design choice so much as the shape of what the DSP has: no
kernel in the vendored set accepts an int8 activation tensor, and the runtime
treats every non-Float arena entry as two bytes per element, so a static-scale
w8a8 graph is not reachable from here. `test_hexagon_quantizer` measures what
this actually computes against the dequantized reference, and the interpreter
in `test/blob_interpreter.py` models the same arithmetic rather than an
idealized fp16 one.

Weights are per-output-channel symmetric, which is the granularity the kernels'
scale operand carries (one scale block covering all of K, kernel
`scale_block_num == 1`). The kernels wired up here are the M == 1 (decode) GEMV
entries and the Q4 M > 1 prefill entry. The GEMV entries read K contiguous
fp16, which for a single row is the layout the graph already carries. Q4
prefill packs the activation in 64-channel tiles and repacks the output, so
`hexagon_ops` admits only the geometries and constant operands that entry
can pack. W8A16 M > 1 stays on the portable kernels because this baseline has
no command-42 packer or emitter.

Usage follows the usual PT2E flow:

    quantizer = get_hexagon_quantizer("q4a16")
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*calibration_inputs)
    quantized = convert_pt2e(prepared)

Which matmuls it annotates, and what a caller should expect of the ones it does
not: `mm`, `addmm`, the two-dimensional `matmul` they are the lowering of, and
`nn.Linear`/`F.linear`, which is rewritten into the `addmm` spelling because its
own lowers to a permuted weight no quantized emitter matches. An annotation is
made only where the quantized emitters can run the matmul, so a projection
outside their gates keeps the fp16 path it would have had without a quantizer
rather than leaving a `dequantize_per_channel` in the graph for a runtime that
has no kernel for it.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.fx import Node
from torchao.quantization.pt2e import PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)

from executorch.backends.hexagon.hexagon_ops import (
    _bias_is_one_value_per_channel,
    _scalar_arg,
    weight_only_matmul_fits,
)

__all__ = [
    "HexagonQuantizer",
    "get_hexagon_quantizer",
    "get_hexagon_quantization_config",
    "get_q4a16_config",
    "get_w8a16_config",
    "SUPPORTED_SCHEMES",
]

#: Scheme name -> weight bit width. The emitter picks the DSP op from this.
SUPPORTED_SCHEMES = {"q4a16": 4, "w8a16": 8}

#: The matmuls the backend has a quantized kernel for, mapped to the argument
#: that holds the weight. mm and addmm both put the weight last, so its last
#: axis is the output-feature axis the per-channel scale is indexed by.
_WEIGHT_ONLY_TARGETS: Dict[Callable, int] = {
    torch.ops.aten.mm.default: 1,
    torch.ops.aten.addmm.default: 2,
}

#: `nn.Linear` and `F.linear`: the spelling every transformer's projections are
#: written in, and not one the table above holds. Its weight is stored `[out, in]`
#: and `to_edge` lowers it to `addmm(b, x, permute_copy(w, [1, 0]))`, so a
#: dequantize on that weight would sit behind a permute `hexagon_ops` does not
#: match. `transform_for_annotation` rewrites an admissible one into the `addmm`
#: the table does hold, over a materialized `[in, out]` constant.
_LINEAR_TARGET = torch.ops.aten.linear.default

_LOGGER = logging.getLogger(__name__)


def _static_shape(value) -> Optional[Tuple[int, ...]]:
    """A tensor's shape as ints, or None.

    None covers both "not a tensor" and "an axis whose extent is dynamic": a
    dynamic axis has no geometry to compare against the kernels' guards, and the
    emitters refuse one for the same reason.
    """
    if not isinstance(value, torch.Tensor):
        return None
    shape = []
    for axis in value.shape:
        if isinstance(axis, torch.SymInt):
            return None
        shape.append(int(axis))
    return tuple(shape)


def _operand(node: Node, index: int) -> Optional[Node]:
    """The node at `index`, if there is one and it is a node."""
    if len(node.args) > index and isinstance(node.args[index], Node):
        return node.args[index]
    return None


def _constant_weight(node: Node, index: int) -> Optional[Node]:
    """The weight operand at `index`, if the program owns it rather than the graph.

    A parameter, buffer or lifted constant is a `get_attr`. That is the same
    question the emitters ask through `HexagonPartitioner`'s
    `is_data_placeholder`, and it is the one that has to be asked *before* the
    annotation: PT2E freezes a constant weight into the stored low-bit tensor
    the emitters pack, but an operand the caller hands in is quantized at run
    time instead, which leaves a `quantize_per_channel` in the graph beside the
    dequantize and no kernel for either.
    """
    weight = _operand(node, index)
    if weight is None or weight.op != "get_attr":
        return None
    return weight


def _matmul_facts(node: Node):
    """(activation, bias, m, k, n) of a matmul the table holds, or None.

    The weight is last in both spellings, so the activation is the operand
    before it -- `mm` is `(activation, weight)` and `addmm` is `(bias,
    activation, weight)`. `alpha` and `beta` are the two arguments the plain path
    already checks: alpha has no kernel here, and beta decides whether the bias
    is one the kernel's own bias operand carries.
    """
    index = _weight_index(node)
    if index is None:
        return None
    weight = _constant_weight(node, index)
    activation = _operand(node, index - 1)
    if weight is None or activation is None:
        return None
    bias = None
    if node.target is torch.ops.aten.addmm.default:
        if _scalar_arg(node, "alpha", 4, 1.0) != 1.0:
            return None
        beta = _scalar_arg(node, "beta", 3, 1.0)
        if beta not in (0.0, 1.0):
            return None
        if beta != 0.0:
            bias = _operand(node, 0)
    activation_shape = _static_shape(activation.meta.get("val"))
    weight_shape = _static_shape(weight.meta.get("val"))
    if activation_shape is None or weight_shape is None:
        return None
    if len(activation_shape) != 2 or len(weight_shape) != 2:
        return None
    if activation_shape[1] != weight_shape[0]:
        return None
    return activation, bias, activation_shape[0], weight_shape[0], weight_shape[1]


def _linear_facts(model: torch.fx.GraphModule, node: Node):
    """(activation, bias, weight, source, m, k, n) of an `aten.linear`, or None.

    The weight is stored `[out, in]`, the other way round from every spelling in
    the table, so `k` is its second axis and `n` its first. `source` is the
    tensor itself, which is what the rewrite transposes.
    """
    activation = _operand(node, 0)
    weight = _constant_weight(node, 1)
    if activation is None or weight is None:
        return None
    if len(node.args) > 2 and node.args[2] is not None and _operand(node, 2) is None:
        return None
    bias = _operand(node, 2)
    source = _owned_tensor(model, str(weight.target))
    activation_shape = _static_shape(activation.meta.get("val"))
    weight_shape = _static_shape(weight.meta.get("val"))
    if source is None or activation_shape is None or weight_shape is None:
        return None
    if len(activation_shape) != 2 or len(weight_shape) != 2:
        return None
    n, k = weight_shape
    if activation_shape[1] != k:
        return None
    return activation, bias, weight, source, activation_shape[0], k, n


def _admits_the_emitters(activation: Node, bias, m, k, n, bits: int) -> bool:
    """Whether the quantized emitters will run this matmul once it is annotated.

    `hexagon_ops.quantized_matmul_is_emittable` answers the same questions of a
    graph that already carries the `dequantize_per_channel` this is deciding
    whether to create; this asks them of the operands, before it exists. Two of
    the answers are shared outright -- `weight_only_matmul_fits` for the shape
    and `_bias_is_one_value_per_channel` for the bias -- so the two cannot drift
    apart on the conditions the kernels' own guards decide. What is left is the
    activation's own: a strided one is read wrong by a kernel that walks K
    contiguously.

    Erring strict is deliberate. A matmul this turns away keeps the fp16 path it
    already had; a matmul it admitted that the emitters then refused would leave
    its dequantize in the graph, and `to_executorch` cannot serialize that.
    """
    value = activation.meta.get("val")
    if not isinstance(value, torch.Tensor) or not value.is_contiguous():
        return False
    if not weight_only_matmul_fits(m, k, n, bits):
        return False
    return bias is None or _bias_is_one_value_per_channel(bias, n)


def _log_unannotated(node: Node, reason: str) -> None:
    """Say why a matmul was left unquantized, when anyone is listening.

    Debug rather than warning, and guarded like the partitioner's own
    diagnostics: a graph that keeps a projection on the portable kernels is the
    answer the backend gives to every gate it does not pass, and an export must
    not start paying for messages nobody asked for.
    """
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            "hexagon: %s stays unquantized: %s, so it keeps the path it has",
            node.name,
            reason,
        )


def _owned_tensor(model: torch.fx.GraphModule, path: str) -> Optional[torch.Tensor]:
    """The tensor a `get_attr` path names, or None if nothing here holds it.

    A parameter or a buffer is what a weight usually is. Anything else is one the
    rewrite cannot transpose, and None leaves that `nn.Linear` exactly as it is
    rather than failing the whole quantization.
    """
    for getter in (model.get_parameter, model.get_buffer):
        try:
            return getter(path)
        except AttributeError:
            continue
    return None


def _transposed_name(weight: Node) -> str:
    """A module attribute name for the transposed copy of `weight`.

    Keyed by the weight it comes from rather than by the node that reads it: two
    `nn.Linear`s over one parameter are one projection stored once, and tied
    weights are common enough in a transformer to be worth the shared constant.
    """
    return "_hexagon_weight_t_" + re.sub(r"\W", "_", str(weight.target))


def _linear_as_addmm(
    model: torch.fx.GraphModule,
    node: Node,
    weight: Node,
    source: torch.Tensor,
    bias,
) -> None:
    """Replace `linear(x, w, b)` with `addmm(b, x, w_t)` over a [in, out] constant.

    A transpose of a constant is a permutation, not arithmetic: the transposed
    weight is the same numbers and `torch.export` drops the original once
    nothing reads it. What it buys is the pattern -- the dequantize lands
    directly on the matmul's weight operand, which is what the emitters match,
    instead of behind the `permute_copy` `to_edge` would have put between them.
    `addmm` reaches the kernel's own bias operand, so a biased projection is one
    command rather than a matmul and an add.
    """
    with torch.no_grad():
        transposed_value = source.detach().t().contiguous()
    name = _transposed_name(weight)
    if not hasattr(model, name):
        model.register_parameter(
            name,
            torch.nn.Parameter(transposed_value.clone(), requires_grad=False),
        )
    graph = model.graph
    with graph.inserting_before(node):
        transposed = graph.create_node("get_attr", name)
        transposed.meta["val"] = weight.meta["val"].t().contiguous()
        if bias is None:
            target, args = torch.ops.aten.mm.default, (node.args[0], transposed)
        else:
            target = torch.ops.aten.addmm.default
            args = (bias, node.args[0], transposed)
        replacement = graph.create_node("call_function", target, args)
        replacement.meta.update(node.meta)
    node.replace_all_uses_with(replacement)
    graph.erase_node(node)


def _matmul_is_two_dimensional(node: Node) -> bool:
    """Whether both of an `aten.matmul`'s operands are 2-D.

    `matmul` is the whole family: 1-D operands, and any number of batch axes
    leading the two the multiply contracts over. The GEMV kernels have neither a
    batch axis nor an M > 1 -- `to_edge` folds a batch into M, which is the
    dimension `hexagon_ops` refuses -- and `2d @ 1d` is a sum of products rather
    than a matmul at all. Only two 2-D operands form the tile these kernels
    read, and that spelling is `mm`. Anything else is left unannotated so it
    keeps the path it has rather than being moved onto a kernel that cannot
    carry it.
    """
    if len(node.args) < 2:
        return False
    operands = (node.args[0], node.args[1])
    if not all(isinstance(operand, Node) for operand in operands):
        return False
    values = [operand.meta.get("val") for operand in operands]
    return all(isinstance(value, torch.Tensor) and value.dim() == 2 for value in values)


def _weight_index(node: Node) -> Optional[int]:
    """The argument a weight-only matmul keeps its weight in, or None.

    `mm` and `addmm` put it last, and so does the 2-D `matmul` they are the
    lowering of; for that one the weight's last axis is the output-feature axis
    the per-channel scale is indexed by, exactly as it is for `mm`.
    """
    index = _WEIGHT_ONLY_TARGETS.get(node.target)
    if index is not None:
        return index
    if node.target is torch.ops.aten.matmul.default and _matmul_is_two_dimensional(
        node
    ):
        return 1
    return None


def _weight_qspec(bits: int, ch_axis: int) -> QuantizationSpec:
    """A per-output-channel symmetric spec for a `bits`-wide weight.

    The values are stored in an int8 tensor whatever the width: PT2E has no
    narrower container, so a 4-bit weight is an int8 tensor whose values all
    fit [-8, 7]. The emitter reads the range back off the dequantize node.
    """
    if bits == 4:
        quant_min, quant_max = -8, 7
    elif bits == 8:
        quant_min, quant_max = -128, 127
    else:
        raise ValueError(f"hexagon: no quantized kernel for a {bits}-bit weight")
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        is_dynamic=False,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    )


def _bits_of(config: QuantizationConfig) -> int:
    span = config.weight.quant_max - config.weight.quant_min + 1
    if span <= 16:
        return 4
    if span <= 256:
        return 8
    raise ValueError(f"hexagon: unsupported weight range of {span} values")


@functools.lru_cache
def get_hexagon_quantization_config(scheme: str = "q4a16") -> QuantizationConfig:
    """The weight-only config for one scheme.

    Only the weight spec is set: the activation specs are None, so the graph's
    activations are left at the width they were exported at and the runtime
    narrows them to fp16 at the arena boundary. What the kernel then does to
    that fp16 row -- per-token int8 quantization -- is not something PT2E can
    express, so no activation observer is inserted and none would be used. The
    ch_axis here is a placeholder -- `HexagonQuantizer.annotate` rebuilds the
    spec with the axis of the operand it is actually annotating.
    """
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"hexagon: unknown scheme {scheme!r}; expected one of "
            f"{sorted(SUPPORTED_SCHEMES)}"
        )
    return QuantizationConfig(
        input_activation=None,
        output_activation=None,
        weight=_weight_qspec(SUPPORTED_SCHEMES[scheme], ch_axis=0),
        bias=None,
        is_qat=False,
    )


def get_q4a16_config() -> QuantizationConfig:
    """int4 weight, fp16 activation."""
    return get_hexagon_quantization_config("q4a16")


def get_w8a16_config() -> QuantizationConfig:
    """int8 weight, fp16 activation."""
    return get_hexagon_quantization_config("w8a16")


class HexagonQuantizer(Quantizer):
    """Annotates matmul weights for the DSP's quantized kernels.

    Only the weight is annotated, so `prepare_pt2e` inserts one observer per
    weight and `convert_pt2e` replaces it with a `dequantize_per_channel` node.
    That node is the pattern `hexagon_ops` matches: its readers are the
    quantized matmuls, which read the stored low-bit weight and its scales
    themselves, so the dequantize emits no command. Fusing it away is what
    "weight-only" means in the command stream -- it does not make the multiply
    an fp16 one, which is the kernel's own int8 arithmetic.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.global_config = config
        self.filter_fn: Optional[Callable[[Node], bool]] = None

    def set_global(self, quantization_config: QuantizationConfig) -> "HexagonQuantizer":
        self.global_config = quantization_config
        return self

    def set_filter_function(
        self, filter_fn: Callable[[Node], bool]
    ) -> "HexagonQuantizer":
        """Quantize only the nodes this returns True for."""
        self.filter_fn = filter_fn
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Rewrite an admissible `linear` into the `addmm` spelling.

        Everything else the table holds is already the spelling the emitters
        match and needs no rewrite. A `linear` the emitters would refuse is left
        exactly as it is, permute and all: that is the fp16 graph the caller
        would have exported without a quantizer, and the alternative is a graph
        carrying a dequantize no kernel here can run.
        """
        config = self.global_config
        if config is None:
            return model
        bits = _bits_of(config)
        for node in list(model.graph.nodes):
            if node.target is not _LINEAR_TARGET:
                continue
            if self.filter_fn is not None and not self.filter_fn(node):
                continue
            facts = _linear_facts(model, node)
            if facts is None:
                _log_unannotated(node, "its weight is not one the export owns")
                continue
            activation, bias, weight, source, m, k, n = facts
            if not _admits_the_emitters(activation, bias, m, k, n, bits):
                _log_unannotated(
                    node,
                    f"the quantized matmul emitters do not take m={m} k={k} n={n} "
                    f"with {bits}-bit weights",
                )
                continue
            _linear_as_addmm(model, node, weight, source, bias)
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        config = self.global_config
        if config is None:
            return model
        bits = _bits_of(config)
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            weight_index = _weight_index(node)
            if weight_index is None or len(node.args) <= weight_index:
                continue
            if self.filter_fn is not None and not self.filter_fn(node):
                continue
            facts = _matmul_facts(node)
            if facts is None:
                _log_unannotated(node, "its weight is not one the export owns")
                continue
            activation, bias, m, k, n = facts
            if not _admits_the_emitters(activation, bias, m, k, n, bits):
                _log_unannotated(
                    node,
                    f"the quantized matmul emitters do not take m={m} k={k} n={n} "
                    f"with {bits}-bit weights",
                )
                continue
            weight = node.args[weight_index]
            # The weight's last axis is the output-feature axis the per-channel
            # scale is indexed by, for both [k, n] and [n, k] spellings.
            spec = _weight_qspec(bits, ch_axis=weight.meta["val"].dim() - 1)
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={weight: spec},
                output_qspec=None,
            )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


def get_hexagon_quantizer(
    scheme: str = "q4a16", config: Optional[QuantizationConfig] = None
) -> HexagonQuantizer:
    """A quantizer for one named scheme, or for a config passed in."""
    return HexagonQuantizer(config or get_hexagon_quantization_config(scheme))
