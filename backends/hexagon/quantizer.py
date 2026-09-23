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
entries: they read K contiguous fp16, which for a single row is the layout the
graph already carries. The prefill (M > 1) entries want an activation blocked
in 64-channel tiles and an output repack, so `hexagon_ops` refuses to delegate
those and they stay on the portable kernels.

Usage follows the usual PT2E flow:

    quantizer = get_hexagon_quantizer("q4a16")
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*calibration_inputs)
    quantized = convert_pt2e(prepared)
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, Optional

import torch
from torch.fx import Node
from torchao.quantization.pt2e import PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
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
        # mm and addmm carry no scalar that has to become a tensor attribute, so
        # there is nothing to rewrite before the observers go in.
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        config = self.global_config
        if config is None:
            return model
        bits = _bits_of(config)
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            weight_index = _WEIGHT_ONLY_TARGETS.get(node.target)
            if weight_index is None or len(node.args) <= weight_index:
                continue
            if self.filter_fn is not None and not self.filter_fn(node):
                continue
            weight = node.args[weight_index]
            if not isinstance(weight, torch.fx.Node):
                continue
            value = weight.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue
            # The weight's last axis is the output-feature axis the per-channel
            # scale is indexed by, for both [k, n] and [n, k] spellings.
            spec = _weight_qspec(bits, ch_axis=value.dim() - 1)
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
