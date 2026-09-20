# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.hexagon.hexagon_ops import (
    _hmx_prepack_enabled,
    EMITTERS,
    pack_hmx_weight,
)
from executorch.backends.hexagon.serialization.blob import BlobBuilder, TensorRef
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
)
from torch.export import ExportedProgram

# node.target -> emitter. The partitioner delegates exactly these, so what the
# DSP is asked to run and what the AOT step can encode cannot drift apart.
SUPPORTED_TARGETS: Dict[Callable, Callable] = EMITTERS

# Emitter contract: append this node's DSP commands to the context and return
# the ref holding its result.
Emitter = Callable[[torch.fx.Node, "BlobContext"], TensorRef]


def bytes_for(numel: int, dtype: torch.dtype) -> int:
    """Byte size of a tensor of this many elements, as the DSP sees it.

    Delegated subgraphs are lowered to contiguous buffers, so shapes only matter
    through their element count.
    """
    return numel * torch.tensor([], dtype=dtype).element_size()


def _val_of(node: torch.fx.Node) -> torch.Tensor:
    val = node.meta.get("val")
    if val is None:
        raise RuntimeError(f"hexagon: node {node.name} has no recorded value")
    return val


def owned_weight(
    program: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    """The tensor behind a placeholder the caller does not pass in.

    A parameter, buffer or lifted constant consumed by a subgraph still arrives
    as a placeholder, but its bytes belong to the program rather than to the
    caller. The partitioner tags those, so EXIR moves their values into this
    program's own state and out of the delegate's arguments; what is left here
    is to find them.
    """
    for fetch in (get_param, get_buffer, get_lifted_tensor_constant):
        tensor = fetch(program, node)
        if tensor is not None:
            return tensor
    return None


def weight_bytes(tensor: torch.Tensor) -> bytes:
    """The bytes the DSP reads out of the weight section.

    The kernels read two bytes per element, and the runtime narrows a fp32
    method input to fp16 on the way into the arena, so a weight stored where
    that input used to be has to hold the same fp16 bytes it would have carried.
    """
    if tensor.dtype == torch.float32:
        tensor = tensor.to(torch.float16)
    return tensor.detach().contiguous().cpu().numpy().tobytes()


class BlobContext:
    """State shared by the emitters while one subgraph is being lowered."""

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        n_inputs: int,
        n_outputs: int,
        output_index: Dict[torch.fx.Node, int],
        weights: Optional[Dict[torch.fx.Node, torch.Tensor]] = None,
    ) -> None:
        self.graph_module = graph_module
        # The tensors behind the placeholders this subgraph owns. torch.export
        # leaves a fake in a lifted weight's metadata, so this is the only way an
        # emitter can look at the value it is being asked to store.
        self.weights: Dict[torch.fx.Node, torch.Tensor] = dict(weights or {})
        self.builder = BlobBuilder(n_inputs, n_outputs)
        self.producer: Dict[torch.fx.Node, TensorRef] = {}
        self._constants: Dict[torch.fx.Node, TensorRef] = {}
        # Placeholder targets whose stored bytes were rewritten into the HMX tile
        # order; their matmul has to say so, because no other kernel reads that.
        self.packed_targets: set = set()
        # Nodes whose value was computed at export time and stored as a weight.
        self.folded: Dict[torch.fx.Node, TensorRef] = {}
        # Those same values, so a consumer that wants to store the tensor itself
        # -- a matmul pre-packing its weight, say -- does not have to read the
        # bytes back out of the blob to get them.
        self.folded_values: Dict[torch.fx.Node, torch.Tensor] = {}
        self._output_index = output_index

    def constant(
        self, node: torch.fx.Node, dtype: torch.dtype = None
    ) -> TensorRef:
        """Materializes a get_attr node into the weight section, once.

        dtype converts the stored tensor, which some ops need: layer norm takes
        gamma and beta as fp32 even when the graph holds them as fp16.
        """
        key = (node, dtype)
        cached = self._constants.get(key)
        if cached is not None:
            return cached
        tensor = self.lifted_value(node)
        if tensor is None:
            raise RuntimeError(f"hexagon: no value for constant {node.name}")
        if dtype is not None:
            tensor = tensor.to(dtype)
        ref = self.builder.add_weights(tensor.contiguous().cpu().numpy().tobytes())
        self._constants[key] = ref
        return ref

    def packed_weight(self, node: torch.fx.Node, tensor, k: int, n: int) -> TensorRef:
        """Materializes a weight in the order the HMX unit reads its tiles.

        The unit can then stream the matrix straight out of the weights section,
        where otherwise the DSP rearranges it into VTCM before every inference,
        and that rearrange is most of what a matmul costs.
        """
        key = (node, "hmx")
        cached = self._constants.get(key)
        if cached is not None:
            return cached
        tensor = tensor.detach().to(torch.float16)
        ref = self.builder.add_weights(pack_hmx_weight(tensor.cpu().numpy(), k, n))
        self._constants[key] = ref
        self.packed_targets.add(getattr(node, "target", node))
        return ref

    def lifted_value(self, node: torch.fx.Node):
        """The tensor behind a lifted parameter, buffer or constant.

        torch.export lifts every weight to a placeholder, so the node carries a
        fake in its metadata and nothing can be folded or pre-packed from it
        until the graph module -- which still holds the real tensor -- is asked.
        """
        tensor = self.weights.get(node)
        if tensor is not None:
            return tensor.detach()
        if node.op == "get_attr":
            return self.graph_module.get_parameter(node.target).detach()
        if not isinstance(node.target, str):
            return None
        for name in ("get_parameter", "get_buffer"):
            try:
                tensor = getattr(self.graph_module, name)(node.target)
            except Exception:
                continue
            if isinstance(tensor, torch.Tensor):
                return tensor.detach()
        value = getattr(self.graph_module, node.target, None)
        return value.detach() if isinstance(value, torch.Tensor) else None

    def scalar(self, value, dtype: torch.dtype = torch.float16) -> TensorRef:
        """Materializes a python scalar as a one-element tensor.

        The DSP reads every operand out of a buffer, so a literal operand has to
        exist in the arena like any other.

        Nothing that carries a torch scalar reaches this today, and a scalar
        multiply is the reason: torch multiplies a half tensor by a float scalar
        in fp32 and rounds the product back, which differs from a fp16 product
        on about one element in six (measured over 65536 random halves), so
        giving the DSP that op changes the output's md5. The op stays on the
        portable kernel and splits the graph where it sits.
        """
        key = ("scalar", float(value), dtype)
        cached = self._constants.get(key)
        if cached is not None:
            return cached
        tensor = torch.tensor([value], dtype=dtype)
        ref = self.builder.add_weights(tensor.numpy().tobytes())
        self._constants[key] = ref
        return ref

    def constant_value(self, node: torch.fx.Node):
        """The tensor behind a constant, whether attribute or lifted input.

        None means this layer cannot see it, which is what a placeholder whose
        value only the runtime holds looks like: nothing is folded then.
        """
        folded = self.folded_values.get(node)
        if folded is not None:
            return folded
        lifted = self.lifted_value(node)
        if lifted is not None:
            return lifted
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and type(value).__name__ != "FakeTensor":
            return value.detach()
        return None

    def folded_weight(self, node: torch.fx.Node, tensor) -> TensorRef:
        """Materializes a tensor computed at export time as a weight."""
        key = ("folded", node)
        cached = self._constants.get(key)
        if cached is not None:
            return cached
        ref = self.builder.add_weights(tensor.contiguous().cpu().numpy().tobytes())
        self._constants[key] = ref
        self.folded[node] = ref
        self.folded_values[node] = tensor.detach()
        return ref

    def operand(self, arg) -> TensorRef:
        """Resolves an operand, whether computed, stored, or a literal."""
        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                return self.constant(arg)
            folded = self.folded.get(arg)
            if folded is not None:
                return folded
            ref = self.producer.get(arg)
            if ref is None:
                raise RuntimeError(f"hexagon: nothing produced {arg.name} yet")
            return ref
        if isinstance(arg, (int, float)) and not isinstance(arg, bool):
            return self.scalar(arg)
        raise RuntimeError(f"hexagon: unsupported operand {arg!r}")

    def result_for(
        self, node: torch.fx.Node, numel: int, dtype: torch.dtype = torch.float16
    ) -> TensorRef:
        """Where this node's result goes.

        A node that is also a graph output writes straight into its output slot,
        so the last op of a chain costs no extra copy.
        """
        index = self._output_index.get(node)
        if index is not None:
            return self.builder.method_output(index, bytes_for(numel, dtype))
        return self.builder.add_activation(bytes_for(numel, dtype))

    def is_method_output(self, node: torch.fx.Node) -> bool:
        """Whether the caller reads this node from an output slot of its own."""
        return node in self._output_index

    def record(self, node: torch.fx.Node, ref: TensorRef) -> TensorRef:
        self.producer[node] = ref
        return ref


class HexagonBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        # No compile specs are defined for this backend yet.
        # The program is copied because emitters may rewrite the graph, while
        # the original still has to serialize into the .pte unchanged.
        program = copy.deepcopy(edge_program)
        graph_module = program.graph_module
        graph = graph_module.graph

        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        output_node = next(node for node in graph.nodes if node.op == "output")
        outputs = _flatten_outputs(output_node)

        # A weight this subgraph owns is stored in the blob, not handed over as
        # an argument. The runtime uploads the blob's weight section once, at
        # delegate init; a method input is copied into the arena on every
        # execute, which for a tower of stacked matmuls is the whole model's
        # weights per inference.
        weights = {
            placeholder: owned_weight(program, placeholder)
            for placeholder in placeholders
        }
        weights = {node: tensor for node, tensor in weights.items() if tensor is not None}
        inputs = [node for node in placeholders if node not in weights]

        context = BlobContext(
            graph_module, len(inputs), len(outputs), dict(outputs), weights
        )

        for node, tensor in weights.items():
            context.producer[node] = context.builder.add_weights(weight_bytes(tensor))

        for index, placeholder in enumerate(inputs):
            value = _val_of(placeholder)
            # A scalar input such as llama.custom_sdpa's start_pos arrives as a
            # SymInt, not a tensor. It still needs a slot: the runtime patches
            # the position into it before the flush.
            if isinstance(value, torch.Tensor):
                # The kernels read activations as fp16, so a fp32 input gets a
                # half-width slot and the runtime narrows it on the way into the
                # arena. Other widths, such as the int64 position read the patch
                # mechanism consumes, keep their own size.
                narrowed = value.dtype == torch.float32
                size = bytes_for(
                    value.numel(), torch.float16 if narrowed else value.dtype
                )
            else:
                size = bytes_for(1, torch.int64)
            context.producer[placeholder] = context.builder.method_input(index, size)

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            emitter = SUPPORTED_TARGETS.get(node.target)
            if emitter is None:
                raise RuntimeError(
                    f"hexagon: no DSP emitter for {node.target}; the partitioner "
                    "should not have delegated it"
                )
            emitter(node, context)

        # Every output index needs a slot, and only an emitter or a placeholder
        # can give it one. Catching it here names the node; `build` can only
        # report the index.
        for node, index in outputs:
            if node not in context.producer:
                raise RuntimeError(
                    f"hexagon: subgraph output {index} is {node.name} "
                    f"({node.op}: {node.target}), which no emitter produced"
                )

        return PreprocessResult(processed_bytes=context.builder.build())


def _flatten_outputs(output_node: torch.fx.Node) -> List[Tuple[torch.fx.Node, int]]:
    """Lists the graph's outputs in flattened order."""
    flat: List[torch.fx.Node] = []

    def visit(value) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, torch.fx.Node):
            flat.append(value)

    visit(output_node.args[0] if output_node.args else [])
    return [(node, index) for index, node in enumerate(flat)]
