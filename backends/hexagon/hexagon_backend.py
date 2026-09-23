# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.hexagon.hexagon_ops import EMITTERS, pack_hmx_weight
from executorch.backends.hexagon.serialization.blob import (
    ALIGNMENT,
    BlobBuilder,
    TensorRef,
)
from executorch.exir._serialize._named_data_store import (
    NamedDataStore,
    NamedDataStoreOutput,
)
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.sym_util import eval_upper_bound
from torch._export.utils import get_buffer, get_lifted_tensor_constant, get_param
from torch.export import ExportedProgram

# node.target -> emitter. The partitioner delegates exactly these, so what the
# DSP is asked to run and what the AOT step can encode cannot drift apart.
SUPPORTED_TARGETS: Dict[Callable, Callable] = EMITTERS

# Emitter contract: append this node's DSP commands to the context and return
# the ref holding its result.
Emitter = Callable[[torch.fx.Node, "BlobContext"], TensorRef]

HMX_PREPACK_SPEC = "hexagon_hmx_prepack"
ATTN_PAGED_SPEC = "hexagon_attn_paged"
EXTERNAL_WEIGHTS_MAX_BYTES_SPEC = "hexagon_external_weights_max_bytes"

#: Every key this backend accepts, so an unknown one is reported by name.
KNOWN_SPEC_KEYS = frozenset(
    {HMX_PREPACK_SPEC, ATTN_PAGED_SPEC, EXTERNAL_WEIGHTS_MAX_BYTES_SPEC}
)


@dataclass(frozen=True)
class HexagonCompileOptions:
    """The choices that change what the blob contains.

    These are compile specs rather than runtime options because each of them
    changes the bytes the .pte carries. A knob a runner could flip at load time
    would let it ask for a layout this program was never compiled for, and the
    DSP would read the weights in the wrong order without saying so. Written
    into the .pte instead, the same value is available to init(), which
    re-derives it from the blob and refuses a load where the two disagree.

    Both sides of every choice therefore come from one object: the partitioner
    stamps these into the delegate, and preprocess reads them back out.
    """

    #: Store each matmul weight in the order the HMX unit reads its tiles. The
    #: reorder is the same size as what it replaces, so the blob does not grow;
    #: what it buys is the unit's weight fill, which is most of what a matmul
    #: costs. No other kernel reads that layout, so a matmul whose weight was
    #: packed has to be the one that says so, and the packer will not pack a
    #: shape that kernel refuses.
    hmx_prepack: bool = True
    #: Run attention through the paged entry point. One page covers the whole
    #: packed cache, so the buffer layout is unchanged and the choice reaches
    #: the kernel as a page size in the command.
    attn_paged: bool = False
    #: Move weights larger than this many bytes out of the .pte and into a .ptd
    #: next to it. None keeps every weight inline, which is what the format did
    #: before this existed. What it saves is the size of the .pte and the work
    #: of copying it to a device; it does not save the rpcmem copy, because a
    #: mapped buffer is the only address the DSP can read.
    external_weights_max_bytes: Optional[int] = None

    def to_compile_specs(self) -> List[CompileSpec]:
        """The delegate's compile specs, in a fixed order."""
        specs = [
            CompileSpec(HMX_PREPACK_SPEC, bytes([1 if self.hmx_prepack else 0])),
            CompileSpec(ATTN_PAGED_SPEC, bytes([1 if self.attn_paged else 0])),
        ]
        if self.external_weights_max_bytes is not None:
            specs.append(
                CompileSpec(
                    EXTERNAL_WEIGHTS_MAX_BYTES_SPEC,
                    int(self.external_weights_max_bytes).to_bytes(8, "little"),
                )
            )
        return specs

    @classmethod
    def from_compile_specs(
        cls, compile_specs: List[CompileSpec]
    ) -> "HexagonCompileOptions":
        """Reads what the .pte carries, rejecting anything it cannot mean.

        No key is mandatory: every one of them has a default, and a caller that
        passes no specs at all is asking for exactly the defaults. What is
        rejected is a key this backend does not define, a key given twice, and
        a value whose width or contents cannot be what the key says.
        """
        options = cls()
        seen = set()
        for spec in compile_specs:
            if spec.key in seen:
                raise ValueError(f"hexagon: compile spec {spec.key} appears twice")
            seen.add(spec.key)
            if spec.key == HMX_PREPACK_SPEC:
                options = replace(options, hmx_prepack=_spec_bool(spec))
            elif spec.key == ATTN_PAGED_SPEC:
                options = replace(options, attn_paged=_spec_bool(spec))
            elif spec.key == EXTERNAL_WEIGHTS_MAX_BYTES_SPEC:
                options = replace(
                    options, external_weights_max_bytes=_spec_uint64(spec)
                )
            else:
                raise ValueError(
                    f"hexagon: unknown compile spec {spec.key!r}; this backend "
                    f"defines {sorted(KNOWN_SPEC_KEYS)}"
                )
        return options


def _spec_payload(spec: CompileSpec, width: int) -> bytes:
    if len(spec.value) != width:
        raise ValueError(
            f"hexagon: compile spec {spec.key} must be {width} bytes, "
            f"got {len(spec.value)}"
        )
    return spec.value


def _spec_bool(spec: CompileSpec) -> bool:
    value = _spec_payload(spec, 1)[0]
    if value not in (0, 1):
        raise ValueError(
            f"hexagon: compile spec {spec.key} must be 0 or 1, got {value}"
        )
    return bool(value)


def _spec_uint64(spec: CompileSpec) -> int:
    # The same boundary Vulkan validates for its external constants cap: a
    # compile spec can bypass parse_compile_options, so the width and the
    # domain are checked here rather than trusted.
    value = int.from_bytes(_spec_payload(spec, 8), byteorder="little")
    if value <= 0:
        raise ValueError(
            f"hexagon: compile spec {spec.key} must be a positive uint64, got {value}"
        )
    return value


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

    A mutated buffer is the exception. It is program state, but it changes every
    call, so its bytes are not a constant this layer may store once: the cache
    has to be read back in (and written out) per execute. Baking it in as a
    weight freezes the empty cache and every later step attends over it.
    """
    signature = program.graph_signature
    target = signature.inputs_to_buffers.get(node.name)
    if target is not None and target in set(signature.buffers_to_mutate.values()):
        return None
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
        dynamic_sequence: Optional[Tuple[int, int, int]] = None,
        dynamic_example: Optional[int] = None,
        options: Optional[HexagonCompileOptions] = None,
    ) -> None:
        self.graph_module = graph_module
        # The tensors behind the placeholders this subgraph owns. torch.export
        # leaves a fake in a lifted weight's metadata, so this is the only way an
        # emitter can look at the value it is being asked to store.
        self.weights: Dict[torch.fx.Node, torch.Tensor] = dict(weights or {})
        # What the exporters chose. The emitters read these rather than the
        # environment, so the blob and the specs that describe it cannot come
        # from two different sets of defaults.
        self.options = options or HexagonCompileOptions()
        self.builder = BlobBuilder(
            n_inputs, n_outputs, self.options.external_weights_max_bytes
        )
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
        self.dynamic_sequence = dynamic_sequence
        self.dynamic_example = dynamic_example
        self.has_symbolic_shape = any(
            isinstance(dim, torch.SymInt)
            for node in graph_module.graph.nodes
            for value in (
                node.meta.get("val") if node.meta.get("val") is not None else (),
            )
            for tensor in (value if isinstance(value, tuple) else (value,))
            for dim in getattr(tensor, "shape", ())
        )
        if dynamic_sequence is not None:
            self.builder.set_dynamic_sequence(*dynamic_sequence, dynamic_example or 0)

    def upper_bound(self, value) -> int:
        return self.upper_dim(value)

    def is_dynamic_dim(self, value) -> bool:
        return isinstance(value, torch.SymInt)

    def upper_dim(self, value) -> int:
        if self.is_dynamic_dim(value):
            return self.dynamic_sequence[2]
        if isinstance(value, torch.SymInt):
            return eval_upper_bound(value)
        return int(value)

    def upper_shape(self, shape, allow_static_fallback: bool = True) -> tuple:
        return tuple(self.upper_dim(value) for value in shape)

    def dynamic_bytes_for_shape(
        self, shape, dtype: torch.dtype, allow_static_fallback: bool = True
    ) -> tuple[int, int, int]:
        """Return bytes as c0 + c1*L + c2*L^2 for one tensor shape."""
        coeffs = [1, 0, 0]
        for dim in shape:
            factor = [0, 1, 0] if self.is_dynamic_dim(dim) else [int(dim), 0, 0]
            product = [0, 0, 0]
            for i, left in enumerate(coeffs):
                for j, right in enumerate(factor):
                    if i + j < len(product):
                        product[i + j] += left * right
            coeffs = product
        itemsize = torch.empty((), dtype=dtype).element_size()
        return tuple(value * itemsize for value in coeffs)

    def activation_for_shape(
        self, shape, dtype: torch.dtype = torch.float16
    ) -> TensorRef:
        upper_numel = 1
        for dim in shape:
            upper_numel *= self.upper_dim(dim)
        return self.builder.add_activation(
            bytes_for(upper_numel, dtype),
            dynamic_layout=self.dynamic_bytes_for_shape(shape, dtype),
        )

    def dynamic_bytes_for_function(
        self, max_bytes: int, function
    ) -> tuple[int, int, int]:
        """Fit a non-negative quadratic byte formula over the sequence length."""
        if (
            self.dynamic_sequence is None
            and self.dynamic_example is None
            and not self.has_symbolic_shape
        ):
            return (int(max_bytes), 0, 0)
        if self.dynamic_sequence is not None:
            upper = int(self.dynamic_sequence[2])
        else:
            upper = max(
                (
                    eval_upper_bound(dim)
                    for node in self.graph_module.graph.nodes
                    for value in (
                        (
                            node.meta.get("val")
                            if node.meta.get("val") is not None
                            else ()
                        ),
                    )
                    for tensor in (value if isinstance(value, tuple) else (value,))
                    for dim in getattr(tensor, "shape", ())
                    if isinstance(dim, torch.SymInt)
                ),
                default=0,
            )
        if upper <= 0:
            return (int(max_bytes), 0, 0)
        middle = max(1, upper // 2)
        at_zero, at_middle, at_upper = (
            int(function(length)) for length in (0, middle, upper)
        )
        if at_zero < 0:
            return (int(max_bytes), 0, 0)
        # Fit c0 + c1*x + c2*x^2 at x=0, upper/2, upper.  Ceil the
        # coefficients so alignment rounding cannot make the runtime buffer
        # smaller than the DSP workspace it describes.
        slope_middle = at_middle - at_zero
        slope_upper = at_upper - at_zero
        numerator = slope_upper * middle - slope_middle * upper
        denominator = middle * upper * (upper - middle)
        c2 = max(0, (numerator + denominator - 1) // denominator)
        c1_numerator = slope_middle - c2 * middle * middle
        c1 = max(0, (c1_numerator + middle - 1) // middle)
        if at_zero + c1 * upper + c2 * upper * upper < at_upper:
            c1 += (
                at_upper - (at_zero + c1 * upper + c2 * upper * upper) + upper - 1
            ) // upper
        return (at_zero, c1, c2)

    def add_dynamic_patch(
        self, op_index: int, param_index: int, scale: int = 1, add: int = 0
    ) -> None:
        if self.dynamic_sequence is not None:
            from executorch.backends.hexagon.serialization.blob import DynamicPatch

            self.builder.add_dynamic_patch(
                DynamicPatch(op_index, param_index, scale, add)
            )

    def constant(self, node: torch.fx.Node, dtype: torch.dtype = None) -> TensorRef:
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
        value = node.meta.get("val")
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, torch.Tensor):
            numel = 1
            for dim in value.shape:
                numel *= self.upper_dim(dim)
        elif isinstance(numel, torch.SymInt):
            numel = eval_upper_bound(numel)
        index = self._output_index.get(node)
        if index is not None:
            layout = (
                self.dynamic_bytes_for_shape(value.shape, dtype)
                if isinstance(value, torch.Tensor)
                else None
            )
            return self.builder.method_output(index, bytes_for(numel, dtype), layout)
        if isinstance(value, torch.Tensor):
            return self.activation_for_shape(value.shape, dtype)
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
        options = HexagonCompileOptions.from_compile_specs(compile_specs)
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
        weights = {
            node: tensor for node, tensor in weights.items() if tensor is not None
        }
        inputs = [node for node in placeholders if node not in weights]

        dynamic_sequence = None
        dynamic_example = None
        dynamic_symbol = None

        finite_constraint_max = 0
        for constraint in program.range_constraints.values():
            if not hasattr(constraint, "upper"):
                continue
            try:
                finite_constraint_max = max(
                    finite_constraint_max, int(constraint.upper)
                )
            except Exception:
                pass

        def symbol_upper(symbol) -> int:
            constraint = next(
                (
                    value
                    for key, value in program.range_constraints.items()
                    if str(key) == str(symbol)
                ),
                None,
            )
            if constraint is not None and hasattr(constraint, "upper"):
                try:
                    return int(constraint.upper)
                except Exception:
                    pass
            return eval_upper_bound(symbol)

        for index, placeholder in enumerate(inputs):
            value = _val_of(placeholder)
            if not isinstance(value, torch.Tensor):
                continue
            dynamic_axes = [
                axis
                for axis, dim in enumerate(value.shape)
                if isinstance(dim, torch.SymInt)
            ]
            if dynamic_axes:
                axis = dynamic_axes[0]
                symbols = {str(value.shape[axis]) for axis in dynamic_axes}
                if len(symbols) != 1:
                    raise RuntimeError(
                        "hexagon: one input cannot contain multiple dynamic symbols"
                    )
                symbol = symbols.pop()
                if dynamic_symbol is None:
                    dynamic_symbol = symbol
                elif symbol != dynamic_symbol:
                    raise RuntimeError(
                        "hexagon: inputs must share one dynamic sequence symbol"
                    )
                if dynamic_sequence is not None:
                    continue
                dynamic_sequence = (
                    index,
                    axis,
                    max(symbol_upper(value.shape[axis]), finite_constraint_max),
                )

        has_symbolic_input = any(
            isinstance(dim, torch.SymInt)
            for placeholder in inputs
            for dim in getattr(_val_of(placeholder), "shape", ())
        )
        if (
            dynamic_sequence is None
            and has_symbolic_input
            and program.range_constraints
        ):
            max_length = max(
                eval_upper_bound(upper) for upper in program.range_constraints.values()
            )
            candidates = []
            for index, placeholder in enumerate(inputs):
                value = _val_of(placeholder)
                if not isinstance(value, torch.Tensor):
                    continue
                for axis, dim in enumerate(value.shape):
                    if isinstance(dim, int) and 1 < dim <= max_length:
                        candidates.append((int(dim), index, axis))
            if candidates:
                example, index, axis = max(
                    candidates,
                    key=lambda item: (
                        sum(item[0] == c[0] for c in candidates),
                        item[0],
                    ),
                )
                dynamic_example = example
                dynamic_sequence = (index, axis, max_length)

        if dynamic_sequence is not None and dynamic_example is None:
            index, axis, _ = dynamic_sequence
            dynamic_example = int(_val_of(inputs[index]).shape[axis])

        context = BlobContext(
            graph_module,
            len(inputs),
            len(outputs),
            dict(outputs),
            weights,
            dynamic_sequence,
            dynamic_example,
            options,
        )

        for node, tensor in weights.items():
            context.producer[node] = context.builder.add_weights(weight_bytes(tensor))

        for index, placeholder in enumerate(inputs):
            value = _val_of(placeholder)
            # A scalar input such as llama.custom_sdpa's start_pos arrives as a
            # SymInt, not a tensor. It still needs a slot: the runtime patches
            # the position into it before the flush.
            if isinstance(value, torch.Tensor):
                is_dynamic_input = (
                    dynamic_sequence is not None and index == dynamic_sequence[0]
                )
                # The kernels read activations as fp16, so a fp32 input gets a
                # half-width slot and the runtime narrows it on the way into the
                # arena. Other widths, such as the int64 position read the patch
                # mechanism consumes, keep their own size.
                narrowed = value.dtype == torch.float32
                max_numel = 1
                for dim in context.upper_shape(
                    value.shape, allow_static_fallback=is_dynamic_input
                ):
                    max_numel *= dim
                size = bytes_for(max_numel, torch.float16 if narrowed else value.dtype)
                dynamic_layout = context.dynamic_bytes_for_shape(
                    value.shape,
                    torch.float16 if narrowed else value.dtype,
                    allow_static_fallback=is_dynamic_input,
                )
            else:
                size = bytes_for(1, torch.int64)
                dynamic_layout = None
            context.producer[placeholder] = context.builder.method_input(
                index, size, dynamic_layout
            )

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

        return PreprocessResult(
            processed_bytes=context.builder.build(),
            data_store_output=_named_data_store(context.builder),
        )


def _named_data_store(builder: BlobBuilder) -> Optional[NamedDataStoreOutput]:
    """The weights that left the .pte, as a store the .ptd writer can serialize.

    One tag, so one file: a Module is handed a single data map at load time, and
    a second shard would be a second file nothing could open. Keys carry the
    weight's content rather than its position, so two delegates that store equal
    bytes share one buffer in the file, and the file's contents do not depend on
    the order the graph happened to reach its weights in.
    """
    blobs = builder.external_weight_data()
    if not blobs:
        return None
    keys = sorted(key for key, _ in blobs)
    tag = (
        "hexagon_weights_" + hashlib.sha256("\0".join(keys).encode("utf-8")).hexdigest()
    )
    store = NamedDataStore()
    for key, data in sorted(blobs):
        store.add_named_data(key, data, alignment=ALIGNMENT, external_tag=tag)
    return store.get_named_data_store_output()


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
