# Rank normalization on the Hexagon convolution boundary

## Finding: the four refused families do not share one root cause

The reading to confirm was that the 370 shape-refused CAMPPlus nodes -- 213
`aten.convolution.default`, 53 `aten.cat.default`, 52
`aten.avg_pool2d.default`, 52 `aten.expand_copy.default` -- all fail for the
same reason, and that one architectural move, normalizing the rank at the op
boundary as a raster blit pair, would reclaim most of them at once.

That holds for the 213 convolutions and for nothing else. Of the 370, **213 are
genuinely one-axis convolutions that a 4-D reading reclaims, and 157 are
refused for reasons no rank reading touches.** The 157 are three different
refusals wearing the same message.

| family | count | is it rank-shaped | measured reason it stays refused |
| --- | --- | --- | --- |
| `aten.convolution.default` | 213 | yes | reclaimed by this change |
| `aten.cat.default` | 53 | no | the fp32 dtype gate; a 3-input cat with 3 of `MAX_CAT_INPUTS = 3` gets a region in fp16 and none in fp32 |
| `aten.avg_pool2d.default` | 52 | no | 128 input channels against `pool_spec`'s exact `POOL_CHANNEL_BLOCK = 64`; needs a new packing, not a rank |
| `aten.expand_copy.default` | 52 | no | `(1,128,1,1) -> (1,128,1,100)` is a periodic tile, which is not a region at any rank |

The convolution family is not a pure unsqueeze/squeeze blit either. All 213
carry a 3-D input **and** a 3-D weight `(O, I, k)`, so the reading has to give
the weight a unit column as well as the activation; a blit pair over the
activation alone would reach a kernel with a weight it cannot index. What makes
the 213 reclaimable is that the DSP's `im2col_convolution_fp16.cc` indexes its
window as `ky = kernel_index / kernelX; kx = kernel_index % kernelX;`, which is
generic in `kernelX`, so `kernelX == 1` is already a walk the kernel performs
and the tap order is torch's cross-correlation order. The 4-D spelling
`(B, C, T, 1)` is therefore not a new kernel but the same command with the
width set to one.

The `cat` count is the sharpest evidence against the uniform claim. Those 53
nodes are within the existing bounds on input count, and the refusal moves
purely on dtype: the same 3-input cat gets a region in fp16 and none in fp32.
A rank reading cannot change a dtype gate, and indeed after this change fp16
has zero `cat` refusals and fp32 has 53.

## What shipped

One reading inside `conv_spec`, not a graph rewrite. No
`ExportedProgram` pass was added, so no node is rewritten and no rank is
changed in the program: the 3-D `(B, C, T)` node is read as the 4-D
`(B, C, T, 1)` one, the graph's own batch stays the command's batch, the time
axis is the height, and the width is one. The one-entry stride, padding and
dilation that the 3-D spelling carries are given the unit width alongside. The
value, the kernel and the result are reshaped with `val.new_empty(shape)`
rather than `unsqueeze`, because a fake tensor's `unsqueeze` inserts a dim of
its own under torch 2.14 and a `view` cannot add one at all.

`ConvSpec` carries a `conv1d` flag. `pack_conv_weight` appends the unit
column to a 3-D weight, and `pack_depthwise_weight` accepts torch's `(C, 1, k)`
one-axis depthwise weight, which is the same bytes as `(C, 1, k, 1)`.
`_emit_convolution` needed no new case: the dynamic length is axis 2 on both
spellings.

This is a spec-level reading, so the emitted command is identical to the 4-D
spelling's own command and the two blobs agree command for command. The 4-D
graph carries exactly one extra command, the trailing alias blit that copies
the 3-D view out of the command's 4-D output buffer, and every other command
matches in type and in every parameter. That is the equivalence control, and it
is asserted per CAMPPlus geometry rather than argued.

## CAMPPlus node counts, before and after

Measured with `to_edge_transform_and_lower` and `HexagonPartitioner`, counting
by `is_node_supported` and decoding the refused family by name.

fp32, 1277 `call_function` nodes:

| | before | after |
| --- | --- | --- |
| supported | 662 | **875** (+213) |
| refused | 615 | 402 |
| `convolution` refused | 213 | **0** |
| `avg_pool2d` refused | 52 | 52 |
| `expand_copy` refused | 52 | 52 |
| `cat` refused | 53 | 53 |

fp16, 1279 nodes: supported 717 to **930** (+213); `convolution` refused 213
to **0**; `cat` refused **0** in both, because the fp32 `cat` count was a
dtype refusal all along.

fp64, 1279 nodes: supported **0** before and after. The dtype gate refuses a
dtype the kernels cannot read, and the one-axis reading does not widen it.
This is the control that shows the change is not a blanket acceptance.

The 213 geometries, for the record: all have a 3-D input, a 3-D weight,
`transposed = False` and `groups = 1`. Input shapes are
`(1,128,100)` x 105, `(1,64,100)` x 52, `(1,512,100)` x 3,
`(1,256,100)` x 2, `(1,288,100)` x 2, `(1,320,100)` x 2. Twelve further
convolutions in the model were already supported and are 4-D.

## Evidence at all three tiers

**Host.** `backends/hexagon/test/test_conv_rank.py` is new, 26 tests.
`pytest backends/hexagon/test/test_conv.py backends/hexagon/test/test_conv_rank.py`
reports **63 passed** with `PYTEST_EXIT=0` in the same run. The tests cover
the delegation itself, the nine CAMPPlus window geometries against their 4-D
spelling command for command, no-bias, depthwise and dilated one-axis forms,
the spec-level equality of the 3-D and 4-D readings, and the 63/64/65 channel
boundary. The boundary is structural and visible in the stream: 63 and 65 carry
a `DSP_OP_ZERO` to clear the ragged lanes and 64 does not, and all three are
bit-exact. Negative tests assert that a transposed one-axis convolution, a
middle group count, an over-VTCM window and a run-time weight each produce **no
delegate at all**, so each still-refused case emits no command. The VTCM gate
moved as the geometry should: over 4864 channels the widest one-axis window
that fits is 8 taps and 9 is refused, where a 5x5 window over the same channels
does not fit. A reverse control hands the 3-D blob the 4-D graph's values with
one element changed and requires the answer to move, so the equality above is
about the bytes that reached the kernel.

**hexagon-sim.** `backends/hexagon/test/test_conv_sim.py` reports **49 passed**
with `PYTEST_EXIT=0`, with `HEXAGON_SDK_ROOT` set and `hexagon_sim._check()`
confirming the SDK first, so an `Unavailable` would have been read as a failure
rather than a skip. The runner's `ConvCase` grew separate `ky/kx` and
`ih/iw` fields and seven one-column cases: 1, 3, 5 and 7 taps, a dilated one,
a 3-channel one and a 2-batch one. Each is compared against `torch.conv1d` on
the 3-D tensor and against `torch.conv2d` on the `(B, C, T, 1)` spelling,
which must agree with each other before the comparison means anything, and
against the unit's own output bit for bit. A second test checks the unit's
weight digest against `pack_conv_weight` called on the 3-D weight, so the
simulator also pins the packing rather than only the product.

The simulator earned its place. On its first run all six padded one-column cases
failed: the runner had handed the kernel `pad_x = pad`, and the kernel
faithifully computed a wider plane. Neither the host interpreter nor a
tolerance check would have caught that, because the wrong answer was a
perfectly consistent answer for a different geometry.

**Device.** `oneplus13-reverse` (OnePlus 13, SM8750, Hexagon v79), through
`lockrun device -- ssh`. The case is CAMPPlus's own shape: a `(1, 128, 100)`
run through a 5-tap window with stride 2 and padding 2, which is 105 of the
213. Weights and inputs are integer-valued in fp16 so every partial sum is
exact, and the reference is computed in fp64 from the same fp16 weights rather
than from the fp16 graph. The runner reported `RUNNER_EXIT=0`, which proves
nothing on its own; the evidence is the file it left.
`ranknorm_out-0.bin` is **12,800 bytes**, 6,400 fp16 elements, which is the
expected `(1, 128, 50)` output. Decoded against the fp64 reference:
**bit-exact, max abs diff 0.0, 0 of 6,400 elements differing.** First eight
values `-514, 1278, 1278, -1410, -514, 1278, -514, -1410`, last four
`-512, 1280, -512, -1152`, identical on both sides. Deployment was under this
worktree's own `hxwin/ranknorm_hx` directory and no other agent's runner or
skel was touched.

## What still refuses, and why

Nothing new is refused. The 157 non-convolution refusals are unchanged, and
each is refused by an arithmetic that a rank reading does not reach: the
`expand_copy` tiles are periodic and a broadcast region cannot express them
(the measured strides are `(4,1,1,0)`, a genuine alias rather than a copy);
the `avg_pool2d` nodes want 128 channels against a pool that is defined for
exactly 64, which is a new packing; the fp32 `cat` nodes are a dtype refusal
that fp16 does not have. The one-axis reading adds its own refusals on top of
the existing gates rather than replacing them: a transposed 3-D convolution, a
group count between 1 and the channel count, a window whose staging does not fit
VTCM, a run-time weight, and a 3-D node that spells its geometry as a pair.

## The reverse direction

It is already free and was not made free by this change: `unsqueeze_copy` and
`squeeze_copy` are in `ALIAS_TARGETS` and emit no command. A 4-D input whose
leading axis is 1 needs no conditional to come back down, which is the
direction the reading was not asked about.

## Files

`src/executorch/backends/hexagon/hexagon_ops.py` carries the reading, the
flag, the two packers and the two helpers `_is_one_axis_geometry` and
`_shape_only`. `hexagon_partitioner.py` has the gate's comment corrected to
say that a 3-D tensor is not a refusal on rank alone. `test_conv_rank.py` is
new, `test_conv_sim.py` and `sim/conv_runner.cpp` grow the one-column tier.
