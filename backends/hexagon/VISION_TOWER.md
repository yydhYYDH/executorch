# Qwen3-VL vision tower on Hexagon

The vision tower runs on the DSP: 24 layers, real checkpoint weights, a small
image, 145 delegates, one .pte, no QNN and no CPU attention.

## What the export does

torch.export cannot see through the tower as HF ships it. HF chunks attention
with cu_seqlens and calls .tolist() on it, and it derives the position
embeddings, the rope tables and the per-image grid at run time. Everything the
grid determines is baked at export instead: the interpolated position
embeddings, from HF's own helper, as one buffer; the 2D axial rope tables as
cos and sin buffers; and the fact that one image is one attention chunk, so
cu_seqlens disappears and the attention becomes a plain non-causal SDPA.

The layer math is HF's, unchanged, and the weights are the checkpoint's
(model.visual.*). Only the forward is rewritten, which keeps one thing true:
the exported module reproduces Qwen3VLVisionModel on the host to max|d| =
1.1e-2 and corr = +0.99998 over the full 24-layer tower, so anything the device
adds is the DSP's, not the module's.

The Conv3d patch embed is rewritten to a matmul by the committed
DecomposePatchEmbed pass, applied through to_edge_transform_and_lower's
transform_passes. Without it the convolution is the tower's one node the
portable kernel rejects.

## What the DSP is asked to run

addmm for the patch embed, qkv, output projection, both MLP layers and both
merger layers; native_layer_norm; bmm and _softmax for the attention; gelu for
the MLP activation; and elementwise mul/add/cat/slice/select for the rope.
The dedicated cache-free vision attention kernel (DSP_OP_VISION_ATTENTION_FP16
= 43, execute_command.cc:737) is dispatched and unused: nothing in the Python
backend emits it, and by the time the graph reaches the partitioner the
attention is already a decomposed bmm + softmax. Using it is the open speed and
precision item below.

## Measured

Device: OnePlus 13 (SM8750, Hexagon v79).

| model | delegates | commands | nonzero rets | corr vs host fp16 |
|---|---|---|---|---|
| tiny ViT (2 layers, hidden 64) | 11 | 89 | 0 | +0.999954 |
| real tower, 2 layers, grid 1x8x8 | 13 | 95 | 0 | +0.997888 |
| real tower, 24 layers, grid 1x8x8 | 145 | 997 | 0 | +0.985257 |

Every run reported exit dN: ok for every delegate and left the phone up.

Where the deviation comes from, measured per stage. The patch embed plus
position embed stage matches to mean|d| = 6e-4 and corr = 0.99999952. Inside
one block, in mean absolute difference on that stage's own values:

| stage | mean abs diff | corr |
|---|---|---|
| layer norm 1 | 5.2e-05 | 0.99999991 |
| qkv matmul | 1.6e-03 | 0.99999325 |
| rope | 1.9e-03 | 0.99999534 |
| attention (bmm + softmax) | 3.8e-03 | 0.99912649 |
| output projection | 6.0e-03 | 0.99987103 |
| mlp | 1.2e-02 | 0.99963410 |

So the layer norm is near exact and the error is carried by the matmuls and
the attention, at 0.02% to 0.1% of the values flowing through them, growing to
a few percent over 24 layers. That reads as a precision profile rather than a
wiring error: nothing is off by more than the rounding of its operands. It is
not settled whether the remaining gap is the DSP matmul's accumulation or the
fp16 activations themselves. On the host, fp16 against fp32 differs by 1.6e-2
on the final output, and the device is 0.77 from the fp16 host, so the DSP does
add something of its own.


## Why the text tower broke: the norm was lowered to fp16 ops

HF's `Qwen3VLTextRMSNorm` is written out by hand, so the exported graph
carries its decomposition: `to(fp32) -> pow(2) -> mean(-1) -> add(eps) ->
rsqrt -> mul -> to(fp16) -> mul(weight)`. The Hexagon kernels are 2-byte
kernels -- the runtime narrows an fp32 operand on the way in -- so every one of
those fp32 intermediates arrives as an fp16 tensor. Two of them cannot hold the
values this model produces:

- The `pow` output saturates at the fp16 maximum. Any activation above 256
  has no square in fp16.
- The reduction narrowed its own result as well:
  `htp_ops_reduce_sum_fp16_inside1_hvx` computed the sum in fp32 and then
  returned it as `__fp16`, so any sum past 65,504 came back as inf before
  the mean could divide it.

Isolating layer 3's input norm on the device pinned the threshold exactly. The
row's sum of squares decided the outcome, not the input magnitude: with a flat
input of 6 the squares are 36 each and the sum is 73,728, and that row turned to
inf, while sums of 51,200 and 62,520 stayed correct and 65,556 turned to inf.
Zeroing row 0 made it clean, and clipping row 0 to 100 did not, so it was never
the size of an individual value in that row.

This matters because the text activations are dominated by outliers of a kind an
fp16 tensor cannot express: layer 2's MLP output peaks at 14,728, whose square
is 2.2e8. The norm then computes x * gamma * inf and the whole row becomes
+-inf, which is what the twelve-stage probe showed -- layer 2 exact, and layer
3's input norm carrying 2,048 non-finite values. The matmul accumulator below is
a real but second-order error beside this one: 0.2% on the same values.

The fix is the pass the backend already ships. `FuseRmsNormPass`
(backends/hexagon/rms_norm.py) matches that exact decomposition anchored on the
weight multiply -- it accepts both the `mul(x, x)` and the `pow(x, 2)`
spelling of the square -- and replaces it with one `et_hexagon.rms_norm`
node, which the emitter lowers to `DSP_OP_LAYER_NORM` with its RMSNorm
flag. One command, accumulating in fp32, with no fp16 tensor in the middle. The
export had simply never installed the pass in `transform_passes`, which is
the only place it can run: it has to be after `to_edge`, because the edge
decomposition table takes `aten.rms_norm` back apart.

Measured after adding it, on the same device and the same input that used to
turn the row to inf:

| graph | bad entries | max abs diff | corr |
| --- | --- | --- | --- |
| isolated norm, real activations (peak 9.2) | 0 | 4.9e-4 | 1.000000 |
| that norm's row 0, the row that was inf | 0 | 0.00000 | 1.000000 |
| layers 2-3, all twelve stages | 0 | 0.0039-0.125 | 0.99993-1.000000 |
| chunk 0-1 of the text tower | 0 of 65,536 | 0.125 | 0.999989 |
| chunk 2-3 | 0 of 65,536 | 0.125 | 1.000000 |
| head, 151,936 logits | 0 of 4,861,952 | 0.362 | 0.999940 |
| the four-layer chain against HF | 0 | 0.366 | argmax 31/32 |

The device op count per delegate drops from 4 to 2 for the norm, because the
mean, the add and the rsqrt leave the graph entirely. The same pass covers the q
and k norms, which removes their zero-padded 128-wide reductions too.

Two kernel fixes were made while chasing this and are kept, because the
reductions that remain in a graph still hit them: the REDUCTION op's vector path
now accumulates in fp32 instead of fp16 `Q6_Vqf16_vadd_VhfVhf` (verified
correct to a sum of 1.8e6), and the innermost reducer divides by the count in
fp32 before narrowing, so a mean of squares past 65,504 no longer returns inf.

## Why the device deviates: the matmul accumulates in fp16

Measured on one block, comparing the device against a host reference that was
run twice, once in fp16 and once in fp32 (max absolute difference):

| stage | device vs fp16 | fp16 vs fp32 |
|---|---|---|
| layer norm 1 | 3.9e-03 | 1.9e-03 |
| qkv matmul | 1.48e-01 | 2.2e-03 |
| attention bmm + softmax | 2.98e-01 | 1.8e-03 |
| output projection | 2.27e-01 | 2.6e-03 |
| mlp fc2 | 8.46e-01 | 4.2e-03 |

fp16 itself costs about 2e-3 per stage. The device costs 10x to 200x that, and
only at the matmuls: the layer norm is not affected. So the gap is not the
model's own fp16 sensitivity and not the emitters' wiring; it is the matmul.

The vendored kernel says why. `htp_ops_loop_matmul_region` (loop_ops.cc:943)
tries an HMX route and then the HVX fast paths, and the HVX fast paths round
the accumulator to fp16 after every multiply-add:

```c
acc = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(acc, prod));   // loop_ops.cc:745
```

Simulating exactly that on the host - accumulate over the 1024 reduction steps,
rounding to fp16 each step - reproduces the device:

| comparison | max abs diff | mean abs diff |
|---|---|---|
| device vs fp32 accumulate | 1.484e-01 | 1.552e-03 |
| device vs fp16 accumulate (simulated) | 7.813e-03 | 3.872e-04 |
| simulated fp16 vs fp32 accumulate | 1.484e-01 | 1.524e-03 |

The simulation's own error from the accurate result is 1.484e-01, the same
number to three digits as the device's. The device is doing what the kernel
says it does.

This is worth more than an explanation: it bounds what any fp16 weight-only
model can achieve on this backend until the accumulator changes, and it applies
to the text tower too, which uses the same kernel. The two candidate fixes are
to make the HVX fast path accumulate in fp32, or to route these shapes to the
HMX path, which the kernel prefers only when the reduction or the output width
is not a multiple of 32 (loop_ops.cc:950). No fix has been written yet.

## Interface the text tower consumes

`get_image_features` returns the merger output plus the three deepstack
features from layers 5, 11 and 17, with no reordering, so the merger output is
already in the order the placeholder positions are filled in. All four are
exported and verified against HF on the host (max abs diff 3.9e-02 to 5.5e-02,
corr 0.99999) and run on the device in one pass:

That was the first skel. With the fp32 accumulator and the reduce_sum_inside1
rename both in, the same four outputs on the same input measure:

| output | max abs diff | mean abs diff | corr |
|---|---|---|---|
| image_embeds | 0.94 | 5.5e-02 | +0.9882 |
| deepstack[5] | 0.41 | 2.5e-02 | +0.9966 |
| deepstack[11] | 0.45 | 3.0e-02 | +0.9971 |
| deepstack[17] | 1.06 | 4.9e-02 | +0.9956 |

All four come from one run of the four-output export (145 delegates, every one
`exit dN: ok`), and its merger output is bit-identical to the one the
merger-only export produces, so the two exports agree exactly.

## The matmul route is a dead end, and the accumulator is the fix

The kernel prefers its HMX route only when the reduction or the output width is
not a multiple of 32 (loop_ops.cc:950), so every shape a transformer uses takes
the fp16-accumulating HVX path. Making it prefer HMX whenever it is eligible
(\`MNN_MATMUL_PREFER_HMX=1\`) improves exactly the shapes it was measured on and
breaks the rest, so the switch stays off:

| shape | old skel (shape rule) | HMX preferred |
|---|---|---|
| 64x1024x1024 | 3.418e-02 max, corr 0.99999 | 1.953e-03 max, corr 0.9999999 |
| 64x1024x1032 | - | 1.953e-03 max |
| 16x1024x4096 | 2.930e-02 max, corr 0.99999 | **1.991e+01 max, corr 0.834** |
| 16x4096x2048 | runs | **aborts, 0x8000040d** |
| 64x4096x4096 | runs | **aborts, 0x8000040d** |

The 17x accuracy gain on the first shape is real and repeatable, but the HMX
general route is not a drop-in: it mis-computes a 4096-wide output and aborts a
4096-wide reduction. \`htp_ops_loop_matmul_hmx_general_eligible\` (loop_ops.cc:560)
checks only positive sizes, two strides and \`E*K*N >= 32768\`, so it accepts shapes
the implementation cannot do - which also means the existing shape rule can reach
that path and abort for a shape as ordinary as an odd output width with a 4096
reduction.

That is why the vision tower's 24-layer export fails in its last delegate under
the preference: the merger's 4096-wide matmuls are the only ones in the model that
hit it. A single transformer block (vpblk) has no such shape and runs fine.

So the accuracy fix has to go where the loss actually is: the multiplier that
rounds the accumulator to fp16 in the HVX fast paths (loop_ops.cc:745). Switching
routes only moves the problem.

Both breaks are now fixed, and they were two independent mistakes. The general
route kept the *whole* packed weight matrix in VTCM (`np*kp*1024*2` bytes = `K*N*2`,
past the unit for every transformer shape) and `vtcm_seq_alloc` never checked its
own bound, so a 4096-wide output corrupted whatever followed VTCM and a 4096-wide
reduction left it entirely. It also issued one deep load of `kp` tiles where the
hardware takes 32 - `attn_hmx_load_k_tiles` (attention_hmx.cc:98) splits long
reductions, the matmul path did not. The route now holds as many 32-column weight
blocks as VTCM takes, splits reductions at 32 tiles, and the allocator returns NULL
instead of overrunning.

| route | 16x1024x4096 + 16x4096x1024 (one MLP block) | 24-layer vision tower |
|---|---|---|
| HVX (default) | max abs 0.0132, bit-identical to the pre-fix build | 145 domains, 12 s |
| HMX general (preference) | max abs 0.0132, within 2.0e-3 of HVX | 145 domains, 11 s |

Preferring HMX still does not pay end to end: the merger's relative L2 error goes
0.1540 to 0.2074 and the 24-layer export 12 s to 11 s. HMX *is* the faster GEMM -
one MLP block (1024x4096, pre-transposed weights, one domain) at 16/64/256 tokens
runs 80/207/676 ms on HVX against 66/131/393 ms on HMX, a 1.2x to 1.7x lead that
is still only 5-16 GFLOP/s - but a GEMM is not where the vision tower spends its
time. A single 64-token transformer block (vpblk.pte, 53 MB, 7 domains) takes
765 ms on HVX, against 275 ms for its 2.15 GFLOP at that measured rate, so roughly
60% of a block is the ops around the GEMM. The route is correct now; the switch
stays off.

## The text tower runs end to end, with the position scheme on the host

Qwen3-VL's text tower cannot be driven by token ids: image features are spliced
into the embedding, and the rotary scheme is M-RoPE, where each frequency pair
takes its position from the (t, h, w) of the token in the merged visual grid
(mrope_section [24, 20, 20], mrope_interleaved), which one input_pos cannot
express. The export therefore takes the embedding and the frequencies:

| input | shape | dtype |
|---|---|---|
| embeddings | [1, 32, 2048] | fp16 |
| input_pos | [32] | int64 |
| rope_cos | [32, 128] | fp16 |
| rope_sin | [32, 128] | fp16 |
| deepstack_0..2 | [1, 32, 2048] | fp16 |

rope_cos/sin are 2D, the shape a freqs_cos[input_pos] gather produces, so the
rope path downstream is untouched. The host computes them with HF's own
Qwen3VLTextRotaryEmbedding, so the position scheme is HF's by construction rather
than a reimplementation. input_pos still indexes the cache and the attention
mask, which is why it stays an input. ModelArgs.rope_from_input selects this,
example_seq_len sets the traced length and deepstack_inputs the number of visual
residuals.

The deepstack residuals are added after text layers 0, 1 and 2 - the 5/11/17
indexes are vision-side, where the features are collected - matching HF's
_deepstack_process.

Verified against a 4-layer HF reference for the prompt
"<|im_start|>user\n<|vision_start|>[16 image tokens]<|vision_end|>What colour is
this image?<|im_end|>\n<|im_start|>assistant\n" with a 128x128 red/blue image
(grid 1x8x8, 16 merged tokens, 32 positions):

| image features | max abs diff | corr | argmax match | next token |
|---|---|---|---|---|
| HF | 2.6e-02 | 1.00000 | 32/32 | exam (ref: exam) |
| DSP vision tower | 4.99e+00 | 0.99273 | 26/32 | exam (ref: exam) |

so the text tower is numerically equivalent to HF when it is fed HF's features,
and the end-to-end path predicts the same token. The whole deviation of the
end-to-end run sits in the 16 image positions and is the vision tower's feature
error above, not the text tower: with HF features every position matches.

## The whole model, end to end, on the device

The text tower runs as fifteen ptes -- fourteen two-layer chunks and the head --
each exported, pushed and deleted in turn so the host stays near one chunk. The
image path is the 24-layer vision export followed by the splice. On the prompt
"<|im_start|>user\n<|vision_start|>[16 image tokens]<|vision_end|>What colour is
this image?<|im_end|>\n<|im_start|>assistant\n" with a 128x128 red/blue image
(grid 1x8x8, 32 positions), against a 28-layer HF reference built by HF itself
(`get_rope_index` for M-RoPE, `deepstack_visual_embeds` and
`visual_pos_masks` for the residuals):

| what ran | max abs diff | corr | argmax | last position |
| --- | --- | --- | --- | --- |
| text tower, HF features | 0.633 | 0.999900 | 32/32 | 785 = HF |
| text tower, DSP vision features | 9.69 | 0.992878 | 30/32 | 785 = HF |

The two positions that differ in the second row are image positions (9 and 11),
and with HF's features every position matches, so the whole deviation of the
product sits in the vision tower's features -- the same conclusion the
four-layer table reached, now at full depth. Both rows now use all four vision
features from one run of the current skel. Before that, with the three deepstack
features still the previous skel's, the same pipeline measured 28/32 at corr
0.991113 with positions 9, 11, 13 and 15 differing. Per-chunk, the device reproduces an eager chain
running the same modules to corr >= 0.99991, with max abs diff equal to one ulp
of the 14,700-scale activation channel that dominates the residual stream (c00:
0.125, c01-c13: 16.0 on a peak near 14,800).

Device-side evidence for one run of the vision export, which is 145 delegates:

    [hexagon] enter d144: ops=25 act=72810496 first=0 count=25
    [hexagon] exit d144: ok

and every text chunk reports the same for its own delegates. The embedding
lookup and the M-RoPE tables are computed on the host, as they are for any
llama-style pipeline; nothing else of the model runs outside the DSP.

## Where the remaining vision error comes from

Per-stage measurement of vision block 0 on the current skel, the device against
the same module run eagerly on the host in fp16:

| stage | peak | max abs diff | corr |
|---|---|---|---|
| norm1 | 7.83 | 3.9e-03 | 1.00000 |
| qkv, fused | 7.76 | 3.9e-03 | 1.00000 |
| q_roped | 7.80 | 3.9e-03 | 1.00000 |
| qk, batched | 26.53 | 1.6e-02 | 1.00000 |
| softmax | 1.000 | 1.1e-01 | 0.99843 |
| attn | 3.72 | 2.8e-01 | 0.99916 |
| proj | 9.13 | 2.2e-01 | 0.99989 |
| res1 | 28.06 | 2.2e-01 | 0.99995 |
## Exporting the tower again

The scripts live under `tmp/` and artifacts are written next to them, never to
`/tmp`. They were recovered from the DSH session store
(`~/.dsh/sessions/--home-yydh-executorch--/*/session.jsonl.zstd`, zstd-compressed,
so a plain grep finds nothing); `tmp/recover_scripts.py` pulls a file back out of
a transcript by its `file_path`, which is how `vit_real_export.py` and
`vl_e2e.py` were brought back.

```bash
cd /home/yydh/executorch/tmp
/home/yydh/miniconda3/envs/et/bin/python vit_real_export.py \\
  --layers 24 --grid 1,8,8 --tag v1vit24 --export
```

The checkpoint path is baked into the script
(`/home/yydh/Models/Qwen3-VL-2B-Instruct`). Lowering uses `DecomposePatchEmbed`
and `HexagonPartitioner`; the run prints HOST PARITY against HF and writes
`v1vit24.pte`, `v1vit24_in.bin` and `v1vit24_ref.bin` under `tmp/`.

`HEXAGON_HMX_PREPACK=1` (with `HEXAGON_HMX_TILE_BUDGET=N`) packs matmul weights
into the HMX tile order at export, but only for shapes the DSP routes to its
general HMX kernel (`hmx_prefers_general`: a non-multiple-of-32 dimension, or a
32-wide one); everywhere else it is gated off because forcing that kernel there
is slower.

Running it on the device, with the tower's own skel directory and the runner
built by `build.sh runner`:

```bash
cd $B/hxwin
export LD_LIBRARY_PATH=$B/libs:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=$B/hxskel_fix1
HEXAGON_ACCT=1 ./executor_runner_p2 --model_path $B/hxvlt/v1vit24.pte \\
  --inputs $B/hxvlt/pixels.bin --output_file out --print_output none
```

Measure with `HEXAGON_ACCT=1` and **not** `HEXAGON_TRACE=1`: the trace prints
every command of every delegate and starts a watchdog thread per delegate, which
costs about 12 s on this tower and once made a 4.9 s inference look like 16 s.
`HEXAGON_ACCT` prints one line per delegate (input bytes, input/flush/call/total
ms) with no other change in behaviour.

Baseline with the fixes in, 128x128 input, 1x8x8 grid, untraced: 145 delegates,
input bytes 690 MB per execution, `call_ms` sum 3.89 s against 3.83 s of DSP
kernel time reported by `optime`, host copies 0.18 s, about 0.8 s of framework
work, and a per-inference wall of 4.9 s (5.9 s including the ~1.0 s load of the
768 MB pte). Output md5 `23ffd2e2` on `hxskel_fix1`, which is the skel this
number belongs to.


Every matmul is at the noise floor, the fused qkv at 3.9e-03 and the batched
QK^T at 1.6e-02 on logits that reach 26.5, both corr 1.00000, and so are the
norms and the rope. The deviation enters at the softmax: its probabilities come
back with max abs diff 0.11 at corr 0.99843, and the device picks a different
winning key in 129 of the 1024 attention rows. The attention state then carries
0.284 of error, which is seventy percent of what the block ends up with (0.414)
and what the image features inherit.

The logits explain the sensitivity. Inside a row the spread is 27.8, so the
distribution is close to one-hot and a row whose top two logits are close is
decided by arithmetic detail rather than by content. The median top1-top2 gap is
0.585 and the tenth percentile is 0.098, so the rows that flip are the near
ties, but an fp16 softmax flips an order of magnitude more of them than the
reference's fp32 one does.

`softmax_ops.cc` computes the whole row in fp16: the max, the exp2 (a
fifth-order qhmath polynomial, whose own error is small), the running sum, the
reciprocal and the final scale. The sum and the reciprocal are the same defect
the two reductions had, and the fix is the same one: keep them in fp32 and
narrow once at the end.

What is left open is the vision tower's own feature error, and it is what the
end-to-end numbers above carry: image_embeds at corr 0.9882 with max abs diff
0.94 on a peak of 8.5. The fp32 accumulator cut that from 5.52 and the deepstack
features from 2.87 to 1.06, so what remains lives in the parts of the vision
tower the accumulator change did not touch.

## Enabling the DSP for the text tower: the hook was in a dead path

The text export produced a pte with zero delegates and ran on the CPU (5.6 s
wall, 100% cpu, no [hexagon] line at all), while the vision pte ran 145 delegates
in 12.3 s. The EXECUTORCH_HEXAGON_PARTITION hook existed only in
_to_edge_and_lower_llama_xnnpack, which this flow never calls: the active flow is
_to_edge_and_lower_llama (export_llama_lib.py:1456), which builds its own
partitioners list. With the hook added there the pte carries 5 delegates
(26 + 63 + 63 + 63 + ... ops) and every one reports "exit dN: ok" - but the
logits come back NaN, and the pte grows from 1.03 GB to 1.86 GB.

The size jump is the clue: the partitioner runs before the fp16 dtype override
reaches the constants, so the delegated weights are serialized fp32 (about
+0.7 GB for the 151936x2048 head alone) while every DSP kernel reads two bytes
per element. The vision export does not hit this because it traces the module in
fp16 to begin with. The fix is to build the text graph the same way, in its own
export script, instead of casting after partitioning.

## What a 24-layer run spends its time on, and the two fixes that came out of it

Same device, same session, old and new runner alternating so the phone's state is
the same for both legs. The model is `v1vit24z.pte` (73 delegates, 1165
commands), `-num_executions 3`, account and phase instrumentation on.

| | before | after |
|---|---|---|
| input copy per execution | 104.4, 104.5 ms | 61.7, 59.8 ms |
| delegate init, `-num_executions 1` | 530 ms | 489 ms |
| output md5 | 18b6997d | 18b6997d, four legs of four |

Neither change re-exports anything.

**The fp32-to-fp16 narrowing runs four lanes at a time.** Every delegated input
that arrives fp32 is narrowed into its fp16 slot, and a layer's residual stream
is 64x1024 values, so that loop was the whole cost of an input copy: 90 ms of the
104 ms per execution. The scalar form rounds by adding 0x1000 to the mantissa and
carrying into the exponent when that overflows, and both steps are the same
integer arithmetic on all four lanes, so the vector form copies it exactly rather
than using `vcvt_f16_f32`, which rounds to nearest even and would change the bits
a re-run produces. Lanes fp16 cannot reach -- subnormal, infinity, overflow --
fall back to the scalar form. The lane arithmetic was checked against the scalar
form over 4 million random patterns plus every mantissa that forces a carry, with
no mismatches.

**The resident block is no longer zero-filled.** Its weights section is memcpy'd
over it a few lines later, its command and sync sections are written by the
descriptor builder, and its group array has its own memset, so the pass over 659
MB bought nothing. The scratch block is still zeroed by the pool: it holds
activation slots, which the blob does not carry.

One execution is 594 ms with both fixes in, and it divides three ways:

| part | ms |
|---|---|
| DSP calls, 73 delegates | 435 |
| narrowing and input copies | 43 |
| portable kernels between delegates | ~116 |
| init, once per load | 489 |

The 116 ms is the guard chain `_safe_softmax` decomposes into: per layer
`eq`, `any`, two `logical_not`, `full_like`, `where` and two `mul.Scalar` --
about eleven small ops, 24 times over, at ~0.4 ms each of dispatch and fp32
elementwise work. It is a no-op on this graph (no mask, so no all-minus-infinity
row) and all three exports that drop it land within fp16 noise of the host
reference: cos 0.9996160 against 0.9996200.

### Merging the delegates costs the DSP twice the time

Four layers, one input file, all four exports run back to back on the same
phone:

| export | delegates | call per execution | input copy |
|---|---|---|---|
| sdpa (the guard chain) | 13 | 94.3 ms | 10.1 ms |
| explicit softmax, every op in one delegate | 1 | 174.0 ms | 0.0 ms |
| one `clamp` fence before the softmax | 5 | 176.9 ms | 0.8 ms |
| `clamp` fences on both sides of the softmax | 9 | 177.4, 179.8 ms | 0.9 ms |

The 24-layer pair says the same thing at scale: 940 ms of DSP time for the
single-delegate export against 463 ms for the split one.

Windowed execution pins down where the extra time lands. Running a suffix of the
single-delegate blob with `HEXAGON_CMD_START` and `HEXAGON_CMD_LIMIT`, and
reading the binary slot out of the per-command profile:

| window | what it holds | binary slot |
|---|---|---|
| [15,+7) | 3 binaries, no matmul | 0.23 ms |
| [25,+7) | 2 binaries, no matmul | 0.17 ms |
| [60,+8) | 3 binaries, no matmul | 0.22 ms |
| [31,+4) | matmul, binary, softmax, matmul | 22.85 ms |
| [31,+48) | one whole layer | 21.8 ms |

A binary that runs right after a score matmul inside the same command group is
charged about 21 ms, once per layer, while the matmul's own slot reports 0.42 ms
for submitting both of its instances. The number is the same in every window that
contains one, and the same at 4 layers and at 24, so it is a stall rather than
work: nothing in the command, the shapes or the loop parameters differs from the
split export, where the same adjacency costs 0.05 ms. The one structural
difference is that in the split export the score matmul and the softmax each sit
alone in a one-command delegate.

That is what the guard chain is doing for the DSP, and it is why removing it --
by exporting `torch.softmax` directly, by fencing with `clone` (which
`to_edge` rewrites to `_clone_dim_order` and the partitioner then absorbs), or
by fencing with one or two exact no-op `clamp`s -- gives back more than the
116 ms it saves. The fp16 case for the fence is otherwise clean: `clamp` to the
fp16 range is exact on fp16 data, and both fence exports reproduce the merged
export's output bit for bit (md5 73e5df61) while the sdpa export differs only in
the last place (md5 cd4fa67d, cos 0.9996160 against 0.9996200).

### A load costs 0.76 ms per MB of pte

| export | blob | delegate init |
|---|---|---|
| four layers, split | 154.6 MB | 254, 258 ms |
| 24 layers, split | 659.6 MB | 636, 661, 640, 606 ms |

Repeating the same load back to back, with 16 GB of the phone's 23 GB in the page
cache, does not move it, so the size law is not file I/O: it is the weights
crossing into rpcmem at about 1.3 GB/s, plus ~140 ms of per-delegate setup that
does not depend on the blob. Only a smaller pte shrinks the first term, which
makes the weight width -- not the delegate structure -- the one lever left on the
load path.

### The weight fill was stride-bound, not byte-bound

The HMX path re-arranges a 32-column weight group into the unit's tile order on
every execution. Storing that order in the blob instead (HEXAGON_HMX_PREPACK,
now on by default) costs nothing -- the reordered bytes are the same size, so the
blob is the same size, and the output is bit-identical -- and it turns the fill
into one straight run per group: a row-major (k, n) weight hands the DSP 32 rows
of 64 bytes at a pitch of n * 2, which is a stride no prefetcher likes.

The copy also had to leave libc's memcpy behind to pay off. A byte-wide copy of a
DDR block ran 2.4x slower than the element rearrange it replaced; the same copy
128 bytes at a time runs 1.7x faster than it.

| 24 layers | load | per image | DSP call |
|---|---|---|---|
| row-major weights | 576, 583 ms | 422, 432, 430 ms | 465 |
| pre-packed weights | 562, 561 ms | 292, 316, 317 ms | 308 |

Both exports reproduce md5 18b6997d, and the same holds at 2 and 12 layers.

### The scale was splitting the graph, and it did not have to

A fp16 tensor times a python float is not a fp16 multiply. torch and the
portable kernel both promote the scalar to fp32, multiply there and round the
product back, which differs from a fp16 product on about one element in six -- so
a fp16 product on the DSP changes the output, and the op stayed on the host. Two
of them per layer is what split the tower into 25 delegates, each paying a copy
in and a copy out.

Doing the fp32 step on the DSP removes the reason to hand it out. The DSP widens
each half to fp32, multiplies by the fp32 scalar and narrows the product back,
which is the same rounding the host path does, so the output is bit-identical and
the whole tower becomes one delegate.

The lane order is the trap: the packed widening needs a `vshuff` in front of it
and the matching `vdeal` behind it, or every element comes back in the wrong
lane. A fp16 multiply would hide that, because a fp16 product of the wrong two
lanes is still a plausible number.

| 24 layers | load | per image | DSP call | host in+out | md5 |
|---|---|---|---|---|---|
| scale on the host, 25 delegates | 559, 567 ms | 248-272 ms | 188 ms | 69 ms | 18b6997d |
| scale on the DSP, 1 delegate | 521, 539 ms | 167-174 ms | 170 ms | 0.2 ms | 18b6997d |

Merging is faster on the DSP side too, so the split was not buying anything: 188
ms of call time became 170 ms, and the delegate init a pte pays per partition
shows up in the load as 559 -> 521 ms. Steady state over eight executions is
167-174 ms an image, from 697 ms at the start.

### Where the remaining 170 ms goes, and why the weights are the floor

One tracing execution of the merged tower, in milliseconds per image (DSP kernel
time per op, from the profile slots; the call is 170 ms in total, so about 17 ms
sits in the command loop and the RPC rather than in a kernel):

| | ms | of which |
|---|---|---|
| matmul (slot 38) | 111 | weight fill 71, output store 26, activation pack 11 |
| binary elementwise (19) | 19.6 | |
| raster blit (3) | 8.7 | |
| unary, the 48 scales (4) | 6.5 | |
| row guard (135) | 3.0 | |
| convert / gelu / softmax | 4.0 | |

The fill is 663 MB an image, which is the whole model read once, and it runs at
9.3 GB/s -- so the matmul is weight-bandwidth-bound rather than compute-bound,
and not by a little. The grid is 1x8x8, so a token block is 64 rows: every weight
byte is used 64 times, while the HMX unit would be happy to use it thousands of
times. At 2 MB of weights per matmul the fill is 215 us of DDR and the compute is
about 34 us.

That is also why the DMA engine was the wrong idea. The fill is the largest
single item, the engine is a second path to DDR that does not touch the vector
unit, and the convolutions, the LSTM and the attention path all move weights with
it. Handing it the same copy here did nothing: a build that samples the head,
middle and tail of the destination and falls back to the vector copy on any
mismatch produced bit-identical output and bit-identical time, which says every
transfer fell back. The scaffolding went with the finding -- an inline copy that
sometimes does nothing is worse than a slow one.

The store slot looked like the next item, and it is not what it looks like. One
column group of 32 columns is written per nt step, so each row of the tile leaves
as a 64-byte store at a 2048-byte stride -- a partial cache line per row -- and
the slot costs 26 ms. Staging the block in VTCM and draining whole 2 KB rows
instead was built and measured: the output stayed bit-identical, but the store
slot went 26.06 -> 32.05 ms and the call 169 -> 176 ms, with the fill unchanged
at 73 ms. Partial-line writes were never the problem. The per-group work is two
tile loads and a drain of a 32x32 accumulator over K=1024, which is about 2 us of
unit time, and at 6900 drains an image that is what the slot is: the unit doing
its job, plus the drain's own store. It is not a locality problem and there is
nothing to win by rewriting it.

That leaves the fill as the only large item, and the honest lever is not a faster
copy but an overlap. The fill is 73 ms of vector-unit DDR reads and the rest of
the matmul is 37 ms of unit work, on two different units that currently run one
after the other: every matmul here is a single weight pass, so there is no next
pass to fill during, and the pass order is fixed by the activation pack, which is
per row block and cannot be repeated per column group. Overlapping them means
staging the activations for every row block in VTCM and splitting the weight fill
per column group so a group's unit work can start while the next group is still
arriving. Measured aggregate DDR use says there is room: the phases together run
at 5.8 GB/s while any phase on its own reaches 9.3.

That overlap was built and measured, and it does not pay either. Every row
block's activation was staged up front -- which also stops the pack from being
repeated once per pass -- and each group's unit loads were issued before the next
group's bytes were moved in, sliced across the row blocks so the vector unit
never sat idle. Output stayed bit-identical, and the split shows why it does not
help:

| | fill | store | call |
|---|---|---|---|
| one copy then compute | 73.06 | 25.97 | 169, 170 |
| fill overlapped with the unit | 69.38 | 32.03 | 169, 178 |

The fill did get cheaper (73.06 -> 69.38, its slices prefetch better than one
long run), but the store slot grew by more than that (25.97 -> 32.03), and it is
the same +6 ms the VTCM-staged store showed. Both rewrites put vector-unit writes
into VTCM next to the unit's own tile reads, and VTCM is one resource: the copy
starves the accumulator, so the time just moves from the fill to the drain.

The per-command overhead is not worth chasing either. The map lookup a command
does is already a small hot array (mmap_mgr.cc), so what is left of the 17 ms
outside the kernels is the profile instrumentation itself, which production runs
do not carry.

What that leaves is the floor: 663 MB of weights an image at 9.3 GB/s is 71 ms,
the units run the rest, and the element-wise ops are already at 8 GB/s. Weight
reuse is the whole story -- 64 rows means every weight byte is used 64 times --
so the levers that remain are the ones that change how many bytes are read
(quantization) or how many rows use them (more rows per weight load).

### Every command's traffic, read off the blob

Guessing at the element-wise ops had run out of road, so the blob was read
directly (tmp/opbytes.py, tmp/opdetail.py -- the op table carries every operand's
space and byte size, so the whole DDR budget is in it):

| type | ops | reads MB | where |
|---|---|---|---|
| batch_matmul | 147 | 692.5 | **weights 657.5**, activations 34.9 |
| binary_elementwise | 390 | 80.9 | activations 79.4, weights 1.6 |
| raster_blit | 408 | 78.6 | activations 78.6 |
| unary | 145 | 28.4 | activations |
| layer_norm | 49 | 6.4 | activations |
| softmax | 24 | 3.2 | activations |
| **total** | 1163 | **890** | plus about as much written back |

Two things this settles. The binaries are not stray arithmetic: 290 of the 390
are broadcast multiplies against a weights vector (1024 or 4096 halves) -- the
RMSNorm gamma and its siblings -- and each pays a full 128 KB read and 128 KB
write of the activation. They cannot be folded into the preceding matmul's
epilogue because the graph rounds to fp16 before the multiply and the epilogue
would round once, which is a different byte.

And the 408 blits are not redundant copies. Reading their regions out of the
table, they are the attention's head split and merge: 96 of them walk a
(64,16,64) operand with strides (1024,64,1) into (16,64,64), and the rest are
their inverses and the three-way QKV cut. That is real data movement through
DDR, about 8.8 ms of it, and the only way out is not to move it: have the
attention matmul read the (rows, heads, dim) layout directly -- the activation
pack already walks rows by stride -- or emit the tower's decomposed attention as
one MNN vision-attention command, which this tree carries but the AOT side never
emits.

### The matmul, split three ways

Which of the two halves of the matmul is the one to attack is a measurement, not
an argument, so the kernel got two build switches that leave the shape of the
work alone and only drop a part of it: one that skips the compute for every
weight pass, one that skips the weight fill. Both produce wrong output on
purpose; the point is the clock.

| build | what runs | per image | call |
|---|---|---|---|
| base | everything | 171, 169, 172 | 170 |
| no compute | fill, packs, every other command | 62, 61, 66 | 63 |
| no fill | compute, packs, every other command | 90, 90, 92 | 90 |

Subtracting: the weight fill is **80 ms**, the unit's own work is **27 ms**, and
everything that is not this matmul is **63 ms**. The three add to 170 exactly.

Two things fall out. The store slot that looked like a locality problem and
weighed 26 ms is the unit's 27 ms seen from the waiter's side -- the store itself
is nearly free, which is why staging it in VTCM could only make things worse. And
the fill and the compute, run separately, cost 63 + 90 = 153 ms; run in the same
kernel, as they are, they cost 170. Interleaving them costs 17 ms more than
running them one after the other, which is exactly what both overlap attempts
measured from the other direction.

So the budget is: weights 80, everything else 63, unit 27. Reaching 120 ms would
mean cutting 50 of the 90 non-weight milliseconds, and even removing every
element-wise op, every layout blit and every pack -- all of them -- lands at
about 117. The ceiling is real: this model's weights are 663 MB an image, and at
the 8.3 GB/s the fill actually achieves that is 80 ms before the unit has done
anything.

### The matmul epilogue fold: implemented, measured, reverted

The element-wise ops that follow the matmuls looked like the one class of work a
matmul's own store could absorb: an mm and a mul(act, gamma) next to each other in
the blob, roughly 240 commands and 13 ms of traffic by the inventory above. It was
built -- the store applies a per-column multiply or add in fp32 between reading the
fp16 tile and writing it back, the vector rides in input slot 5 of BATCH_MATMUL
(which no path reads) and the mode in params[28], past the packed loop descriptor;
the encoder folds by appending to the command the matmul just emitted, guarded on a
single user and on that command writing exactly the operator's result. Two commands
per layer survived the guards, and the reason is worth keeping:

* The consumers of a matmul are not the element-wise ops. Every mul in a layer reads
  a cat, a squeeze_copy, b_cos or b_sin; the adds read a mul or another add. A
  binary that sits directly after an mm in the blob is a coincidence of emission
  order, not a data dependency -- reading adjacency out of the blob is not reading
  the graph.
* What the fold did catch was one add(mm_out, bias) per layer: 24 commands of 1163.
  Measured in one session, back to back, 201.7 ms without it and 201.4 ms with it.

The verdict was still a revert, because the fold is not bit-exact. The folded output
differs from the unfolded one (md5 331b38e4 against 18b6997d): the DSP narrows with
Q6_Vhf_vcvt_VsfVsf, and that does not round the same way as the portable fp16 add in
the binary kernel. A single-rounding epilogue is only exact when the operator it
replaces is itself single-rounding, which the graph's fp16 -> fp32 -> op -> fp16
sequence is not.

Both sides were restored and checked: the skeleton rebuilds to
48cdc5d129b4d85292e67fc3ba080fad and a fresh 24-layer export is byte-identical to the
reference blob (5581899de2eb095d4744cc29ac427671). Note the device ran at 201 ms for
this build tonight against 170 ms earlier -- the A/B above is like-for-like, but
absolute numbers from different sessions are not comparable. Cheaper than folding is
deleting: the 78.6 MB of raster blits and the 80.8 MB of binaries only exist between
layout ops, so the traffic to attack is the layout, not the arithmetic.

### Where the call actually goes: 101 ms before any kernel runs

Two diagnostic skeletons answered the question the round-5 split left open. Skipping
the movement of all 408 raster blits (keeping their dispatch) leaves the call at
201.6 ms against a 201.4 ms base, and skipping all 389 element-wise ops leaves it at
202.8. Neither family costs anything measurable, so the 63 ms that the no-compute leg
was carrying is not their arithmetic.

Returning instead at the very top of execute_single_command -- before the mmap
lookup, the cache clean and the flatbuffer parse -- for every one of the 1163
commands lands at 101.1 ms, and cutting the same function at three later points
(after the mmap, after the clean, after the parse) changes it by less than 0.3 ms:
101.7, 101.2, 101.0. The per-command path is free. The 101 ms is spent outside it,
in the group entry (VTCM acquire, mmap init, both sync_group_tensors calls, the
profile setup) or in the FastRPC call itself, and it is a floor that does not depend
on what the commands do.

Read against the 201 ms base, that is: ~100 ms of kernels, of which the weight fill
is ~80 and the unit ~27, and ~101 ms that the command group pays whatever it is
asked to run. The obvious next measurement is the same bisection one level up,
inside htp_ops_execute_command_group, which separates the group's own setup from
the transport around it. The device was running ~200 ms this session against the
170 ms recorded earlier, so these numbers compare within the session only.

### What the group entry costs, and what it does not

The round-7 number needed correcting. Cutting the group entry at seven points,
back to back on one device session:

* return before the VTCM guard: 101.3, 0.79, 0.61 ms across three executions. The
  first call carries a one-time ~100 ms, the rest cost nothing. That also explains
  the 63 ms and 101 ms floors measured earlier, which were first-call effects.
* keep only the matmul kernel and dispatch the other 1015 commands to nothing:
  202.1, 204.0, 201.4 ms against a 201.5 ms base. Every blit, cat, permute, unary,
  norm and softmax in the tower is free. The whole call is the matmul path.
* return just before the command loop, after the group's own setup: 103.0, 101.2,
  101.0, 101.2 ms. So the group entry costs ~100 ms on every call, not just the
  first, and it is spent before the first command runs.

Three candidates for that ~100 ms were removed and none of them moved the number:
keeping the VTCM window acquired across calls and building the fd-to-pointer table
once (201.7 against a 201.5 ms base), and dropping the pre-execute sync group
entirely, which is the flatbuffer naming every operand of all 1163 ops including
the 663 MB of weights (201.6 against a 201.4 ms base). The two statements left in
that window -- the group's mmap lookup and a qurt_mem_cache_clean over its 14 KB of
command entries -- should each cost microseconds, so the next move is to stamp
HAP_perf_get_time_us around each of them rather than infer from more cut points.

Read together: 201 ms is ~100 ms of group entry that no candidate explains, plus
~101 ms of matmul (fill ~80, unit ~27), plus nothing at all for the other 1015
commands. The layout and elementwise work that earlier rounds treated as targets is
already off the critical path, and the fill remains the largest thing that is
actually understood.

### The call, itemised

Stamping the DSP with the profile counters it already carries (slots 200-212,
cumulative microseconds, read back with HEXAGON_PHASE=1) gives the whole call in one
run of the verified build, output md5 18b6997d:

| item | ms |
| --- | --- |
| weight fill (200) | 72.8 |
| activation pack (201) | 11.2 |
| unit tile loads (202) | 0.4 |
| output store (203) | 26.0 |
| matmul kernel, all commands (204) | 111.6 |
| whole htp_ops_batch_matmul (208) | 113.3 |
| group entry, guard to loop (211) | 13.5 |
| VTCM guard (212) | 0.019 |
| **call_ms** | **204.4** |

The bracket around htp_ops_batch_matmul settles two questions that earlier rounds
left open. There are 147 batch_matmul commands and the HMX plan path fires for 48 of
them, but `htp_ops_loop_matmul_batch_hmx_prepare` costs 68 us in total across all
147, and `hmx_manager_enable_execution` plus `hmx_unit_acquire` cost 125 us in total.
Per-command HMX setup is not a cost, and neither is the matmul dispatch: the kernel
is 111.6 of the 113.3 ms the function spends.

So the matmul path is 113.3 ms and the group entry is 13.5 ms, against a call of
204.4. That leaves ~78 ms that no instrumentation in either place accounts for, and
it scales with the command count: 1163 commands at ~66 us is 77 ms, and a loop of
1163 commands that return at the very top of execute_single_command also costs ~87 ms
above the entry. The per-command overhead is real and it is not the kernel.

### The 101/201 staircase was the host watchdog

Superseding the section below. A traced delegate starts a ProbeWatchdog thread; its
Loop slept in 100 ms slices and the destructor called join() on it, so every traced
invoke paid whatever was left of the slice as dead time. Measured on the same session
and the same skeleton, HEXAGON_TRACE=1, three executions each: watchdog unset 201.9 /
201.4 / 201.5, HEXAGON_WATCHDOG_SECONDS=60 204.3 / 201.6 / 203.8, and
HEXAGON_WATCHDOG_SECONDS=0 171.7 / 171.0 / 169.6. The staircase is a host thread,
not the DSP, which is why removing DSP work never moved it and why the documented
167-174 ms steady state reappeared the moment tracing was off.

The wait is now interruptible: Loop blocks in cv_.wait_for(lock, seconds_, stop_) and
the destructor sets the flag under the same mutex and notifies before joining, so join
returns immediately instead of at the end of a slice. With the default interval and
HEXAGON_TRACE=1 the patched runner measures 162.4 / 173.2 / 173.0 / 173.4 / 172.0 /
172.6 over six executions, output md5 18b6997d, and zero watchdog dumps -- the dump
path is untouched, it just waits on the condition variable. Tracing now costs ~3 ms
instead of ~35 ms.

Untraced, which is how the tower actually runs, the same build measures 163.8 / 168.9 /
170.1 / 171.6 / 168.5 / 169.8 ms with in_ms 0.06 and total_ms equal to call_ms to
within 0.1 ms. There is no host overhead left to find: the ~90 ms gap between call_ms
and total_ms that round 7 saw was tracing, the probe ring, and this watchdog.

Everything in the sections below that reads a DSP-side phase counter still stands --
those numbers are read inside the call and never included the join. What changes is
the per-image figure: 169 ms untraced, of which ~140 ms is measured DSP kernel work
and 67.8 ms of that is the weight fill.

### call_ms is not a metric: it is bimodal

Bracketing the dispatch itself settles where the loop's time goes, and then four
models over the same skeleton overturn the number the brackets live in. The dispatch
of all 1163 commands is 147.6 ms (slot 209), of which the 147 matmul commands are
110.0 (slot 208) -- so the other 1016 commands really do cost 37.6 ms, whatever the
legs that removed their kernels said. The whole per-command prologue -- mmap, cache
clean, flatbuffer parse -- is 0.27 ms across all 1163 (slot 210).

Then the same build over four exports:

| model | ops | call_ms, three executions |
| --- | --- | --- |
| seq1, 2 layers | 59 | 101.1, 101.4, 101.3 |
| v1l4n, 4 layers | 205 | 201.4, 203.7, 201.6 |
| v1vit12m, 12 layers | 587 | 102.5, 102.9, 101.2 |
| m1, 24 layers | 1163 | 201.4, 201.6, 200.9 |

Two layers and twelve layers cost the same 101 ms; four layers cost 201. The value
does not follow the command count, the weight bytes or the layer count -- it is
bimodal at ~101 and ~201 ms, and it is stable within a model across executions.

That is the artefact the last three rounds were chasing. A call whose DSP-side work
is a few milliseconds reports 101 ms, which is why every leg that removed work --
blits, element-wise, sync groups, cache maintenance, VTCM, the group's own entry --
measured the same number and read as a null result. Those nulls are uninformative,
not negative, and the 63 ms and 101 ms floors of rounds 5 to 8 are the same artefact
seen through a different cut point.

The authoritative numbers are the DSP's own phase counters, because they are read
inside the call and are not quantised: fill 70.9, pack 10.6, unit 0.4, store 26.0,
matmul kernel 108.5, all matmul commands 110.0, all commands 147.6, prologue 0.3.
That is ~146 ms of measured work in a 201 ms call, and any A/B from here has to be
read either from these slots or from many repetitions of end-to-end total_ms, never
from a single call_ms.

### Per-family, and one hypothesis tested properly

The profile array is indexed by op type, so the family split needs no build at all,
just HEXAGON_PHASE=1 and reading slots below 40. Over one execution of the 24-layer
tower (md5 18b6997d):

| op type | family | ms |
| --- | --- | --- |
| 38 | matmul | 106.3 |
| 19 | binary elementwise | 18.5 |
| 3 | raster blit | 8.6 |
| 4 | unary | 6.5 |
| 8 | layer norm | 1.4 |
| 28 | softmax | 1.2 |

Inside the matmul: fill 67.8, output store 26.0, activation pack 10.0, unit 0.4.

Target (3) turns out to be largely done. The elementwise broadcast path does not
fall to element-at-a-time code: both scalar-operand chunk functions splat the
operand with Q6_Vh_vsplat_R and run vector loops over ADD, ADD_RELU, SUB, MUL,
SQUARED_DIFFERENCE, MAX, MIN and MUL_SILU, with a scalar tail. The right-hand
multiply uses Q6_Vhf_vmpy_VhfVhf precisely because it rounds like scalar a * b;
widening to qf16 and narrowing back does not.

So the 18.5 ms is traffic, not arithmetic -- and the dispatch hypothesis was worth
testing anyway, since the worker pool is entered whenever a row has 2048 or more
elements and nearly every op in this graph does. Raising that threshold to 262144
(HTP_OPS_BINARY_MT_MIN_FP16_ELEMS) so small ops take the serial vector path made the
family *slower*: 21.1 ms against 18.5 ms in the same session, with the output md5
unchanged. The pool earns its dispatch cost at these sizes. Thresholds restored and
the skeleton rebuilt byte-identical to 48cdc5d1.

Measured work for the whole graph in this session: 67.8 + 26.0 + 10.0 + 0.4 = 104.2
of matmul, plus 18.5 + 8.6 + 6.5 + 1.4 + 1.2 = 36.2 of everything else, so ~140 ms.
The call reports 201. The gap is the artefact above, not work.

### Roofline: what the numbers come to

`tmp/roofline.py` sums the blob's own declaration: 40.06 GOPs of matmul over 147
matmul commands, 659.0 MB of weights in the weight space, 231.1 MB of activation
operands read or written once, so 890.1 MB of traffic per image and an arithmetic
intensity of 45.0 flop/byte.

| quantity | value |
| --- | --- |
| matmul work | 40.06 GFLOP (24 layers, 1.67 GFLOP/layer) |
| weight traffic | 659.0 MB declared, 663.7 MB read by the fill |
| activation operand traffic | 231.1 MB |
| flop byte | 45.0 |
| matmul throughput | 376.8 GFLOP/s (40.06 GOPs / 106.3 ms) |
| weight streaming rate | 9.79 GB/s (663.7 MB / 67.8 ms) |
| whole graph | 285.3 GFLOP/s and 6.37 GB/s over 140.4 ms of measured work |
| bandwidth floor | 894.8 MB / 9.79 GB/s = 91.4 ms per image |
| efficiency against that floor | 65% |
| pte load | 658.8 MB / ~530 ms = 1.24 GB/s |

The tile shapes say why. Every heavy matmul has m = 64 -- one 8x8 patch grid --
against n of 1024 to 4096: 24 of (m=64, k=1024, n=4096) at 537 MFLOP, 24 of the
reverse down-projection, 24 of (k=1024, n=3072) for QKV, and 48 tiny batched 64x64x64
attention tiles at 8.4 MFLOP. Sixty-four rows cannot amortise a 1024x4096 weight
matrix, so the graph is a stack of matrix-vector-shaped products: 663.7 MB of weights
are read per image to do 40 GFLOP. At the 9.79 GB/s the fill measures, 45 flop/byte
caps the machine at 440 GFLOP/s, and the matmul phase runs at 377, which is 86% of
that ceiling. The kernels are not the problem and further kernel tuning cannot move
it: only fewer bytes per flop can, which is what int8 would do.

The same arithmetic gives the goal's floor. Weights alone cost 67.8 ms at the
measured rate and activations another 23.6 ms, so ~91 ms per image is the floor for
this graph in fp16 on this DSP. Measured work is 140.4 ms -- 65% of the way there --
and the <=90 ms target sits just under the floor, reachable only by cutting bytes.

### Where this leaves the tower

Every family's time is consistent with moving its operands once or twice at the rate
the weight fill establishes. The fill moves 663,748,608 bytes in 67.8 ms, which is
9.79 GB/s, and that is the machine's reference number. Against it: blits 78.6 MB in
8.6 ms (9.1 GB/s), the matmul output store ~226 MB in 26.0 ms (8.7 GB/s), the binary
family's read/write traffic in 18.5 ms (7.7 GB/s), unary in 6.5 ms (8.7 GB/s).
Nothing here is leaving an order of magnitude on the table, and nothing here is
element-at-a-time: the elementwise broadcast path splats and runs native fp16 vector
ops, the store takes its fast path, and the matmul dispatch costs 137 us in total.

So the tower is at the DSP's DRAM bandwidth, ~140 ms of measured work per image,
against a call that reports 201 ms because of the bimodal artefact. The three named
targets are answered: (1) the fill is not serialised against compute, it *is* the
bandwidth floor; (2) the per-command cost is 0.27 ms of prologue plus the kernels
themselves, not a fixed per-command tax; (3) the store and pack are vectorized and
bandwidth-bound. The remaining lever is fewer bytes, which is two things: quantize,
or remove the materialisation of intermediates so a layout op's write and its
consumer's read both disappear. The second is bit-exact by construction and is the
one to try next -- the 1016 non-matmul commands move ~110 MB between them for 36 ms,
and fusing a permute or a clone into its consumer needs no arithmetic change at all.

Four things were removed on the way to that conclusion, all of them nulls: keeping
the VTCM window acquired and the fd table built across calls (201.7 against 201.5),
dropping the pre-execute sync group (201.6 against 201.4), dropping the group's 14 KB
command-entry invalidation (201.8 against 201.4, and the output md5 stays 18b6997d
exactly), and dropping the post-execute sync flush (203.7 against 204.1, md5 still
18b6997d). The last two are worth keeping in mind as deletions: all of that cache
maintenance is redundant for this model, it is simply not where the time goes.
