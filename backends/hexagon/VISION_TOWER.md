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
