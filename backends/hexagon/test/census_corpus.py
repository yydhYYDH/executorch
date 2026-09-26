"""The nineteen hand-written geometries the gate census is measured over.

These lived in a scratch directory, which is why PARTITION_GATES.md section 9 had
to say that a census whose point is re-derivability had a corpus nobody could
re-derive.  They sit beside the other model fixtures now, byte for byte what the
scratch copy was, so a fresh worktree can rebuild every count in
PARTITION_GATES.md section 4 from the tree alone: census_gates.py drives this
module through the same instrument test_partition_gates.py carries.

Every weight here is randomly initialised.  No checkpoint is loaded and none is
faked: what the census measures is which nodes the partitioner delegates, and
that is a property of the graph's targets, shapes and dtypes, not of the values
in the weights.  No accuracy claim of any kind is made anywhere in this file.

`torchvision` and `timm` are not installed in this environment, so the "resnet18
shape" and "mobilenet_v2 shape" CNNs are written out by hand from the published
architectures rather than instantiated from a library.  Their channel widths,
kernel sizes, strides and paddings follow the published definitions; the random
weights and the absent checkpoint are the reason the classification head is
small.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

F16 = torch.float16


def _half(module):
    return module.eval().to(F16)


# ---------------------------------------------------------------------------
# CNN classifier, resnet18 shape
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet18Shape(nn.Module):
    """The published resnet18 geometry, including its ceil_mode stem pool."""

    def __init__(self, num_classes=10, ceil_mode=True):
        super().__init__()
        self.ceil_mode = ceil_mode
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1, ceil_mode=ceil_mode),
        )
        self.layer1 = self._stage(64, 64, 2, 1)
        self.layer2 = self._stage(64, 128, 2, 2)
        self.layer3 = self._stage(128, 256, 2, 2)
        self.layer4 = self._stage(256, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        layers += [_BasicBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# CNN classifier, mobilenet_v2 shape
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        self.use_residual = stride == 1 and in_ch == out_ch
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden)]
            layers += [nn.ReLU6(inplace=False)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class MobileNetV2Shape(nn.Module):
    """The published mobilenet_v2 geometry: depthwise 3x3, relu6, residuals."""

    #: (expand, out_ch, repeats, stride)
    _SETTINGS = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
    ]

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=False)
        )
        blocks = []
        in_ch = 32
        for expand, out_ch, repeats, stride in self._SETTINGS:
            for index in range(repeats):
                blocks.append(
                    _InvertedResidual(
                        in_ch, out_ch, stride if index == 0 else 1, expand
                    )
                )
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 320, 1, 1, 0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=False),
        )
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(torch.flatten(x, 1))


# ---------------------------------------------------------------------------
# Transformer encoder, deit_tiny shape
# ---------------------------------------------------------------------------


class TransformerEncoderTiny(nn.Module):
    """deit_tiny's numbers (d=192, heads=3, ffn=768, 12 layers, pre-norm)."""

    def __init__(self, depth=12, d_model=192, nhead=3, dim_ff=768, num_classes=10):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 197, d_model))
        self.proj = nn.Linear(3 * 16 * 16, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, patches):
        # patches: [1, 196, 3*16*16], already flattened by a patch embed.
        x = self.proj(patches)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


# ---------------------------------------------------------------------------
# Diffusion up-block / VAE decoder
# ---------------------------------------------------------------------------


class UpBlock(nn.Module):
    """nearest upsample -> 3x3 conv -> norm -> activation, plus a residual."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        hidden = F.silu(self.norm(self.conv(up)))
        return hidden + self.skip(up)


class VaeDecoder(nn.Module):
    def __init__(self, in_ch=64, mid_ch=64, out_ch=32):
        super().__init__()
        self.mid = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.up1 = UpBlock(mid_ch, mid_ch)
        self.up2 = UpBlock(mid_ch, out_ch)
        self.out = nn.Conv2d(out_ch, 3, 3, 1, 1)

    def forward(self, z):
        x = self.mid(z)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


class UpsampleExactToNearest(nn.Module):
    """The nearest-exact spelling, which export decomposes into index math."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest-exact")


def _cnn_inputs():
    return (torch.randn(1, 3, 64, 64, dtype=F16),)


def _deit_inputs():
    return (torch.randn(1, 196, 3 * 16 * 16, dtype=F16),)


def _vae_inputs(square=True):
    shape = (1, 64, 16, 16) if square else (1, 64, 12, 20)
    return (torch.randn(*shape, dtype=F16),)


def _vision_tower_inputs():
    return (torch.randn(1, 3, 32, 32, dtype=F16),)


def _audio_inputs():
    # A log-mel front end's output: [batch, mel, frames].
    return (torch.randn(1, 80, 64, dtype=F16),)


# ---------------------------------------------------------------------------
# LLM: a small config-built causal LM, prefill and decode shapes
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight, spelled out.

    `nn.RMSNorm` lowers to `aten.rms_norm`, which this backend does not have a
    target for; the fused `et_hexagon.rms_norm` is inserted by an opt-in pass.
    What a model authored without that pass produces is this expression, and
    that is what the census needs to see.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        inner = x.to(torch.float32)
        hidden = inner * torch.rsqrt(inner.pow(2).mean(-1, keepdim=True) + self.eps)
        return (hidden.to(x.dtype) * self.weight)


class CausalSelfAttention(nn.Module):
    """Explicit matmul / mask-add / softmax / matmul attention.

    Written this way rather than as `scaled_dot_product_attention` because the
    explicit form is the one that reaches this backend's wired ops; the sdpa
    spelling has no emitter at all and is measured separately.
    """

    def __init__(self, d_model, nhead, head_dim):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, 3 * nhead * head_dim, bias=False)
        self.out = nn.Linear(nhead * head_dim, d_model, bias=False)

    def forward(self, x, mask, past=None):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(self.nhead * self.head_dim, dim=2)
        query = query.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        key = key.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        value = value.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, value)
        hidden = hidden.transpose(1, 2).reshape(batch, tokens, self.nhead * self.head_dim)
        return self.out(hidden), (key, value)


class TinyLM(nn.Module):
    """A decoder-only causal LM built from a config, with random weights."""

    def __init__(self, vocab=1024, d_model=128, nhead=2, head_dim=64, layers=2, ffn=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(d_model),
                        "attn": CausalSelfAttention(d_model, nhead, head_dim),
                        "norm2": RMSNorm(d_model),
                        "gate": nn.Linear(d_model, ffn, bias=False),
                        "up": nn.Linear(d_model, ffn, bias=False),
                        "down": nn.Linear(ffn, d_model, bias=False),
                    }
                )
            )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def _block(self, block, x, mask, past):
        hidden = block["norm1"](x)
        attn, cache = block["attn"](hidden, mask, past)
        x = x + attn
        hidden = block["norm2"](x)
        hidden = block["down"](F.silu(block["gate"](hidden)) * block["up"](hidden))
        return x + hidden, cache

    def forward(self, tokens, mask, past_keys, past_values):
        x = self.embed(tokens)
        caches = []
        for index, block in enumerate(self.blocks):
            past = None
            if past_keys is not None:
                past = (past_keys[index], past_values[index])
            x, cache = self._block(block, x, mask, past)
            caches.append(cache)
        return self.head(self.norm(x))


def _llm_common(vocab=1024, d_model=128, nhead=2, head_dim=64, layers=2, ffn=512, frozen=True):
    model = TinyLM(vocab, d_model, nhead, head_dim, layers, ffn)
    torch.manual_seed(0)
    for parameter in model.parameters():
        if parameter.dim() >= 2:
            nn.init.normal_(parameter, std=0.02)
        else:
            parameter.data.fill_(1.0)
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def _causal_mask(tokens, total, dtype=F16):
    """Additive mask over `total` keys for the `tokens` queries at the end."""
    offset = total - tokens
    rows = offset + torch.arange(tokens).unsqueeze(1)
    cols = torch.arange(total).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros(tokens, total, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def llm_prefill():
    """A 64-token prefill: the cache it writes is empty on the way in."""
    model = _half(_llm_common())
    tokens = 64
    inputs = (
        torch.randint(0, 1024, (1, tokens)),
        _causal_mask(tokens, tokens),
        torch.zeros(2, 1, 2, 0, 64, dtype=F16),
        torch.zeros(2, 1, 2, 0, 64, dtype=F16),
    )
    return model, inputs


def llm_decode():
    model = _half(_llm_common())
    inputs = (
        torch.randint(0, 1024, (1, 1)),
        _causal_mask(1, 33),
        torch.randn(2, 1, 2, 32, 64, dtype=F16),
        torch.randn(2, 1, 2, 32, 64, dtype=F16),
    )
    return model, inputs


# ---------------------------------------------------------------------------
# Cheap extra families: a vision tower and an audio encoder
# ---------------------------------------------------------------------------


class VisionTower(nn.Module):
    """Patch embed by convolution, then a few pre-norm encoder layers."""

    def __init__(self, d_model=128, nhead=4, dim_ff=512, layers=2, patch=16):
        super().__init__()
        self.patch = nn.Conv2d(3, d_model, patch, patch, bias=False)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)

    def forward(self, image):
        x = self.patch(image)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(1))


class AudioEncoder(nn.Module):
    """A log-mel conv stack, then encoder layers, then mean pooling."""

    def __init__(self, mel=80, d_model=128, nhead=4, dim_ff=512, layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(mel, d_model, 3, 1, 1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, 2, 1)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 8)

    def forward(self, mel):
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class _SdpaSpelling(nn.Module):
    """One attention block spelled `scaled_dot_product_attention`."""

    def __init__(self):
        super().__init__()
        self.lm = _llm_common()
        self.attn = self.lm.blocks[0]["attn"]

    def forward(self, tokens):
        x = self.lm.embed(tokens)
        hidden = self.lm.blocks[0]["norm1"](x)
        query, key, value = self.attn.qkv(hidden).split(
            self.attn.nhead * self.attn.head_dim, dim=2
        )
        batch, length, _ = hidden.shape
        reshaper = lambda t: t.view(  # noqa: E731
            batch, length, self.attn.nhead, self.attn.head_dim
        ).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            reshaper(query), reshaper(key), reshaper(value), is_causal=True
        )
        return out.transpose(1, 2).reshape(batch, length, -1)


def _tiny_lm_sdpa():
    return _half(_SdpaSpelling()), (torch.randint(0, 1024, (1, 64)),)


from executorch.backends.hexagon.fold_batch_norm import FoldBatchNormIntoConv  # noqa: E402
from executorch.backends.hexagon.mul_silu import FuseMulSiluPass  # noqa: E402
from executorch.backends.hexagon.rms_norm import FuseRmsNormPass  # noqa: E402


def _bnfold():
    return [FoldBatchNormIntoConv()]


def _llm_fusions():
    return [FuseRmsNormPass(), FuseMulSiluPass()]


#: The opt-in transform passes each model is measured with.  A model absent here
#: is measured on the bare path -- `to_edge_transform_and_lower` with the
#: partitioner and nothing else -- which is what a caller who just points the
#: partitioner at a model gets.
PASSES = {
    "cnn_resnet18_bnfold": _bnfold(),
    "cnn_resnet18_staticpool_bnfold": _bnfold(),
    "cnn_mobilenetv2_bnfold": _bnfold(),
    "llm_prefill_fused": _llm_fusions(),
    "llm_decode_fused": _llm_fusions(),
    "transformer_encoder_2layer_fused": _llm_fusions(),
}

REGISTRY = {
    "cnn_resnet18": lambda: (_half(ResNet18Shape()), _cnn_inputs()),
    "cnn_resnet18_bnfold": lambda: (_half(ResNet18Shape()), _cnn_inputs()),
    "cnn_resnet18_staticpool_bnfold": lambda: (
        _half(ResNet18Shape(ceil_mode=False)),
        _cnn_inputs(),
    ),
    "cnn_mobilenetv2_bnfold": lambda: (_half(MobileNetV2Shape()), _cnn_inputs()),
    "llm_prefill_fused": llm_prefill,
    "llm_decode_fused": llm_decode,
    "transformer_encoder_2layer_fused": lambda: (
        _half(TransformerEncoderTiny(depth=2)),
        _deit_inputs(),
    ),
    "cnn_resnet18_staticpool": lambda: (
        _half(ResNet18Shape(ceil_mode=False)),
        _cnn_inputs(),
    ),
    "cnn_mobilenetv2": lambda: (_half(MobileNetV2Shape()), _cnn_inputs()),
    "transformer_encoder": lambda: (_half(TransformerEncoderTiny()), _deit_inputs()),
    "transformer_encoder_2layer": lambda: (
        _half(TransformerEncoderTiny(depth=2)),
        _deit_inputs(),
    ),
    "vae_upblock": lambda: (_half(VaeDecoder()), _vae_inputs(True)),
    "vae_upblock_nonsquare": lambda: (_half(VaeDecoder()), _vae_inputs(False)),
    "upsample_nearest_exact": lambda: (
        _half(UpsampleExactToNearest()),
        _vae_inputs(True),
    ),
    "llm_prefill": llm_prefill,
    "llm_decode": llm_decode,
    "lm_sdpa_spelling": _tiny_lm_sdpa,
    "vision_tower": lambda: (_half(VisionTower()), _vision_tower_inputs()),
    "audio_encoder": lambda: (_half(AudioEncoder()), _audio_inputs()),
}

DESCRIPTIONS = {
    "cnn_resnet18": "resnet18 geometry written out by hand, ceil_mode stem pool as published",
    "cnn_resnet18_staticpool": "the same, with ceil_mode=False on the stem pool",
    "cnn_mobilenetv2": "mobilenet_v2 geometry written out by hand: depthwise 3x3, relu6, residuals",
    "transformer_encoder": "deit_tiny numbers: d=192, heads=3, ffn=768, 12 pre-norm layers, 197 tokens",
    "transformer_encoder_2layer": "the same stack at 2 layers",
    "vae_upblock": "nearest x2 upsample, 3x3 conv, GroupNorm, SiLU, residual; 16x16 input",
    "vae_upblock_nonsquare": "the same block on 12x20 -> 24x40 -> 48x80",
    "upsample_nearest_exact": "interpolate(mode='nearest-exact'), the decomposed spelling",
    "llm_prefill": "tiny causal LM, 64-token prefill, explicit attention; random weights",
    "llm_decode": "the same LM, 1-token decode over a 32-token cache; random weights",
    "lm_sdpa_spelling": "one attention block spelled scaled_dot_product_attention",
    "vision_tower": "conv patch embed, 2 pre-norm encoder layers, mean pool",
    "audio_encoder": "log-mel conv stack, 2 encoder layers, mean pool",
    "cnn_resnet18_bnfold": "resnet18 with FoldBatchNormIntoConv in transform_passes",
    "cnn_resnet18_staticpool_bnfold": "the static-pool resnet18 with the batch-norm fold",
    "cnn_mobilenetv2_bnfold": "mobilenet_v2 with FoldBatchNormIntoConv in transform_passes",
    "llm_prefill_fused": "the prefill graph with FuseRmsNormPass and FuseMulSiluPass",
    "llm_decode_fused": "the decode graph with FuseRmsNormPass and FuseMulSiluPass",
    "transformer_encoder_2layer_fused": "the two-layer encoder with the norm and silu fusions",
}


# ---------------------------------------------------------------------------
# The twentieth and twenty-first graphs: a Qwen3 built from a transformers config
# ---------------------------------------------------------------------------

#: The two Qwen3 geometries, at one layer and at twenty-eight.  These were the
#: other half of what section 9 could not hand a reader: the census quoted counts
#: for a model nobody could build, because the config was in a scratch file.  The
#: index dtype is a parameter and not a constant because the int32 column is a
#: control, not the deployment shape -- see CENSUS-index-dtype, which is the
#: reason two census files that agreed on their total still hid an 18-node error.
QWEN3_GEOMETRIES = {
    "qwen3_1L": dict(
        layers=1, vocab=151936, hidden=1024, inter=3072, heads=16, kv=8, head_dim=128
    ),
    "qwen3_28L": dict(
        layers=28, vocab=151936, hidden=1024, inter=3072, heads=16, kv=8, head_dim=128
    ),
}


def qwen3(layers, vocab=151936, hidden=1024, inter=3072, heads=16, kv=8, head_dim=128,
          ids_dtype=None):
    """(module, inputs) for Qwen3ForCausalLM at `layers`, fp16, eager attention.

    `ids_dtype` defaults to int64, the dtype a real tokenizer produces and the
    dtype every count in PARTITION_GATES.md is taken at.  Pass torch.int32 for the
    control column, and say so when quoting it: the two columns differ in exactly
    two buckets, and a figure lifted out of the wrong one is not a rounding
    difference.
    """
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        max_position_embeddings=4096,
        use_cache=False,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).eval().to(torch.float16)
    dtype = torch.int64 if ids_dtype is None else ids_dtype
    return model, (torch.randint(0, 1024, (1, 32), dtype=dtype),)


def qwen3_from_registry(label, ids_dtype=None):
    return qwen3(ids_dtype=ids_dtype, **QWEN3_GEOMETRIES[label])
