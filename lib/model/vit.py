import math
import types
import einops
from einops import rearrange, repeat
from absl import logging
from flash_attn import flash_attn_func
from timm.models.layers import to_2tuple, trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def broadcat(tensors, dim=-1):
    """Broadcast and concatenate tensors along a dimension."""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)])
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    """Rotate half of the hidden dims for RoPE."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionRotaryEmbeddingFast(nn.Module):
    """2D Vision Rotary Position Embedding."""
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, theta=10000, num_cls_token=0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            num_img, rope_dim = cos_img.shape
            cos_pad = torch.ones(num_cls_token, rope_dim, dtype=cos_img.dtype)
            sin_pad = torch.zeros(num_cls_token, rope_dim, dtype=sin_img.dtype)
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0))
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0))
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE and RMSNorm on Q/K."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, rope_apply_fn):
        input_dtype = x.dtype
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope_apply_fn(q)
        k = rope_apply_fn(k)

        q, k, v = [u.to(torch.bfloat16).permute(0, 2, 1, 3) for u in [q, k, v]]
        x = flash_attn_func(q, k, v, dropout_p=self.attn_drop_rate)
        x = einops.rearrange(x, "B L H D -> B L (H D)")

        x = x.to(input_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2*hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        hidden = F.silu(x2) * x1
        return self.w3(self.ffn_dropout(hidden))


class PostNormAdaGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim, hidden_dim, drop=0.0, activation_fn=F.silu):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(dim, hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

        self.ffn_dropout = nn.Dropout(drop)
        self.activation_fn = activation_fn

    def forward(self, x, skip):
        x1 = self.w1(skip)
        x2 = self.w2(x)
        hidden = self.activation_fn(x2) * x1
        return self.w3(self.ffn_dropout(hidden))


class PreNormAdaGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim, hidden_dim, drop=0.0, activation_fn=F.silu):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(dim, hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)

        self.norm = RMSNorm(dim, eps=1e-6)
        self.ffn_dropout = nn.Dropout(drop)
        self.activation_fn = activation_fn

    def forward(self, x, skip):
        x1 = self.w1(skip)
        x2 = self.w2(self.norm(x))
        hidden = self.activation_fn(x2) * x1
        return self.w3(self.ffn_dropout(hidden))


class Block(nn.Module):
    """Encoder block: RMSNorm + RoPE Attention + SwiGLU FFN."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 attn_max_seqlen=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @torch.compile
    def forward(self, x, rope_apply_fn=None):
        x = x + self.attn(self.norm1(x), rope_apply_fn=rope_apply_fn)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Final output layer with adaLN modulation."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)

    @torch.compile
    def forward(self, x):
        x = self.linear(self.norm_final(x))
        return x


class GateBlock(nn.Module):
    def __init__(
        self, 
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0, 
        gate_activation_fn=F.silu,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.gate = PostNormAdaGLU(
            hidden_size,
            mlp_hidden_dim,
            drop=proj_drop,
            activation_fn=gate_activation_fn,
        )

        self.skip_linear = nn.Linear(2*hidden_size, hidden_size)
        self.skip_norm = nn.LayerNorm(hidden_size)

    @torch.compile
    def forward(self, x, skip, rope_apply_fn=None):
        skip = self.gate(x, skip)
        x = self.skip_linear(torch.cat([skip, x], dim=-1))
        x = self.skip_norm(x)

        x = x + self.attn(self.norm1(x), rope_apply_fn=rope_apply_fn)
        x = x + self.mlp(self.norm2(x))
        return x


class ResGateBlock(nn.Module):
    def __init__(
        self, 
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        gate_activation_fn=F.silu,

    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = RoPEAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  attn_drop=attn_drop, proj_drop=proj_drop)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.gate = PreNormAdaGLU(
            hidden_size,
            mlp_hidden_dim,
            drop=proj_drop,
            activation_fn=gate_activation_fn,
        )

    @torch.compile
    def forward(self, x, skip, rope_apply_fn=None):
        x = x + self.gate(x, skip)
        x = x + self.attn(self.norm1(x), rope_apply_fn=rope_apply_fn)
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """
    ViT-based denoising model with encoder-decoder architecture.
    Encoder uses RoPE attention, decoder uses residual skip connections with adaLN.
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        tokens=1,
        num_classes=1000,
        decoder_depth=6,
        decoder_fn = GateBlock,
        # gate relevant hyper-parameters
        gate_activation_fn=F.silu,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.in_chans = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.dtype = torch.float32
        self.num_patches = (image_size // patch_size) ** 2
        self.tokens = tokens
        self.extras = 1 + tokens

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size,
            in_chans=in_channels, embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.num_patches, embed_dim))
        self.cls_tokens = nn.Parameter(torch.zeros(1, tokens, embed_dim)) if tokens  > 0  else None
        self.time_embed = nn.Identity()

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        half_head_dim = embed_dim // num_heads // 2
        hw_seq_len = image_size // patch_size
        self.rope_apply_fn = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.extras,
        )
        self.decoder_blocks = nn.ModuleList([
            decoder_fn(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                gate_activation_fn=gate_activation_fn,
            ) for _ in range(decoder_depth)
        ])

        self.class_embed = nn.Embedding(num_classes + 1, embed_dim) if num_classes > 0 else None
        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels)
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.kaiming_normal_(
            self.patch_embed.proj.weight.view(
                [self.patch_embed.proj.weight.shape[0], -1]
            ), 
            a=1,
            mode='fan_in'
        )
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0.)

        if hasattr(self, "pos_embed"):
            trunc_normal_(self.pos_embed, std=0.02)
        if self.tokens != 0:
            nn.init.normal_(self.cls_tokens, std=1e-6)
        if self.class_embed is not None:
            nn.init.normal_(self.class_embed.weight, std=0.02)

        self.final_layer.linear.weight.detach().zero_()
        self.final_layer.linear.bias.detach().zero_()

    def get_num_param(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def no_weight_decay(self):
        return {"cls_token"}

    def unpatchify(self, x):
        """Reconstruct image from patch tokens."""
        num_patch = int(math.sqrt(self.num_patches))
        return einops.rearrange(
            x, "b (m n) (p1 p2 c) -> b c (m p1) (n p2)",
            m=num_patch, n=num_patch,
            p1=self.patch_size, p2=self.patch_size,
            c=self.in_chans,
        )

    def forward(self, x, t, y=None, **kwargs):
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        class_cond = self.class_embed(y).unsqueeze(1)
        prefix_tokens = self.cls_tokens + class_cond
        x = torch.cat((prefix_tokens, x), dim=1)

        time_token = self.time_embed(timestep_embedding(t, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        x = x + self.pos_embed
        skip = x

        for blk in self.blocks:
            x = blk(x, self.rope_apply_fn)

        for blk in self.decoder_blocks:
            x = blk(x, skip, self.rope_apply_fn)

        x = x[:, self.extras:]
        x = self.final_layer(x)
        x = self.unpatchify(x)

        return x


@torch.no_grad()
def reinitialize_block(self):
    enc_scale_factor =  1/math.sqrt(2*len(self.blocks))
    dec_scale_factor =  1/ math.sqrt(2*len(self.decoder_blocks))
    for block in self.blocks:

        nn.init.kaiming_normal_(block.attn.qkv.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(block.attn.proj.weight, a=1, mode='fan_in')
        block.attn.proj.weight.mul_( enc_scale_factor )

        if block.attn.qkv.bias is not None:
            nn.init.constant_(block.attn.qkv.bias, 0)
        if block.attn.proj.bias is not None:
            nn.init.constant_(block.attn.proj.bias, 0)

        nn.init.kaiming_normal_(block.mlp.w12.weight, a=1, mode='fan_in') 
        nn.init.kaiming_normal_(block.mlp.w3.weight, a=1, mode='fan_in')
        block.mlp.w3.weight.mul_( enc_scale_factor )

        if block.mlp.w12.bias is not None:
            nn.init.constant_(block.mlp.w12.bias, 0)
        if block.mlp.w3.bias is not None:
            nn.init.constant_(block.mlp.w3.bias, 0)

    for block in self.decoder_blocks:
        nn.init.kaiming_normal_(block.gate.w1.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(block.gate.w2.weight, a=1, mode='fan_in')
        nn.init.constant_(block.gate.w3.weight, 0)

        if block.gate.w1.bias is not None:
            nn.init.constant_(block.gate.w1.bias, 0)
        if block.gate.w2.bias is not None:
            nn.init.constant_(block.gate.w2.bias, 0)
        if block.gate.w3.bias is not None:
            nn.init.constant_(block.gate.w3.bias, 0)

        nn.init.kaiming_normal_(block.attn.qkv.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(block.attn.proj.weight, a=1, mode='fan_in')
        block.attn.proj.weight.mul_(dec_scale_factor)

        if block.attn.qkv.bias is not None:
            nn.init.constant_(block.attn.qkv.bias, 0)
        if block.attn.proj.bias is not None:
            nn.init.constant_(block.attn.proj.bias, 0)

        nn.init.kaiming_normal_(block.mlp.w12.weight, a=1, mode='fan_in') 
        nn.init.kaiming_normal_(block.mlp.w3.weight, a=1, mode='fan_in')
        block.mlp.w3.weight.mul_(dec_scale_factor)

        if block.mlp.w12.bias is not None:
            nn.init.constant_(block.mlp.w12.bias, 0)
        if block.mlp.w3.bias is not None:
            nn.init.constant_(block.mlp.w3.bias, 0)

    logging.info("Initialize block weights with reloaded intialization function")


def pitb(image_size=256, patch_size=16):
    param_dict = dict(
        in_channels=3,
        mlp_ratio=4,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=6,
        num_heads=12,
        tokens=1,
        num_classes=1000,
        decoder_depth=6,

        decoder_fn=GateBlock,
        gate_activation_fn=F.silu,

    )
    return ViT(**param_dict)


def pitl(image_size=256, patch_size=16):
    param_dict = dict(
        in_channels=3,
        mlp_ratio=4,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=1024,
        depth=10,
        num_heads=16,
        tokens=1,
        num_classes=1000,
        decoder_depth=10,

        decoder_fn=ResGateBlock,
        gate_activation_fn=F.silu,
    )
    model = ViT(**param_dict)
    model._initialize_weights = types.MethodType(reinitialize_block, model)
    model._initialize_weights() # initailize again
    return model


def pitxl(image_size=256, patch_size=16):
    param_dict = dict(
        in_channels=3,
        mlp_ratio=4,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=1152,
        depth=15,
        num_heads=16,
        tokens=1,
        num_classes=1000,
        decoder_depth=15,

        decoder_fn=ResGateBlock,
        gate_activation_fn=F.silu,
    )
    model = ViT(**param_dict)
    model._initialize_weights = types.MethodType(reinitialize_block, model)
    model._initialize_weights() # initailize again
    return model


def pitg(image_size=256, patch_size=16):
    param_dict = dict(
        in_channels=3,
        mlp_ratio=6,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=1536,
        depth=15,
        num_heads=16,
        tokens=1,
        num_classes=1000,
        decoder_depth=15,

        decoder_fn=ResGateBlock,
        gate_activation_fn=F.silu,
    )
    model = ViT(**param_dict)
    model._initialize_weights = types.MethodType(reinitialize_block, model)
    model._initialize_weights() # initailize again
    return model


models = {
    "PiT-B": pitb,
    "PiT-L": pitl,
    "PiT-XL": pitxl,
    "PiT-G": pitg,
}