"""ViT encoder con bloques WaveletAttention para JEPA."""
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from module import WaveletAttention, FeedForward


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple = ()


class WaveBlock(nn.Module):
    """Bloque ViT con WaveletAttention (sin CLS — solo patch tokens)."""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WaveletAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, output_attentions=False):
        attn_out = self.attn(self.norm1(x), output_attentions=output_attentions)
        if output_attentions:
            x_attn, attn = attn_out
            x = x + x_attn
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class WaveViTEncoder(nn.Module):
    """ViT-tiny con WaveletAttention. Mean-pool → pseudo-CLS en [:, 0] para JEPA."""

    def __init__(self, img_size=224, patch_size=14, in_channels=3,
                 dim=192, depth=12, num_heads=3, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.blocks = nn.ModuleList([
            WaveBlock(dim, num_heads, mlp_dim or dim * 4, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, interpolate_pos_encoding=False, output_attentions=False):
        # ponytail: interpolate_pos_encoding ignorado; siempre 224x224 → 16x16=256 patches.
        # Si se usan tamaños variables, interpolar pos_embed aquí.
        x = self.patch_embed(x)                       # (B, dim, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)             # (B, N, dim)
        x = x + self.pos_embed
        attentions = []
        for blk in self.blocks:
            if output_attentions:
                x, attn = blk(x, output_attentions=True)
                attentions.append(attn)
            else:
                x = blk(x)
        x = self.norm(x)
        cls = x.mean(dim=1, keepdim=True)             # (B, 1, dim) pseudo-CLS
        x = torch.cat([cls, x], dim=1)                # (B, N+1, dim)
        return EncoderOutput(last_hidden_state=x, attentions=tuple(attentions))

    def __call__(self, x, interpolate_pos_encoding=False, output_attentions=False):
        return self.forward(x, interpolate_pos_encoding, output_attentions)


if __name__ == "__main__":
    m = WaveViTEncoder(img_size=224, patch_size=14, dim=192, depth=2, num_heads=3)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out.last_hidden_state.shape == (2, 257, 192), out.last_hidden_state.shape
    print(f"WaveViTEncoder OK: {out.last_hidden_state.shape}, params={sum(p.numel() for p in m.parameters()):,}")