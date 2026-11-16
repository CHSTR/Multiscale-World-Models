"""Encoder piramidal multiescala para LiftWM.

Aplica un ``LiftingScheme`` invertible a los píxeles y convierte cada sub-banda en
una grilla de tokens espaciales ``(B, N_l, D_l)``. NO hay pooling global: cada escala
conserva su resolución espacial para que el predictor y el planning CEM trabajen por
escala (lo grueso = estructura/posición, lo fino = orientación/bordes/contacto).

Escalas (índice ``level``):
  * ``level=0``: aproximación LL_L (la más gruesa, resolución img/2^L).
  * ``level=l`` (1..L): detalle ``details[L-l]`` (LH/HL/HH apilados, 3C canales).
    ``level=L`` es el detalle más fino (resolución img/2).

Cada escala: stem conv (patchify ``stem_patch``) -> tokens + pos_embed aprendible ->
``stem_blocks`` bloques transformer con atención espacial NO causal.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from module import Attention, FeedForward


@dataclass
class ScaleTokens:
    tokens: torch.Tensor          # (B, N_l, D_l)
    grid_hw: tuple[int, int]      # N_l = gh_l * gw_l
    level: int                    # 0 = más gruesa ... L = más fina


@dataclass
class PyramidOutput:
    scales: list[ScaleTokens]     # len L+1; índice 0 = approx, índice L = detalle fino


class EncoderBlock(nn.Module):
    """Bloque transformer con atención espacial NO causal (is_causal=False)."""

    def __init__(self, dim, heads=6, dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim or dim * 4, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.mlp(self.norm2(x))
        return x


class PyramidEncoder(nn.Module):
    def __init__(self, levels: int = 3, in_channels: int = 3,
                 dims: list[int] | None = None, stem_blocks: int = 2,
                 img_size: int = 224, learnable_lifting: bool = False,
                 stem_patch: int = 4, heads: int = 6, dim_head: int = 64,
                 dropout: float = 0.0, lifting: nn.Module | None = None):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.dims = list(dims) if dims is not None else [192] * (levels + 1)
        assert len(self.dims) == levels + 1, "dims debe tener L+1 entradas"
        self.stem_patch = stem_patch
        self.stem_blocks = stem_blocks
        self.img_size = img_size

        if lifting is None:
            from wavelet.lifting import LiftingScheme
            lifting = LiftingScheme(levels, in_channels, init="haar",
                                    learnable=learnable_lifting)
        self.lifting = lifting

        self.stems = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for l in range(levels + 1):
            in_ch = in_channels if l == 0 else 3 * in_channels
            res = self._res(l)
            assert res % stem_patch == 0, (
                f"resolución {res} de escala {l} no divisible por stem_patch={stem_patch}")
            gh = gw = res // stem_patch
            n = gh * gw
            dim = self.dims[l]
            self.stems.append(nn.Conv2d(in_ch, dim, stem_patch, stem_patch))
            self.pos_embeds.append(nn.Parameter(torch.randn(1, n, dim) * 0.02))
            self.blocks.append(nn.ModuleList([
                EncoderBlock(dim, heads, dim_head, dim * 4, dropout)
                for _ in range(stem_blocks)
            ]))
            self.norms.append(nn.LayerNorm(dim))

    def _res(self, level: int) -> int:
        if level == 0:
            return self.img_size // (2 ** self.levels)
        return self.img_size // (2 ** (self.levels - level + 1))

    def forward(self, pixels: torch.Tensor) -> PyramidOutput:
        p = self.lifting(pixels)
        scales = []
        for l in range(self.levels + 1):
            feat = p.approx if l == 0 else p.details[self.levels - l]
            x = self.stems[l](feat)                 # (B, dim, gh, gw)
            gh, gw = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)        # (B, N, dim)
            x = x + self.pos_embeds[l][:, :x.size(1)]
            for blk in self.blocks[l]:
                x = blk(x)
            x = self.norms[l](x)
            scales.append(ScaleTokens(x, (gh, gw), l))
        return PyramidOutput(scales=scales)


if __name__ == "__main__":
    enc = PyramidEncoder(levels=3, stem_blocks=1)
    out = enc(torch.randn(2, 3, 224, 224))
    for s in out.scales:
        print(f"level={s.level} grid={s.grid_hw} tokens={tuple(s.tokens.shape)}")
