"""Encoder multiescala ConvNeXt para ``PyramidJEPA`` (rama ``convnext-ms``).

Usa las **etapas jerárquicas** de ConvNeXt como escalas del world model, en vez de
una pirámide wavelet explícita: cada etapa ``i`` produce un mapa a resolución
``img / 2^(i+2)`` (224 -> 56/28/14/7). Las etapas elegidas se devuelven como
``PyramidOutput`` (misma interfaz que ``wavelet.pyramid_encoder.PyramidEncoder``),
así que ``PyramidJEPA``, ``HierarchicalPredictor``, las pérdidas por escala y el
costo CEM por escala se reutilizan sin cambios.

Convención de niveles (igual que LiftWM): ``level=0`` es la escala **más gruesa**
(la etapa más profunda), ``level=n-1`` la más fina.

    stages=[1, 2, 3], img=224  ->  level 0: etapa 3, 7x7   (49 tokens)
                                   level 1: etapa 2, 14x14 (196)
                                   level 2: etapa 1, 28x28 (784)

Inicialización (``init``):
  * ``dinov3``: pesos DINOv3 ConvNeXt (tiny/small/base/large, destilados del
    ViT-7B, LVD-1689M). Pesos: ``ckpt_path`` > ``$DINOV3_CONVNEXT_CKPT`` > hub.
  * ``scratch``: misma arquitectura, init aleatoria (trunc_normal), todo entrenable.
  DINOv2 no publicó modelos ConvNeXt (sólo ViT), por eso no hay opción ``dinov2``.

Ajuste del backbone (``finetune``, sólo aplica a ``init=dinov3``):
  * ``lora``: LoRA manual en ``pwconv1/pwconv2`` + normas + stem entrenables
    (mismo esquema que ``DinoV3ConvNeXtEncoder``).
  * ``full``: todo el backbone entrenable.
  * ``frozen``: backbone congelado; sólo el neck por escala aprende.

Neck por escala: ``LayerNorm(dim_etapa) -> Linear(dim_etapa, out_dim)``. Las etapas
intermedias de DINOv3 ConvNeXt no tienen norma propia (``norms[0..2] = Identity``),
así que la LayerNorm del neck las estandariza antes de proyectar a ``out_dim``.

Costo: ``out_grids`` (opcional) aplica ``adaptive_avg_pool2d`` por escala para
fijar la grilla de tokens (p. ej. ``[7, 14, 14]`` deja la escala fina en 196
tokens en vez de 784). Es la perilla principal de memoria/tiempo del predictor.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.encoders.vit_lora import (
    apply_lora_to_vit,
    count_total_parameters,
    count_trainable_parameters,
    freeze_all,
    freeze_except_lora_norm_patch,
)
from wavelet.pyramid_encoder import PyramidOutput, ScaleTokens

_SIZES = ("tiny", "small", "base", "large")
_INITS = ("dinov3", "scratch")
_FINETUNE = ("lora", "full", "frozen")


def _build_backbone(size: str, init: str, ckpt_path: Optional[str],
                    drop_path_rate: float) -> nn.Module:
    """Construye el ConvNeXt de DINOv3 (pesos preentrenados o init aleatoria)."""
    if init == "scratch":
        from src.dinov3.models.convnext import ConvNeXt, convnext_sizes
        model = ConvNeXt(in_chans=3, drop_path_rate=drop_path_rate,
                         layer_scale_init_value=1e-6, **convnext_sizes[size])
        model.init_weights()
        return model

    from src.dinov3.hub import backbones as hub
    build = getattr(hub, f"dinov3_convnext_{size}")
    resolved = ckpt_path or os.environ.get("DINOV3_CONVNEXT_CKPT") or None
    if resolved is not None:
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"DINOv3 ConvNeXt ckpt no encontrado: {resolved}. Pasa "
                "model.encoder.ckpt_path=<ruta>, exporta DINOV3_CONVNEXT_CKPT=<ruta> "
                "o deja ckpt_path=null para descargar por hub.")
        return build(weights=resolved)
    return build(pretrained=True)


def _expand_stem(model: nn.Module, in_chans: int) -> None:
    """Expande el stem conv (k=4, s=4) a ``in_chans`` canales (RGB copiado, resto = media)."""
    stem = model.downsample_layers[0]
    conv = stem[0]
    old = conv.weight.detach()
    new = nn.Conv2d(in_chans, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :old.shape[1]].copy_(old)
        new.weight[:, old.shape[1]:].copy_(old.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    stem[0] = new


class ConvNeXtPyramidEncoder(nn.Module):
    """ConvNeXt jerárquico -> tokens espaciales por escala (``PyramidOutput``)."""

    def __init__(
        self,
        size: str = "tiny",
        init: str = "dinov3",
        ckpt_path: Optional[str] = None,
        finetune: str = "lora",
        stages: Sequence[int] = (1, 2, 3),
        out_dim: int = 192,
        out_grids: Optional[Sequence[int]] = None,
        img_size: int = 224,
        drop_path_rate: float = 0.0,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        lora_targets: Sequence[str] = ("pwconv1", "pwconv2"),
        starlet_levels: int = 0,
        starlet_filter: str = "b3",
    ):
        super().__init__()
        if size not in _SIZES:
            raise ValueError(f"size='{size}' no válido; usa uno de {_SIZES}")
        if init not in _INITS:
            raise ValueError(f"init='{init}' no válido; usa uno de {_INITS}")
        if finetune not in _FINETUNE:
            raise ValueError(f"finetune='{finetune}' no válido; usa uno de {_FINETUNE}")
        stages = sorted({int(s) for s in stages}, reverse=True)   # grueso -> fino
        if not stages or any(s < 0 or s > 3 for s in stages):
            raise ValueError(f"stages={stages} debe ser un subconjunto no vacío de [0, 1, 2, 3]")
        if init == "scratch" and finetune != "full":
            print(f"[ConvNeXtPyramidEncoder] init=scratch con finetune={finetune}: "
                  "el backbone aleatorio quedaría (casi) congelado. ¿Querías finetune=full?")

        self.size = size
        self.init = init
        self.finetune = finetune
        self.stages = stages                      # índice de etapa por nivel (level 0 = más gruesa)
        self.levels = len(stages) - 1             # contrato con PyramidJEPA: n_scales = levels + 1
        self.out_dim = int(out_dim)
        self.img_size = int(img_size)
        self.last_stage = max(stages)

        if out_grids is not None:
            out_grids = [int(g) for g in out_grids]
            if len(out_grids) != len(stages):
                raise ValueError(f"out_grids necesita {len(stages)} entradas (una por escala, "
                                 f"de gruesa a fina); recibió {out_grids}")
        self.out_grids = out_grids

        # Frontend starlet opcional (paridad con DinoV3ConvNeXtEncoder).
        self.starlet_levels = int(starlet_levels)
        self.starlet_filter = starlet_filter
        if self.starlet_levels > 0:
            self.level_weights = nn.Parameter(torch.ones(self.starlet_levels + 1))

        self.model = _build_backbone(size, init, ckpt_path, drop_path_rate)
        self.stage_dims = list(self.model.embed_dims)

        # Las etapas posteriores a la última usada no se ejecutan: se podan.
        for i in range(self.last_stage + 1, 4):
            self.model.downsample_layers[i] = nn.Identity()
            self.model.stages[i] = nn.Identity()

        in_chans = 3 * (self.starlet_levels + 1) if self.starlet_levels > 0 else 3
        if in_chans != 3:
            _expand_stem(self.model, in_chans)
        self.in_chans = in_chans

        # Política de entrenamiento del backbone.
        self._lora_layers = []
        if finetune == "lora":
            self._lora_layers = apply_lora_to_vit(
                self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                targets=list(lora_targets), add_convnext=True)
            freeze_except_lora_norm_patch(self.model)
        elif finetune == "frozen":
            freeze_all(self.model)
        if finetune != "full" and in_chans != 3:
            # El stem expandido debe aprender a leer los canales wavelet.
            for p in self.model.downsample_layers[0].parameters():
                p.requires_grad_(True)

        # Neck por escala (keyed por nivel).
        self.necks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.stage_dims[s], eps=1e-6),
                          nn.Linear(self.stage_dims[s], self.out_dim))
            for s in stages
        ])

        print(
            f"ConvNeXtPyramidEncoder(size={size}, init={init}, finetune={finetune}, "
            f"stages={stages}, grids={self.grid_sizes()}, out_dim={self.out_dim}, "
            f"in_chans={in_chans}, lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,}]"
        )

    # ------------------------------------------------------------------ #
    def grid_sizes(self) -> list[int]:
        """Lado de la grilla de tokens por nivel (grueso -> fino) para ``img_size``."""
        if self.out_grids is not None:
            return list(self.out_grids)
        return [self.img_size // (2 ** (s + 2)) for s in self.stages]

    def forward(self, pixels: torch.Tensor) -> PyramidOutput:
        x = pixels
        if self.starlet_levels > 0:
            from wavelet.starlet_torch import starlet_conv4d
            x = starlet_conv4d(x, self.starlet_levels, scale=self.level_weights,
                               filter=self.starlet_filter)

        feats = {}
        for i in range(self.last_stage + 1):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            if i in self.stages:
                feats[i] = x                                   # (B, C_i, H_i, W_i)

        scales = []
        for level, s in enumerate(self.stages):
            f = feats[s]
            if self.out_grids is not None:
                g = self.out_grids[level]
                if f.shape[-2:] != (g, g):
                    f = F.adaptive_avg_pool2d(f, (g, g))
            gh, gw = f.shape[-2:]
            tok = f.flatten(2).transpose(1, 2)                 # (B, N, C_i)
            if s == 3:
                tok = self.model.norm(tok)                     # norma final preentrenada
            tok = self.necks[level](tok)                       # (B, N, out_dim)
            scales.append(ScaleTokens(tok, (int(gh), int(gw)), level))
        return PyramidOutput(scales=scales)


if __name__ == "__main__":
    enc = ConvNeXtPyramidEncoder(init="scratch", finetune="full")
    out = enc(torch.randn(2, 3, 224, 224))
    for s in out.scales:
        print(f"level={s.level} grid={s.grid_hw} tokens={tuple(s.tokens.shape)}")
