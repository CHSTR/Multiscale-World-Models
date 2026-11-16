"""Fábricas Hydra para los componentes por escala de LiftWM.

Construyen ``nn.ModuleDict`` keyed por ``str(level)`` (0..n_scales-1) sin tener que
anidar ``_target_: torch.nn.ModuleDict`` en el YAML. ``n_scales`` debe coincidir con
``encoder.levels + 1``.

* ``scale_mlp_dict``: projector / pred_proj token-a-token por escala.
* ``hier_nca`` / ``hier_vit``: predictor jerárquico con un NCA o un ViT por escala
  (intercambiables; misma interfaz espacial).
"""
from __future__ import annotations

from torch import nn

from wavelet.hier_predictor import HierarchicalPredictor


def scale_mlp_dict(n_scales: int, input_dim: int, output_dim: int,
                   hidden_dim: int) -> nn.ModuleDict:
    from module import MLP
    return nn.ModuleDict({
        str(l): MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        for l in range(n_scales)
    })


def hier_nca(n_scales: int, **kwargs) -> HierarchicalPredictor:
    from nca import LatentNCAPredictor
    return HierarchicalPredictor(nn.ModuleDict({
        str(l): LatentNCAPredictor(**kwargs) for l in range(n_scales)
    }))


def hier_vit(n_scales: int, **kwargs) -> HierarchicalPredictor:
    from wavelet.vit_predictor import ViTSpatialPredictor
    return HierarchicalPredictor(nn.ModuleDict({
        str(l): ViTSpatialPredictor(**kwargs) for l in range(n_scales)
    }))
