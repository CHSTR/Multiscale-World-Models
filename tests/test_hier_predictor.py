"""Tests del predictor jerárquico (por escala).

``HierarchicalPredictor`` sólo reparte por nivel: debe funcionar con cualquier
predictor que cumpla la interfaz espacial (NCA, ViT, ...). Aquí se usa el NCA
como predictor concreto y en ``test_vit_predictor`` se repite con el ViT.
"""

import torch
from torch import nn

from nca import LatentNCAPredictor
from wavelet.hier_predictor import HierarchicalPredictor


def _mk(dims=(64, 64), grids=((8, 8), (14, 14))):
    preds = nn.ModuleDict({str(l): LatentNCAPredictor(latent_dim=d, act_dim=32,
        hidden_channels=8, update_dim=32, steps=[2,4]) for l,d in enumerate(dims)})
    return HierarchicalPredictor(preds), grids


def test_forward_shapes():
    hp, grids = _mk()
    emb = {0: torch.randn(2,3,64,64), 1: torch.randn(2,3,196,64)}
    out = hp.forward(emb, torch.randn(2,3,32), grids)
    assert out[0].shape==(2,3,64,64) and out[1].shape==(2,3,196,64)


def test_only_last():
    hp, grids = _mk()
    emb = {0: torch.randn(2,3,64,64), 1: torch.randn(2,3,196,64)}
    out = hp.forward(emb, torch.randn(2,3,32), grids, only_last=True)
    assert out[0].shape[1]==1


def test_rollout():
    hp, grids = _mk()
    ctx = {0: torch.randn(2,2,64,64), 1: torch.randn(2,2,196,64)}
    out = hp.rollout(ctx, torch.randn(2,3,32), grids, history=2)
    assert out[0].shape==(2,3,64,64)
