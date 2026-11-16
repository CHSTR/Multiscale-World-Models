"""Tests del predictor ViT espacial (alternativa intercambiable al NCA)."""

import torch

from wavelet.vit_predictor import ViTSpatialPredictor


def test_forward_shapes():
    p = ViTSpatialPredictor(latent_dim=64, act_dim=32, depth=2, heads=4, dim_head=16)
    x = torch.randn(2,3,196,64); c = torch.randn(2,3,32)
    assert p(x, c, grid_hw=(14,14)).shape == (2,3,196,64)
    assert p(x, c, grid_hw=(14,14), only_last=True).shape == (2,1,196,64)


def test_rollout():
    p = ViTSpatialPredictor(latent_dim=64, act_dim=32, depth=2, heads=4, dim_head=16)
    ctx = torch.randn(2,2,196,64); seq = torch.randn(2,3,32)
    out = p.rollout(ctx, seq, grid_hw=(14,14), history=2)
    assert out.shape == (2,3,196,64)


def test_eval_determinism():
    p = ViTSpatialPredictor(latent_dim=64, act_dim=32, depth=2, heads=4, dim_head=16).eval()
    ctx = torch.randn(2,2,196,64); seq = torch.randn(2,3,32)
    a = p.rollout(ctx, seq, grid_hw=(14,14)); b = p.rollout(ctx, seq, grid_hw=(14,14))
    assert torch.equal(a, b)


def test_rollout_matches_chained_forward():
    # Con history=1 la velocidad es cero, como teacher forcing con T=1: el
    # rollout debe ser exactamente encadenar forward+only_last.
    p = ViTSpatialPredictor(latent_dim=64, act_dim=32, depth=2, heads=4,
                            dim_head=16).eval()
    ctx = torch.randn(2, 2, 196, 64); seq = torch.randn(2, 3, 32)
    out = p.rollout(ctx, seq, grid_hw=(14, 14), post=lambda y: y, history=1)
    cur, chain = ctx, []
    for m in range(seq.size(1)):
        nxt = p(cur[:, -1:], seq[:, m:m+1], grid_hw=(14, 14), only_last=True)
        cur = torch.cat([cur, nxt], dim=1)
        chain.append(nxt[:, 0])
    assert torch.allclose(out, torch.stack(chain, dim=1), atol=1e-6)


def test_hier_with_vit():
    from torch import nn
    from wavelet.hier_predictor import HierarchicalPredictor
    preds = nn.ModuleDict({str(l): ViTSpatialPredictor(latent_dim=64, act_dim=32,
        depth=2, heads=4, dim_head=16) for l in range(2)})
    hp = HierarchicalPredictor(preds)
    emb = {0: torch.randn(2,3,64,64), 1: torch.randn(2,3,196,64)}
    grids = {0:(8,8), 1:(14,14)}
    out = hp.forward(emb, torch.randn(2,3,32), grids)
    assert out[1].shape == (2,3,196,64)
