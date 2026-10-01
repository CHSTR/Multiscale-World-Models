"""Proto JEPA frame-level — Task 1: PrototypeHead + sinkhorn; Task 2: predict_proto."""

import torch
from torch import nn


def test_prototype_head_shapes():
    from module import PrototypeHead

    m = PrototypeHead(num_prototypes=64, dim=192, tau=0.1)
    h = torch.randn(4, 3, 192)
    logits, pi, z = m(h)
    assert logits.shape == (4, 3, 64)
    assert pi.shape == (4, 3, 64)
    assert z.shape == (4, 3, 192)
    assert torch.allclose(pi.sum(-1), torch.ones(4, 3), atol=1e-5)


def test_sinkhorn_rows_sum_to_one():
    from module import sinkhorn

    q = sinkhorn(torch.randn(8, 64) / 0.1, eps=0.05, iters=3)
    assert q.shape == (8, 64)
    assert torch.isfinite(q).all()
    assert torch.allclose(q.sum(-1), torch.ones(8), atol=1e-3)


def make_mini_jepa(proto_head=None, dim=16):
    """Helper Task 2: JEPA mínimo sobre CPU con ARPredictor/Embedder reales."""
    from jepa import JEPA
    from module import ARPredictor, Embedder

    predictor = ARPredictor(
        num_frames=3, depth=1, heads=2, mlp_dim=32,
        input_dim=dim, hidden_dim=dim, output_dim=dim,
        dim_head=8, dropout=0.0, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=4, smoothed_dim=8, emb_dim=dim)
    return JEPA(
        encoder=nn.Identity(), predictor=predictor,
        action_encoder=action_encoder, projector=nn.Identity(),
        pred_proj=nn.Identity(), proto_head=proto_head,
    )


def test_predict_proto_shapes_and_predict_intact():
    from module import PrototypeHead

    jepa = make_mini_jepa(proto_head=PrototypeHead(64, 16, tau=0.1), dim=16)
    emb = torch.randn(2, 3, 16)
    act = jepa.action_encoder(torch.randn(2, 3, 4))
    z, logits = jepa.predict_proto(emb, act)
    assert z.shape == (2, 3, 16) and logits.shape == (2, 3, 64)
    p1 = jepa.predict(emb, act)
    assert p1.shape == (2, 3, 16) and torch.isfinite(p1).all()


def test_predict_proto_none_head_raises_but_predict_works():
    import pytest

    jepa = make_mini_jepa(proto_head=None, dim=16)
    emb = torch.randn(2, 3, 16)
    act = jepa.action_encoder(torch.randn(2, 3, 4))
    assert jepa.predict(emb, act).shape == (2, 3, 16)
    with pytest.raises(RuntimeError):
        jepa.predict_proto(emb, act)


def test_proto_loss_terms_finite():
    from module import _proto_losses as proto_losses

    out = proto_losses(
        z=torch.randn(4, 2, 16), logits=torch.randn(4, 2, 64),
        tgt=torch.randn(4, 2, 16),
        prototypes=torch.randn(64, 16), tau=0.1, eps=0.05, iters=3,
    )
    assert torch.isfinite(out["mse"]) and torch.isfinite(out["ce"])
    assert out["loss"] == out["mse"] + out["ce"]  # pesos 1.0 en el helper
    assert torch.isfinite(out["perplexity"]) and out["perplexity"] > 1.0
    assert torch.isfinite(out["entropy"]) and out["entropy"] > 0.0
