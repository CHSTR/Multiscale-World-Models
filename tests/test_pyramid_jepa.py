import torch
from torch import nn

from jepa_pyramid import PyramidJEPA, pyramid_losses
from wavelet.pyramid_encoder import PyramidOutput, ScaleTokens


class MockEnc(nn.Module):
    levels = 1

    def forward(self, pixels, **kw):
        n = pixels.shape[0]
        return PyramidOutput(scales=[
            ScaleTokens(torch.randn(n, 64, 192), (8, 8), 0),
            ScaleTokens(torch.randn(n, 196, 192), (14, 14), 1),
        ])


class MockPred(nn.Module):
    def forward(self, emb, act_emb, grid_hw, only_last=False):
        return {l: (v[:, :1] if only_last else v) for l, v in emb.items()}

    def rollout(self, ctx, seq, grid_hw, **kw):
        return {l: v[:, -1:].expand(-1, seq.shape[1], -1, -1) for l, v in ctx.items()}


def _model():
    proj = nn.ModuleDict({str(l): nn.Identity() for l in range(2)})
    return PyramidJEPA(MockEnc(), MockPred(), nn.Identity(), proj, proj)


def test_encode_dict():
    m = _model()
    info = {"pixels": torch.randn(2, 3, 3, 224, 224), "action": torch.randn(2, 3, 4)}
    out = m.encode(info)
    assert isinstance(out["emb"], dict)
    assert out["emb"][0].shape == (2, 3, 64, 192)
    assert out["emb"][1].shape == (2, 3, 196, 192)
    assert m.grid_hw[1] == (14, 14)


def test_predict_dict():
    m = _model()
    emb = {0: torch.randn(2, 3, 64, 192), 1: torch.randn(2, 3, 196, 192)}
    m.grid_hw = {0: (8, 8), 1: (14, 14)}
    pred = m.predict(emb, torch.randn(2, 3, 192))
    assert set(pred) == {0, 1} and pred[1].shape == (2, 3, 196, 192)


def test_criterion_shape_and_determinism():
    m = _model()
    pred = {0: torch.randn(2, 5, 4, 64, 192), 1: torch.randn(2, 5, 4, 196, 192)}
    goal = {0: torch.randn(2, 4, 64, 192), 1: torch.randn(2, 4, 196, 192)}
    info = {"predicted_emb": pred, "goal_emb": goal}
    c1 = m.criterion(info)
    c2 = m.criterion(info)
    assert c1.shape == (2, 5) and torch.equal(c1, c2)


def test_scale_weights_effect():
    m = _model()
    m.scale_weights.copy_(torch.tensor([0.0, 1.0]))
    pred = {0: torch.randn(2, 5, 4, 64, 192), 1: torch.randn(2, 5, 4, 196, 192)}
    goal = {0: torch.randn(2, 4, 64, 192), 1: torch.randn(2, 4, 196, 192)}
    c = m.criterion({"predicted_emb": pred, "goal_emb": goal})
    pred2 = {0: torch.randn_like(pred[0]), 1: pred[1]}
    assert torch.allclose(c, m.criterion({"predicted_emb": pred2, "goal_emb": goal}))


def test_sigreg_per_scale_shape():
    from module import sigreg_input
    emb_l = torch.randn(2, 3, 196, 192)
    x = sigreg_input(emb_l, max_tokens=8)
    assert x.shape[0] == 3 and x.shape[-1] == 192 and x.shape[1] <= 2 * 8


def test_pyramid_losses_runs():
    from omegaconf import OmegaConf
    m = _model()
    emb = {0: torch.randn(2, 4, 64, 192), 1: torch.randn(2, 4, 196, 192)}
    pred = {0: torch.randn(2, 3, 64, 192), 1: torch.randn(2, 3, 196, 192)}
    tgt = {0: torch.randn(2, 3, 64, 192), 1: torch.randn(2, 3, 196, 192)}
    cfg = OmegaConf.create({"history_size": 3, "loss": {}})
    batch = {"action": torch.randn(2, 4, 4)}
    extra, logs = pyramid_losses(m, batch, emb, pred, tgt, cfg)
    assert extra.ndim == 0 and "copy_loss_l0" in logs
