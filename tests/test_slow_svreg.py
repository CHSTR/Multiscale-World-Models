"""Unit tests for Slow SVReg (R1), IS weighting & non-BN projectors (R2)."""

from functools import partial
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from jepa import JEPA
from jepa_pyramid import PyramidJEPA
from module import ARPredictor, Embedder, GroupNorm1d, MLP, SIGReg, SVReg, SlownessLoss
from prioritized_replay import ReplayConfig, ReplayState
from tests.test_prioritized_replay import FakeEncoder, _import_train
from wavelet.factories import hier_vit, scale_mlp_dict
from wavelet.pyramid_encoder import PyramidOutput, ScaleTokens
from wavelet.vit_predictor import ViTSpatialPredictor


# ---------------------------------------------------------------------- #
# 1. SVReg
# ---------------------------------------------------------------------- #
def test_svreg_unilateral_floor_and_collapse():
    reg = SVReg(num_proj=128, eps=1e-8)

    # Large variance along all directions -> directional std >= 1 -> exact 0 loss.
    torch.manual_seed(0)
    z_wide = torch.randn(3, 32, 16) * 5.0
    loss_wide = reg(z_wide)
    assert loss_wide.dtype == torch.float32
    assert float(loss_wide) == 0.0

    # Collapsed / small variance -> loss > 0.
    z_small = torch.randn(3, 32, 16) * 0.05
    loss_small = reg(z_small)
    assert float(loss_small) > 0.5


def test_svreg_collapse_gradients_finite_and_no_stop_grad():
    reg = SVReg(num_proj=64, eps=1e-8)

    # Near collapse: finite, non-NaN, nonzero gradient flows directly to input.
    z_near = (torch.randn(2, 8, 12) * 1e-4).requires_grad_(True)
    loss_near = reg(z_near)
    loss_near.backward()
    assert z_near.grad is not None
    assert torch.isfinite(z_near.grad).all()
    assert z_near.grad.abs().sum() > 0

    # Exact constant collapse: finite gradient (no NaN from sqrt(0)).
    z_const = torch.full((2, 8, 12), 3.0, requires_grad=True)
    loss_const = reg(z_const)
    loss_const.backward()
    assert z_const.grad is not None
    assert torch.isfinite(z_const.grad).all()


def test_svreg_edge_cases_n1_empty_and_bf16():
    reg = SVReg(num_proj=64, eps=1e-8)

    # N == 1 must not divide by zero or return NaN.
    z_n1 = torch.randn(3, 1, 8, requires_grad=True)
    loss_n1 = reg(z_n1)
    assert torch.isfinite(loss_n1)
    expected = (1.0 - (1e-8 ** 0.5)) ** 2
    assert torch.allclose(loss_n1, torch.tensor(expected), atol=1e-5)
    loss_n1.backward()
    assert torch.isfinite(z_n1.grad).all()

    # Empty tensor returns 0 safely.
    assert float(reg(torch.empty(2, 0, 8))) == 0.0

    # bf16 autocast must still compute in float32 cleanly.
    z_bf16 = torch.randn(2, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_bf16 = reg(z_bf16)
    assert loss_bf16.dtype == torch.float32
    assert torch.isfinite(loss_bf16)
    loss_bf16.backward()
    assert torch.isfinite(z_bf16.grad).all()


@pytest.mark.parametrize("pooled", [False, True])
def test_svreg_is_weights(pooled):
    reg = SVReg(num_proj=32, eps=1e-8, pooled=pooled)
    x = torch.randn(2, 6, 8) * 0.2

    torch.manual_seed(0)
    unweighted = reg(x)
    torch.manual_seed(0)
    ones_weighted = reg(x, weights=torch.ones(6))
    assert torch.allclose(unweighted, ones_weighted, atol=1e-6)

    # Weight 2 on sample 0 == duplicating sample 0.
    x_dup = torch.cat([x, x[:, :1]], dim=1)
    w = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    torch.manual_seed(1)
    w_loss = reg(x, weights=w)
    torch.manual_seed(1)
    dup_loss = reg(x_dup)
    assert torch.allclose(w_loss, dup_loss, atol=1e-6)

    # Zero-sum weights fall back safely to uniform without NaN.
    torch.manual_seed(2)
    z_loss = reg(x, weights=torch.zeros(6))
    torch.manual_seed(2)
    u_loss = reg(x)
    assert torch.isfinite(z_loss)
    assert torch.allclose(z_loss, u_loss, atol=1e-6)


# ---------------------------------------------------------------------- #
# 2. SlownessLoss
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("D", [16, 192, 10, 5])
def test_slowness_acts_only_on_last_eighth(D):
    slow = SlownessLoss(slow_frac=0.125, eps=1e-6)
    d_slow = max(1, int(D * 0.125))
    torch.manual_seed(0)
    emb = torch.randn(4, 6, D)
    base = slow(emb)

    # Mutating the first D - d_slow ("free") dims must NOT change SlownessLoss.
    emb_mut_free = emb.clone()
    emb_mut_free[..., :-d_slow] += torch.randn_like(emb_mut_free[..., :-d_slow]) * 5.0
    assert torch.allclose(base, slow(emb_mut_free), atol=1e-6)

    # Mutating the last d_slow ("slow") dims MUST change SlownessLoss.
    emb_mut_slow = emb.clone()
    emb_mut_slow[..., -d_slow:] *= 0.1
    emb_mut_slow[..., -d_slow:] += torch.randn_like(emb_mut_slow[..., -d_slow:])
    assert not torch.allclose(base, slow(emb_mut_slow), atol=1e-4)


def test_slowness_edge_cases_and_white_noise():
    slow = SlownessLoss(slow_frac=0.125, eps=1e-6)

    # T < 2 returns 0.0 safely.
    assert float(slow(torch.randn(4, 1, 16))) == 0.0
    assert float(slow(torch.randn(4, 0, 16))) == 0.0

    # Constant / zero-variance input must not produce NaN/Inf.
    z_const = torch.full((4, 5, 16), 2.5, requires_grad=True)
    loss_const = slow(z_const)
    assert float(loss_const.detach()) == 0.0
    assert torch.isfinite(loss_const)
    loss_const.backward()
    assert torch.isfinite(z_const.grad).all()

    # i.i.d. noise -> R_slow ~ 1.0 (since E||b_{t+1}-b_t||^2 = 2 tr Var(b)).
    torch.manual_seed(42)
    z_iid = torch.randn(256, 8, 32)
    r_iid = slow(z_iid)
    assert abs(float(r_iid) - 1.0) < 0.05

    # 4D spatial tensor + bf16 autocast.
    z_4d = torch.randn(3, 4, 9, 16, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        r_4d = slow(z_4d)
    assert r_4d.dtype == torch.float32
    assert torch.isfinite(r_4d)
    r_4d.backward()
    assert torch.isfinite(z_4d.grad).all()


@pytest.mark.parametrize("shape", [(6, 5, 16), (6, 5, 4, 16)])
def test_slowness_is_weights(shape):
    slow = SlownessLoss(slow_frac=0.125, eps=1e-6)
    torch.manual_seed(3)
    emb = torch.randn(*shape)

    assert torch.allclose(slow(emb), slow(emb, weights=torch.ones(shape[0])), atol=1e-6)

    emb_dup = torch.cat([emb, emb[:1]], dim=0)
    w = torch.tensor([2.0] + [1.0] * (shape[0] - 1))
    assert torch.allclose(slow(emb, weights=w), slow(emb_dup), atol=1e-6)

    # Zero-sum weights fall back safely to uniform.
    assert torch.allclose(slow(emb, weights=torch.zeros(shape[0])), slow(emb), atol=1e-6)


# ---------------------------------------------------------------------- #
# 3. Gated Residual Predictors (ARPredictor, ViTSpatialPredictor, PyramidJEPA)
# ---------------------------------------------------------------------- #
def test_ar_predictor_gated_residual():
    torch.manual_seed(0)
    pred = ARPredictor(
        num_frames=4, depth=1, heads=2, mlp_dim=16, input_dim=8,
        hidden_dim=8, output_dim=8, dim_head=4, dropout=0.0,
        gated_residual=True, gate_init=0.1,
    )
    assert hasattr(pred, "res_gate")
    assert pred.res_gate.shape == (8,)
    assert torch.allclose(pred.res_gate, torch.full((8,), 0.1))

    x = torch.randn(2, 3, 8)
    c = torch.randn(2, 3, 8)
    with torch.no_grad():
        out = pred(x, c)
        assert out.shape == x.shape
        saved = pred.res_gate.clone()
        pred.res_gate.zero_()
        out_zero = pred(x, c)
        assert torch.allclose(out_zero, x)
        pred.res_gate.fill_(2.0)
        out_two = pred(x, c)
        pred.res_gate.copy_(saved)
        assert torch.allclose(out - x, 0.05 * (out_two - x), atol=1e-5)

    # Gradient flow to both res_gate and input x (no stop-gradient).
    x_g = x.clone().requires_grad_(True)
    pred.zero_grad()
    out_g = pred(x_g, c)
    out_g.square().mean().backward()
    assert pred.res_gate.grad is not None and pred.res_gate.grad.abs().sum() > 0
    assert x_g.grad is not None and x_g.grad.abs().sum() > 0

    # JEPA with pred_proj=None preserves z_t + c * f(...) in predict() and rollout().
    jepa = JEPA(
        encoder=FakeEncoder(hw=8, d=8),
        predictor=pred,
        action_encoder=Embedder(input_dim=4, emb_dim=8),
        projector=None,
        pred_proj=None,
    ).eval()
    assert isinstance(jepa.pred_proj, nn.Identity)
    emb_ctx = torch.randn(2, 3, 8)
    act_ctx = torch.randn(2, 3, 8)
    assert torch.equal(jepa.predict(emb_ctx, act_ctx), pred(emb_ctx, act_ctx))
    info = {
        "pixels": torch.randn(1, 2, 3, 3, 8, 8),
        "action": torch.randn(1, 2, 3, 4),
    }
    ro = jepa.rollout(info, torch.randn(1, 2, 5, 4), history_size=3)
    assert ro["predicted_emb"].shape == (1, 2, 6, 8)

    with pytest.raises(ValueError, match="gated_residual requires output_dim"):
        ARPredictor(
            num_frames=4, depth=1, heads=2, mlp_dim=16, input_dim=8,
            hidden_dim=8, output_dim=16, dim_head=4, gated_residual=True,
        )


def test_vit_spatial_predictor_gated_residual_forward_and_rollout():
    torch.manual_seed(0)
    p = ViTSpatialPredictor(
        latent_dim=16, act_dim=8, depth=2, heads=2, dim_head=8,
        gated_residual=True, gate_init=0.1,
    ).eval()
    assert p.res_gate.shape == (16,)
    assert torch.allclose(p.res_gate, torch.full((16,), 0.1))
    assert p.aux == {}

    x = torch.randn(2, 3, 9, 16, requires_grad=True)
    c = torch.randn(2, 3, 8)
    out = p(x, c, grid_hw=(3, 3))
    assert p.aux == {}
    assert out.shape == (2, 3, 9, 16)

    # Rollout with history=1 matches chained forward(..., only_last=True).
    ctx = torch.randn(2, 2, 9, 16)
    seq = torch.randn(2, 3, 8)
    r_out = p.rollout(ctx, seq, grid_hw=(3, 3), history=1)
    cur, chain = ctx, []
    for m in range(seq.size(1)):
        nxt = p(cur[:, -1:], seq[:, m:m + 1], grid_hw=(3, 3), only_last=True)
        cur = torch.cat([cur, nxt], dim=1)
        chain.append(nxt[:, 0])
    assert torch.allclose(r_out, torch.stack(chain, dim=1), atol=1e-6)

    # Zero gate produces exact identity skip in both forward and rollout.
    with torch.no_grad():
        saved = p.res_gate.clone()
        p.res_gate.zero_()
        assert torch.allclose(p(x, c, grid_hw=(3, 3)), x)
        r_zero = p.rollout(ctx, seq, grid_hw=(3, 3))
        for m in range(seq.size(1)):
            assert torch.allclose(r_zero[:, m], ctx[:, -1])
        p.res_gate.copy_(saved)

    # Gradient flow to both res_gate and input x through rollout.
    p.zero_grad()
    ctx_g = ctx.clone().requires_grad_(True)
    r_g = p.rollout(ctx_g, seq, grid_hw=(3, 3))
    r_g.square().mean().backward()
    assert p.res_gate.grad is not None and p.res_gate.grad.abs().sum() > 0
    assert ctx_g.grad is not None and ctx_g.grad.abs().sum() > 0


def test_pyramid_jepa_pred_proj_none_and_gated_residual():
    class TinyPyrEnc(nn.Module):
        levels = 1

        def forward(self, pixels, **kw):
            n = pixels.shape[0]
            return PyramidOutput(scales=[
                ScaleTokens(torch.randn(n, 4, 16), (2, 2), 0),
                ScaleTokens(torch.randn(n, 9, 16), (3, 3), 1),
            ])

    pred = hier_vit(
        n_scales=2, latent_dim=16, act_dim=8, depth=1, heads=2, dim_head=8,
        gated_residual=True, gate_init=0.1,
    )
    proj = scale_mlp_dict(2, 16, 16, 32, norm_fn=partial(GroupNorm1d, num_groups=4))
    m = PyramidJEPA(TinyPyrEnc(), pred, Embedder(4, 8, 8), projector=proj, pred_proj=None)

    info = {"pixels": torch.randn(2, 4, 3, 8, 8), "action": torch.randn(2, 4, 4)}
    out = m.encode(info)
    ctx = {l: v[:, :3] for l, v in out["emb"].items()}
    preds = m.predict(ctx, out["act_emb"][:, :3])
    assert set(preds) == {0, 1}
    assert preds[0].shape == (2, 3, 4, 16)

    r_preds = m.predict_rollout(ctx, out["act_emb"][:, :2])
    assert r_preds[1].shape == (2, 2, 9, 16)

    cost = m.get_cost(
        {
            "pixels": torch.randn(1, 2, 3, 3, 8, 8),
            "goal": torch.randn(1, 2, 1, 3, 8, 8),
            "action": torch.randn(1, 2, 3, 4),
        },
        torch.randn(1, 2, 4, 4),
    )
    assert cost.shape == (1, 2) and torch.isfinite(cost).all()


# ---------------------------------------------------------------------- #
# 4. Non-BatchNorm Projector Options & MLP Flattening
# ---------------------------------------------------------------------- #
def test_non_batchnorm_projectors_and_mlp_flatten():
    # GroupNorm1d works with MLP directly, via partial, and on 3D/4D inputs.
    m_gn = MLP(16, 32, 16, norm_fn=GroupNorm1d)
    assert isinstance(m_gn.net[1], GroupNorm1d) and m_gn.net[1].num_groups == 8

    m_gn_partial = MLP(16, 32, 16, norm_fn=partial(GroupNorm1d, num_groups=4))
    assert m_gn_partial.net[1].num_groups == 4

    x3 = torch.randn(2, 3, 16)
    assert torch.equal(m_gn(x3), m_gn.net(x3.reshape(-1, 16)).reshape(2, 3, 16))
    x4 = torch.randn(2, 3, 5, 16)
    assert torch.equal(m_gn(x4), m_gn.net(x4.reshape(-1, 16)).reshape(2, 3, 5, 16))

    # GroupNorm1d standalone on 3D tensor.
    gn = GroupNorm1d(32, num_groups=8)
    x_gn3 = torch.randn(2, 4, 32)
    assert torch.equal(gn(x_gn3), nn.GroupNorm(8, 32)(x_gn3.reshape(-1, 32)).reshape(2, 4, 32))

    # LayerNorm and Identity in MLP & scale_mlp_dict.
    m_ln = MLP(16, 32, 16, norm_fn=nn.LayerNorm)
    m_id = MLP(16, 32, 16, norm_fn=None)
    assert isinstance(m_ln.net[1], nn.LayerNorm)
    assert isinstance(m_id.net[1], nn.Identity)

    d_default = scale_mlp_dict(2, 16, 16, 32)
    assert isinstance(d_default["0"].net[1], nn.LayerNorm)
    d_gn = scale_mlp_dict(2, 16, 16, 32, norm_fn=GroupNorm1d)
    assert isinstance(d_gn["1"].net[1], GroupNorm1d)
    d_id = scale_mlp_dict(2, 16, 16, 32, norm_fn=nn.Identity)
    assert isinstance(d_id["0"].net[1], nn.Identity)


# ---------------------------------------------------------------------- #
# 5. Hydra Config Composition & lejepa_forward Integration
# ---------------------------------------------------------------------- #
def test_hydra_config_composition_matrix():
    try:
        import stable_pretraining  # noqa: F401
    except Exception:
        pass
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    cfg_dir = os.path.abspath("config/train")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        base = compose(config_name="lewm", overrides=[])
        assert base.loss.type == "sigreg"
        assert base.loss.sigreg.weight == pytest.approx(0.09)
        assert base.model.predictor.gated_residual is False

        nca = compose(config_name="lewm", overrides=["+experiment=nca_frozen", "data=pusht"])
        assert nca.loss.type == "sigreg"
        assert nca.loss.sigreg.weight == pytest.approx(0.0)
        assert nca.loss.gate_l1.weight == pytest.approx(0.01)

        for ov in [
            [
                "loss=slow_svreg", "model=lewm", "model.predictor.gated_residual=true",
                "model.pred_proj=null", "model.projector.norm_fn._target_=torch.nn.LayerNorm",
            ],
            [
                "loss=slow_svreg", "model=dinov3_lora", "replay=per_is",
                "model.predictor.gated_residual=true", "model.pred_proj=null",
                "model.projector.norm_fn._target_=module.GroupNorm1d",
            ],
            [
                "loss=slow_svreg", "model=lewm", "model.predictor.gated_residual=true",
                "model.pred_proj=null", "model.projector.norm_fn=null",
            ],
            [
                "loss=slow_svreg", "model=convnext_ms", "data=ogb",
                "model.predictor.gated_residual=true", "model.pred_proj=null",
            ],
            [
                "loss=slow_svreg", "model=convnext_ms_scratch", "data=msdial",
                "model.predictor.gated_residual=true", "model.pred_proj=null",
            ],
            [
                "loss=slow_svreg", "model=liftwm_vit",
                "model.predictor.gated_residual=true", "model.pred_proj=null",
            ],
        ]:
            cfg = compose(config_name="lewm", overrides=ov)
            assert cfg.loss.type == "slow_svreg"
            assert cfg.loss.svreg.weight == pytest.approx(30.0)
            assert cfg.loss.slow.weight is None
            assert cfg.model.predictor.gated_residual is True
            assert cfg.model.pred_proj is None

        # Verify projector instantiation with GroupNorm1d / LayerNorm / Identity.
        cfg_gn = compose(config_name="lewm", overrides=[
            "loss=slow_svreg", "model=lewm", "model.predictor.gated_residual=true",
            "model.pred_proj=null", "model.projector.norm_fn._target_=module.GroupNorm1d",
        ])
        proj_gn = instantiate(cfg_gn.model.projector)
        assert isinstance(proj_gn.net[1], GroupNorm1d)


def test_lejepa_forward_slow_svreg_jepa_and_is_weights():
    train = _import_train()
    torch.manual_seed(0)
    HS, K, D, A = 3, 5, 16, 4
    model = JEPA(
        encoder=FakeEncoder(hw=8, d=16),
        predictor=ARPredictor(
            num_frames=HS, depth=1, heads=2, mlp_dim=16, input_dim=D,
            hidden_dim=D, output_dim=D, dim_head=4, dropout=0.0,
            gated_residual=True, gate_init=0.1,
        ),
        action_encoder=Embedder(input_dim=A, emb_dim=D),
        projector=MLP(16, 32, D, norm_fn=partial(GroupNorm1d, num_groups=4)),
        pred_proj=None,
    )

    class Mod:
        def __init__(self, m):
            self.model = m
            self.sigreg = SIGReg(knots=17, num_proj=32)
            self.svreg = SVReg(num_proj=32, eps=1e-8)
            self.slowness = SlownessLoss(slow_frac=0.125, eps=1e-6)
            self.logged = {}

        def log_dict(self, d, **kw):
            self.logged.update(d)

    cfg = OmegaConf.create({
        "history_size": HS,
        "num_preds": 1,
        "loss": {
            "type": "slow_svreg",
            "svreg": {"weight": 30.0, "max_tokens": 6, "kwargs": {"num_proj": 32}},
            "slow": {"enabled": True, "weight": None, "kwargs": {"slow_frac": 0.125}},
        },
    })

    # 1) Standard batch without replay: beta = 1 / 30.0.
    batch = {
        "pixels": torch.randn(4, HS + 1, 3, 8, 8),
        "action": torch.randn(4, HS + 1, A),
    }
    mod = Mod(model)
    out = train.lejepa_forward(mod, dict(batch), "fit", cfg)
    expected_total = (
        out["pred_loss"] + 30.0 * out["svreg_loss"] + (1.0 / 30.0) * out["slow_loss"]
    )
    assert torch.allclose(out["loss"], expected_total, atol=1e-6)
    assert "fit/svreg_loss" in mod.logged
    assert "fit/slow_loss" in mod.logged
    assert "fit/pred_gate_mean" in mod.logged
    assert "fit/pred_gate_absmax" in mod.logged
    assert float(mod.logged["fit/pred_gate_mean"]) == pytest.approx(0.1)

    model.zero_grad()
    out["loss"].backward()
    assert model.predictor.res_gate.grad is not None
    assert model.predictor.res_gate.grad.abs().sum() > 0

    # 2) lambda = 0 guard: beta must be 0.0 (no ZeroDivisionError), finite loss.
    cfg_zero = OmegaConf.merge(cfg, {"loss": {"svreg": {"weight": 0.0}}})
    out_zero = train.lejepa_forward(Mod(model), dict(batch), "fit", cfg_zero)
    assert torch.isfinite(out_zero["loss"])
    assert float(out_zero["svreg_loss"]) == 0.0
    assert torch.allclose(out_zero["loss"], out_zero["pred_loss"], atol=1e-6)

    # 3) PER + H-net IS weights: w_sig passed to BOTH SVReg and SlownessLoss.
    model_eval = model.eval()
    prob = torch.tensor([0.05, 0.1, 0.2, 0.25, 0.2, 0.2])
    full_batch = {
        "pixels": torch.randn(6, HS + K, 3, 8, 8),
        "action": torch.randn(6, HS + K, A),
        "idx": torch.arange(6),
        "sample_prob": prob.double(),
    }
    rc = ReplayConfig(enabled=True, sampler="prioritized", horizon=K, warmup_epochs=0.0,
                      chunk_batches=1, log_every=1, dump_topk=2)
    rc.validate()
    st = ReplayState(rc, n=6)
    st.beta = 1.0
    torch.manual_seed(7)
    out_is = train.lejepa_forward(Mod(model_eval), dict(full_batch), "fit", cfg, replay=st)
    with torch.no_grad():
        enc = model_eval.encode({
            "pixels": full_batch["pixels"][:, :HS + 1],
            "action": full_batch["action"][:, :HS + 1],
        })
        w_sig = (1.0 / prob).float()
        w_sig = w_sig / w_sig.mean()
        torch.manual_seed(7)
        # Note: predict runs before svreg, which doesn't consume torch RNG in eval().
        ref_svreg = SVReg(num_proj=32, eps=1e-8)(enc["emb"].transpose(0, 1), weights=w_sig)
        ref_slow = SlownessLoss(slow_frac=0.125, eps=1e-6)(enc["emb"], weights=w_sig)
    assert torch.allclose(out_is["svreg_loss"], ref_svreg, atol=1e-5)
    assert torch.allclose(out_is["slow_loss"], ref_slow, atol=1e-5)


def test_lejepa_forward_slow_svreg_pyramid_jepa():
    train = _import_train()
    torch.manual_seed(0)
    HS, D, A = 3, 16, 4

    class TinyPyrEnc(nn.Module):
        levels = 1

        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(3 * 8 * 8, D)
            self.l1 = nn.Linear(3 * 8 * 8, D)

        def forward(self, pixels, **kw):
            flat = pixels.flatten(1)
            n = flat.shape[0]
            t0 = self.l0(flat)[:, None, :].expand(n, 4, D)
            t1 = self.l1(flat)[:, None, :].expand(n, 9, D)
            return PyramidOutput(scales=[
                ScaleTokens(t0, (2, 2), 0),
                ScaleTokens(t1, (3, 3), 1),
            ])

    model = PyramidJEPA(
        encoder=TinyPyrEnc(),
        predictor=hier_vit(
            n_scales=2, latent_dim=D, act_dim=D, depth=1, heads=2, dim_head=8,
            gated_residual=True, gate_init=0.1,
        ),
        action_encoder=Embedder(input_dim=A, emb_dim=D),
        projector=scale_mlp_dict(2, D, D, 32),
        pred_proj=None,
    )

    class Mod:
        def __init__(self, m):
            self.model = m
            self.svreg = SVReg(num_proj=32, eps=1e-8)
            self.slowness = SlownessLoss(slow_frac=0.125, eps=1e-6)
            self.logged = {}

        def log_dict(self, d, **kw):
            self.logged.update(d)

    cfg = OmegaConf.create({
        "history_size": HS,
        "num_preds": 1,
        "loss": {
            "type": "slow_svreg",
            "svreg": {"weight": 10.0, "max_tokens": 8, "kwargs": {"num_proj": 32}},
            "slow": {"enabled": True, "weight": None, "kwargs": {"slow_frac": 0.125}},
        },
    })
    batch = {
        "pixels": torch.randn(2, HS + 1, 3, 8, 8),
        "action": torch.randn(2, HS + 1, A),
    }
    mod = Mod(model)
    out = train.lejepa_forward(mod, dict(batch), "fit", cfg)
    assert torch.isfinite(out["loss"])
    assert "fit/svreg_loss" in mod.logged
    assert "fit/slow_loss" in mod.logged
    assert "fit/pred_gate_mean" in mod.logged
    assert "fit/pred_gate_absmax" in mod.logged
    assert float(mod.logged["fit/pred_gate_mean"]) == pytest.approx(0.1)

    model.zero_grad()
    out["loss"].backward()
    assert model.predictor.predictors["0"].res_gate.grad is not None
    assert model.predictor.predictors["1"].res_gate.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bf16_autocast_float32_stability():
    svreg = SVReg(num_proj=64, eps=1e-8).cuda()
    slow = SlownessLoss(slow_frac=0.125, eps=1e-6).cuda()
    emb = (torch.randn(4, 5, 16, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        l_sv = svreg(emb.transpose(0, 1))
        l_sl = slow(emb)
    assert l_sv.dtype == torch.float32 and torch.isfinite(l_sv)
    assert l_sl.dtype == torch.float32 and torch.isfinite(l_sl)
    (l_sv + l_sl).backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
