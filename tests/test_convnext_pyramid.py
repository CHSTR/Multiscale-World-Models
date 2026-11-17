"""Tests del encoder ConvNeXt multiescala (rama convnext-ms).

No descargan pesos: la ruta ``init=dinov3`` se prueba con un checkpoint falso
(ConvNeXt aleatorio guardado en disco) cargado por el mismo hub de DINOv3.
"""
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.encoders.convnext_pyramid_encoder import ConvNeXtPyramidEncoder
from src.encoders.vit_lora import LoRALinear
from wavelet.pyramid_encoder import PyramidOutput


@pytest.fixture
def fake_dinov3_ckpt(tmp_path, monkeypatch):
    """ConvNeXt-tiny aleatorio guardado como si fuera el ckpt DINOv3."""
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_home"))
    from src.dinov3.models.convnext import ConvNeXt, convnext_sizes
    torch.manual_seed(0)
    m = ConvNeXt(**convnext_sizes["tiny"])
    m.init_weights()
    path = tmp_path / "dinov3_convnext_tiny_fake.pth"
    torch.save(m.state_dict(), path)
    return str(path), m.state_dict()


def _scratch(**kw):
    kw.setdefault("init", "scratch")
    kw.setdefault("finetune", "full")
    return ConvNeXtPyramidEncoder(**kw)


def test_default_scales_coarse_to_fine():
    enc = _scratch()
    out = enc(torch.randn(2, 3, 224, 224))
    assert isinstance(out, PyramidOutput) and len(out.scales) == 3
    assert enc.levels == 2
    assert [s.level for s in out.scales] == [0, 1, 2]
    assert [s.grid_hw for s in out.scales] == [(7, 7), (14, 14), (28, 28)]
    assert enc.grid_sizes() == [7, 14, 28]
    for s in out.scales:
        assert s.tokens.shape == (2, s.grid_hw[0] * s.grid_hw[1], 192)


def test_out_grids_pooling():
    enc = _scratch(out_grids=[7, 14, 14], out_dim=64)
    out = enc(torch.randn(1, 3, 224, 224))
    assert [s.grid_hw for s in out.scales] == [(7, 7), (14, 14), (14, 14)]
    assert all(s.tokens.shape[-1] == 64 for s in out.scales)


def test_unused_stages_are_pruned():
    enc = _scratch(stages=[2, 1])
    assert enc.stages == [2, 1] and enc.levels == 1
    assert isinstance(enc.model.stages[3], torch.nn.Identity)
    out = enc(torch.randn(1, 3, 224, 224))
    assert [s.grid_hw for s in out.scales] == [(14, 14), (28, 28)]


def test_invalid_args():
    with pytest.raises(ValueError):
        _scratch(stages=[4])
    with pytest.raises(ValueError):
        _scratch(out_grids=[7, 14])
    with pytest.raises(ValueError):
        _scratch(size="huge")


def test_scratch_full_all_trainable_and_grad_flow():
    enc = _scratch(stages=[1, 2])
    assert all(p.requires_grad for p in enc.parameters())
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    sum(s.tokens.sum() for s in enc(x).scales).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_dinov3_ckpt_loads_weights(fake_dinov3_ckpt):
    path, sd = fake_dinov3_ckpt
    enc = ConvNeXtPyramidEncoder(init="dinov3", ckpt_path=path, finetune="full")
    got = enc.model.stages[2][0].dwconv.weight
    assert torch.equal(got, sd["stages.2.0.dwconv.weight"])


def test_dinov3_lora_freezing(fake_dinov3_ckpt):
    path, _ = fake_dinov3_ckpt
    enc = ConvNeXtPyramidEncoder(init="dinov3", ckpt_path=path, finetune="lora")
    loras = [m for m in enc.model.modules() if isinstance(m, LoRALinear)]
    assert len(loras) == 2 * (3 + 3 + 9 + 3)          # pwconv1/2 en todos los bloques
    assert not enc.model.stages[1][0].dwconv.weight.requires_grad
    assert not loras[0].weight.requires_grad and loras[0].lora_A.requires_grad
    assert all(p.requires_grad for p in enc.necks.parameters())


def test_dinov3_frozen_only_necks_train(fake_dinov3_ckpt):
    path, _ = fake_dinov3_ckpt
    enc = ConvNeXtPyramidEncoder(init="dinov3", ckpt_path=path, finetune="frozen")
    assert not any(p.requires_grad for p in enc.model.parameters())
    assert all(p.requires_grad for p in enc.necks.parameters())


def test_missing_ckpt_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConvNeXtPyramidEncoder(init="dinov3", ckpt_path=str(tmp_path / "nope.pth"))


def test_starlet_frontend_expands_stem():
    enc = _scratch(stages=[3], starlet_levels=2)
    assert enc.in_chans == 9
    assert enc.model.downsample_layers[0][0].in_channels == 9
    out = enc(torch.randn(1, 3, 224, 224))
    assert out.scales[0].grid_hw == (7, 7)


# ---------------------------------------------------------------------- #
# Integración con PyramidJEPA vía Hydra
# ---------------------------------------------------------------------- #
def _cfg(model, extra=()):
    cfg_dir = os.path.abspath("config/train")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="lewm", overrides=[
            f"model={model}", "data=tworoom", "model.action_encoder.input_dim=8", *extra,
        ])


def test_scratch_config_inherits_base():
    cfg = _cfg("convnext_ms_scratch")
    assert cfg.model.encoder.init == "scratch"
    assert cfg.model.encoder.finetune == "full"
    assert list(cfg.model.encoder.stages) == [1, 2, 3]
    assert cfg.model.predictor.n_scales == 3


def test_config_forward_predict(fake_dinov3_ckpt):
    path, _ = fake_dinov3_ckpt
    m = instantiate(_cfg("convnext_ms", [f"model.encoder.ckpt_path={path}"]).model)
    info = {"pixels": torch.randn(2, 4, 3, 224, 224), "action": torch.randn(2, 4, 8)}
    out = m.encode(info)
    assert sorted(out["emb"]) == [0, 1, 2]
    assert out["emb"][2].shape == (2, 4, 28 * 28, 192)
    pred = m.predict({l: v[:, :3] for l, v in out["emb"].items()}, out["act_emb"][:, :3])
    assert {l: v.shape for l, v in pred.items()} == {
        0: (2, 3, 49, 192), 1: (2, 3, 196, 192), 2: (2, 3, 784, 192)}


def test_get_cost_planning_path_and_ckpt_roundtrip(tmp_path):
    cfg = _cfg("convnext_ms_scratch", [
        "model.encoder.out_grids=[7,7,14]", "model.predictor.depth=1"])
    m = instantiate(cfg.model).eval()
    B, S, H, T = 1, 2, 3, 5
    info = {
        "pixels": torch.randn(B, S, H, 3, 224, 224),
        "goal": torch.randn(B, S, 1, 3, 224, 224),
        "action": torch.randn(B, S, H, 8),
    }
    with torch.no_grad():
        cost = m.get_cost(info, torch.randn(B, S, T, 8))
    assert cost.shape == (B, S) and torch.isfinite(cost).all()

    p = tmp_path / "w.pt"
    torch.save(m.state_dict(), p)
    m2 = instantiate(cfg.model).eval()
    m2.load_state_dict(torch.load(p))
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for s1, s2 in zip(m.encoder(x).scales, m2.encoder(x).scales):
            assert torch.allclose(s1.tokens, s2.tokens, atol=1e-5)
