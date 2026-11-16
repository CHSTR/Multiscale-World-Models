import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _cfg(model, extra=()):
    cfg_dir = os.path.abspath("config/train")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="lewm", overrides=[
            f"model={model}", "data=tworoom", "model.action_encoder.input_dim=8", *extra,
        ])


def test_liftwm_nca_forward():
    m = instantiate(_cfg("liftwm").model)
    info = {"pixels": torch.randn(2, 4, 3, 224, 224), "action": torch.randn(2, 4, 8)}
    out = m.encode(info)
    assert isinstance(out["emb"], dict) and len(out["emb"]) == 4
    pred = m.predict(out["emb"], out["act_emb"])
    assert set(pred) == set(out["emb"])


def test_liftwm_vit_forward():
    m = instantiate(_cfg("liftwm_vit").model)
    info = {"pixels": torch.randn(2, 4, 3, 224, 224), "action": torch.randn(2, 4, 8)}
    out = m.encode(info)
    pred = m.predict(out["emb"], out["act_emb"])
    assert set(pred) == set(out["emb"])
    assert pred[3].shape[2] == 28 * 28   # escala fina


def test_checkpoint_roundtrip(tmp_path):
    cfg = _cfg("liftwm")
    m1 = instantiate(cfg.model)
    p = tmp_path / "c.pt"
    torch.save(m1.state_dict(), p)
    m2 = instantiate(cfg.model)
    m2.load_state_dict(torch.load(p))
    x = torch.randn(1, 3, 224, 224)
    o1, o2 = m1.encoder(x), m2.encoder(x)
    for s1, s2 in zip(o1.scales, o2.scales):
        assert torch.allclose(s1.tokens, s2.tokens, atol=1e-5)


def test_get_cost_planning_path():
    speed = ["model.encoder.stem_blocks=0", "model.predictor.steps=[2,2]"]
    m = instantiate(_cfg("liftwm", speed).model).eval()
    B, S, H, T = 1, 2, 3, 5
    info = {
        "pixels": torch.randn(B, S, H, 3, 224, 224),
        "goal": torch.randn(B, S, 1, 3, 224, 224),
        "action": torch.randn(B, S, H, 8),
    }
    cost = m.get_cost(info, torch.randn(B, S, T, 8))
    assert cost.shape == (B, S)
    assert torch.isfinite(cost).all()
