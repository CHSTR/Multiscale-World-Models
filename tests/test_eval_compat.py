from pathlib import Path
import functools
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

import eval as eval_mod
from jepa import JEPA
from jepa_pyramid import PyramidJEPA
from module import ARPredictor, Embedder, GroupNorm1d, MLP
from wavelet.factories import hier_vit, scale_mlp_dict
from wavelet.pyramid_encoder import PyramidOutput, ScaleTokens


EVAL_CFG_DIR = str(Path(__file__).resolve().parents[1] / "config" / "eval")


def test_all_eval_configs_resolve_portably():
    for name in ("cube", "msdial", "pusht", "reacher", "tworoom"):
        with initialize_config_dir(EVAL_CFG_DIR, version_base=None):
            cfg = compose(config_name=name)
            OmegaConf.resolve(cfg)
            assert cfg.solver._target_ == "stable_worldmodel.planning.CEMSolver"
            assert "cost" in cfg.solver.keys()
            assert "/home/chr" not in str(cfg)

    with initialize_config_dir(EVAL_CFG_DIR, version_base=None):
        cfg_adam = compose(config_name="msdial", overrides=["solver=adam"])
        OmegaConf.resolve(cfg_adam)
        assert cfg_adam.solver._target_ == "stable_worldmodel.planning.GradientSolver"
        assert "cost" in cfg_adam.solver.keys()


def test_msdial_env_registered_via_eval_import():
    spec = gym.spec("swm/MultiScaleDial-v0")
    assert spec is not None


def test_resolve_results_path_uses_checkpoints_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    with initialize_config_dir(EVAL_CFG_DIR, version_base=None):
        cfg_pt = compose(config_name="msdial", overrides=["policy=my_run/weights_epoch_1.pt"])
        assert eval_mod.resolve_results_path(cfg_pt) == tmp_path / "checkpoints" / "my_run"

        ckpt_dir = tmp_path / "checkpoints" / "dir_run"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg_dir = compose(config_name="msdial", overrides=["policy=dir_run"])
        assert eval_mod.resolve_results_path(cfg_dir) == ckpt_dir

        cfg_out = compose(
            config_name="msdial",
            overrides=[f"+output.dir={tmp_path / 'custom_out'}"],
        )
        assert eval_mod.resolve_results_path(cfg_out) == tmp_path / "custom_out"


class _TinyEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x, **_kw):
        b = x.size(0)
        return types.SimpleNamespace(last_hidden_state=torch.ones(b, 5, self.dim, device=x.device))


class _TinyPyramidEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.levels = 1
        self.dim = dim

    def forward(self, x, **_kw):
        b = x.size(0)
        return PyramidOutput(
            scales=[
                ScaleTokens(torch.ones(b, 4, self.dim, device=x.device), (2, 2), 0),
                ScaleTokens(torch.ones(b, 9, self.dim, device=x.device), (3, 3), 1),
            ]
        )


def test_instantiate_cem_solver_and_cost_batch_gt_1():
    dim = 8
    jepa = JEPA(
        _TinyEncoder(dim),
        ARPredictor(
            num_frames=3,
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            depth=1,
            heads=2,
            mlp_dim=16,
            dim_head=4,
            gated_residual=True,
        ),
        Embedder(input_dim=2, emb_dim=dim),
        projector=MLP(dim, 16, output_dim=dim, norm_fn=functools.partial(GroupNorm1d, num_groups=2)),
        pred_proj=None,
    ).eval()

    pyr = PyramidJEPA(
        _TinyPyramidEncoder(dim),
        hier_vit(n_scales=2, latent_dim=dim, act_dim=dim, depth=1, heads=2, dim_head=4, mlp_dim=16, gated_residual=True),
        Embedder(input_dim=2, emb_dim=dim),
        projector=scale_mlp_dict(2, dim, dim, 16, norm_fn=functools.partial(GroupNorm1d, num_groups=2)),
        pred_proj=None,
    ).eval()

    with initialize_config_dir(EVAL_CFG_DIR, version_base=None):
        cfg = compose(
            config_name="msdial",
            overrides=["solver.device=cpu", "solver.num_samples=4", "solver.n_steps=2", "solver.topk=2"],
        )
        solver_jepa = eval_mod.instantiate_solver(cfg, jepa)
        solver_pyr = eval_mod.instantiate_solver(cfg, pyr)
        assert solver_jepa.cost is jepa
        assert solver_pyr.cost is pyr

    # Verify JEPA.get_cost handles batch_size > 1 and non-tensor goal_* keys cleanly
    B, S, H, T = 2, 3, 1, 3
    info = {
        "pixels": torch.zeros(B, S, H, 3, 16, 16),
        "goal": torch.zeros(B, S, 1, 3, 16, 16),
        "goal_meta": "non-tensor",
    }
    actions = torch.zeros(B, S, T, 2)
    cost_jepa = jepa.get_cost(info, actions)
    assert cost_jepa.shape == (B, S)
    assert torch.isfinite(cost_jepa).all()

    info_pyr = {
        "pixels": torch.zeros(B, S, H, 3, 16, 16),
        "goal": torch.zeros(B, S, 1, 3, 16, 16),
        "goal_meta": "non-tensor",
    }
    cost_pyr = pyr.get_cost(info_pyr, actions)
    assert cost_pyr.shape == (B, S)
    assert torch.isfinite(cost_pyr).all()
