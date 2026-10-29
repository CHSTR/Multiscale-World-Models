import types

import numpy as np
import pytest
import torch
from torch import nn

from probe_train import (
    GROUP_SPECS,
    drop_nan_rows,
    extract_features,
    fit_mlp,
    fit_ridge,
    make_rows,
    mse_and_pearson,
    predict_mlp,
    sample_and_split,
    standardize,
)


class FakeDataset:
    def __init__(self, ep_idx):
        self.ep_idx = np.asarray(ep_idx)

    def __len__(self):
        return len(self.ep_idx)

    def get_col_data(self, name):
        assert name == "ep_idx"
        return self.ep_idx


def test_group_specs_cover_eight_groups():
    assert len(GROUP_SPECS) == 8
    assert GROUP_SPECS["joint_pos"] == ["proprio_joint_pos"]
    assert set(GROUP_SPECS["gripper"]) == {
        "proprio_gripper_opening",
        "proprio_gripper_contact",
        "proprio_gripper_vel",
    }


def test_split_has_no_episode_leakage_and_is_deterministic():
    ep = np.repeat(np.arange(20), 10)
    ds = FakeDataset(ep)
    tr1, va1 = sample_and_split(ds, n_frames=100, val_frac=0.2, seed=0)
    tr2, va2 = sample_and_split(ds, n_frames=100, val_frac=0.2, seed=0)
    assert tr1 == tr2 and va1 == va2
    assert set(ep[tr1]).isdisjoint(set(ep[va1]))
    assert len(tr1) + len(va1) == 100


class StubEncoder(nn.Module):
    """Devuelve [CLS(3), storage(2), patches(4)]."""

    def __init__(self, n_patches=4):
        super().__init__()
        self.n_patches = n_patches

    def forward(self, x, **kw):
        b = x.size(0)
        cls = torch.full((b, 1, 3), 10.0)
        storage = torch.zeros(b, 2, 3)
        patches = torch.arange(self.n_patches, dtype=torch.float32).view(1, -1, 1).expand(b, -1, 3)
        return types.SimpleNamespace(last_hidden_state=torch.cat([cls, storage, patches], dim=1))


class StubModel(nn.Module):
    def __init__(self, n_patches=4):
        super().__init__()
        self.encoder = StubEncoder(n_patches)


class TinyDataset:
    def __init__(self, n=6):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "pixels": torch.zeros(1, 3, 224, 224),
            "proprio_joint_pos": torch.full((1, 6), float(i)),
        }


def test_patch_mean_uses_last_tokens_not_tokens_after_cls():
    ds = TinyDataset()
    out = extract_features(
        StubModel(n_patches=4),
        ds,
        list(range(6)),
        source="patch_mean",
        batch_size=3,
        target_columns=["proprio_joint_pos"],
    )
    # patches valen 0,1,2,3 -> media 1.5; si tomara storage+patches daria 1.0
    assert out["features"].shape == (6, 3)
    assert np.allclose(out["features"], 1.5)


def test_targets_are_concatenated_and_cls_is_used():
    ds = TinyDataset()
    out = extract_features(
        StubModel(),
        ds,
        [0, 1],
        source="cls",
        batch_size=2,
        target_columns=["proprio_joint_pos"],
    )
    assert np.allclose(out["features"], 10.0)
    assert out["targets"]["proprio_joint_pos"].shape == (2, 6)


def test_mse_and_pearson_perfect_and_random():
    y = np.arange(100.0).reshape(-1, 1)
    mse, r = mse_and_pearson(y, y)
    assert mse == pytest.approx(0.0)
    assert r == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    y_rand = rng.normal(size=(100, 1))
    _, r_rand = mse_and_pearson(y, y_rand)
    assert abs(r_rand) < 0.2


def test_mse_and_pearson_constant_target_returns_zero():
    y = np.ones((50, 1))
    mse, r = mse_and_pearson(y, np.zeros((50, 1)))
    assert np.isfinite(mse)
    assert r == 0.0


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 16))
    W = rng.normal(size=(16, 3))
    Y = X @ W
    W_hat, b_hat, alpha = fit_ridge(X[:1500], Y[:1500], X[1500:], Y[1500:], [1e-4, 1e-2, 1.0])
    _, Zva, _ = standardize(X[:1500], X[1500:])
    pred = Zva @ W_hat + b_hat
    _, r = mse_and_pearson(Y[1500:], pred)
    assert r > 0.99


def test_standardize_zero_variance_dim_is_safe():
    X = np.ones((20, 4))
    X[:, 1] = np.arange(20)
    Z, _, _ = standardize(X, X)
    assert np.isfinite(Z).all()


def test_drop_nan_rows_removes_rows_with_any_nan_target():
    Y = {"a": np.array([[1.0], [np.nan]]), "b": np.array([[0.0], [1.0]])}
    mask = drop_nan_rows(Y)
    assert mask.tolist() == [True, False]


def test_mlp_beats_linear_on_nonlinear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8)).astype("float32")
    Y = np.stack([X[:, 0] * X[:, 1], np.tanh(X[:, 2])], axis=1)
    W, b, _ = fit_ridge(X[:3000], Y[:3000], X[3000:], Y[3000:], [1e-2])
    _, Zva, _ = standardize(X[:3000], X[3000:])
    _, r_lin = mse_and_pearson(Y[3000:], Zva @ W + b)

    model = fit_mlp(X[:3000], Y[:3000], X[3000:], Y[3000:], epochs=80, patience=10)
    pred = predict_mlp(model, X[3000:])
    _, r_mlp = mse_and_pearson(Y[3000:], pred)
    assert r_mlp > r_lin + 0.25
    assert r_mlp > 0.8


def test_rows_have_expected_columns_and_group_dim():
    y = np.random.default_rng(0).normal(size=(50, 3))
    rows = make_rows("ckpt.pt", "cls", "ridge", y, y, {"block_pos": ["a", "b", "c"]})
    assert {r["group"] for r in rows} == {"block_pos"}
    assert rows[0]["dim"] == 3
    assert rows[0]["pearson"] == pytest.approx(1.0)
    assert rows[0]["mse_z"] == pytest.approx(0.0)
