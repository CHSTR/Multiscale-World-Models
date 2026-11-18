"""Tests de las sondas ridge por episodio (Task 6 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_probe.py -q
"""
import numpy as np
import pytest


def test_probe_recovers_linear_map():
    from swm_msdial.probe import ridge_fit, r2

    rng = np.random.default_rng(0)
    n, d, k = 2000, 32, 4
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    y = X @ W + 0.01 * rng.normal(size=(n, k))  # mapa lineal + ruido leve
    W_hat = ridge_fit(X[:1600], y[:1600], lam=1e-6)
    y_hat = X[1600:] @ W_hat
    assert r2(y[1600:], y_hat) > 0.95
    # señal nula: targets independientes -> R^2 ~ 0
    y_null = rng.normal(size=(n, k))
    W_null = ridge_fit(X[:1600], y_null[:1600], lam=1e-6)
    assert r2(y_null[1600:], X[1600:] @ W_null) < 0.05


def test_split_episodes_disjoint():
    from swm_msdial.probe import split_episodes

    rng = np.random.default_rng(0)
    ep_idx = rng.integers(0, 100, size=10000)
    train_eps, val_eps = split_episodes(ep_idx, val_frac=0.2, seed=0)
    assert set(train_eps) & set(val_eps) == set()
    assert len(val_eps) == 20 and len(train_eps) == 80
