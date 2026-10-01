"""Probes lineales (ridge) con split por episodio y selección de alpha por GroupKFold (solo numpy)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def episode_split(ep_ids: np.ndarray, test_frac: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Máscaras train/test separando EPISODIOS completos (evita fuga entre frames vecinos)."""
    uniq = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    test_eps = rng.choice(uniq, size=max(1, int(round(test_frac * uniq.size))), replace=False)
    test = np.isin(ep_ids, test_eps)
    return ~test, test


def expand_angles(Y: np.ndarray, angle_dims: Sequence[int], names: Optional[List[str]] = None):
    """Reemplaza cada dimensión angular (radianes) por (sin, cos). Devuelve Y', nombres y el mapa."""
    names = names or [f"d{i}" for i in range(Y.shape[1])]
    cols, out_names, angle_pairs = [], [], {}
    for i in range(Y.shape[1]):
        if i in angle_dims:
            angle_pairs[names[i]] = (len(cols), len(cols) + 1)
            cols += [np.sin(Y[:, i]), np.cos(Y[:, i])]
            out_names += [f"{names[i]}_sin", f"{names[i]}_cos"]
        else:
            cols.append(Y[:, i])
            out_names.append(names[i])
    return np.stack(cols, axis=1), out_names, angle_pairs


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Ridge con intercepto (centrando). Devuelve (W, b)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    d = X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    b = ym - xm @ W
    return W, b


def r2_per_dim(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)


def select_alpha(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas=DEFAULT_ALPHAS,
                 k: int = 5, seed: int = 0) -> float:
    """Alpha con mejor R² medio en validación cruzada agrupada por episodio."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), min(k, uniq.size))
    best, best_score = alphas[0], -np.inf
    for a in alphas:
        scores = []
        for f in folds:
            va = np.isin(groups, f)
            if va.all() or not va.any():
                continue
            W, b = ridge_fit(X[~va], Y[~va], a)
            scores.append(np.nanmean(r2_per_dim(Y[va], X[va] @ W + b)))
        s = float(np.mean(scores)) if scores else -np.inf
        if s > best_score:
            best, best_score = a, s
    return best


def fit_probe(X: np.ndarray, Y: np.ndarray, ep_ids: np.ndarray, test_frac: float = 0.2,
              angle_dims: Sequence[int] = (), names: Optional[List[str]] = None,
              alphas=DEFAULT_ALPHAS, seed: int = 0) -> Dict[str, object]:
    """Ajusta un probe lineal X -> Y con split por episodio. X se estandariza con stats de train."""
    Yx, out_names, angle_pairs = expand_angles(Y, angle_dims, names)
    tr, te = episode_split(ep_ids, test_frac, seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    alpha = select_alpha(Xs[tr], Yx[tr], ep_ids[tr], alphas, seed=seed)
    W, b = ridge_fit(Xs[tr], Yx[tr], alpha)
    pred_te, pred_tr = Xs[te] @ W + b, Xs[tr] @ W + b
    res: Dict[str, object] = {
        "alpha": alpha,
        "names": out_names,
        "r2_test": r2_per_dim(Yx[te], pred_te),
        "r2_train": r2_per_dim(Yx[tr], pred_tr),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "angle_mae_deg": {},
    }
    for name, (i, j) in angle_pairs.items():
        true = np.arctan2(Yx[te, i], Yx[te, j])
        est = np.arctan2(pred_te[:, i], pred_te[:, j])
        err = np.angle(np.exp(1j * (est - true)))
        res["angle_mae_deg"][name] = float(np.degrees(np.abs(err)).mean())
    return res
