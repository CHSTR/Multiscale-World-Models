"""Estadística para comparar tasas de éxito con episodios emparejados (sin dependencias de swm)."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """IC de Wilson para una proporción (mejor que el normal cerca de 0 o 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """p bilateral exacto de McNemar: b = (A éxito, B fallo), c = (A fallo, B éxito)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired_sign_flip(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """p bilateral para H0: E[d] = 0 con d = diferencias emparejadas (test de permutación de signo).

    Exacto si hay <= 16 diferencias no nulas; si no, Monte Carlo con ``n_perm`` permutaciones.
    Sirve con cualquier número de réplicas por episodio (d en [-1, 1]).
    """
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    obs = abs(nz.sum())
    if nz.size <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=nz.size)))
        stats = np.abs(signs @ nz)
        return float(np.mean(stats >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, nz.size))
    stats = np.abs(signs @ nz)
    return float((1 + np.sum(stats >= obs - 1e-12)) / (n_perm + 1))


def bootstrap_mean_ci(d: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                      seed: int = 0) -> Tuple[float, float]:
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def holm(pvals: Sequence[float]) -> List[float]:
    """Corrección de Holm-Bonferroni (controla FWER), devuelve p ajustados en el orden original."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


EpisodeKey = Tuple[int, int, int]  # (eval_seed, episode_idx, start_step)


def per_key_success(records: List[Dict[str, np.ndarray]]) -> Dict[EpisodeKey, List[float]]:
    """Agrupa éxitos por episodio a través de réplicas (p. ej. semillas de entrenamiento)."""
    out: Dict[EpisodeKey, List[float]] = {}
    for r in records:
        for s, e, t, ok in zip(r["eval_seed"], r["episode_idx"], r["start_step"], r["success"]):
            out.setdefault((int(s), int(e), int(t)), []).append(float(ok))
    return out


def compare(ref: Dict[EpisodeKey, List[float]], other: Dict[EpisodeKey, List[float]],
            seed: int = 0) -> Dict[str, float]:
    """Compara ``other`` contra ``ref`` sobre los episodios comunes.

    d_k = mean(éxito other en k) - mean(éxito ref en k). Reporta Δ en pp, IC bootstrap,
    p de permutación de signo y, si hay exactamente una réplica por episodio en ambos, McNemar exacto.
    """
    keys = sorted(set(ref) & set(other))
    if not keys:
        raise ValueError("no hay episodios en común: ¿mismo eval seed y num_eval?")
    a = np.array([np.mean(ref[k]) for k in keys])
    b = np.array([np.mean(other[k]) for k in keys])
    d = b - a
    lo, hi = bootstrap_mean_ci(d, seed=seed)
    res = {
        "n_pairs": len(keys),
        "delta_pp": 100 * float(d.mean()),
        "ci_lo_pp": 100 * lo,
        "ci_hi_pp": 100 * hi,
        "p_signflip": paired_sign_flip(d, seed=seed),
    }
    if all(len(ref[k]) == 1 and len(other[k]) == 1 for k in keys):
        b_cnt = int(np.sum((b == 1) & (a == 0)))
        c_cnt = int(np.sum((b == 0) & (a == 1)))
        res.update({"only_other": b_cnt, "only_ref": c_cnt, "p_mcnemar": mcnemar_exact(b_cnt, c_cnt)})
    return res
