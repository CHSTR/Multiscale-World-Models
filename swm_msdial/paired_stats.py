"""Estadística pareada para evals msdial (spec §3): Δpp con IC bootstrap,
McNemar exacto y corrección de Holm. Consume los results.jsonl de la eval.

Uso:
  /home/chr/le-wm/.venv/bin/python swm_msdial/paired_stats.py \
      --run-a <dir_lewm>/results.jsonl --run-b <dir_ms>/results.jsonl \
      --key success_dial --out reports/pareado.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------ núcleo
def paired(a: np.ndarray, b: np.ndarray, rng=None,
           n_boot: int = 10000, ci: float = 0.95) -> dict:
    """Comparación pareada de éxitos por episodio (mismo set fijo de eval).

    a, b: arrays bool (n,) con los éxitos por episodio de cada run.
    Devuelve Δpp (a-b), IC bootstrap por episodio, McNemar exacto
    (discordantes b/c) y el conteo por discordancia.
    """
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    assert a.shape == b.shape
    n = len(a)
    delta = a.mean() - b.mean()

    diffs = a.astype(float) - b.astype(float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boots, [alpha, 1 - alpha])

    b_only_a = int((a & ~b).sum())   # solo gana A
    c_only_b = int((~a & b).sum())   # solo gana B
    m = b_only_a + c_only_b
    if m == 0:
        p = 1.0
    else:  # McNemar exacto (dos colas) sobre Bin(m, 0.5)
        k = min(b_only_a, c_only_b)
        p = min(1.0, 2.0 * binom.cdf(k, m, 0.5))

    return {
        "n": n,
        "delta_pp": float(delta * 100.0),
        "ci_low": float(ci_low * 100.0),
        "ci_high": float(ci_high * 100.0),
        "b_only_a": b_only_a,
        "c_only_b": c_only_b,
        "mcnemar_p": float(p),
    }


def holm(pvals: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni: devuelve p-ajustados en el orden dado."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj_sorted = p[order] * (m - np.arange(m))
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj.tolist()


# ------------------------------------------------------------------- I/O
def load_successes(jsonl_path: str, key: str, row: int = 0) -> np.ndarray:
    """Lee la línea `row` del results.jsonl y devuelve bool (n,)."""
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == row:
                return np.asarray(json.loads(line)[key], bool)
    raise IndexError(f"results.jsonl tiene <= {row} líneas: {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="results.jsonl del run A")
    ap.add_argument("--run-b", required=True, help="results.jsonl del run B")
    ap.add_argument("--row-a", type=int, default=0)
    ap.add_argument("--row-b", type=int, default=0)
    ap.add_argument("--key", default="successes",
                    choices=["successes", "success_pos", "success_dial"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = paired(load_successes(a.run_a, a.key, a.row_a),
                 load_successes(a.run_b, a.key, a.row_b))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print("->", a.out)


if __name__ == "__main__":
    main()
