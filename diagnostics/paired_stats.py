"""Experimento 1: comparación pareada de tasas de éxito a partir de las trazas de evaluación.

Cada ``--run ETIQUETA=GLOB`` agrupa uno o más ``traces_*.npz`` (p. ej. las semillas de entrenamiento
de una variante). Los episodios se emparejan por (eval_seed, episode_idx, start_step): con el mismo
``seed`` de evaluación todos los modelos ven exactamente los mismos estados iniciales y goals.

Reporta por variante: éxito (%), IC de Wilson, media ± desviación entre réplicas; y contra la
referencia: Δ en pp con IC bootstrap, p de permutación de signo (Holm entre comparaciones) y
McNemar exacto cuando hay una réplica por episodio.

Uso:
    python -m diagnostics.paired_stats \
        --run "DINOv3=ckpts/cube__dinov3-rgb__s*/traces_*_clean_seed42.npz" \
        --run "DINOv3+Starlet=ckpts/cube__dinov3-starlet-legacy__s*/traces_*_clean_seed42.npz" \
        --ref DINOv3 --out cube_report.md
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np

from diagnostics.stats_core import compare, holm, per_key_success, wilson_ci


def load_traces(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    missing = [k for k in ("ep_success", "ep_episode_idx", "ep_start_step", "ep_eval_seed") if k not in d]
    if missing:
        raise ValueError(f"{path}: faltan {missing}; evalúa con la versión nueva de viz/eval_reward.py")
    return {"success": d["ep_success"].astype(bool), "episode_idx": d["ep_episode_idx"],
            "start_step": d["ep_start_step"], "eval_seed": d["ep_eval_seed"]}


def summarize(label: str, records: List[Dict[str, np.ndarray]]) -> Dict[str, object]:
    succ = np.concatenate([r["success"] for r in records])
    k, n = int(succ.sum()), int(succ.size)
    lo, hi = wilson_ci(k, n)
    per_rep = [100 * r["success"].mean() for r in records]
    return {"variant": label, "files": len(records), "episodes": n, "success_pct": 100 * k / n,
            "wilson_lo": 100 * lo, "wilson_hi": 100 * hi,
            "rep_mean": float(np.mean(per_rep)),
            "rep_std": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="ETIQUETA=GLOB (repetible)")
    ap.add_argument("--ref", help="etiqueta de referencia (por defecto la primera)")
    ap.add_argument("--out", help="ruta .md; además escribe un .csv al lado")
    args = ap.parse_args(argv)

    runs: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for spec in args.run:
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"sin archivos para {label!r}: {pattern}")
        runs[label] = [load_traces(f) for f in files]
    ref = args.ref or next(iter(runs))
    if ref not in runs:
        raise SystemExit(f"--ref {ref!r} no está entre {list(runs)}")

    summaries = [summarize(lbl, recs) for lbl, recs in runs.items()]
    keyed = {lbl: per_key_success(recs) for lbl, recs in runs.items()}
    comps = []
    for lbl in runs:
        if lbl == ref:
            continue
        c = compare(keyed[ref], keyed[lbl])
        c["variant"] = lbl
        comps.append(c)
    if comps:
        for c, p in zip(comps, holm([c["p_signflip"] for c in comps])):
            c["p_holm"] = p

    lines = ["| Variante | Archivos | Episodios | Éxito % | IC 95% (Wilson) | Media ± DE réplicas |",
             "|---|---|---|---|---|---|"]
    for s in summaries:
        rep = f"{s['rep_mean']:.1f} ± {s['rep_std']:.1f}" if s["files"] > 1 else f"{s['rep_mean']:.1f}"
        lines.append(f"| {s['variant']} | {s['files']} | {s['episodes']} | {s['success_pct']:.1f} | "
                     f"[{s['wilson_lo']:.1f}, {s['wilson_hi']:.1f}] | {rep} |")
    if comps:
        lines += ["", f"Comparaciones pareadas contra **{ref}** (Δ = variante − referencia):", "",
                  "| Variante | Pares | Δ pp | IC 95% bootstrap | p (signo) | p Holm | McNemar (b/c, p) |",
                  "|---|---|---|---|---|---|---|"]
        for c in comps:
            mc = (f"{c['only_other']}/{c['only_ref']}, {c['p_mcnemar']:.3g}"
                  if "p_mcnemar" in c else "—")
            lines.append(f"| {c['variant']} | {c['n_pairs']} | {c['delta_pp']:+.1f} | "
                         f"[{c['ci_lo_pp']:+.1f}, {c['ci_hi_pp']:+.1f}] | {c['p_signflip']:.3g} | "
                         f"{c['p_holm']:.3g} | {mc} |")
    report = "\n".join(lines)
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        with out.with_suffix(".csv").open("w", newline="") as f:
            fields = sorted({k for c in comps for k in c} | {"variant"})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(comps)


if __name__ == "__main__":
    main()
