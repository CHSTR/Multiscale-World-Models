"""Visual de eval MultiScaleDial a partir de los artefactos guardados.

No corre modelos: lee
  - sets de eval H5 (frames de inicio/meta en las filas 0 y 25),
  - eval_meta.json (pares start/goal),
  - traces.npz de cada run (posición por paso, distancias, acciones),
  - results.jsonl (éxitos por episodio).

Uso:
  ./.venv_hf/bin/python swm_msdial/viz_eval.py --out reports/msdial_eval_viz.png
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DS = Path.home() / ".stable_worldmodel/datasets/msdial"
CK = Path.home() / ".stable_worldmodel/checkpoints"

COARSE = ("msdial_k1_balanced__lewm__s0", "msdial_eval_k1_balanced_coarse.h5")
FINE1 = ("msdial_k1_balanced__lewm__s0", "msdial_eval_k1_balanced_fine.h5")
FINE2 = ("msdial_k2_balanced__lewm__s0", "msdial_eval_k2_balanced_fine.h5")


def load_results(run):
    f = CK / run / "results.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def pick_result(run, tag):
    """results.jsonl del run: la línea cuyo dataset contiene `tag`."""
    for j in load_results(run):
        if tag in j["dataset_name"]:
            return j
    return None


def start_goal_images(name):
    with h5py.File(DS / name) as f:
        px = f["pixels"]
        return np.asarray(px[0]), np.asarray(px[25])


def traces(run):
    d = np.load(CK / run / "traces.npz")
    return {k.replace("trace_", ""): d[k] for k in d.files if k.startswith("trace_")}


def meta_pairs(name):
    meta = json.loads((DS / "eval_meta.json").read_text())
    return meta["sets"].get(name, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/msdial_eval_viz.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    # (1) pares inicio/meta de los 3 modos (k=1): qué cambia en cada modo
    for j, mode in enumerate(("coarse", "fine", "both")):
        ax = fig.add_subplot(gs[0, j])
        s, g = start_goal_images(f"msdial_eval_k1_balanced_{mode}.h5")
        ax.imshow(np.concatenate([s, np.full((8, 224, 3), 255, np.uint8), g], 0))
        ax.set_title(f"(A) k=1 {mode}: inicio (arriba) → meta (abajo)", fontsize=9)
        ax.axis("off")

    # (2) trayectorias: k1 coarse (falla posición) vs k2 fine (resuelve el dial)
    for j, (run, name, label) in enumerate((
            (*COARSE, "k=1 coarse: trayectorias (pos 2%)"),
            (*FINE2, "k=2 fine: trayectorias (dial 100%)"),
    )):
        ax = fig.add_subplot(gs[1, j])
        tr = traces(run)
        pairs = meta_pairs(name)
        for i in range(6):
            p = tr["proprio"][:, i]
            ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, alpha=0.8)
            ax.plot(p[0, 0], p[0, 1], "o", c="gray", ms=4)
            if i < len(pairs):
                g = pairs[i]["goal"]
                ax.plot(g[0], g[1], "*", c="k", ms=8)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # (3) curvas de distancia al goal por paso (k1 coarse)
    ax = fig.add_subplot(gs[1, 2])
    tr = traces(COARSE[0])
    res = pick_result(COARSE[0], "coarse")
    succ = np.array(res["successes"], bool) if res else np.zeros(tr["distance"].shape[1], bool)
    for i in range(tr["distance"].shape[1]):
        c = "tab:green" if succ[i] else "tab:red"
        ax.plot(tr["distance"][:, i], c=c, lw=0.9, alpha=0.7)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("paso del rollout")
    ax.set_ylabel("‖p − p_goal‖ (km)")
    ax.set_title(f"(C) k=1 coarse: distancia (verde=éxito, {succ.sum()}/{len(succ)})", fontsize=9)
    ax.grid(alpha=0.2)

    # (4) resumen de dv/vt por eval disponible
    ax = fig.add_subplot(gs[2, :])
    rows = []
    for run, tag, name in (
            (*COARSE, "k1 coarse"), (*FINE1, "k1 fine"), (*FINE2, "k2 fine")):
        r = pick_result(run, tag)
        if r:
            rows.append((name, np.mean(r["success_pos"]) * 100,
                         np.mean(r["success_dial"]) * 100,
                         np.mean(r["successes"]) * 100, len(r["successes"])))
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.25, [r[1] for r in rows], 0.25, label="pos", color="tab:blue")
    ax.bar(x, [r[2] for r in rows], 0.25, label="dial", color="tab:orange")
    ax.bar(x + 0.25, [r[3] for r in rows], 0.25, label="pos ∧ dial", color="tab:green")
    ax.set_xticks(x, [f"{l}\n(n={r[4]})" for l, r in zip(labels, rows)], fontsize=9)
    ax.set_ylabel("% éxito (ever)")
    ax.set_title("(D) éxito por componente — 2 epochs, seed 0", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle("MultiScaleDial — resultados visuales (2 epochs, lewm)", fontsize=13)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
