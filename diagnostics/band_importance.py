"""Experimento 4 (offline): importancia de cada banda del front-end para la pérdida de predicción.

Para cada banda (y el grupo "details"), y para cada modo de ablación, mide sobre el split de
validación de entrenamiento (misma semilla => mismo split que train.py):
  - Δ pérdida de predicción (contexto ablacionado, target limpio), media ± error estándar
  - deriva del embedding del contexto (1 - cos)
  - Integrated Gradients sobre un multiplicador por banda (atribución que suma L(todo) - L(nada))

Uso (desde la raíz del repo):
    python -m diagnostics.band_importance data=ogb +policy=<run>/weights_epoch_10.pt \
        loader.batch_size=16 +diag.max_batches=30 +diag.modes=[zero,noise] +diag.ig_steps=16

Salida: CSV ``band_importance_<ckpt>.csv`` junto a los resultados de evaluación del checkpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from diagnostics.common import band_importance_batch, build_val_loader, load_world_model


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg: DictConfig):
    import stable_pretraining  # noqa: F401  (registra resolvers usados por los configs)
    import stable_worldmodel as swm

    diag = cfg.get("diag") or {}
    device = diag.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    modes = list(diag.get("modes", ["zero", "noise"]))
    max_batches = int(diag.get("max_batches", 30))
    ig_steps = int(diag.get("ig_steps", 16))
    policy = cfg.policy

    model = load_world_model(policy, device)
    loader = build_val_loader(cfg)
    names = None
    acc = {}
    ig_sum, n_ig, loss_all, loss_none = None, 0, [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = band_importance_batch(model, batch, cfg.history_size, cfg.num_preds, modes=modes,
                                    ig_steps=ig_steps, seed=cfg.seed + i)
        names = out["names"]
        for key, v in out["ablations"].items():
            d = acc.setdefault(key, {"delta": [], "drift": []})
            d["delta"].append(v["delta"])
            d["drift"].append(v["drift"])
        ig_sum = out["ig"] if ig_sum is None else ig_sum + out["ig"]
        n_ig += 1
        loss_all.append(float(out["loss_all"]))
        loss_none.append(float(out["loss_none"]))
        print(f"[{i + 1}/{max_batches}] base={out['loss_all']:.4f}")

    base = float(np.mean(loss_all))
    ig = (ig_sum / max(n_ig, 1)).numpy()
    rows = []
    for (gname, mode), v in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        delta = torch.cat(v["delta"]).numpy()
        drift = torch.cat(v["drift"]).numpy()
        idx = names.index(gname) if gname in names else None
        rows.append({
            "band": gname,
            "mode": mode,
            "delta_loss": float(delta.mean()),
            "delta_loss_se": float(delta.std(ddof=1) / np.sqrt(delta.size)),
            "delta_rel_pct": 100 * float(delta.mean()) / base,
            "drift_1mcos": float(drift.mean()),
            "ig": float(ig[idx]) if idx is not None else float("nan"),
            "ig_frac_pct": 100 * float(ig[idx]) / float(ig.sum()) if idx is not None and ig.sum() != 0 else float("nan"),
            "n": int(delta.size),
        })

    out_dir = Path(diag.get("out_dir") or Path(swm.data.utils.get_cache_dir(), str(policy)).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"band_importance_{Path(str(policy)).stem}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "policy": str(policy), "base_loss": base, "loss_no_bands": float(np.mean(loss_none)),
        "ig_sum": float(ig.sum()), "completeness_gap": float(np.mean(loss_all) - np.mean(loss_none)),
    }
    (out_dir / f"band_importance_{Path(str(policy)).stem}.json").write_text(json.dumps(summary, indent=2))

    print(f"\npérdida base={base:.4f}  sin bandas={summary['loss_no_bands']:.4f}  "
          f"Σ IG={summary['ig_sum']:.4f} (debe ≈ {summary['completeness_gap']:.4f}; si no, sube +diag.ig_steps)")
    print(f"{'banda':>8} {'modo':>6} {'Δpérdida':>10} {'±se':>8} {'Δ%':>7} {'1-cos':>7} {'IG%':>6}")
    for r in rows:
        print(f"{r['band']:>8} {r['mode']:>6} {r['delta_loss']:>10.4f} {r['delta_loss_se']:>8.4f} "
              f"{r['delta_rel_pct']:>7.1f} {r['drift_1mcos']:>7.4f} {r['ig_frac_pct']:>6.1f}")
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    run()
