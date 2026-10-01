"""Experimento 8: error de predicción latente vs horizonte (rollout open-loop con acciones reales).

Sobre el split de validación de train.py, con secuencias de ``history_size + horizon`` frames:
codifica todos los frames (referencia), parte de los ``history_size`` primeros y predice
autorregresivamente con las acciones reales, igual que ``JEPA.rollout``. Reporta por paso k:
MSE, MSE normalizado por la varianza del latente y 1 - cos.

Uso:
    python -m diagnostics.rollout_error data=pusht +policy=<run>/weights_epoch_10.pt seed=<seed> \
        data.dataset.num_steps=13 +rollout.horizon=10 +rollout.max_batches=20 loader.batch_size=32

(``data.dataset.num_steps`` debe ser history_size + horizon.)
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from diagnostics.common import build_val_loader, encode_pixels, load_world_model


@torch.no_grad()
def rollout_errors(model, batch, history: int, horizon: int):
    pixels = batch["pixels"]
    action = torch.nan_to_num(batch["action"], 0.0)
    assert pixels.size(1) >= history + horizon, "secuencias demasiado cortas: sube data.dataset.num_steps"
    z = encode_pixels(model, pixels[:, : history + horizon])
    emb = z[:, :history]
    act = action[:, :history]
    mse, cos = [], []
    for k in range(horizon):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -history:], act_emb[:, -history:])[:, -1:]
        true = z[:, history + k: history + k + 1]
        mse.append((pred - true).pow(2).mean(dim=(1, 2)))
        cos.append(1 - F.cosine_similarity(pred, true, dim=-1).mean(dim=1))
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, action[:, history + k: history + k + 1]], dim=1)
    var = z.var(dim=(0, 1)).mean()
    return torch.stack(mse, 1).cpu(), torch.stack(cos, 1).cpu(), float(var)


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg: DictConfig):
    import stable_pretraining  # noqa: F401
    import stable_worldmodel as swm

    rcfg = cfg.get("rollout") or {}
    horizon = int(rcfg.get("horizon", 10))
    max_batches = int(rcfg.get("max_batches", 20))
    device = rcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = load_world_model(cfg.policy, device)
    loader = build_val_loader(cfg)

    mses, coss, vars_ = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        m, c, v = rollout_errors(model, batch, cfg.history_size, horizon)
        mses.append(m)
        coss.append(c)
        vars_.append(v)
    mse = torch.cat(mses).numpy()
    cos = torch.cat(coss).numpy()
    var = float(np.mean(vars_))

    out_dir = Path(rcfg.get("out_dir") or Path(swm.data.utils.get_cache_dir(), str(cfg.policy)).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rollout_error_{Path(str(cfg.policy)).stem}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mse", "mse_se", "nmse", "one_minus_cos", "n"])
        for k in range(horizon):
            w.writerow([k + 1, mse[:, k].mean(), mse[:, k].std(ddof=1) / np.sqrt(len(mse)),
                        mse[:, k].mean() / var, cos[:, k].mean(), len(mse)])
    print(f"{'paso':>4} {'MSE':>9} {'nMSE':>7} {'1-cos':>7}")
    for k in range(horizon):
        print(f"{k + 1:>4} {mse[:, k].mean():>9.4f} {mse[:, k].mean() / var:>7.3f} {cos[:, k].mean():>7.4f}")
    print(f"-> {out_csv}")


if __name__ == "__main__":
    run()
