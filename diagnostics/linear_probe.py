"""Experimento 6: probes lineales del latente al estado verdadero del entorno.

Codifica frames sueltos del dataset con el encoder del checkpoint y ajusta ridge (alpha por
validación cruzada agrupada por episodio) desde dos features:
  - ``cls``: CLS del encoder (384 en DINO)
  - ``emb``: salida del projector (el latente que usa el predictor, 192)
hacia cada columna de estado (proprio, state, observation, ...). El split train/test es por EPISODIO.
Las dimensiones angulares se codifican como (sin, cos) y además se reporta el error angular medio.

Uso (desde la raíz del repo):
    python -m diagnostics.linear_probe data=pusht +policy=<run>/weights_epoch_10.pt \
        +probe.max_frames=20000 +probe.targets=[state,proprio] '+probe.angle_dims={state:[4]}'

Por defecto: targets = keys_to_cache del data config sin "action"; en PushT, state[4] (ángulo del
bloque) se trata como ángulo. Salida: ``probe_<ckpt>.csv``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.common import frame_transform, load_world_model, to_numpy
from diagnostics.probe_core import fit_probe


def _frames_to_batch(frames, tf) -> torch.Tensor:
    arr = to_numpy(frames)
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C): ToImage espera HWC en numpy
    return torch.stack([tf(f) for f in arr])


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg: DictConfig):
    import stable_pretraining  # noqa: F401
    import stable_worldmodel as swm

    probe = OmegaConf.to_container(cfg.get("probe") or {}, resolve=True)
    device = probe.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    name = cfg.data.dataset.name
    targets = probe.get("targets") or [k for k in cfg.data.dataset.keys_to_cache if k != "action"]
    angle_dims = probe.get("angle_dims")
    if angle_dims is None:
        angle_dims = {"state": [4]} if "pusht" in name else {}
    max_frames = int(probe.get("max_frames", 20000))
    test_frac = float(probe.get("test_frac", 0.2))
    bs = int(probe.get("batch_size", 128))

    model = load_world_model(cfg.policy, device)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(name, cache_dir=cache_dir, keys_to_cache=list(targets))
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_all = np.asarray(dataset.get_col_data(ep_col))
    Y_all = {t: np.asarray(dataset.get_col_data(t), dtype=np.float64) for t in targets}
    valid = np.ones(ep_all.shape[0], dtype=bool)
    for t in targets:
        Yt = Y_all[t].reshape(Y_all[t].shape[0], -1)
        valid &= ~np.isnan(Yt).any(axis=1)
    rng = np.random.default_rng(cfg.seed)
    idx = np.flatnonzero(valid)
    if idx.size > max_frames:
        idx = np.sort(rng.choice(idx, size=max_frames, replace=False))

    tf = frame_transform(cfg.img_size)
    feats = {"cls": [], "emb": []}
    with torch.no_grad():
        for s in range(0, idx.size, bs):
            rows = dataset.get_row_data(idx[s: s + bs])
            x = _frames_to_batch(rows["pixels"], tf).to(device)
            out = model.encoder(x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            feats["cls"].append(cls.float().cpu().numpy())
            feats["emb"].append(model.projector(cls).float().cpu().numpy())
            print(f"codificados {min(s + bs, idx.size)}/{idx.size}", end="\r")
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    ep = ep_all[idx]

    rows_out = []
    for t in targets:
        Y = Y_all[t][idx].reshape(idx.size, -1)
        for fname, X in feats.items():
            res = fit_probe(X, Y, ep, test_frac=test_frac, angle_dims=angle_dims.get(t, []),
                            names=[f"{t}[{i}]" for i in range(Y.shape[1])], seed=cfg.seed)
            for dim_name, r2te, r2tr in zip(res["names"], res["r2_test"], res["r2_train"]):
                rows_out.append({"target": t, "dim": dim_name, "feature": fname,
                                 "r2_test": float(r2te), "r2_train": float(r2tr),
                                 "alpha": res["alpha"], "n_train": res["n_train"], "n_test": res["n_test"]})
            for ang, mae in res["angle_mae_deg"].items():
                rows_out.append({"target": t, "dim": f"{ang}_mae_deg", "feature": fname,
                                 "r2_test": float("nan"), "r2_train": float("nan"), "alpha": res["alpha"],
                                 "n_train": res["n_train"], "n_test": res["n_test"], "angle_mae_deg": mae})

    out_dir = Path(probe.get("out_dir") or Path(swm.data.utils.get_cache_dir(), str(cfg.policy)).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"probe_{Path(str(cfg.policy)).stem}.csv"
    fields = ["target", "dim", "feature", "r2_test", "r2_train", "alpha", "n_train", "n_test", "angle_mae_deg"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"\n{'dimensión':>22} {'feature':>7} {'R² test':>8} {'R² train':>9} {'MAE°':>7}")
    for r in rows_out:
        mae = r.get("angle_mae_deg")
        print(f"{r['dim']:>22} {r['feature']:>7} {r['r2_test']:>8.3f} {r['r2_train']:>9.3f} "
              f"{(f'{mae:.1f}' if mae is not None else ''):>7}")
    print(f"-> {out_csv}")


if __name__ == "__main__":
    run()
