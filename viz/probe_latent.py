"""Probe lineal: ¿el latente conserva la posición del agente y del bloque?

Para cada checkpoint se codifican los mismos frames, se ajusta una regresión
ridge latente -> ``state`` (o la columna que se pida) y se reporta el R² en
episodios no vistos. Sirve para detectar el colapso "el encoder descarta lo que
se mueve": si el R² del agente/bloque cae al agregar gate o NCA, el modelo está
tirando justo la información dinámica.

Features por checkpoint:
  * ``grid``: todos los tokens aplanados (N*D) en modelos espaciales, o el
    vector ``(D,)`` en el JEPA global (CLS).
  * ``mean``: promedio de los tokens (sólo modelos espaciales). Es una cota
    inferior: la posición suele estar en *qué* token se activa, no en el promedio.

Split por episodios (``ep_offset``/``ep_len`` del h5) para que train/val/test no
compartan trayectorias. ``λ`` se elige en val y el R² se reporta en test.

Uso (comparar variantes con los mismos frames):

    python viz/probe_latent.py \\
        --ckpt dataset/checkpoints/lewm_dinov3/weights_epoch_10.pt \\
        --ckpt dataset/checkpoints/nca/weights_epoch_10.pt \\
        --ckpt dataset/checkpoints/nca_gate_idm/weights_epoch_10.pt \\
        --target state --angle-dims 4 --n-samples 4000

En Push-T ``state`` suele ser ``[agente_x, agente_y, bloque_x, bloque_y,
bloque_ángulo]``; verifica el orden en tu h5. ``--angle-dims`` reemplaza esas
columnas por (sin, cos) para que el ángulo sea lineal-predecible.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


# --------------------------------------------------------------------------- #
#  Ridge (sin dependencias de swm: testeable aislado)
# --------------------------------------------------------------------------- #


def _standardize(train, *others):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp_min(1e-6)
    return [(x - mu) / sd for x in (train, *others)]


def _ridge_fit_predict(Xtr, Ytr, Xev, lam):
    """Ridge sin intercepto sobre datos estandarizados. Usa la forma dual si d > n."""
    n, d = Xtr.shape
    if d > n:
        K = Xtr @ Xtr.T
        alpha = torch.linalg.solve(K + lam * torch.eye(n, device=K.device, dtype=K.dtype), Ytr)
        return (Xev @ Xtr.T) @ alpha
    A = Xtr.T @ Xtr + lam * torch.eye(d, device=Xtr.device, dtype=Xtr.dtype)
    W = torch.linalg.solve(A, Xtr.T @ Ytr)
    return Xev @ W


def r2_per_dim(y, yhat):
    ss_res = (y - yhat).pow(2).sum(0)
    ss_tot = (y - y.mean(0, keepdim=True)).pow(2).sum(0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def ridge_probe(X, Y, split, lambdas=LAMBDAS):
    """X: (n, d), Y: (n, k), split: (n,) con 0=train, 1=val, 2=test.

    Devuelve ``{"lambda", "r2_val", "r2_test" (lista por dim), "r2_test_mean"}``.
    """
    X, Y = X.double(), Y.double()
    tr, va, te = (split == 0), (split == 1), (split == 2)
    Xtr, Xva, Xte = _standardize(X[tr], X[va], X[te])
    Ytr, Yva, Yte = _standardize(Y[tr], Y[va], Y[te])
    best = None
    for lam in lambdas:
        r2v = r2_per_dim(Yva, _ridge_fit_predict(Xtr, Ytr, Xva, lam)).mean().item()
        if best is None or r2v > best[1]:
            best = (lam, r2v)
    lam = best[0]
    r2t = r2_per_dim(Yte, _ridge_fit_predict(Xtr, Ytr, Xte, lam))
    return {
        "lambda": lam,
        "r2_val": best[1],
        "r2_test": [round(v, 4) for v in r2t.tolist()],
        "r2_test_mean": round(r2t.mean().item(), 4),
    }


def expand_angles(Y, angle_dims):
    """Reemplaza las columnas en ``angle_dims`` por (sin, cos). Devuelve (Y, nombres)."""
    cols, names = [], []
    for j in range(Y.size(1)):
        if j in angle_dims:
            cols += [torch.sin(Y[:, j]), torch.cos(Y[:, j])]
            names += [f"{j}_sin", f"{j}_cos"]
        else:
            cols.append(Y[:, j])
            names.append(str(j))
    return torch.stack(cols, 1), names


def episode_split(ep_off, ep_len, n_samples, seed, fracs=(0.7, 0.15, 0.15), min_len=1):
    """Elige ``n_samples`` filas, separando train/val/test por episodio."""
    rng = np.random.default_rng(seed)
    eps = np.flatnonzero(np.asarray(ep_len) >= min_len)
    rng.shuffle(eps)
    n_tr = int(len(eps) * fracs[0])
    n_va = int(len(eps) * fracs[1])
    groups = [eps[:n_tr], eps[n_tr:n_tr + n_va], eps[n_tr + n_va:]]
    rows, split = [], []
    for s, (g, f) in enumerate(zip(groups, fracs)):
        if len(g) == 0:
            raise ValueError("muy pocos episodios para el split train/val/test")
        cand = np.concatenate([np.arange(ep_off[e], ep_off[e] + ep_len[e]) for e in g])
        k = min(len(cand), max(1, int(round(n_samples * f))))
        pick = np.sort(rng.choice(cand, size=k, replace=False))
        rows.append(pick)
        split.append(np.full(k, s))
    return np.concatenate(rows), np.concatenate(split)


# --------------------------------------------------------------------------- #
#  Datos y modelo
# --------------------------------------------------------------------------- #


def to_chw(x):
    a = torch.as_tensor(x).float()
    if a.ndim == 3 and a.shape[-1] in (1, 3) and a.shape[0] not in (1, 3):
        a = a.permute(2, 0, 1)
    return a


def load_dataset(cfg, name, targets, cache_dir):
    import stable_worldmodel as swm
    import stable_worldmodel.data.formats.hdf5  # noqa: F401
    from omegaconf import OmegaConf

    from utils import get_img_preprocessor

    dcfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    default_name = dcfg.pop("name")
    dcfg.pop("fraction", None)
    dcfg.pop("keys_to_merge", None)
    dcfg["num_steps"] = 1
    dcfg["keys_to_load"] = ["pixels", *targets]
    dcfg["keys_to_cache"] = list(targets)
    ds = swm.data.load_dataset(name or default_name, transform=None, cache_dir=cache_dir, **dcfg)
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    return ds


def episode_bounds(ds):
    try:
        return np.asarray(ds.get_col_data("ep_offset")), np.asarray(ds.get_col_data("ep_len"))
    except Exception:
        print("[probe] AVISO: sin ep_offset/ep_len; uso bloques contiguos de 100 filas como 'episodios'")
        n = len(ds)
        off = np.arange(0, n, 100)
        return off, np.minimum(100, n - off)


@torch.no_grad()
def encode_rows(model, ds, rows, batch_size, device):
    """Devuelve dict con ``grid`` (n, N*D o D) y, si es espacial, ``mean`` (n, D)."""
    grid, mean = [], []
    for i in range(0, len(rows), batch_size):
        pix = torch.stack([to_chw(ds.get_row_data(int(r))["pixels"]) for r in rows[i:i + batch_size]])
        emb = model.encode({"pixels": pix.unsqueeze(1).to(device)})["emb"][:, 0].float()
        if emb.ndim == 3:  # (b, N, D) espacial
            mean.append(emb.mean(1).cpu())
            grid.append(emb.reshape(emb.size(0), -1).cpu())
        else:
            grid.append(emb.cpu())
    out = {"grid": torch.cat(grid)}
    if mean:
        out["mean"] = torch.cat(mean)
    return out


def load_targets(ds, rows, targets):
    ys = []
    for t in targets:
        ys.append(torch.stack([torch.as_tensor(ds.get_row_data(int(r))[t]).float().reshape(-1)
                               for r in rows]))
    return torch.cat(ys, 1)


def load_model(ckpt, device):
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def main():
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True, help="repetible: un checkpoint por variante")
    p.add_argument("--config", action="append", default=None,
                   help="config.yaml de cada run (default: junto a cada ckpt)")
    p.add_argument("--data", default=None, help="nombre del dataset (default: el del config)")
    p.add_argument("--target", action="append", default=None, help="columna(s) a predecir (default: state)")
    p.add_argument("--angle-dims", type=int, nargs="*", default=[], help="columnas angulares -> (sin, cos)")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON de salida (default: probe_results.json junto al 1er ckpt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    targets = args.target or ["state"]
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    configs = args.config or [str(Path(c).parent / "config.yaml") for c in args.ckpt]
    assert len(configs) == len(args.ckpt), "pasa un --config por cada --ckpt (o ninguno)"

    rows = split = Y = names = None
    results = []
    for ckpt, cfg_path in zip(args.ckpt, configs):
        cfg = OmegaConf.load(cfg_path)
        ds = load_dataset(cfg, args.data, targets, cache_dir)
        if rows is None:  # mismos frames para todas las variantes
            ep_off, ep_len = episode_bounds(ds)
            rows, split = episode_split(ep_off, ep_len, args.n_samples, args.seed)
            Y, names = expand_angles(load_targets(ds, rows, targets), set(args.angle_dims))
            print(f"[probe] {len(rows)} frames (train/val/test = "
                  f"{(split == 0).sum()}/{(split == 1).sum()}/{(split == 2).sum()}), targets {names}")

        model = load_model(ckpt, args.device)
        feats = encode_rows(model, ds, rows, args.batch_size, args.device)
        split_t = torch.as_tensor(split)
        for kind, X in feats.items():
            r = ridge_probe(X, Y, split_t)
            r.update({"ckpt": ckpt, "features": kind, "dim": X.size(1), "targets": names})
            results.append(r)
            print(f"[probe] {Path(ckpt).parent.name}/{Path(ckpt).name} [{kind}, d={X.size(1)}] "
                  f"R² test = {r['r2_test_mean']:.3f}  por dim {dict(zip(names, r['r2_test']))}  λ={r['lambda']:g}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = Path(args.out) if args.out else Path(args.ckpt[0]).parent / "probe_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[probe] resultados -> {out}")


if __name__ == "__main__":
    main()
