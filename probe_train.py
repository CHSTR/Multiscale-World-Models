"""Probing lineal (ridge) y no-lineal (MLP) sobre latentes congelados de Cube.

Protocolo (ver docs/superpowers/specs/2026-10-29-probe-dinov3-cube-design.md):
  - 200k (configurable) frames muestreados uniformes.
  - Split 80/20 POR EPISODIO (sin leakage).
  - Features: CLS (384d) y patch-mean (384d, media de los ultimos n_patches tokens).
  - Probes: ridge cerrada (alpha elegido en validacion) y MLP 2 capas.
  - Metricas por grupo fisico: MSE (z-units y unidades originales) y Pearson r.

Uso:
  python probe_train.py \
      --ckpt <ruta/weights_epoch_5.pt> [--ckpt ...] \
      --source cls --source patch_mean \
      --n-frames 50000 --out-dir outputs/probe
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
#  Definicion de grupos fisicos
# --------------------------------------------------------------------------- #

GROUP_SPECS: dict[str, list[str]] = {
    "joint_pos": ["proprio_joint_pos"],
    "joint_vel": ["proprio_joint_vel"],
    "effector_pos": ["proprio_effector_pos"],
    "effector_yaw": ["proprio_effector_yaw"],
    "gripper": ["proprio_gripper_opening", "proprio_gripper_contact", "proprio_gripper_vel"],
    "block_pos": ["privileged_block_0_pos"],
    "block_quat": ["privileged_block_0_quat"],
    "block_yaw": ["privileged_block_0_yaw"],
}

EXTRA_DIAG_SPECS: dict[str, list[str]] = {
    "gripper_contact": ["proprio_gripper_contact"],
}


def group_targets_for_specs(
    targets_by_col: dict[str, np.ndarray],
    specs: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Concatena las columnas de cada grupo en `specs` respetando sus dimensiones."""
    return {
        group: np.concatenate([targets_by_col[c] for c in cols], axis=1)
        for group, cols in specs.items()
    }


def all_target_columns() -> list[str]:
    cols: list[str] = []
    for c in GROUP_SPECS.values():
        cols.extend(c)
    return cols


# --------------------------------------------------------------------------- #
#  Dataset / split
# --------------------------------------------------------------------------- #


def load_cube_dataset(cache_dir: str | None = None, img_size: int = 224):
    import stable_worldmodel as swm

    from utils import get_img_preprocessor

    cache = cache_dir or os.environ.get("LOCAL_DATASET_DIR") or os.path.expanduser(
        "~/.stable_worldmodel"
    )
    dataset = swm.data.load_dataset(
        "ogbench/cube_single_expert.h5",
        transform=None,
        cache_dir=cache,
        keys_to_load=["pixels", *all_target_columns()],
        num_steps=1,
        frameskip=1,
    )
    dataset.transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=img_size
    )
    return dataset


def sample_and_split(dataset, n_frames: int, val_frac: float = 0.2, seed: int = 0):
    """Muestrea n_frames filas y las parte por episodio (80/20)."""
    n = min(int(n_frames), len(dataset))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)

    ep = np.asarray(dataset.get_col_data("ep_idx"))[idx]
    episodes = np.unique(ep)
    perm = rng.permutation(episodes)
    n_val = max(1, int(round(val_frac * len(episodes))))
    val_eps = set(perm[:n_val].tolist())

    is_val = np.array([int(e) in val_eps for e in ep])
    return idx[~is_val].tolist(), idx[is_val].tolist()


def group_targets(targets_by_col: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatena las columnas de cada grupo respetando sus dimensiones reales."""
    return group_targets_for_specs(targets_by_col, GROUP_SPECS)


def drop_nan_rows(targets: dict[str, np.ndarray]) -> np.ndarray:
    """Mascara True para filas sin NaN en ninguna columna de ningun grupo."""
    n = len(next(iter(targets.values())))
    mask = np.ones(n, dtype=bool)
    for v in targets.values():
        v = np.asarray(v)
        mask &= ~np.isnan(v).any(axis=1)
    return mask


# --------------------------------------------------------------------------- #
#  Extraccion de features
# --------------------------------------------------------------------------- #


def _patch_size(model) -> int | None:
    enc = model.encoder
    for obj in (enc, getattr(enc, "model", None), getattr(enc, "vit", None)):
        if obj is None:
            continue
        p = getattr(obj, "patch_size", None)
        if p:
            return int(p)
        cfg = getattr(obj, "config", None)
        p = getattr(cfg, "patch_size", None) if cfg is not None else None
        if p:
            return int(p)
    return None


def _n_patches(model, seq_len: int) -> int:
    """Cantidad de tokens de parche (los ULTIMOS de la secuencia)."""
    enc = model.encoder
    n = getattr(enc, "n_patches", None)  # stub de tests
    if n:
        return int(n)
    img = int(getattr(enc, "img_size", 224) or 224)
    p = _patch_size(model)
    if p:
        return (img // p) ** 2
    n_store = getattr(getattr(enc, "model", None), "n_storage_tokens", 0) or 0
    n_dyn = getattr(enc, "n_dyn_tokens", 0) or 0
    return seq_len - 1 - n_store - n_dyn


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encoder_forward(encoder, pixels):
    try:
        return encoder(pixels, interpolate_pos_encoding=True)
    except TypeError:
        return encoder(pixels)


@torch.no_grad()
def extract_features(
    model,
    dataset,
    indices,
    source: str = "cls",
    batch_size: int = 256,
    target_columns: list[str] | None = None,
    transform=None,
):
    """Devuelve features (N,D) y targets crudos por columna."""
    cols = target_columns if target_columns is not None else all_target_columns()
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = _model_device(model)
    model.eval()

    feats: list[np.ndarray] = []
    tgts: dict[str, list[np.ndarray]] = {c: [] for c in cols}

    for batch in loader:
        pixels = batch["pixels"].float()
        pixels_4d = pixels.reshape(-1, *pixels.shape[-3:]).to(device)
        if source == "emb":
            emb = model.encode({"pixels": pixels_4d.unsqueeze(1)})["emb"]
            if isinstance(emb, dict):
                f = torch.cat(
                    [
                        (
                            v.mean(dim=2)[:, 0]
                            if v.ndim == 4
                            else (v[:, 0] if v.ndim == 3 else v)
                        )
                        for v in (emb[k] for k in sorted(emb.keys()))
                    ],
                    dim=-1,
                )
            else:
                if emb.ndim == 4:
                    emb = emb.mean(dim=2)
                f = emb[:, 0] if emb.ndim == 3 else emb
        else:
            out = _encoder_forward(model.encoder, pixels_4d)
            tokens = getattr(out, "last_hidden_state", out)
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
            if not torch.is_tensor(tokens):
                raise ValueError(
                    f"source={source!r} requires an encoder with last_hidden_state; "
                    f"use --source emb for {type(model).__name__} ({type(out).__name__})."
                )

            if source == "cls":
                f = tokens[:, 0]
            elif source in ("patch_mean", "patchmean"):
                f = tokens[:, -_n_patches(model, tokens.size(1)) :].mean(dim=1)
            else:
                raise ValueError(f"source {source!r} no soportado (cls|patch_mean|emb)")

        feats.append(f.float().cpu().numpy())
        for c in cols:
            v = batch[c].float()
            v = v.reshape(v.size(0), -1)
            tgts[c].append(v.numpy())

    return {
        "features": np.concatenate(feats).astype(np.float32),
        "targets": {c: np.concatenate(v).astype(np.float32) for c, v in tgts.items()},
    }


# --------------------------------------------------------------------------- #
#  Metricas
# --------------------------------------------------------------------------- #


def mse_and_pearson(y_true, y_pred) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    mse = float(np.mean((y_true - y_pred) ** 2))
    rs: list[float] = []
    for j in range(y_true.shape[1]):
        a, b = y_true[:, j], y_pred[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs.append(0.0)
            continue
        r = float(np.corrcoef(np.stack([a, b]))[0, 1])
        rs.append(0.0 if not np.isfinite(r) else r)
    return mse, float(np.mean(rs)) if rs else 0.0


def standardize(train, other):
    train = np.asarray(train, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd, sd.squeeze(0)


# --------------------------------------------------------------------------- #
#  Probes
# --------------------------------------------------------------------------- #


def fit_ridge(
    X_tr,
    Y_tr,
    X_va,
    Y_va,
    alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
):
    """Ridge cerrada sobre X estandarizada; alpha elegido por Pearson en val.

    Devuelve (W, b, alpha) con prediccion ``Z_va @ W + b`` en unidades RAW de Y.
    """
    Ztr, Zva, _ = standardize(X_tr, X_va)
    Y_tr = np.asarray(Y_tr, dtype=np.float64)
    Y_va = np.asarray(Y_va, dtype=np.float64)
    if Y_tr.ndim == 1:
        Y_tr = Y_tr[:, None]
        Y_va = Y_va[:, None]

    # centrar Y solo para el intercepto (equivale a b = mean(Y_train))
    y_mu = Y_tr.mean(0, keepdims=True)
    Yc = Y_tr - y_mu

    eye = np.eye(Ztr.shape[1])
    best_r, best_alpha, best_W = -np.inf, alphas[0], None
    for alpha in alphas:
        W = np.linalg.solve(Ztr.T @ Ztr + alpha * eye, Ztr.T @ Yc)
        _, r = mse_and_pearson(Y_va, Zva @ W + y_mu)
        if r > best_r:
            best_r, best_alpha, best_W = r, alpha, W
    return best_W, y_mu.squeeze(0), float(best_alpha)


def _to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class MLPProbe(nn.Module):
    """MLP sobre features estandarizadas que devuelve predicciones en unidades RAW."""

    def __init__(self, net, x_mu, x_sd, y_mu, y_sd):
        super().__init__()
        self.net = net
        self.register_buffer("x_mu", torch.as_tensor(x_mu, dtype=torch.float32))
        self.register_buffer("x_sd", torch.as_tensor(x_sd, dtype=torch.float32))
        self.register_buffer("y_mu", torch.as_tensor(y_mu, dtype=torch.float32))
        self.register_buffer("y_sd", torch.as_tensor(y_sd, dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu


def fit_mlp(
    X_tr,
    Y_tr,
    X_va,
    Y_va,
    hidden: int = 256,
    epochs: int = 60,
    patience: int = 8,
    batch_size: int = 4096,
    seed: int = 0,
):
    """MLP 2 capas con early stopping. Devuelve un MLPProbe en unidades RAW."""
    Ztr, Zva, x_sd = standardize(X_tr, X_va)
    x_mu = np.asarray(X_tr, dtype=np.float64).mean(0)
    Ttr, Tva, y_sd = standardize(Y_tr, Y_va)
    y_mu = np.asarray(Y_tr, dtype=np.float64).mean(0)
    if Ttr.ndim == 1:
        Ttr = Ttr[:, None]
        Tva = Tva[:, None]
        y_mu = np.atleast_1d(y_mu)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xt, Yt = _to_tensor(Ztr, device), _to_tensor(Ttr, device)
    Xv, Yv = _to_tensor(Zva, device), _to_tensor(Tva, device)

    net = nn.Sequential(
        nn.Linear(Xt.shape[1], hidden),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, Yt.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = Xt.shape[0]
    best_state, best_val, bad = None, np.inf, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            j = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(Xt[j]), Yt[j])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xv), Yv).item()
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    probe = MLPProbe(net, x_mu, x_sd, y_mu, y_sd).to(device)
    probe.eval()
    return probe


@torch.no_grad()
def predict_mlp(model, X) -> np.ndarray:
    device = _model_device(model)
    model.eval()
    preds = []
    for i in range(0, len(X), 8192):
        preds.append(model(_to_tensor(X[i : i + 8192], device)).cpu().numpy())
    return np.concatenate(preds)


# --------------------------------------------------------------------------- #
#  Filas de resultado
# --------------------------------------------------------------------------- #


def make_rows(
    checkpoint: str,
    source: str,
    probe: str,
    y_true,
    y_pred,
    group_specs: dict[str, list[str]],
    var_raw=None,
) -> list[dict]:
    """Una fila por grupo. Acepta arrays completos (se cortan por grupo) o dicts."""
    rows = []
    offset = 0
    for group, cols in group_specs.items():
        if isinstance(y_true, dict):
            yt, yp = y_true[group], y_pred[group]
            var = None if var_raw is None else var_raw.get(group)
            dim = int(np.asarray(yt).shape[1])
        else:
            dim = len(cols)
            yt = np.asarray(y_true)[:, offset : offset + dim]
            yp = np.asarray(y_pred)[:, offset : offset + dim]
            var = None if var_raw is None else np.asarray(var_raw)[offset : offset + dim]
            offset += dim

        sq = np.mean((yt - yp) ** 2, axis=0)
        mse_raw = float(np.mean(sq))
        if var is None:
            mse_z = mse_raw
        else:
            v_arr = np.asarray(var, dtype=np.float64)
            v_arr = np.where(v_arr < 1e-12, 1.0, v_arr)
            mse_z = float(np.mean(sq / v_arr))
        _, r = mse_and_pearson(yt, yp)

        rows.append(
            {
                "checkpoint": Path(checkpoint).parent.name + "/" + Path(checkpoint).name,
                "source": source,
                "probe": probe,
                "group": group,
                "dim": dim,
                "mse_z": mse_z,
                "mse_raw": mse_raw,
                "pearson": r,
                "run": int(time.time()),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
#  Run completo
# --------------------------------------------------------------------------- #


def _cache_path(out_dir: Path, ckpt: str, n_frames: int, seed: int) -> Path:
    stem = Path(ckpt).parent.name + "__" + Path(ckpt).stem
    return out_dir / f"features_{stem}_n{n_frames}_s{seed}.npz"


def _load_or_extract(
    model, dataset, ckpt: str, out_dir: Path, n_frames: int, val_frac: float,
    seed: int, sources: list[str], force: bool,
):
    cache = _cache_path(out_dir, ckpt, n_frames, seed)
    cols = all_target_columns()
    if cache.exists() and not force:
        with np.load(cache) as data:
            tr = data["train_idx"]
            va = data["val_idx"]
            cached_sources = {
                k[:-6]
                for k in data.files
                if k.endswith("_train") and f"{k[:-6]}_val" in data.files
            }
            all_cached = {
                s: (data[f"{s}_train"], data[f"{s}_val"])
                for s in cached_sources
            }
            tgt_tr = {c: data[f"tgt_train_{c}"] for c in cols}
            tgt_va = {c: data[f"tgt_val_{c}"] for c in cols}

        missing = [s for s in sources if s not in all_cached]
        if missing:
            tr_list = tr.tolist()
            va_list = va.tolist()
            for s in missing:
                a = extract_features(model, dataset, tr_list, source=s)
                b = extract_features(model, dataset, va_list, source=s)
                all_cached[s] = (a["features"], b["features"])
            np.savez_compressed(
                cache,
                train_idx=np.asarray(tr),
                val_idx=np.asarray(va),
                **{f"{s}_train": pair[0] for s, pair in all_cached.items()},
                **{f"{s}_val": pair[1] for s, pair in all_cached.items()},
                **{f"tgt_train_{c}": tgt_tr[c] for c in tgt_tr},
                **{f"tgt_val_{c}": tgt_va[c] for c in tgt_va},
            )
            print(f"[probe] cache actualizado ({missing}) -> {cache}")
        feats = {s: all_cached[s] for s in sources}
        return tr, va, feats, tgt_tr, tgt_va

    tr, va = sample_and_split(dataset, n_frames, val_frac, seed)
    feats, tgt_tr, tgt_va = {}, {}, {}
    for s in sources:
        a = extract_features(model, dataset, tr, source=s)
        b = extract_features(model, dataset, va, source=s)
        feats[s] = (a["features"], b["features"])
        if not tgt_tr:
            tgt_tr, tgt_va = a["targets"], b["targets"]

    np.savez_compressed(
        cache,
        train_idx=np.asarray(tr),
        val_idx=np.asarray(va),
        **{f"{s}_train": feats[s][0] for s in sources},
        **{f"{s}_val": feats[s][1] for s in sources},
        **{f"tgt_train_{c}": tgt_tr[c] for c in tgt_tr},
        **{f"tgt_val_{c}": tgt_va[c] for c in tgt_va},
    )
    print(f"[probe] cache -> {cache}")
    return tr, va, feats, tgt_tr, tgt_va


def evaluate_group_probes(
    ckpt: str,
    source: str,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Ytr_s: dict[str, np.ndarray],
    Yva_s: dict[str, np.ndarray],
    specs: dict[str, list[str]],
    var: dict[str, np.ndarray],
    seed: int = 0,
) -> list[dict]:
    _, Zva, _ = standardize(Xtr, Xva)
    rows: list[dict] = []

    pred_va: dict[str, np.ndarray] = {}
    for group in specs:
        W, b, _ = fit_ridge(Xtr, Ytr_s[group], Xva, Yva_s[group])
        pred_va[group] = Zva @ W + b
    rows += make_rows(ckpt, source, "ridge", Yva_s, pred_va, specs, var)

    pred_va = {}
    for group in specs:
        m = fit_mlp(Xtr, Ytr_s[group], Xva, Yva_s[group], seed=seed)
        pred_va[group] = predict_mlp(m, Xva)
    rows += make_rows(ckpt, source, "mlp", Yva_s, pred_va, specs, var)
    return rows


def run_probe(
    ckpt: str,
    sources: list[str],
    n_frames: int = 50_000,
    val_frac: float = 0.2,
    out_dir: str | Path = "outputs/probe",
    cache_dir: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> list[dict]:
    import stable_worldmodel as swm

    from utils import get_img_preprocessor  # noqa: F401  (documental)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(ckpt).expanduser()
    ckpt_arg = str(p.resolve()) if p.exists() else ckpt

    print(f"[probe] cargando {ckpt_arg}")
    model = swm.wm.utils.load_pretrained(ckpt_arg, cache_dir=cache_dir)
    model = model.eval().requires_grad_(False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_cube_dataset(cache_dir=cache_dir)
    _, _, feats, tgt_tr, tgt_va = _load_or_extract(
        model, dataset, ckpt_arg, out_dir, n_frames, val_frac, seed, sources, force
    )

    Ytr = group_targets(tgt_tr)
    Yva = group_targets(tgt_va)
    var = {g: Ytr[g].var(axis=0) for g in GROUP_SPECS}

    Ytr_extra = group_targets_for_specs(tgt_tr, EXTRA_DIAG_SPECS)
    Yva_extra = group_targets_for_specs(tgt_va, EXTRA_DIAG_SPECS)
    var_extra = {g: Ytr_extra[g].var(axis=0) for g in EXTRA_DIAG_SPECS}

    rows: list[dict] = []
    for s in sources:
        Xtr, Xva = feats[s]
        ok_tr = drop_nan_rows(tgt_tr)
        ok_va = drop_nan_rows(tgt_va)
        Xtr = Xtr[ok_tr]
        Xva = Xva[ok_va]
        Ytr_s = {g: v[ok_tr] for g, v in Ytr.items()}
        Yva_s = {g: v[ok_va] for g, v in Yva.items()}

        rows += evaluate_group_probes(
            ckpt_arg, s, Xtr, Xva, Ytr_s, Yva_s, GROUP_SPECS, var, seed=seed
        )
        if EXTRA_DIAG_SPECS:
            Ytr_ex_s = {g: v[ok_tr] for g, v in Ytr_extra.items()}
            Yva_ex_s = {g: v[ok_va] for g, v in Yva_extra.items()}
            rows += evaluate_group_probes(
                ckpt_arg, s, Xtr, Xva, Ytr_ex_s, Yva_ex_s, EXTRA_DIAG_SPECS, var_extra, seed=seed
            )

        print(f"[probe] {s}: ridge+mlp listos ({len(rows)} filas acumuladas)")

    import pandas as pd

    stem = Path(ckpt_arg).parent.name + "__" + Path(ckpt_arg).stem
    csv_path = out_dir / f"{stem}_{'-'.join(sources)}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[probe] resultados -> {csv_path}")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Probes lineales/MLP sobre latentes de Cube")
    p.add_argument("--ckpt", action="append", required=True, help="checkpoint (repetible)")
    p.add_argument("--source", action="append", choices=["cls", "patch_mean", "emb"], default=None)
    p.add_argument("--n-frames", type=int, default=50_000)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--out-dir", default="outputs/probe")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="reextraer features ignorando cache")
    args = p.parse_args(argv)
    if not args.source:
        args.source = ["cls", "patch_mean"]
    return args


def main(argv=None):
    args = parse_args(argv)
    for ckpt in args.ckpt:
        run_probe(
            ckpt=ckpt,
            sources=args.source,
            n_frames=args.n_frames,
            val_frac=args.val_frac,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
