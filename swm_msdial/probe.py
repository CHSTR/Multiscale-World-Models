"""Sonda lineal ridge latente -> estado, split por episodio (spec §3/§7.7).

Reporta R² de [p, sinθ, cosθ] (columna `observation`) desde el embedding del
encoder, en episodios NO vistos en el ajuste. Para modelos por escala
(`emb` dict, p.ej. convnext_ms) reporta además R² por escala.

Uso:
  /home/chr/le-wm/.venv/bin/python swm_msdial/probe.py \
      --policy ~/.stable_worldmodel/checkpoints/<run>/weights_epoch_2.pt \
      --data ~/.stable_worldmodel/datasets/msdial/msdial_val_k2_balanced.h5 \
      --out reports/probe_<run>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------ núcleo
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Ridge por mínimos cuadrados: (XᵀX + λI)⁻¹ Xᵀy (sin intercepto)."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def split_episodes(ep_idx: np.ndarray, val_frac: float = 0.2,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split por EPISODIO (no por ventana): episodios disjuntos."""
    eps = np.unique(ep_idx)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(eps) * val_frac)))
    val_eps = rng.choice(eps, size=n_val, replace=False)
    train_eps = np.setdiff1d(eps, val_eps)
    return train_eps, val_eps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------- I/O
def encode_frames_with_model(model, pixels: np.ndarray, batch: int = 64) -> dict:
    """Codifica frames (N,H,W,3 uint8) -> {"emb": (N,D)} | {"emb_<l>": ..., "emb": ...}."""
    import torch

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    out: dict = {}
    with torch.no_grad():
        for i in range(0, len(pixels), batch):
            px = torch.tensor(pixels[i:i + batch], device=device, dtype=torch.float32) / 255.0
            px = (px.permute(0, 3, 1, 2) - mean) / std
            info = {"pixels": px.unsqueeze(1)}
            emb = model.encode(info)["emb"]
            if isinstance(emb, dict):  # modelos por escala (PyramidJEPA)
                for l in sorted(emb.keys()):
                    v = emb[l]
                    if v.ndim == 4:
                        v = v.mean(dim=2)
                    if v.ndim == 3:
                        v = v[:, 0]
                    out.setdefault(f"emb_{l}", []).append(
                        v.float().cpu().numpy())
            else:
                if emb.ndim == 4:
                    emb = emb.mean(dim=2)
                if emb.ndim == 3:
                    emb = emb[:, 0]
                out.setdefault("emb", []).append(
                    emb.float().cpu().numpy())

    res = {k: np.concatenate(v) for k, v in out.items()}
    if "emb" not in res and res:
        scale_keys = sorted(
            (k for k in res if k.startswith("emb_")),
            key=lambda k: int(k.split("_", 1)[1]) if k.split("_", 1)[1].isdigit() else k,
        )
        if scale_keys:
            res["emb"] = np.concatenate([res[k] for k in scale_keys], axis=-1)
    return res


def encode_frames(policy_path: str, pixels: np.ndarray,
                  batch: int = 64, cache_dir: str | None = None) -> dict:
    """Codifica frames (N,H,W,3 uint8) -> {"emb": (N,D)} | {"emb_<l>": ...}."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(policy_path).expanduser()
    ckpt_arg = str(p.resolve()) if p.exists() else policy_path
    model = __import__("stable_worldmodel").wm.utils.load_pretrained(
        ckpt_arg, cache_dir=cache_dir).to(device).eval()
    return encode_frames_with_model(model, pixels, batch=batch)


def probe_checkpoint(policy_path: str, data_path: str,
                     target_col: str = "observation",
                     val_frac: float = 0.2, max_frames: int = 20000,
                     batch: int = 64, seed: int = 0,
                     cache_dir: str | None = None) -> dict:
    import hdf5plugin  # noqa: F401 — registra blosc antes de importar h5py
    import h5py

    with h5py.File(data_path) as f:
        n = min(len(f["ep_idx"]), max_frames)
        pixels = f["pixels"][:n]
        y = f[target_col][:n]
        ep = f["ep_idx"][:n]

    train_eps, val_eps = split_episodes(ep, val_frac, seed)
    tr = np.isin(ep, train_eps)
    va = np.isin(ep, val_eps)

    embs = encode_frames(policy_path, pixels, batch=batch, cache_dir=cache_dir)
    report = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
              "target": target_col, "r2": {}}
    for name, lat in embs.items():
        W = ridge_fit(lat[tr], y[tr])
        report["r2"][name] = r2(y[va], lat[va] @ W)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="observation")
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = probe_checkpoint(a.policy, a.data, target_col=a.target,
                           max_frames=a.max_frames)
    print(json.dumps(rep, indent=2))
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()
