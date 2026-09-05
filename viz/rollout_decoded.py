"""Rollout decodificado del world model (Predictor rollout, sin tiempo real).

Carga un world model (JEPA) congelado, un decoder a posteriori (``decoder.pt``
de ``decoder_train.py``) y un dataset. Codifica ``--ctx`` observaciones como
contexto (GT), genera ``--n`` latentes futuros de forma autoregresiva
condicionados a las acciones reales del dataset (open-loop), y los decodifica
a imagen. Compone una figura de una fila: ``ctx`` frames GT de contexto +
``n`` frames del rollout decodificado.

Notas:
  - El decoder debe estar entrenado con ``--source emb`` (el espacio
    post-projector), porque el rollout predice latentes en ese mismo espacio.
  - ``--data`` selecciona el dataset por nombre literal (p. ej.
    ``ogbench/cube_single_expert.h5``, ``pusht_expert_train.h5``,
    ``tworoom.h5``, ``reacher.h5``); si se omite, usa el ``config.yaml`` del run.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registra el formato HDF5

from viz.decoder_train import CLSDecoder, denormalize
from utils import get_img_preprocessor


def load_dataset(cfg, name, cache_dir):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if name is not None:
        dataset_cfg.pop("name")
    else:
        name = dataset_cfg.pop("name")
    dataset_cfg.pop("fraction", None)
    dataset_cfg.pop("keys_to_merge", None)  # proprio virtual no existe crudo en el h5
    dataset_cfg["num_steps"] = 1  # indexamos frame a frame
    ds = swm.data.load_dataset(name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    return ds


def load_decoder(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    decoder = CLSDecoder(
        cls_dim=ck["cls_dim"],
        img_size=ck["img_size"],
        patch_size=ck["patch_size"],
        dim=ck["dim"],
        heads=ck["heads"],
        depth=ck["depth"],
    ).to(device).eval()
    decoder.load_state_dict(ck["state_dict"])
    return decoder, ck.get("source", "cls"), ck["cls_dim"]


def to_chw(x):
    """Acepta (H,W,3) HWC o (3,H,W) CHW y devuelve tensor (3,H,W)."""
    a = torch.as_tensor(x).float()
    if a.ndim == 3 and a.shape[-1] in (1, 3) and a.shape[0] not in (1, 3):
        a = a.permute(2, 0, 1)  # HWC -> CHW
    return a


def to_hwc(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="world model (.pt) — ruta absoluta o relativa al cache")
    p.add_argument("--config", default=None, help="config.yaml del run (default: junto al ckpt)")
    p.add_argument("--decoder", required=True, help="decoder.pt entrenado con --source emb")
    p.add_argument("--data", default=None, help="nombre literal del dataset (default: del config.yaml)")
    p.add_argument("--start", type=int, default=500, help="índice de frame del primer contexto")
    p.add_argument("--ctx", type=int, default=3, help="nº de frames de contexto (GT)")
    p.add_argument("--n", type=int, default=6, help="nº de latentes futuros a decodificar")
    p.add_argument("--out", default=None, help="ruta de salida del PNG")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    # config del run
    cfg_path = Path(args.config) if args.config else Path(args.ckpt).parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No encuentro {cfg_path}. Pasa --config.")
    cfg = OmegaConf.load(cfg_path)
    print(f"[rollout] config: {cfg_path}")

    # world model congelado
    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # decoder
    decoder, dec_source, dec_dim = load_decoder(args.decoder, device)
    if dec_source != "emb":
        print(f"[rollout] AVISO: decoder entrenado con source={dec_source}; "
              f"para rollout se espera --source emb (post-proyector).")
    print(f"[rollout] decoder source={dec_source} cls_dim={dec_dim}")

    # dataset
    ds = load_dataset(cfg, args.data, cache_dir)
    n_rows = len(ds)
    ep_col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_arr = np.asarray(ds.get_col_data(ep_col))
    ep = int(ep_arr[args.start])
    ep_rows = np.nonzero(ep_arr == ep)[0]
    ep_end = int(ep_rows[-1]) + 1
    need = args.start + args.ctx + args.n
    if need > ep_end:
        raise ValueError(
            f"start={args.start}+ctx={args.ctx}+n={args.n}={need} sale del episodio {ep} "
            f"(termina en {ep_end - 1}). Baja --start o --n."
        )
    print(f"[rollout] dataset rows={n_rows} | episodio {ep} [{ep_rows[0]}..{ep_end - 1}]")

    # contexto GT + acciones (bloqueadas: action por-step repetida frameskip veces)
    ds_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    frameskip = int(ds_cfg["frameskip"])
    ctx_pix = torch.stack([to_chw(ds.get_row_data(args.start + i)["pixels"]) for i in range(args.ctx)])
    raw_acts = torch.stack(
        [torch.as_tensor(ds.get_row_data(args.start + i)["action"]).float() for i in range(args.ctx + args.n)]
    )
    acts = raw_acts.repeat(1, frameskip)  # (ctx+n, act_dim*frameskip) tile del bloque
    print(f"[rollout] frameskip={frameskip} act_dim={raw_acts.size(-1)} act_block={acts.size(-1)}")
    info = {
        "pixels": ctx_pix.unsqueeze(0).float().to(device),
        "action": acts[: args.ctx].unsqueeze(0).float().to(device),
    }
    with torch.no_grad():
        out = model.encode(info)
    emb = out["emb"]          # (1, ctx, D)
    act_emb = out["act_emb"]  # (1, ctx, A_emb)
    if dec_dim != emb.size(-1):
        raise ValueError(
            f"decoder cls_dim={dec_dim} != latente predicho {emb.size(-1)}. "
            f"Entrena el decoder con --source emb sobre el mismo modelo."
        )

    # rollout autoregresivo open-loop
    preds = []
    history_size = int(cfg.history_size)
    with torch.no_grad():
        for t in range(args.n):
            ctx_e = emb[:, -history_size:]      # (1, HS, D)
            ctx_a = act_emb[:, -history_size:]  # (1, HS, A)
            next_lat = model.predict(ctx_e, ctx_a)[:, -1:]  # (1, 1, D)
            emb = torch.cat([emb, next_lat], dim=1)
            preds.append(next_lat)
            ae = model.action_encoder(acts[args.ctx + t].view(1, 1, -1).float().to(device))
            act_emb = torch.cat([act_emb, ae], dim=1)
    z = torch.cat(preds, dim=1)[0]  # (n, D)

    # decodificar
    with torch.no_grad():
        recons = denormalize(decoder(z).float()).cpu()
    recons = recons.clamp(0, 1)

    # figura: ctx GT + n rollout
    n_panels = args.ctx + args.n
    fig, axes = plt.subplots(1, n_panels, figsize=(2.0 * n_panels, 2.3))
    for i in range(args.ctx):
        axes[i].imshow(to_hwc(ctx_pix[i]))
        axes[i].set_title(f"ctx {i - args.ctx}", fontsize=9)
        axes[i].axis("off")
    for j in range(args.n):
        axes[args.ctx + j].imshow(to_hwc(recons[j]))
        axes[args.ctx + j].set_title(f"roll +{j + 1}", fontsize=9)
        axes[args.ctx + j].axis("off")
    fig.suptitle(
        f"{Path(args.ckpt).parent.name} | {args.data or cfg.data.dataset.get('name', '')} | "
        f"decoder source={dec_source} | start={args.start}",
        fontsize=10,
    )
    plt.tight_layout()

    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"rollout_ctx{args.ctx}_n{args.n}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[rollout] imagen guardada: {out}")


if __name__ == "__main__":
    main()