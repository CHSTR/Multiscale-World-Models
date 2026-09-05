"""Visualizacion de rollouts open-loop del predictor, decodificados a pixeles.

Replica las Figs. 7 y 9 del paper LeWM: se codifican N frames de contexto, el
predictor genera autoregresivamente los siguientes latentes condicionado a las
acciones reales, y cada latente se decodifica con el CLSDecoder entrenado a
posteriori. Fila superior = real, fila inferior = imaginado.

IMPORTANTE - espacio latente
----------------------------
El predictor opera en el espacio POST-projector: en jepa.py, encode() devuelve
emb = projector(cls) y predict() devuelve pred_proj(predictor(...)), y la
perdida compara pred_emb contra emb. Por tanto el decoder tiene que haberse
entrenado con `--source emb`. Un decoder de `--source cls` recibiria vectores
de otra distribucion y produciria basura.

    python decoder_train.py --ckpt <ckpt> --source emb
    python decoder_rollout.py --ckpt <ckpt> --decoder <out_dir>/decoder.pt

Uso
---
    python decoder_rollout.py \
        --ckpt dataset/checkpoints/lewm/weights_epoch_10.pt \
        --decoder dataset/checkpoints/lewm/decoder_emb/decoder.pt \
        --start 685385 --context 3 --steps 5 --plot-error
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_pretraining as spt
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registra el formato HDF5

from decoder_train import (
    CLSDecoder,
    build_dataset,
    denormalize,
    encode_frames,
    load_world_model,
)
from utils import ZScoreNormalizer, get_img_preprocessor


# --------------------------------------------------------------------------- #
#  Carga
# --------------------------------------------------------------------------- #


def load_decoder(path, device):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("source") != "emb":
        raise ValueError(
            f"El decoder fue entrenado con source={ckpt.get('source')!r}. Para decodificar "
            "rollouts del predictor hace falta source='emb' (espacio post-projector). "
            "Reentrena con: python decoder_train.py --ckpt <ckpt> --source emb"
        )
    decoder = CLSDecoder(
        cls_dim=ckpt["cls_dim"],
        img_size=ckpt["img_size"],
        patch_size=ckpt["patch_size"],
        dim=ckpt["dim"],
        heads=ckpt["heads"],
        depth=ckpt["depth"],
    )
    decoder.load_state_dict(ckpt["state_dict"])
    return decoder.to(device).eval().requires_grad_(False)


def episode_column(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def episode_span(dataset, row):
    """Devuelve [start, end) del episodio que contiene `row`.

    Usa ep_offset/ep_len (columnas pequenas, presentes en los h5 de lewm) en vez
    de leer la columna por-fila, que en datasets con columnas comprimidas (p.ej.
    pusht) puede fallar con un error de filtro al leerla completa.
    """
    ep_off = np.asarray(dataset.get_col_data("ep_offset"))
    ep_len = np.asarray(dataset.get_col_data("ep_len"))
    e = int(np.searchsorted(ep_off, row, side="right") - 1)
    return int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])


def collect_sequence(dataset, start, length, stride, act_normalizer):
    """Toma `length` filas consecutivas y devuelve (pixels, actions) de la trayectoria.

    Cada fila del dataset es una sub-trayectoria de T frames; aqui se usa el
    frame 0 y el bloque de accion 0 de cada fila, de modo que filas consecutivas
    equivalen a pasos latentes consecutivos. Si tu HDF5 usa otra convencion de
    stride entre filas, ajusta --row-stride.
    """
    col = episode_column(dataset)
    idx = [start + k * stride for k in range(length)]

    lo, hi = episode_span(dataset, start)
    if idx[-1] >= hi:
        raise ValueError(
            f"Las filas {idx[0]}..{idx[-1]} cruzan un limite de episodio "
            f"(episodio [{lo}, {hi})). Elige otro --start."
        )

    pixels, actions = [], []
    for i in idx:
        row = dataset[i]
        px = row["pixels"]
        act = torch.as_tensor(np.asarray(row["action"]), dtype=torch.float32)
        pixels.append(px[0] if px.ndim == 4 else px)
        actions.append(act[0] if act.ndim >= 2 else act)

    pixels = torch.stack(pixels)  # (L, C, H, W)
    actions = torch.nan_to_num(torch.stack(actions), 0.0)  # (L, frameskip * A)

    # La fila del dataset entrega la accion ya agrupada en bloques de frameskip
    # (p.ej. 10 = 5 pasos x 2D), pero las estadisticas z-score se calculan sobre
    # la accion CRUDA sin agrupar (2D). En train.py el normalizador se aplica
    # antes del aplanado; aqui lo aplicamos por sub-accion para que coincida.
    a_dim = act_normalizer.mean.size(-1)
    if actions.size(-1) != a_dim:
        assert actions.size(-1) % a_dim == 0, (
            f"bloque de accion de {actions.size(-1)} no es multiplo de la dim cruda {a_dim}"
        )
        L_, blk = actions.shape
        actions = act_normalizer(actions.reshape(L_, blk // a_dim, a_dim)).reshape(L_, blk)
    else:
        actions = act_normalizer(actions)

    return pixels, actions


# --------------------------------------------------------------------------- #
#  Rollout
# --------------------------------------------------------------------------- #


@torch.no_grad()
def open_loop_rollout(model, pixels, actions, n_context, n_steps, history_size):
    """Codifica el contexto y predice n_steps latentes autoregresivamente.

    Sigue la misma logica de truncado que JEPA.rollout: en cada paso solo se
    alimentan los ultimos `history_size` embeddings y acciones, y se toma la
    ultima prediccion. Devuelve (emb_pred, emb_real), ambos (1, n_context+n_steps, D).
    """
    device = next(model.parameters()).device
    pixels = pixels.to(device)
    actions = actions.to(device).unsqueeze(0)  # (1, L, A)

    emb_real = encode_frames(model, pixels, source="emb").unsqueeze(0)  # (1, L, D)
    emb = emb_real[:, :n_context].clone()

    HS = history_size
    for t in range(n_steps):
        act_emb = model.action_encoder(actions[:, : emb.size(1)])
        pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)

    return emb, emb_real[:, : emb.size(1)]


# --------------------------------------------------------------------------- #
#  Figura
# --------------------------------------------------------------------------- #


def save_rollout_figure(real_px, imag_px, n_context, frameskip, path, title=None):
    real = denormalize(real_px).cpu().numpy().transpose(0, 2, 3, 1)
    imag = denormalize(imag_px).cpu().numpy().transpose(0, 2, 3, 1)
    L = len(real)

    fig, axes = plt.subplots(2, L, figsize=(1.9 * L, 4.4))
    axes = axes.reshape(2, -1)
    for i in range(L):
        axes[0, i].imshow(real[i])
        axes[1, i].imshow(imag[i])
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            for s in axes[r, i].spines.values():
                s.set_edgecolor("tab:red" if i >= n_context else "0.7")
                s.set_linewidth(1.8 if i >= n_context else 0.8)
        axes[1, i].set_xlabel(f"T={i * frameskip}", fontsize=8)

    axes[0, 0].set_ylabel("Real", fontsize=10)
    axes[1, 0].set_ylabel("Imagined", fontsize=10)
    axes[0, n_context - 1].set_title("| contexto", fontsize=8, loc="right", color="0.4")
    axes[0, n_context].set_title("open loop |", fontsize=8, loc="left", color="tab:red")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_error_curve(err, n_context, frameskip, path):
    steps = np.arange(len(err)) * frameskip
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(steps, err, marker="o", color="tab:red")
    ax.axvline((n_context - 1) * frameskip, ls="--", color="0.5", lw=1)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("MSE latente (pred vs real)")
    ax.set_title("Error de rollout open-loop")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="ruta relativa al cache dir del world model")
    p.add_argument("--decoder", required=True, help="decoder.pt entrenado con --source emb")
    p.add_argument("--config", default=None, help="config.yaml del run (por defecto: junto al ckpt)")
    p.add_argument("--start", type=int, required=True, help="indice de fila inicial en el dataset")
    p.add_argument("--context", type=int, default=None, help="frames de contexto (default: cfg.history_size)")
    p.add_argument("--steps", type=int, default=5, help="pasos latentes a predecir")
    p.add_argument("--row-stride", type=int, default=1, help="separacion entre filas consecutivas")
    p.add_argument("--plot-error", action="store_true", help="guarda tambien la curva de MSE latente")
    p.add_argument("--out-dir", default="outputs/rollout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(swm.data.utils.get_cache_dir())

    cfg_path = Path(args.config) if args.config else (cache_dir / args.ckpt).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    n_context = args.context or cfg.history_size
    history_size = cfg.history_size
    frameskip = cfg.data.dataset.frameskip

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_world_model(args.ckpt, device)
    decoder = load_decoder(args.decoder, device)

    # El dataset se construye sin normalizador de acciones (build_dataset solo
    # toca pixels), asi que lo recuperamos aparte: el predictor fue entrenado
    # con acciones z-scored y alimentarlo con acciones crudas rompe el rollout.
    #
    # Nota: get_column_normalizer() devuelve un WrapTorchTransform, cuyo metodo
    # .transform tiene firma (x, params) y no es un callable simple. Aqui se
    # replica su calculo interno para obtener un ZScoreNormalizer directo.
    dataset = build_dataset(cfg)
    _col = torch.from_numpy(np.array(dataset.get_col_data("action")))
    _col = _col[~torch.isnan(_col).any(dim=1)]
    act_scaler = ZScoreNormalizer(
        _col.mean(0, keepdim=True).clone(), _col.std(0, keepdim=True).clone()
    )

    L = n_context + args.steps
    pixels, actions = collect_sequence(dataset, args.start, L, args.row_stride, act_scaler)
    print(f"[rollout] {n_context} frames de contexto + {args.steps} pasos open-loop "
          f"(frameskip={frameskip}, history_size={history_size})")

    emb_pred, emb_real = open_loop_rollout(
        model, pixels, actions, n_context, args.steps, history_size
    )

    with torch.no_grad():
        imag_px = decoder(emb_pred[0]).float()

    tag = f"{Path(args.ckpt).parent.name}_start{args.start}"
    save_rollout_figure(
        pixels.to(device), imag_px, n_context, frameskip,
        out_dir / f"rollout_{tag}.png",
        title=f"{Path(args.ckpt).parent.name} — fila {args.start}",
    )

    err = (emb_pred - emb_real).pow(2).mean(dim=-1)[0].cpu().numpy()
    print("[rollout] MSE latente por paso: " + "  ".join(f"{e:.4f}" for e in err))

    if args.plot_error:
        save_error_curve(err, n_context, frameskip, out_dir / f"error_{tag}.png")

    print(f"[rollout] listo. Salidas en {out_dir}")


if __name__ == "__main__":
    main()
