"""Genera las anclas pre-cargadas para el Space.

Lee el dataset HDF5 local de pusht (con el mismo preprocesamiento de imagen y
z-score de acciones que el entrenamiento), elige varios clips bien separados de
episodios distintos y guarda cada ancla de forma autonoma dentro de hf_space/:

    hf_space/anchors/a0/
        preview_0.png            # frame de contexto (denormalizado, para la UI)
        preview_1.png
        preview_2.png
        pixels.pt                # (n_context,3,224,224) ya normalizados (ImageNet)
        emb.pt                   # (1, n_context, D) latentes post-projector pre-codificados
        acts.npz                 # acciones normalizadas (n_context-1, n_sub*a_dim)
        meta.json                # descripcion de la ancla

Asi el Space no necesita el dataset: codifica el ancla una sola vez aqui, y la
app solo reconstruye imagenes con el decoder.

Uso:
    .venv_hf/bin/python make_anchors.py --out hf_space/anchors
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import stable_worldmodel as swm

from decoder_rollout import collect_sequence, episode_span, episode_column
from decoder_train import build_dataset, denormalize, encode_frames
from utils import ZScoreNormalizer


def build_act_scaler(dataset, a_dim):
    col = torch.from_numpy(np.array(dataset.get_col_data("action")))
    col = col[~torch.isnan(col).any(dim=1)]
    col = col.reshape(-1, a_dim)
    return ZScoreNormalizer(col.mean(0, keepdim=True).clone(), col.std(0, keepdim=True).clone())


def pick_anchors(dataset, n_anchors=6, context=3, rng_seed=0, min_span=12):
    """Elige n_anchors filas de episodios distintos, repartidos por el dataset."""
    ep_off = np.asarray(dataset.get_col_data("ep_offset"))
    ep_len = np.asarray(dataset.get_col_data("ep_len"))
    rng = np.random.default_rng(rng_seed)

    # episodios con margen suficiente y fila inicial aleatoria dentro de ellos
    cand = []
    for e in range(len(ep_off)):
        lo = int(ep_off[e])
        hi = lo + int(ep_len[e])
        if hi - lo < min_span:
            continue
        cand.append(lo + rng.integers(0, hi - lo - context - 1))

    # esparce la seleccion por todo el dataset (episodios tempranos y tardios)
    cand = sorted(cand)
    n = len(cand)
    step = n / max(1, n_anchors)
    sel = sorted({cand[int(k * step) % n] for k in range(n_anchors)})
    return [int(s) for s in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models_server/checkpoints/dinov3_s_lora_pusht/weights_epoch_10.pt")
    ap.add_argument("--config", default="models_server/checkpoints/dinov3_s_lora_pusht/config.yaml")
    ap.add_argument("--out", default="hf_space/anchors")
    ap.add_argument("--n-anchors", type=int, default=6)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    ckpt_rel = args.ckpt.replace("models_server/checkpoints/", "")
    model = swm.wm.utils.load_pretrained(ckpt_rel, cache_dir="models_server")
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    dataset = build_dataset(cfg)
    a_dim = 2
    act_scaler = build_act_scaler(dataset, a_dim)
    col = torch.from_numpy(np.array(dataset.get_col_data("action"))).reshape(-1, a_dim)
    act_std = float(col.std(0).mean())
    chance = 2.0

    starts = pick_anchors(dataset, args.n_anchors, args.context)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(starts):
        try:
            pixels, acts = collect_sequence(dataset, start, args.context, 1, act_scaler)
        except ValueError as ex:
            print(f"[skip] start={start}: {ex}")
            continue

        with torch.no_grad():
            emb = encode_frames(model, pixels.to(device), source="emb").unsqueeze(0)

        adir = out / f"a{i}"
        adir.mkdir(parents=True, exist_ok=True)

        # pixels normalizados -> tensores para la app
        torch.save(pixels.clone(), adir / "pixels.pt")
        # latentes pre-codificados
        torch.save(emb.clone(), adir / "emb.pt")
        # acciones normalizadas (n_context-1, a_dim*n_sub)
        torch.save(acts[: args.context - 1].clone(), adir / "acts.pt")

        # previews denormalizados (ImageNet) para la UI
        frames = denormalize(pixels).cpu().numpy().transpose(0, 2, 3, 1)
        for j, img in enumerate(frames):
            from PIL import Image
            Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(
                adir / f"preview_{j}.png"
            )

        meta = {
            "index": i,
            "start_row": start,
            "dataset": "pusht_expert_train.h5",
            "context": args.context,
            "a_dim": a_dim,
            "n_sub": 5,
            "act_std": act_std,
            "act_mean": float(act_scaler.mean[0, 0]),
            "chance": chance,
            "frames": [f"preview_{j}.png" for j in range(len(frames))],
        }
        (adir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[ok] anchor a{i}  start={start}  emb={tuple(emb.shape)}  acts={tuple(acts.shape)}")


if __name__ == "__main__":
    main()