"""Visualización específica del predictor NCA de LiftWM.

Dos GIFs sobre un frame de cube con el checkpoint NCA entrenado:

1. ``nca_internal_idx*.gif``: dinámica celular INTERNA. Muestra la norma del
   latente en cada uno de los K pasos locales de la regla NCA (paso 0 = estado
   inicial), por escala. Se ve cómo la información se propaga por la grilla.

2. ``nca_rollout_idx*.gif``: rollout multi-paso del world model. Por escala,
   error |predicho − real| de los tokens en cada paso del mundo (horizonte),
   con clim fijo para ver cómo crece el error con el horizonte.

Uso:
  /home/chr/le-wm/.venv/bin/python viz/liftwm_nca_viz.py --idx 0 --ckpt liftwm_nca_cube/weights_epoch_1.pt
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


_DS = None


def _dataset():
    global _DS
    if _DS is None:
        _DS = swm.data.load_dataset("ogbench/cube_single_expert.h5", keys_to_cache=["action"])
    return _DS


def get_frame_action(idx, size=224, n_act=5):
    ds = _dataset()
    row = ds.get_row_data(idx)
    tf = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size),
    ])
    img = tf(row["pixels"])
    acts = [np.asarray(ds.get_row_data(j)["action"], dtype=np.float32).reshape(-1)
            for j in range(idx, min(idx + n_act, len(ds)))]
    return img, np.concatenate(acts)  # frameskip*action_dim (p.ej. 25 en cube)


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _heatmap_grid(fig, axes, grids, norms, title, clim, cmap="viridis"):
    for k, ax in enumerate(axes):
        gh, gw = grids[k]
        ax.imshow(norms[k], cmap=cmap, vmin=clim[k][0], vmax=clim[k][1])
        ax.set_title(f"escala {k} · {gh}×{gw}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()


def internal_gif(model, idx, args):
    img, act_np = get_frame_action(idx)
    A = model.action_encoder.patch_embed.in_channels
    act = torch.tensor(act_np[-A:], dtype=torch.float32).reshape(1, 1, A)

    with torch.no_grad():
        info = {"pixels": img.unsqueeze(0).unsqueeze(0), "action": act}
        out = model.encode(info)
        emb, act_emb, grids = out["emb"], out["act_emb"], model.grid_hw
        per_scale_steps = []
        for l in sorted(emb):
            p = model.predictor.predictors[str(l)]
            gh, gw = grids[l]
            x_f = emb[l][0, 0].unsqueeze(0)          # (1, N, D)
            v_f = torch.zeros_like(x_f)              # T=1 -> velocidad 0
            s = p._init_state(x_f, v_f, gh, gw)
            shift, scale = p._modulation(act_emb[0, 0].unsqueeze(0))
            states = [p._latent_out(s)]
            for i in range(p.num_steps()):
                s, _ = p.nca_step(s, shift, scale, i)
                states.append(p._latent_out(s))
            norm_steps = [st[0].norm(dim=-1).reshape(gh, gw).float().cpu().numpy()
                          for st in states]
            per_scale_steps.append(norm_steps)

    ns = len(per_scale_steps)
    K = len(per_scale_steps[0]) - 1
    grids = [model.grid_hw[l] for l in sorted(emb)]
    clim = []
    for k in range(ns):
        pool = np.concatenate([n.ravel() for n in per_scale_steps[k]])
        clim.append((float(np.percentile(pool, 2)), float(np.percentile(pool, 98))))

    import imageio.v2 as imageio
    fig, axes = plt.subplots(1, ns, figsize=(3.2 * ns, 3.6), squeeze=False)
    axrow = axes[0]
    arrays = []
    for i in range(K + 1):
        norms = [per_scale_steps[k][i] for k in range(ns)]
        _heatmap_grid(fig, axrow, grids, norms,
                      f"NCA dinámica celular interna — paso {i}/{K} (norma del latente)", clim)
        arrays.append(fig_to_array(fig))
    plt.close(fig)
    out = os.path.join(args.out, f"nca_internal_idx{idx}.gif")
    imageio.mimsave(out, arrays, duration=1.0 / args.fps, loop=0)
    print(f"GIF NCA interno ({K + 1} pasos) ->", out)


def rollout_gif(model, idx, args):
    img, _ = get_frame_action(idx)
    A = model.action_encoder.patch_embed.in_channels
    B, S, T = 1, 1, args.nsteps
    with torch.no_grad():
        info = {"pixels": img.unsqueeze(0).unsqueeze(0),
                "action": torch.zeros(1, 1, A)}
        out = model.encode(info)
        ctx = out["emb"]                                   # {l:(B,1,N,D)}
        act_seq = model.action_encoder(torch.zeros(B, T, A))  # (B,T,192)
        pred = model.predict_rollout(ctx, act_seq, history=1)   # {l:(B,T,N,D)}
        # verdad: codificar frames futuros con actions neutras (misma escala temporal)
        true = {}
        for t in range(1, T + 1):
            j = idx + t * args.stride
            im2, _ = get_frame_action(j)
            o2 = model.encode({"pixels": im2.unsqueeze(0).unsqueeze(0)})
            for l in o2["emb"]:
                true.setdefault(l, []).append(o2["emb"][l][:, 0])
        err = {l: torch.stack([(pred[l][0, t] - true[l][t]).abs().norm(dim=-1)
                               for t in range(T)]).float().cpu().numpy()
               for l in pred}

    levels = sorted(err)
    ns = len(levels)
    grids = [model.grid_hw[l] for l in levels]
    clim = []
    for l in levels:
        p = np.concatenate([e.ravel() for e in err[l]])
        clim.append((0.0, float(np.percentile(p, 98)) or 1.0))

    import imageio.v2 as imageio
    fig, axes = plt.subplots(1, ns, figsize=(3.2 * ns, 3.6), squeeze=False)
    axrow = axes[0]
    arrays = []
    for t in range(T):
        norms = [err[l][t].reshape(model.grid_hw[l]) for l in levels]
        _heatmap_grid(fig, axrow, grids, norms,
                      f"NCA rollout — error |pred−real| paso {t + 1}/{T} del mundo",
                      clim, cmap="magma")
        arrays.append(fig_to_array(fig))
    plt.close(fig)
    out = os.path.join(args.out, f"nca_rollout_idx{idx}.gif")
    imageio.mimsave(out, arrays, duration=1.0 / args.fps, loop=0)
    print(f"GIF NCA rollout ({T} pasos) ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="liftwm_nca_cube/weights_epoch_1.pt")
    ap.add_argument("--out", type=str, default="outputs/liftwm_viz")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--nsteps", type=int, default=6, help="pasos del mundo en rollout")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--mode", choices=["internal", "rollout", "both"], default="both")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = swm.wm.utils.load_pretrained(args.ckpt).to("cpu").eval()
    if args.mode in ("internal", "both"):
        internal_gif(model, args.idx, args)
    if args.mode in ("rollout", "both"):
        rollout_gif(model, args.idx, args)


if __name__ == "__main__":
    main()
