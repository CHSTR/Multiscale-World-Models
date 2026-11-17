"""Visualización específica del predictor NCA de LiftWM.

Dos GIFs sobre un frame de cube con el checkpoint NCA entrenado:

1. ``nca_internal_idx*.gif``: dinámica celular INTERNA. Norma del latente en cada
   uno de los K pasos locales de la regla NCA (paso 0 = estado inicial), por
   escala. Primer panel = frame real de referencia.

2. ``nca_rollout_idx*.gif``: rollout multi-paso. Error |predicho − real| por
   escala en cada paso del mundo. Primer panel = frame real de ese paso.

La figure, los títulos y el layout se crean UNA vez; en cada frame sólo se
actualizan los datos (set_data) y el texto del título — así los paneles no se
mueven entre frames (el tight_layout por frame reacomodaba los axes).

Uso:
  /home/chr/le-wm/.venv/bin/python viz/liftwm_nca_viz.py --idx 0 \
      --ckpt liftwm_nca_cube/weights_epoch_1.pt
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
    return img, np.concatenate(acts)


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _make_figure(grids, suptitle, cmap, clim, has_real=True, clabel=""):
    """Crea la figure UNA vez con panel fijo para el real + uno por escala + colorbar.

    Todos los paneles comparten la misma escala de color (min/max de los clim
    por escala), así la única colorbar a la derecha es válida para todos.
    """
    ns = len(grids)
    ncol = ns + (1 if has_real else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol + 1.0, 3.9), squeeze=False)
    ax = axes[0]
    h_real = None
    if has_real:
        h_real = ax[0].imshow(np.zeros((224, 224, 3)))
        ax[0].set_title("frame real", fontsize=9)
        ax[0].axis("off")
    vmin = min(c[0] for c in clim)
    vmax = max(c[1] for c in clim)
    handles = []
    for k in range(ns):
        gh, gw = grids[k]
        h = ax[k + int(has_real)].imshow(np.zeros((gh, gw)), cmap=cmap,
                                        vmin=vmin, vmax=vmax)
        ax[k + int(has_real)].set_title(f"escala {k} · {gh}×{gw}", fontsize=9)
        ax[k + int(has_real)].axis("off")
        handles.append(h)
    st = fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    cb = fig.colorbar(handles[0], cax=cax)
    cb.set_label(clabel, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return fig, h_real, handles, st


def internal_gif(model, idx, args):
    img, act_np = get_frame_action(idx)
    A = model.action_encoder.patch_embed.in_channels
    act = torch.tensor(act_np[-A:], dtype=torch.float32).reshape(1, 1, A)

    with torch.no_grad():
        out = model.encode({"pixels": img.unsqueeze(0).unsqueeze(0), "action": act})
        emb, act_emb = out["emb"], out["act_emb"]
        levels = sorted(emb)
        grids = [model.grid_hw[l] for l in levels]
        per_scale_steps = []
        for l in levels:
            p = model.predictor.predictors[str(l)]
            old_eval = p.eval_steps
            if args.steps > 0:
                p.eval_steps = args.steps   # más pasos internos en inferencia
            gh, gw = model.grid_hw[l]
            x_f = emb[l][0, 0].unsqueeze(0)
            v_f = torch.zeros_like(x_f)
            s = p._init_state(x_f, v_f, gh, gw)
            shift, scale = p._modulation(act_emb[0, 0].unsqueeze(0))
            states = [p._latent_out(s)]
            for i in range(p.num_steps()):
                s, _ = p.nca_step(s, shift, scale, i)
                states.append(p._latent_out(s))
            per_scale_steps.append([
                st[0].norm(dim=-1).reshape(gh, gw).float().cpu().numpy() for st in states
            ])
            p.eval_steps = old_eval

    K = len(per_scale_steps[0]) - 1
    clim = []
    for k in range(len(levels)):
        pool = np.concatenate([n.ravel() for n in per_scale_steps[k]])
        clim.append((float(np.percentile(pool, 2)), float(np.percentile(pool, 98))))

    fig, h_real, handles, st = _make_figure(
        grids, "NCA dinámica celular interna — paso 0", "viridis", clim,
        clabel="‖latente‖₂")
    h_real.set_data(img.permute(1, 2, 0).numpy())
    frames = []
    for i in range(K + 1):
        for k, h in enumerate(handles):
            h.set_data(per_scale_steps[k][i])
        st.set_text(f"NCA dinámica celular interna — paso {i}/{K} (norma del latente)")
        frames.append(fig_to_array(fig))
    plt.close(fig)

    import imageio.v2 as imageio
    out_path = os.path.join(args.out, f"nca_internal_idx{idx}.gif")
    imageio.mimsave(out_path, frames, duration=1.0 / args.fps, loop=0)
    print(f"GIF NCA interno ({K + 1} pasos) ->", out_path)


def rollout_gif(model, idx, args):
    A = model.action_encoder.patch_embed.in_channels
    B, T = 1, args.nsteps
    with torch.no_grad():
        img0, _ = get_frame_action(idx)
        out = model.encode({"pixels": img0.unsqueeze(0).unsqueeze(0),
                            "action": torch.zeros(1, 1, A)})
        ctx = out["emb"]
        act_seq = model.action_encoder(torch.zeros(B, T, A))
        pred = model.predict_rollout(ctx, act_seq, history=1)

        true_imgs, true = [], {}
        for t in range(1, T + 1):
            im, _ = get_frame_action(idx + t * args.stride)
            true_imgs.append(im)
            o2 = model.encode({"pixels": im.unsqueeze(0).unsqueeze(0)})
            for l in o2["emb"]:
                true.setdefault(l, []).append(o2["emb"][l][:, 0])
        err = {l: torch.stack([(pred[l][0, t] - true[l][t]).abs().norm(dim=-1)
                               for t in range(T)]).float().cpu().numpy() for l in pred}

    levels = sorted(err)
    grids = [model.grid_hw[l] for l in levels]
    clim = []
    for l in levels:
        pool = err[l].ravel()
        clim.append((0.0, float(np.percentile(pool, 98)) or 1.0))

    fig, h_real, handles, st = _make_figure(
        grids, "NCA rollout — error |pred−real| paso 1", "magma", clim,
        clabel="‖pred − real‖₂")
    frames = []
    for t in range(T):
        h_real.set_data(true_imgs[t].permute(1, 2, 0).numpy())
        for k, l in enumerate(levels):
            handles[k].set_data(err[l][t].reshape(model.grid_hw[l]))
        st.set_text(f"NCA rollout — error |pred−real| paso {t + 1}/{T} del mundo")
        frames.append(fig_to_array(fig))
    plt.close(fig)

    import imageio.v2 as imageio
    out_path = os.path.join(args.out, f"nca_rollout_idx{idx}.gif")
    imageio.mimsave(out_path, frames, duration=1.0 / args.fps, loop=0)
    print(f"GIF NCA rollout ({T} pasos) ->", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="liftwm_nca_cube/weights_epoch_1.pt")
    ap.add_argument("--out", type=str, default="outputs/liftwm_viz")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--nsteps", type=int, default=6, help="pasos del mundo en rollout")
    ap.add_argument("--steps", type=int, default=0,
                    help="pasos internos del NCA en el GIF interno (0 = los del ckpt)")
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
