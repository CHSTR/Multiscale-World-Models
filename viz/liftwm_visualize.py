"""Visualización de lo multiescala / lifting para LiftWM.

Produce dos figuras sobre un frame real de cube:
  1. Pirámide lifting (Haar): frame original + aproximación LL + sub-bandas de
     detalle LH/HL/HH por nivel (de grueso a fino). Captura la descomposición
     wavelet invertible que alimenta el encoder.
  2. Normas de los tokens por escala del encoder entrenado: muestra que la
     estructura espacial se conserva por escala (sin pooling global).

Uso:
  /home/chr/le-wm/.venv/bin/python viz/liftwm_visualize.py --idx 0 \
      --ckpt liftwm_vit_cube/weights_epoch_1.pt --out outputs/liftwm_viz
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
from wavelet.lifting import LiftingScheme


def get_frame(idx: int, size: int = 224) -> torch.Tensor:
    ds = swm.data.load_dataset("ogbench/cube_single_expert.h5", keys_to_cache=["action"])
    row = ds.get_row_data(idx)
    tf = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size),
    ])
    return tf(row["pixels"])  # (3, size, size) en [0, 1]


def _show(ax, t, title, diverging=False):
    a = t.detach().float().cpu().numpy()
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3:
        a = a.mean(0)
    if diverging:
        v = float(np.abs(a).max()) or 1.0
        ax.imshow(a, cmap="RdBu_r", vmin=-v, vmax=v)
    else:
        ax.imshow(a, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def build_pyramid_fig(img: torch.Tensor, levels: int):
    lift = LiftingScheme(levels=levels, in_channels=3, init="haar", learnable=False)
    p = lift(img.unsqueeze(0))
    C = img.shape[0]
    fig, axes = plt.subplots(levels + 1, 4, figsize=(11, 3 * (levels + 1)))

    axes[0, 0].imshow(img.permute(1, 2, 0).numpy())
    axes[0, 0].set_title(f"original {img.shape[-1]}×{img.shape[-1]}", fontsize=9)
    axes[0, 0].axis("off")
    _show(axes[0, 1], p.approx, f"LL approx {p.approx.shape[-1]}×{p.approx.shape[-1]}")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # filas 1..L: detalles de más grueso (details[L-1]) a más fino (details[0])
    for r in range(1, levels + 1):
        di = levels - r
        det = p.details[di][0]
        res = det.shape[-1]
        LH, HL, HH = det[:C], det[C:2 * C], det[2 * C:]
        axes[r, 0].text(0.5, 0.5, f"nivel {r}\n{res}×{res}",
                        ha="center", va="center", fontsize=10)
        axes[r, 0].axis("off")
        _show(axes[r, 1], LH, f"LH (horizontal)", diverging=True)
        _show(axes[r, 2], HL, f"HL (vertical)", diverging=True)
        _show(axes[r, 3], HH, f"HH (diagonal)", diverging=True)

    fig.tight_layout()
    return fig


def pyramid_fig(img: torch.Tensor, levels: int, out: str):
    fig = build_pyramid_fig(img, levels)
    fig.suptitle("Pirámide lifting (Haar, L=3): aproximación + sub-bandas de detalle por escala\n"
                 "(detalle en colormap divergente; fino arriba-de-abajo en resolución)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def get_trajectory(idx: int, nframes: int, stride: int, size: int = 224):
    """Frames consecutivos de un mismo episodio (ordenados por step_idx)."""
    ds = swm.data.load_dataset("ogbench/cube_single_expert.h5", keys_to_cache=["action"])
    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep = ds.get_col_data(col)
    step = ds.get_col_data("step_idx")
    target_ep = ep[idx]
    rows = np.where(ep == target_ep)[0]
    rows = rows[np.argsort(step[rows])]
    pos = int(np.where(rows == idx)[0][0])
    sel = rows[pos:pos + nframes * stride:stride]
    tf = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size),
    ])
    return [tf(ds.get_row_data(int(i))["pixels"]) for i in sel]


def make_gif(args):
    import imageio.v2 as imageio
    frames_px = get_trajectory(args.idx, args.nframes, args.stride)
    arrays = []
    for t, img in enumerate(frames_px):
        fig = build_pyramid_fig(img, args.levels)
        fig.suptitle(f"Pirámide lifting en el tiempo — frame {t + 1:02d}/{len(frames_px):02d} "
                     f"(Haar, L={args.levels})", fontsize=12)
        fig.tight_layout()
        arrays.append(fig_to_array(fig))
        plt.close(fig)
    out = os.path.join(args.out, f"pyramid_traj_idx{args.idx}.gif")
    imageio.mimsave(out, arrays, duration=1.0 / args.fps, loop=0)
    print(f"GIF pirámide lifting ({len(arrays)} frames) ->", out)


def make_tokens_gif(args):
    """GIF de las normas de tokens del encoder entrenado por escala, en el tiempo."""
    import imageio.v2 as imageio
    model = swm.wm.utils.load_pretrained(args.ckpt).to("cpu").eval()
    frames = get_trajectory(args.idx, args.nframes, args.stride)

    norms, grids = [], None
    with torch.no_grad():
        for img in frames:
            po = model.encoder(img.unsqueeze(0))
            grids = [s.grid_hw for s in po.scales]
            norms.append([s.tokens[0].norm(dim=-1).reshape(s.grid_hw).float().cpu().numpy()
                          for s in po.scales])
    ns = len(grids)
    # clim fijo por escala (percentiles sobre toda la trayectoria) para que el
    # cambio sea visible sin saturar: vmin/vmin robustos a outliers.
    pool = [np.concatenate([n[k].ravel() for n in norms]) for k in range(ns)]
    vmin = [float(np.percentile(p, 2)) for p in pool]
    vmax = [float(np.percentile(p, 98)) or 1.0 for p in pool]

    fig, axes = plt.subplots(1, ns + 1, figsize=(3.2 * (ns + 1), 3.8))
    h_orig = axes[0].imshow(frames[0].permute(1, 2, 0).numpy())
    axes[0].set_title("original", fontsize=10)
    axes[0].axis("off")
    handles = []
    for k in range(ns):
        gh, gw = grids[k]
        h = axes[k + 1].imshow(norms[0][k], cmap="viridis", vmin=vmin[k], vmax=vmax[k])
        axes[k + 1].set_title(f"escala {k} · {gh}×{gw}", fontsize=10)
        axes[k + 1].axis("off")
        handles.append(h)
    st = fig.suptitle("Tokens del encoder por escala", fontsize=12)
    fig.tight_layout()

    arrays = []
    for t, img in enumerate(frames):
        h_orig.set_data(img.permute(1, 2, 0).numpy())
        for k in range(ns):
            handles[k].set_data(norms[t][k])
        st.set_text(f"Tokens del encoder por escala — frame {t + 1}/{len(frames)} "
                    f"(norma, clim fijo por escala)")
        arrays.append(fig_to_array(fig))
    plt.close(fig)
    out = os.path.join(args.out, f"tokens_traj_idx{args.idx}.gif")
    imageio.mimsave(out, arrays, duration=1.0 / args.fps, loop=0)
    print(f"GIF tokens por escala ({len(arrays)} frames) ->", out)


def tokens_fig(model, img: torch.Tensor, out: str):
    device = next(model.parameters()).device
    with torch.no_grad():
        po = model.encoder(img.unsqueeze(0).to(device))
    scales = po.scales
    fig, axes = plt.subplots(1, len(scales), figsize=(4 * len(scales), 4.2))
    for ax, s in zip(axes, scales):
        gh, gw = s.grid_hw
        norm = s.tokens[0].norm(dim=-1).reshape(gh, gw).float().cpu().numpy()
        ax.imshow(norm, cmap="viridis")
        ax.set_title(f"escala {s.level} · {gh}×{gw} = {gh*gw} tokens", fontsize=10)
        ax.axis("off")
    fig.suptitle("Norma de tokens por escala (encoder entrenado): estructura espacial preservada, sin pooling global",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="liftwm_vit_cube/weights_epoch_1.pt")
    ap.add_argument("--out", type=str, default="outputs/liftwm_viz")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--gif", action="store_true", help="GIF de la pirámide lifting en el tiempo")
    ap.add_argument("--tokens-gif", action="store_true", help="GIF de tokens del encoder por escala")
    ap.add_argument("--nframes", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.gif:
        make_gif(args)
        return

    if args.tokens_gif:
        make_tokens_gif(args)
        return

    img = get_frame(args.idx)

    pout = os.path.join(args.out, f"pyramid_idx{args.idx}.png")
    pyramid_fig(img, args.levels, pout)
    print("pirámide lifting ->", pout)

    model = swm.wm.utils.load_pretrained(args.ckpt).to("cpu").eval()
    tout = os.path.join(args.out, f"tokens_idx{args.idx}.png")
    tokens_fig(model, img, tout)
    print("tokens por escala ->", tout)


if __name__ == "__main__":
    main()
