"""GIF de las acciones elegidas por el planner CEM (desde traces.npz de un eval).

Los evals guardan `trace_action (n_envs, pasos, action_dim)`: la acción que el
CEM seleccionó y ejecutó en cada paso del mundo. Este script toma un episodio
y anima:

  * frame real de referencia (geo_background_image),
  * barras de la acción actual (dim 0..A-1),
  * mapa de la secuencia completa de acciones (ejes: paso × dim),
  * curva de distancia al goal con marcador en el paso actual.

No re-corre el CEM: usa los traces ya guardados (~segundos).

Uso:
  /home/chr/le-wm/.venv/bin/python viz/liftwm_actions_viz.py \
      --run liftwm_nca_cube --env 0 --fps 8
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import imageio.v2 as imageio


def pick_env(d: dict, env: int | None):
    succ = d["ep_success"]
    if env is not None:
        return int(env)
    ok = np.where(succ)[0]
    return int(ok[0]) if ok.size else 0


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="liftwm_nca_cube",
                    help="nombre de carpeta en ~/.stable_worldmodel/<run>/traces.npz")
    ap.add_argument("--env", type=int, default=None,
                    help="índice de episodio (default: primer episodio exitoso)")
    ap.add_argument("--out", default="outputs/liftwm_viz")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    path = os.path.expanduser(f"~/.stable_worldmodel/{args.run}/traces.npz")
    d = dict(np.load(path, allow_pickle=True))
    env = pick_env(d, args.env)
    acts = d["trace_action"][env]                 # (T, A)
    dist = d["trace_distance"][env]                # (T,)
    bg = d["geo_background_image"][env]            # (H, W, 3)
    T, A = acts.shape
    steps = np.arange(T)
    vmax = float(np.max(np.abs(acts))) or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                            gridspec_kw={"width_ratios": [1.1, 1.0, 2.0, 1.4]})

    # panel 1: frame real
    axes[0].imshow(bg)
    axes[0].set_title("frame real", fontsize=9)
    axes[0].axis("off")

    # panel 2: barras de la acción actual
    bars = axes[1].bar(np.arange(A), acts[0], color="steelblue")
    axes[1].set_ylim(-vmax * 1.1, vmax * 1.1)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xticks(np.arange(A))
    axes[1].set_xlabel("dimensión de acción", fontsize=8)
    axes[1].set_title("acción ejecutada", fontsize=9)

    # panel 3: secuencia completa de acciones (paso × dim)
    im = axes[2].imshow(acts.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    vline = axes[2].axvline(0, color="white", lw=1.5)
    axes[2].set_xlabel("paso del mundo", fontsize=8)
    axes[2].set_ylabel("dim acción", fontsize=8)
    axes[2].set_title("secuencia de acciones", fontsize=9)

    # panel 4: distancia al goal
    axes[3].plot(steps, dist, color="gray", lw=1)
    pt, = axes[3].plot([0], [dist[0]], "ro", ms=6)
    axes[3].set_xlabel("paso del mundo", fontsize=8)
    axes[3].set_title("distancia al goal", fontsize=9)

    st = fig.suptitle(f"Acciones del planner CEM — episodio {env} (éxito={bool(d['ep_success'][env])})",
                      fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    cax = fig.add_axes([0.955, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("valor de acción", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    frames = []
    for t in range(T):
        for b, v in zip(bars, acts[t]):
            b.set_height(v)
        vline.set_xdata([t, t])
        pt.set_data([t], [dist[t]])
        st.set_text(f"Acciones del planner CEM — episodio {env} · paso {t + 1}/{T} "
                    f"(éxito={bool(d['ep_success'][env])})")
        frames.append(fig_to_array(fig))
    plt.close(fig)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"actions_{args.run}_env{env}.gif")
    imageio.mimsave(out, frames, duration=1.0 / args.fps, loop=0)
    print(f"GIF acciones ({T} pasos, env={env}) ->", out)


if __name__ == "__main__":
    main()
