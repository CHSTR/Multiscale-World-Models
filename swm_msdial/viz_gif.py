"""GIF de episodios de eval de MultiScaleDial.

Lee trace_pixels + proprio + terminated del traces.npz de un run y el frame
meta del H5 de eval (fila 0 = inicio, fila 25 = meta). Genera un GIF por
episodio: izquierda = rollout con estela del agente, derecha = meta.

Requiere que la eval se haya corrido con captura de píxeles
(trace_pixels presente; ver viz/eval_traces.py).

Uso:
  /home/chr/le-wm/.venv/bin/python swm_msdial/viz_gif.py \
      --run msdial_k2_balanced__lewm__s0 \
      --eval msdial/msdial_eval_k2_balanced_fine.h5 \
      --out-dir reports/msdial_gifs
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

CK = Path.home() / ".stable_worldmodel/checkpoints"
DS = Path.home() / ".stable_worldmodel/datasets"
UP = 2  # upscale de frames 112 -> 224


def draw_frame(px, goal_px, trail, goal_xy, t, success, size=224):
    img = Image.fromarray(px).resize((size, size), Image.NEAREST)
    g = Image.fromarray(goal_px).resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (2 * size + 8, size), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(g, (size + 8, 0))
    d = ImageDraw.Draw(canvas)

    # estela del agente (coords normalizadas -> px del frame 112 -> upscale)
    if len(trail) > 1:
        pts = [(float(x) * (size / 224) * 224, float(y) * (size / 224) * 224)
               for x, y in [(p[0], p[1]) for p in trail]]
        pts = [(x, y) for x, y in pts]
        d.line(pts, fill=(255, 220, 0), width=2)
    # meta (★) en el frame derecho
    gx, gy = float(goal_xy[0]) * size, float(goal_xy[1]) * size
    r = 6
    d.line([(gx - r, gy), (gx + r, gy)], fill=(0, 0, 0), width=2)
    d.line([(gx, gy - r), (gx, gy + r)], fill=(0, 0, 0), width=2)
    d.rectangle([4, 4, 150, 22], fill=(255, 255, 255))
    d.text((8, 7), f"step {t}  {'OK' if success else ''}", fill=(0, 0, 0))
    d.text((size + 12, 7), "meta", fill=(0, 0, 0))
    return np.asarray(canvas)


def frames_for_episode(tr, goal_px, goal_xy, ep, success):
    T = tr["pixels"].shape[0]
    out, trail = [], []
    for t in range(T):
        trail.append(tr["proprio"][t, ep])
        out.append(draw_frame(tr["pixels"][t, ep], goal_px, trail, goal_xy,
                              t, success))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="msdial_k2_balanced__lewm__s0")
    ap.add_argument("--eval", default="msdial/msdial_eval_k2_balanced_fine.h5")
    ap.add_argument("--out-dir", default="reports/msdial_gifs")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    run_dir = CK / a.run
    tr = {k.replace("trace_", ""): v for k, v in np.load(run_dir / "traces.npz").items()}
    assert "pixels" in tr, "traces.npz no tiene trace_pixels: re-correr la eval"
    lines = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    name = Path(a.eval).name
    res = next(j for j in reversed(lines)
               if Path(j["dataset_name"]).name == name)
    succ = np.asarray(res["successes"], bool)

    with h5py.File(DS / a.eval) as f:
        goal_px = np.asarray(f["pixels"][25])
        # el par (start, goal) vive en el propio H5: filas 0 y 25 de raw_state
        lens = f["ep_len"][()]
        offs = f["ep_offset"][()]
        goal_xy_eps = np.stack([f["raw_state"][int(o + L - 1)][:2]
                                for o, L in zip(offs, lens)])
    goal_px = goal_px[::2, ::2]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    if succ.any():
        picks.append(("exito", int(np.where(succ)[0][0])))
    if (~succ).any():
        picks.append(("fallo", int(np.where(~succ)[0][0])))
    for tag, ep in picks[:2]:
        goal_xy = goal_xy_eps[ep]
        frames = frames_for_episode(tr, goal_px, goal_xy, ep, bool(succ[ep]))
        mode = Path(a.eval).stem.split("_")[-1]
        path = out / f"{a.run}__{mode}__ep{ep}_{tag}.gif"
        imageio.mimsave(path, frames, duration=1 / a.fps, loop=0)
        print(f"[{tag}] ep {ep} -> {path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
