"""Sets de eval fijos para MultiScaleDial (spec §3: pareado).

Cada episodio del set tiene EXACTAMENTE `goal_offset + 1` = 26 filas:
fila 0 = (pixels render(start), state=start), filas 1..24 = filler estático
en start, fila 25 = (pixels render(goal), state=goal). La eval de swm usa
`arr[0]` como init y `arr[-1]` como goal (`_extract_init_goal`), así que el
filler no contamina nada; estos episodios NUNCA se usan para entrenar.

Los mismos (start, goal) para todos los modelos: el archivo se genera una
vez y se comparte; meta en `eval_meta.json`.

Uso:
  /home/chr/le-wm/.venv/bin/python swm_msdial/gen_eval_set.py --levels 2 \
      --style balanced --episodes 200 --out-dir ~/.stable_worldmodel/datasets/msdial
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm_msdial.env import EPS_P, EPS_TH, MultiScaleDialEnv  # noqa: E402

GOAL_OFFSET = 25


def gen_eval_set(levels, style, episodes, seed, out_dir,
                 modes=("coarse", "fine", "both")):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = "+".join(str(x) for x in sorted(levels))
    env = MultiScaleDialEnv(levels=tuple(levels), dial_style=style)
    rng = np.random.default_rng(seed)
    n_rows = episodes * (GOAL_OFFSET + 1)

    meta = {
        "goal_offset": GOAL_OFFSET,
        "tolerances": {"eps_p": EPS_P, "eps_th": EPS_TH},
        "levels": list(levels),
        "dial_style": style,
        "seed": seed,
        "sets": {},
    }

    def write(mode, pairs):
        name = f"msdial_eval_k{k}_{style}_{mode}.h5"
        with h5py.File(out / name, "w") as f:
            ds_px = f.create_dataset(
                "pixels", (n_rows, 224, 224, 3), np.uint8,
                chunks=(min(100, n_rows), 224, 224, 3), **hdf5plugin.Blosc())
            ds_st = f.create_dataset("raw_state", (n_rows, 2 + env.K), np.float64)
            ds_ob = f.create_dataset("observation", (n_rows, 2 + 2 * env.K),
                                     np.float64)
            ds_pr = f.create_dataset("proprio", (n_rows, 2 + 2 * env.K),
                                     np.float64)
            ds_ac = f.create_dataset("action", (n_rows, 2 + env.K), np.float32)
            ds_ep = f.create_dataset("ep_idx", (n_rows,), np.int32)
            ds_si = f.create_dataset("step_idx", (n_rows,), np.int64)
            ds_el = f.create_dataset("ep_len", (episodes,), np.int32)
            ds_of = f.create_dataset("ep_offset", (episodes,), np.int64)
            ds_id = f.create_dataset("id", (n_rows,), np.int64)
            ds_rw = f.create_dataset("reward", (n_rows,), np.float64)

            for i, (start, goal) in enumerate(pairs):
                row = i * (GOAL_OFFSET + 1)
                # fila 0: start
                env.reset(seed=int(rng.integers(1 << 31)))
                env.set_state(start)
                px_start = np.asarray(env.render(), dtype=np.uint8)
                ob_start = np.asarray(env._obs()["state"], dtype=np.float64)
                # fila 25: goal
                env.set_state(goal)
                px_goal = np.asarray(env.render(), dtype=np.uint8)
                ob_goal = np.asarray(env._obs()["state"], dtype=np.float64)
                for t in range(GOAL_OFFSET + 1):
                    r = row + t
                    last = t == GOAL_OFFSET
                    ds_px[r] = px_goal if last else px_start
                    ds_st[r] = goal if last else start
                    ds_ob[r] = ob_goal if last else ob_start
                    ds_pr[r] = ob_goal if last else ob_start
                    ds_ac[r] = 0.0
                    ds_ep[r] = i
                    ds_si[r] = t
                    ds_id[r] = r
                    ds_rw[r] = 0.0
                ds_el[i] = GOAL_OFFSET + 1
                ds_of[i] = row
        meta["sets"][name] = [
            {"start": [float(x) for x in s], "goal": [float(x) for x in g]}
            for s, g in pairs
        ]
        print(f"{mode:>6}: {episodes} eps -> {out / name}")

    for mode in modes:
        env.reset(seed=seed)
        pairs = []
        for _ in range(episodes):
            start = env.sample_state(rng)
            goal = env.sample_goal(start, mode, rng)
            pairs.append((start.copy(), goal.copy()))
        write(mode, pairs)

    # merge con misma ruta: generar otro nivel no debe borrar las entradas previas
    meta_path = out / "eval_meta.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            sets = prev.get("sets", {})
            sets.update(meta["sets"])
            meta["sets"] = sets
        except (json.JSONDecodeError, AttributeError):
            pass
    meta_path.write_text(json.dumps(meta, indent=2))
    print("meta ->", meta_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[2])
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["coarse", "fine", "both"])
    ap.add_argument("--out-dir",
                   default="~/.stable_worldmodel/datasets/msdial")
    a = ap.parse_args()
    gen_eval_set(a.levels, a.style, a.episodes, a.seed,
                 Path(a.out_dir).expanduser(), tuple(a.modes))


if __name__ == "__main__":
    main()
