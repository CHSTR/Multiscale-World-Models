"""Recolección de datos offline para MultiScaleDial en formato H5 swm
(esquema clon de tworoom.h5: mismas columnas, compresión blosc).

Train y val en archivos separados con estados disjuntos: los episodios de
val usan seeds con offset 10_000_000 (split por episodio, no por ventana).

Política: controlador proporcional ruidoso hacia waypoints aleatorios
(en p y θ) + fracción de acciones uniformes (spec §6.3).

La última fila de cada episodio queda sin sucesor (par (pixels_t, action_t)
sin s_{t+1}): válido para ventanas JEPA, documentado en el plan.

Uso:
  /home/chr/le-wm/.venv/bin/python swm_msdial/collect.py --levels 2 \
      --style balanced --episodes 1800 --val-episodes 200 --steps 100 \
      --out-dir ~/.stable_worldmodel/datasets/msdial
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm_msdial.env import MultiScaleDialEnv, scripted_policy  # noqa: E402

VAL_SEED_OFFSET = 10_000_000

# Esquema exacto (Task 2 del plan): nombre -> (dtype, extra)
SCHEMA = {
    "pixels": np.uint8,
    "action": np.float32,
    "state": np.float64,
    "observation": np.float64,
    "proprio": np.float64,
    "ep_idx": np.int32,
    "step_idx": np.int64,
    "ep_len": np.int32,
    "ep_offset": np.int64,
    "id": np.int64,
    "reward": np.float64,
}


def _write_h5(path: Path, episodes, steps: int, levels, style, seed: int,
              val: bool):
    env = MultiScaleDialEnv(levels=tuple(levels), dial_style=style,
                            max_steps=steps)
    rng = np.random.default_rng(seed)
    n = episodes * steps
    with h5py.File(path, "w") as f:
        ds = {}
        for col, dt in SCHEMA.items():
            shape = (episodes,) if col in ("ep_len", "ep_offset") else (n,)
            if col == "pixels":
                ds[col] = f.create_dataset(
                    col, (n, env.H, env.H, 3), dtype=dt,
                    chunks=(min(100, n), env.H, env.H, 3),
                    **hdf5plugin.Blosc())
            elif col == "action":
                ds[col] = f.create_dataset(col, (n, 2 + env.K), dtype=dt,
                                           chunks=(min(1000, n), 2 + env.K))
            elif col in ("state",):
                ds[col] = f.create_dataset(col, (n, 2 + env.K), dtype=dt)
            elif col in ("observation", "proprio"):
                ds[col] = f.create_dataset(col, (n, 2 + 2 * env.K), dtype=dt)
            else:
                ds[col] = f.create_dataset(col, shape, dtype=dt)

        row = 0
        for ep in range(episodes):
            ep_seed = int(rng.integers(1 << 31))
            if val:
                ep_seed += VAL_SEED_OFFSET
            obs, info = env.reset(seed=ep_seed)
            st = info["state"]
            target = env.sample_state(rng)
            for t in range(steps):
                if t % 25 == 0:
                    target = env.sample_state(rng)
                if rng.random() < 0.3:
                    act = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                else:
                    act = scripted_policy(st, target, rng)
                obs_arr = np.asarray(obs["pixels"], dtype=np.uint8)
                ds["pixels"][row] = obs_arr
                ds["action"][row] = act
                ds["state"][row] = st
                obs_v = np.asarray(obs["state"], dtype=np.float64)
                ds["observation"][row] = obs_v
                ds["proprio"][row] = obs_v
                ds["ep_idx"][row] = ep
                ds["step_idx"][row] = t
                ds["id"][row] = row
                ds["reward"][row] = 0.0
                row += 1
                obs, _, _, _, info = env.step(act)
                st = info["state"]
            ds["ep_len"][ep] = steps
            ds["ep_offset"][ep] = ep * steps
    return path


def collect(levels, style, episodes, val_episodes, steps, seed, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = "+".join(str(x) for x in sorted(levels))
    t0 = time.time()
    train_path = _write_h5(out / f"msdial_train_k{k}_{style}.h5",
                           episodes, steps, levels, style, seed, val=False)
    val_path = _write_h5(out / f"msdial_val_k{k}_{style}.h5",
                         val_episodes, steps, levels, style, seed, val=True)
    dt = time.time() - t0
    n = (episodes + val_episodes) * steps
    print(f"{n} frames en {dt:.1f}s ({n / dt:.0f} fps)")
    print("train ->", train_path)
    print("val   ->", val_path)
    return train_path, val_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, nargs="+", default=[2])
    p.add_argument("--style", default="balanced")
    p.add_argument("--episodes", type=int, default=1800)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--random-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir",
                   default="~/.stable_worldmodel/datasets/msdial")
    a = p.parse_args()
    collect(a.levels, a.style, a.episodes, a.val_episodes, a.steps, a.seed,
            Path(a.out_dir).expanduser())


if __name__ == "__main__":
    main()
