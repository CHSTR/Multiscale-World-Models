#!/usr/bin/env python3
"""Utilidad liviana para datasets state-only compatibles con ``stable-envs`` online render.

Dos modos:
1. ``strip``: copia un HDF5 de stable-worldmodel eliminando columnas ``pixels*``
   (p. ej. reduce PushT de varios GB a ~159 MB).
2. ``collect``: recolecta episodios rápidos state-only directamente desde
   ``stable-envs`` (``tworoom`` o ``pusht``) con política aleatoria/suave,
   útil para pruebas rápidas o smoke tests en Google Colab sin descargar GB.

Ejemplos:
    python scripts/make_states_only_h5.py strip pusht_expert_train.h5 pusht_state.h5
    python scripts/make_states_only_h5.py collect --env tworoom --episodes 200 --steps 100 \\
        --out swm_home/datasets/tworoom_state.h5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def strip_pixels(src: Path, dst: Path, episodes: int | None = None) -> dict:
    import h5py

    t0 = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        ep_len = fi["ep_len"][:]
        if episodes is not None:
            ep_len = ep_len[:episodes]
        ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        rows = int(ep_len.sum())
        fo.create_dataset("ep_len", data=ep_len.astype(np.int32))
        fo.create_dataset("ep_offset", data=ep_offset)
        kept = []
        for key in fi.keys():
            if key in ("ep_len", "ep_offset") or key.startswith("pixels"):
                continue
            src_ds = fi[key]
            dst_ds = fo.create_dataset(key, shape=(rows, *src_ds.shape[1:]), dtype=src_ds.dtype)
            for start in range(0, rows, 1 << 16):
                stop = min(rows, start + (1 << 16))
                dst_ds[start:stop] = src_ds[start:stop]
            kept.append(key)
    elapsed = time.perf_counter() - t0
    return {"rows": rows, "episodes": len(ep_len), "columns": kept, "seconds": round(elapsed, 2)}


def collect_states(env_name: str, dst: Path, episodes: int, steps: int, seed: int = 0, device: str = "cpu") -> dict:
    import h5py
    import stable_envs as snv
    import torch

    t0 = time.perf_counter()
    env_key = env_name.lower()
    if env_key == "tworoom":
        snv_id = "TwoRoom"
    elif env_key == "pusht":
        snv_id = "PushT"
    else:
        raise ValueError(f"Unsupported --env {env_name!r}; expected 'tworoom' or 'pusht'")

    rng = np.random.default_rng(seed)
    env = snv.make(snv_id, num_envs=1, device=device)
    spec = env.action_spec
    low = np.broadcast_to(np.asarray(spec.low, np.float32), spec.shape)
    high = np.broadcast_to(np.asarray(spec.high, np.float32), spec.shape)
    act_dim = int(np.prod(spec.shape))

    all_actions: list[np.ndarray] = []
    all_proprio: list[np.ndarray] = []
    all_state: list[np.ndarray] = []
    all_ep_idx: list[np.ndarray] = []
    all_step_idx: list[np.ndarray] = []
    ep_lens: list[int] = []

    try:
        for ep in range(episodes):
            state = env.reset(seed=int(seed + ep))
            # Smooth random walk / persistent actions for better coverage
            cur_act = rng.uniform(low, high).astype(np.float32)
            ep_acts = np.zeros((steps, act_dim), dtype=np.float32)
            ep_props = []
            ep_states = []

            for t in range(steps):
                if env_key == "tworoom":
                    pos = snv.get_field(state, "agent/position")[0].detach().cpu().numpy().astype(np.float32)
                    ep_props.append(pos)
                    ep_states.append(pos)
                else:
                    obs = snv.get_field(state, "current.observation")[0].detach().cpu().numpy().astype(np.float32)
                    ep_states.append(obs)
                    ep_props.append(obs[[0, 1, 5, 6]])

                if t % 5 == 0 or rng.random() < 0.25:
                    cur_act = rng.uniform(low, high).astype(np.float32)
                ep_acts[t] = cur_act
                act_t = torch.as_tensor(cur_act, dtype=spec.dtype, device=env.device).reshape(1, -1)
                out = env.step(act_t, frames=None)
                state = out["state"]

            ep_acts[-1] = np.nan  # convention: last step has no outgoing action
            all_actions.append(ep_acts)
            all_proprio.append(np.stack(ep_props, axis=0))
            all_state.append(np.stack(ep_states, axis=0))
            all_ep_idx.append(np.full(steps, ep, dtype=np.int64))
            all_step_idx.append(np.arange(steps, dtype=np.int64))
            ep_lens.append(steps)
    finally:
        env.close()

    ep_len_arr = np.asarray(ep_lens, dtype=np.int32)
    ep_offset_arr = np.concatenate([[0], np.cumsum(ep_len_arr)[:-1]]).astype(np.int64)
    actions_arr = np.concatenate(all_actions, axis=0)
    proprio_arr = np.concatenate(all_proprio, axis=0)
    state_arr = np.concatenate(all_state, axis=0)
    ep_idx_arr = np.concatenate(all_ep_idx, axis=0)
    step_idx_arr = np.concatenate(all_step_idx, axis=0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst, "w") as fo:
        fo.create_dataset("ep_len", data=ep_len_arr)
        fo.create_dataset("ep_offset", data=ep_offset_arr)
        fo.create_dataset("episode_idx", data=ep_idx_arr)
        fo.create_dataset("step_idx", data=step_idx_arr)
        fo.create_dataset("action", data=actions_arr)
        fo.create_dataset("proprio", data=proprio_arr)
        if env_key == "pusht":
            fo.create_dataset("state", data=state_arr)

    elapsed = time.perf_counter() - t0
    return {
        "env": snv_id,
        "rows": int(ep_len_arr.sum()),
        "episodes": episodes,
        "steps": steps,
        "dst": str(dst),
        "seconds": round(elapsed, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_strip = sub.add_parser("strip", help="Strip pixels* columns from an existing HDF5 dataset")
    p_strip.add_argument("src", type=Path)
    p_strip.add_argument("dst", type=Path)
    p_strip.add_argument("--episodes", type=int, default=None)

    p_col = sub.add_parser("collect", help="Collect a lightweight state-only HDF5 dataset via stable-envs")
    p_col.add_argument("--env", choices=["tworoom", "pusht"], required=True)
    p_col.add_argument("--episodes", type=int, default=200)
    p_col.add_argument("--steps", type=int, default=100)
    p_col.add_argument("--seed", type=int, default=0)
    p_col.add_argument("--device", default="cpu")
    p_col.add_argument("--out", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "strip":
        info = strip_pixels(args.src, args.dst, episodes=args.episodes)
    else:
        info = collect_states(args.env, args.out, episodes=args.episodes, steps=args.steps, seed=args.seed, device=args.device)
    print(info)


if __name__ == "__main__":
    main()
