"""Oráculo dentro del wrapper REAL de swm (spec §7.2, Task 4 Step 6).

Corre con el venv de entrenamiento:
  /home/chr/le-wm/.venv/bin/python swm_msdial/smoke_eval_integration.py

Ejercita el camino completo de la eval PERO con el controlador oráculo
(estado verdadero) en lugar del CEM: registro gym + swm.World + MegaWrapper
+ `_extract_init_goal` + `_apply_callables` con los callables del config +
terminated/ever_* a través del pool. 50 pasos de entorno (presupuesto eval).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    import stable_worldmodel as swm
    import swm_msdial  # noqa: F401 — registra swm/MultiScaleDial-v0
    from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

    from swm_msdial.env import scripted_policy
    from swm_msdial.gen_eval_set import gen_eval_set

    N = 8
    BUDGET = 50
    out_dir = Path("/tmp/opencode/msdial_t4_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_eval_set((2,), "balanced", N, seed=0, out_dir=out_dir,
                 modes=("fine",))
    eval_file = out_dir / "msdial_eval_k2_balanced_fine.h5"

    world = swm.World(
        env_name="swm/MultiScaleDial-v0",
        num_envs=N,
        image_shape=(224, 224),
        max_episode_steps=2 * BUDGET,
        levels=(2,),
        dial_style="balanced",
    )

    ds = swm.data.load_dataset(str(eval_file), keys_to_cache=["action"])
    init, goal, _ = _extract_init_goal(
        ds, np.arange(N), np.zeros(N, dtype=int), np.full(N, 25, dtype=int))

    world.reset(seed=init.get("seed"))

    callables = [
        {"method": "_set_state",
         "args": {"state": {"value": "raw_state"}}},
        {"method": "_set_goal_state",
         "args": {"goal_state": {"value": "goal_raw_state"}}},
    ]
    merged = {**init, **goal}
    for i in range(N):
        env_init = {k: v[i] for k, v in merged.items()}
        _apply_callables(world.envs.envs[i].unwrapped, callables, env_init)

    rng = np.random.default_rng(0)
    goals = goal["goal_raw_state"]
    alive = np.ones(N, dtype=bool)
    ever = np.zeros(N, dtype=bool)
    for t in range(BUDGET):
        acts = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            if not alive[i]:
                continue
            st = np.asarray(world.infos["agent_state"][i]).ravel()
            acts[i] = scripted_policy(st, goals[i], rng, noise=0.0)
        _, _, terms, _, infos = world.envs.step(
            acts, mask=alive if not alive.all() else None)
        ever |= terms
        alive &= ~terms

    ever_pos = np.asarray(infos["ever_pos"]).ravel()[:N]
    ever_dial = np.asarray(infos["ever_dial"]).ravel()[:N]
    print(f"terminated (OR sobre 50 pasos): {ever.sum()}/{N}")
    print(f"ever_pos: {ever_pos.sum()}/{N}   ever_dial: {ever_dial.sum()}/{N}")
    assert ever.sum() >= N - 1, f"oráculo debería terminar casi todo: {ever}"
    assert ever_dial.sum() >= N - 1, f"dial no alcanzado: {ever_dial}"
    print("SMOKE OK: pipeline de eval completo funciona con el wrapper real")


if __name__ == "__main__":
    main()
