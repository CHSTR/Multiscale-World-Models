"""Eval CEM de un checkpoint sobre PushT renderizado por stable-envs.

Adaptación del ejemplo oficial ``stable-envs/examples/lewm/eval.py``: envuelve
un entorno de stable-envs en la interfaz de tarea de stable-worldmodel y lo
pasa por ``swm.World``. El costo CEM es el del propio modelo
(``PyramidJEPA.get_cost``), así que sirve para los modelos multiescala.

    PYTHONPATH=. python eval_snv.py \
      --policy cnms_pusht_online_3ep_b32/weights_epoch_3.pt \
      --num-eval 8 --budget 30 --samples 64 --steps 8 --topk 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ENV_ID = "stable-envs-eval/Task-v0"
DEFAULT_DATA = "/home/chr/stable-envs-test/pusht_state.h5"
DEFAULT_POLICY = "cnms_pusht_online_3ep_b32/weights_epoch_3.pt"

ENV_CFG = {
    "id": "PushT",
    "kwargs": {},
    "state_column": "state",
    "leaves": [
        {"leaf": "current.body.agent.position", "columns": [0, 2]},
        {"leaf": "current.body.block.position", "columns": [2, 4]},
        {"leaf": "current.body.block.angle", "columns": [4, 5]},
    ],
    "zero": ["current.render_segments", "current.render_segment_count"],
}
EVAL_CFG = {
    "observation_leaf": "current.observation",
    "proprio_columns": [0, 1, 5, 6],
    "reset_options": {"state": "state", "goal_state": "goal_state"},
    "keys_to_cache": ["action", "proprio", "state"],
    "seed": 42,
}


def make_task_env(**kwargs):
    """gym.Env de una instancia de stable-envs con la interfaz de tarea de swm."""
    import gymnasium as gym
    import torch
    from stable_envs import get_field

    from online_render import StateRenderer

    class Task(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, env_cfg, eval_cfg, device="cpu", size=224,
                     render_mode="rgb_array"):
            self.render_mode = render_mode
            leaves = [(i["leaf"], *i["columns"]) for i in env_cfg["leaves"]]
            self.renderer = StateRenderer(
                env_cfg["id"], leaves, list(env_cfg.get("zero") or ()),
                device=device, size=size,
                env_kwargs=dict(env_cfg.get("kwargs") or {}),
            )
            self.env = self.renderer.env
            self.state_column = env_cfg["state_column"]
            self.observation_leaf = str(eval_cfg["observation_leaf"])
            self.proprio = list(eval_cfg["proprio_columns"])
            self.reset_columns = dict(eval_cfg["reset_options"])
            spec = self.env.action_spec
            low, high = (np.broadcast_to(np.asarray(v, np.float32), spec.shape)
                         for v in (spec.low, spec.high))
            self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
            dim = int(get_field(self.env.get_state(), self.observation_leaf).shape[-1])
            box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)
            self.observation_space = gym.spaces.Dict(
                {"proprio": box(len(self.proprio)), "state": box(dim)})
            self._goal_row = None
            self._goal = None

        def reset_options_from_dataset(self, init_row, goal_row):
            self._goal_row = np.asarray(goal_row[f"goal_{self.state_column}"], np.float32)
            rows = {**init_row, **goal_row}
            return {name: np.asarray(rows[col], np.float64)
                    for name, col in self.reset_columns.items()}

        def _observation(self, state):
            obs = get_field(state, self.observation_leaf)[0].detach().cpu().numpy().astype(np.float64)
            return {"proprio": obs[self.proprio], "state": obs}

        def _info(self):
            return {"goal": self._goal}

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            state = self.env.reset(seed=seed, options=options)
            if self._goal_row is not None:
                import torch
                self._goal = self.renderer(torch.as_tensor(self._goal_row)[None])[0].cpu().numpy()
            else:
                self._goal = self.render()
            return self._observation(state), self._info()

        def step(self, action):
            import torch
            act = torch.as_tensor(np.asarray(action), dtype=self.env.action_spec.dtype,
                                  device=self.env.device)
            out = self.env.step(act.reshape(1, -1), frames=None)
            return (self._observation(out["state"]), float(out["reward"][0]),
                    bool(out["terminated"][0]), bool(out["truncated"][0]), self._info())

        def render(self):
            size = self.renderer.size
            return self.env.render(**({} if size is None else {"size": size}))[0].cpu().numpy()

        def close(self):
            self.renderer.close()

    return Task(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--policy", default=DEFAULT_POLICY, help=".pt con config.json al lado, o 'random'")
    p.add_argument("--num-eval", type=int, default=8)
    p.add_argument("--goal-offset", type=int, default=20)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--env-device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from gymnasium.envs.registration import register, registry
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point=make_task_env)

    dataset = swm.data.load_dataset(str(Path(a.data).resolve()),
                                    keys_to_cache=list(EVAL_CFG["keys_to_cache"]))
    names = set(dataset.column_names) | set(getattr(dataset, "_schema_names", ()))
    episode_col = "episode_idx" if "episode_idx" in names else "ep_idx"
    episodes = dataset.get_col_data(episode_col)
    steps = dataset.get_col_data("step_idx")

    process = {}
    for col in EVAL_CFG["keys_to_cache"]:
        data = dataset.get_col_data(col)
        scaler = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    _, inverse = np.unique(episodes, return_inverse=True)
    last = np.zeros(inverse.max() + 1, np.int64)
    np.maximum.at(last, inverse, steps)
    max_start = last[inverse] - a.goal_offset
    valid = np.nonzero(steps <= max_start)[0]
    rng = np.random.default_rng(EVAL_CFG["seed"])
    picked = np.sort(valid[rng.choice(len(valid), size=a.num_eval, replace=False)])
    eval_episodes, eval_starts = episodes[picked], steps[picked]

    world = swm.World(
        ENV_ID,
        num_envs=a.num_eval,
        image_shape=(224, 224),
        max_episode_steps=2 * a.budget,
        env_cfg=ENV_CFG,
        eval_cfg=EVAL_CFG,
        device=a.env_device,
        size=224,
    )

    if a.policy == "random":
        policy = swm.policy.RandomPolicy()
    else:
        image = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ])
        model = swm.wm.utils.load_pretrained(a.policy)
        model = model.to(a.device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        solver = swm.planning.CEMSolver(
            cost=model, batch_size=1, num_samples=a.samples,
            n_steps=a.steps, topk=a.topk, device=a.device, seed=EVAL_CFG["seed"],
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
            process=process,
            transform={"pixels": image, "goal": image},
        )
    world.set_policy(policy)

    t0 = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_starts.tolist(),
        goal_offset=a.goal_offset,
        eval_budget=a.budget,
    )
    seconds = time.perf_counter() - t0
    world.close()
    result = {
        "policy": a.policy,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": [bool(x) for x in metrics["episode_successes"]],
        "seconds": round(seconds, 1),
    }
    print(json.dumps(result))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
