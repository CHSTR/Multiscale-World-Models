"""Eval CEM de un checkpoint sobre entornos renderizados por stable-envs.

Soporta presets livianos sin MuJoCo (``pusht`` y ``tworoom``) y un YAML custom
vía ``--env-config``. Envuelve un entorno de ``stable-envs`` en la interfaz de
tarea de ``stable-worldmodel`` y lo evalúa con ``swm.World`` + ``CEMSolver``.

Ejemplos:
    PYTHONPATH=. python eval_snv.py --env pusht --data datasets/pusht_state.h5 \\
      --policy ab_pusht_online_lewm_slow_svreg/weights_epoch_10.pt \\
      --num-eval 50 --budget 50 --samples 300 --steps 30 --topk 30

    PYTHONPATH=. python eval_snv.py --env tworoom --data datasets/tworoom_state.h5 \\
      --policy random --num-eval 8 --budget 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ENV_ID = "stable-envs-eval/Task-v0"

PRESETS: dict[str, dict[str, Any]] = {
    "pusht": {
        "default_filename": "pusht_state.h5",
        "fallback_filenames": ["pusht_expert_train.h5"],
        "env_cfg": {
            "id": "PushT",
            "kwargs": {},
            "state_column": "state",
            "leaves": [
                {"leaf": "current.body.agent.position", "columns": [0, 2]},
                {"leaf": "current.body.block.position", "columns": [2, 4]},
                {"leaf": "current.body.block.angle", "columns": [4, 5]},
            ],
            "zero": ["current.render_segments", "current.render_segment_count"],
        },
        "eval_cfg": {
            "observation_leaf": "current.observation",
            "proprio_columns": [0, 1, 5, 6],
            "reset_options": {"state": "state", "goal_state": "goal_state"},
            "keys_to_cache": ["action", "proprio", "state"],
            "seed": 42,
        },
    },
    "tworoom": {
        "default_filename": "tworoom_state.h5",
        "fallback_filenames": ["tworoom.h5"],
        "env_cfg": {
            "id": "TwoRoom",
            "kwargs": {},
            "state_column": "proprio",
            "leaves": [
                {"leaf": "agent/position", "columns": [0, 2]},
            ],
            "zero": [],
        },
        "eval_cfg": {
            "observation_leaf": "agent/position",
            "proprio_columns": [0, 1],
            "reset_options": {"state": "proprio", "target_state": "goal_proprio"},
            "keys_to_cache": ["action", "proprio"],
            "seed": 42,
        },
    },
}

# Backwards-compatible aliases (default PushT preset)
ENV_CFG = PRESETS["pusht"]["env_cfg"]
EVAL_CFG = PRESETS["pusht"]["eval_cfg"]


def resolve_data_path(env_name: str, explicit_data: str | None, cache_dir: str | None = None) -> Path:
    """Resolve dataset path portably using --data, cache_dir, $LOCAL_DATASET_DIR, or $STABLEWM_HOME."""
    if explicit_data:
        p = Path(explicit_data).expanduser()
        if p.exists():
            return p.resolve()

    preset = PRESETS[env_name]
    filenames = [explicit_data] if explicit_data else []
    filenames += [preset["default_filename"], *preset.get("fallback_filenames", ())]

    roots: list[Path] = []
    for raw in (cache_dir, os.environ.get("LOCAL_DATASET_DIR"), os.environ.get("STABLEWM_HOME"), "~/.stable_worldmodel"):
        if raw:
            base = Path(raw).expanduser()
            roots.extend([base / "datasets", base])

    for root in roots:
        for fname in filenames:
            if not fname:
                continue
            cand = root / fname
            if cand.is_file():
                return cand.resolve()
            if cand.is_dir() and (cand / Path(fname).name).is_file():
                return (cand / Path(fname).name).resolve()

    searched = ", ".join(str(r) for r in roots[:4])
    raise FileNotFoundError(
        f"No se encontró el dataset para env={env_name!r} (buscado: {filenames} en {searched}). "
        "Pasá --data /ruta/al/archivo.h5 o configurá LOCAL_DATASET_DIR / STABLEWM_HOME."
    )


def get_preset_configs(env_name: str, env_config_path: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (env_cfg, eval_cfg) from built-in preset or an external YAML file."""
    if env_config_path:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(OmegaConf.load(env_config_path), resolve=True)
        env_cfg = dict(raw["env_cfg"] if "env_cfg" in raw else raw["render"]["env"])
        if "state_column" not in env_cfg and "render" in raw:
            env_cfg["state_column"] = raw["render"]["state_column"]
        eval_cfg = dict(raw.get("eval_cfg") or PRESETS[env_name]["eval_cfg"])
        return env_cfg, eval_cfg

    preset = PRESETS[env_name]
    return dict(preset["env_cfg"]), dict(preset["eval_cfg"])


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
                self._goal = self.renderer(torch.as_tensor(self._goal_row)[None])[0].cpu().numpy()
            else:
                self._goal = self.render()
            return self._observation(state), self._info()

        def step(self, action):
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--env", choices=sorted(PRESETS.keys()), default="pusht",
                   help="Preset de entorno stable-envs (pusht | tworoom)")
    p.add_argument("--env-config", default=None,
                   help="YAML opcional para sobreescribir env_cfg/eval_cfg")
    p.add_argument("--data", default=None,
                   help="Ruta al .h5 state-only (o nombre relativo bajo LOCAL_DATASET_DIR/STABLEWM_HOME)")
    p.add_argument("--cache-dir", default=None,
                   help="Raíz de cache para checkpoints/datasets (por defecto $STABLEWM_HOME)")
    p.add_argument("--policy", default="random",
                   help="Ruta o nombre relativo del .pt con config.json al lado, o 'random'")
    p.add_argument("--num-eval", type=int, default=8)
    p.add_argument("--goal-offset", type=int, default=20)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--env-device", default="cpu")
    p.add_argument("--out", default=None, help="Archivo JSONL donde anexar métricas")
    return p


def main() -> None:
    a = build_parser().parse_args()

    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from gymnasium.envs.registration import register, registry
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    env_cfg, eval_cfg = get_preset_configs(a.env, a.env_config)
    eval_cfg["seed"] = a.seed
    data_path = resolve_data_path(a.env, a.data, cache_dir=a.cache_dir)

    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point=make_task_env)

    dataset = swm.data.load_dataset(str(data_path),
                                    keys_to_cache=list(eval_cfg["keys_to_cache"]))
    names = set(dataset.column_names) | set(getattr(dataset, "_schema_names", ()))
    episode_col = "episode_idx" if "episode_idx" in names else "ep_idx"
    episodes = dataset.get_col_data(episode_col)
    steps = dataset.get_col_data("step_idx")

    process = {}
    for col in eval_cfg["keys_to_cache"]:
        data = dataset.get_col_data(col)
        valid_rows = data[~np.isnan(data).any(axis=1)] if data.ndim > 1 else data[~np.isnan(data)]
        scaler = preprocessing.StandardScaler().fit(valid_rows)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    _, inverse = np.unique(episodes, return_inverse=True)
    last = np.zeros(inverse.max() + 1, np.int64)
    np.maximum.at(last, inverse, steps)
    max_start = last[inverse] - a.goal_offset
    valid = np.nonzero(steps <= max_start)[0]
    rng = np.random.default_rng(a.seed)
    n_eval = min(a.num_eval, len(valid))
    picked = np.sort(valid[rng.choice(len(valid), size=n_eval, replace=False)])
    eval_episodes, eval_starts = episodes[picked], steps[picked]

    world = swm.World(
        ENV_ID,
        num_envs=n_eval,
        image_shape=(224, 224),
        max_episode_steps=2 * a.budget,
        env_cfg=env_cfg,
        eval_cfg=eval_cfg,
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
        policy_path = Path(a.policy).expanduser()
        if not policy_path.is_file() and a.cache_dir:
            cand = Path(a.cache_dir).expanduser() / "checkpoints" / a.policy
            if cand.is_file():
                policy_path = cand
        model_ref = str(policy_path) if policy_path.is_file() else a.policy
        model = swm.wm.utils.load_pretrained(model_ref)
        model = model.to(a.device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        solver = swm.planning.CEMSolver(
            cost=model, batch_size=1, num_samples=a.samples,
            n_steps=a.steps, topk=a.topk, device=a.device, seed=a.seed,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=a.horizon,
                receding_horizon=a.receding_horizon,
                action_block=a.action_block,
            ),
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

    ep_succ = [bool(x) for x in metrics["episode_successes"]]
    result = {
        "env": a.env,
        "dataset_name": str(data_path),
        "policy": a.policy,
        "seed": a.seed,
        "num_eval": n_eval,
        "success_rate": float(metrics["success_rate"]),
        "successes": ep_succ,
        "episode_successes": ep_succ,
        "seconds": round(seconds, 1),
    }
    print(json.dumps(result))
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
