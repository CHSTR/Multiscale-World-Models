import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, __file__[:__file__.rfind("/viz")])

import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from viz.eval_traces import evaluate_with_traces, save_traces, format_summary
from diagnostics.perturb import ImagePerturbation


def img_transform(cfg, perturb: bool = False):
    """ToImage -> [0, 1] -> (perturbación opcional) -> Normalize ImageNet -> Resize."""
    steps = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    pcfg = cfg.get("perturb") or {}
    if perturb and pcfg.get("kind", "none") != "none":
        steps.append(ImagePerturbation(pcfg.get("kind"), pcfg.get("strength", 0.0), seed=cfg.seed))
    steps += [
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ]
    return transforms.Compose(steps)


def apply_band_ablation(model, cfg) -> list | None:
    """Ablación de bandas del front-end en test (ej. ablation.bands=[w1] ablation.mode=zero)."""
    acfg = cfg.get("ablation") or {}
    bands = acfg.get("bands")
    if bands is None or (hasattr(bands, "__len__") and len(bands) == 0):
        return None
    enc = model.encoder
    if not hasattr(enc, "set_band_ablation"):
        raise ValueError(f"{type(enc).__name__} no soporta ablación de bandas")
    bands = list(bands) if not isinstance(bands, (str, int)) else [bands]
    idx = enc.set_band_ablation(bands, mode=acfg.get("mode", "zero"), seed=cfg.seed)
    names = [enc.band_names[i] for i in idx]
    print(f"[ablation] bandas {names} (idx={idx}) modo={acfg.get('mode', 'zero')}")
    return names


def run_tag(cfg) -> str:
    """Etiqueta legible de la condición evaluada (para nombrar trazas y filas del jsonl)."""
    tag = (cfg.get("output") or {}).get("tag")
    if tag:
        return str(tag)
    parts = []
    acfg = cfg.get("ablation") or {}
    if acfg.get("bands") is not None:
        b = acfg["bands"]
        b = [b] if isinstance(b, (str, int)) else list(b)
        parts.append(f"abl-{acfg.get('mode', 'zero')}-" + "+".join(map(str, b)))
    pcfg = cfg.get("perturb") or {}
    if pcfg.get("kind", "none") != "none":
        parts.append(f"{pcfg['kind']}{pcfg.get('strength', 0)}")
    return "_".join(parts) or "clean"


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=dataset_path,
        keys_to_cache=cfg.dataset.keys_to_cache,
    )
    return dataset

@hydra.main(version_base=None, config_path="../config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform (la perturbación de robustez aplica a observaciones y, si perturb.goal, al goal)
    pcfg = cfg.get("perturb") or {}
    transform = {
        "pixels": img_transform(cfg, perturb=True),
        "goal": img_transform(cfg, perturb=bool(pcfg.get("goal", True))),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        ablated = apply_band_ablation(model, cfg)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()
        ablated = None

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = evaluate_with_traces(
        world, dataset=dataset,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    end_time = time.time()

    metrics = results["aggregate"]
    print(metrics)

    # Claves para emparejar episodios entre modelos (mismo seed => mismos estados iniciales y goals).
    n_ep = len(eval_episodes)
    results["per_episode"]["episode_idx"] = np.asarray(eval_episodes)
    results["per_episode"]["start_step"] = np.asarray(eval_start_idx)
    results["per_episode"]["eval_seed"] = np.full(n_ep, cfg.seed)

    tag = run_tag(cfg)
    stem = Path(str(cfg.policy)).stem if cfg.policy != "random" else "random"
    traces_path = results_path / f"traces_{stem}_{tag}_seed{cfg.seed}.npz"
    save_traces(traces_path, results)

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": str(cfg.policy),
        "env": str(cfg.world.env_name),
        "tag": tag,
        "seed": int(cfg.seed),
        "ablation": ablated,
        "ablation_mode": (cfg.get("ablation") or {}).get("mode"),
        "perturb": OmegaConf.to_container(cfg.perturb, resolve=True) if cfg.get("perturb") else None,
        "num_eval": int(cfg.eval.num_eval),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        "plan_horizon": int(cfg.plan_config.horizon),
        "solver_num_samples": int(cfg.solver.get("num_samples", -1)) if cfg.get("solver") else None,
        "traces": str(traces_path),
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
    }
    with (results_path / "results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(format_summary(results["aggregate"]))
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
