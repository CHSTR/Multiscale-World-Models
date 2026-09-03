"""Evaluación con trazas por paso para stable_worldmodel (modo dataset-driven).

Replica el comportamiento de ``World._evaluate_from_dataset`` pero engancha un
``on_step`` que acumula trazas por paso (posición del agente, acción, distancia,
terminación, ...) y deriva métricas por episodio: pasos hasta el éxito, error
mínimo/final, error integrado y magnitud media de la acción.

Además guarda un bloque ``geometry`` con la posición del goal y del inicio y, si
se pide, las imágenes de fondo y del goal, todo en el MISMO sistema de
coordenadas que la trayectoria, para poder superponer directamente.

Pensado para swm/TwoRoom-v1:
  - Señal de error: ``info['distance_to_target']``; éxito si baja del umbral
    (16.0 por defecto). El ``reward`` es siempre 0.0 y NO se usa como error.
  - Sistema de coordenadas tipo imagen: origen (0,0) arriba-izquierda, rango
    0..IMG_SIZE (224), y CRECE HACIA ABAJO. El goal se fija en
    ``env.target_position`` vía el callable ``_set_goal_state`` y es contra lo
    que se mide ``distance_to_target``.

Nota: usa internos (``_extract_init_goal``, ``_apply_callables``) que pueden
cambiar entre versiones menores. Fija la versión de ``stable-worldmodel`` y
apóyate en el sanity check de ``success_rate`` frente a ``world.evaluate``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

TWOROOM_SUCCESS_THRESHOLD = 16.0


def _drop_time_dim(arr: Any) -> np.ndarray:
    """Quita la dimensión temporal (axis=1, tamaño 1) que la librería añade a
    cada clave de ``world.infos`` (convención ``(n_envs, time=1, ...)``)."""
    a = np.asarray(arr)
    if a.ndim >= 2 and a.shape[1] == 1:
        a = a[:, 0, ...]
    return a


def _to_hwc_uint8(img: Any) -> np.ndarray:
    """Normaliza una imagen a (H, W, 3) uint8 (acepta (3,H,W) o (H,W,3))."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    return a.astype(np.uint8)


def evaluate_with_traces(
    world,
    dataset,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset: int,
    eval_budget: int,
    callables: list[dict] | None = None,
    success_threshold: float = TWOROOM_SUCCESS_THRESHOLD,
    save_images: bool = True,
) -> dict:
    """Evalúa la policy adjunta capturando trazas por paso y geometría.

    Returns:
        dict con cuatro bloques:
          - ``'aggregate'``: métricas escalares agregadas.
          - ``'per_episode'``: arrays de largo ``n`` (uno por episodio).
          - ``'traces'``: arrays ``(T, n[, dim])`` por paso, incluida ``proprio``
            (posición del agente DESPUÉS de cada paso).
          - ``'geometry'``: ``goal_pos`` y ``start_pos`` (n, 2), ``img_size`` y,
            si ``save_images``, ``goal_image`` y ``background_image`` (n, H, W, 3).
    """
    n = len(episodes_idx)
    assert n == world.num_envs, (
        f"num_envs ({world.num_envs}) debe ser igual a "
        f"len(episodes_idx) ({n})."
    )

    # --- 1. Reconstruir estados inicial y objetivo desde el dataset ---------
    init_state, goal_state, _ = _extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    world.reset(seed=init_state.get("seed"))

    # --- 2. Aplicar callables de setup por env ------------------------------
    if callables:
        merged = {**init_state, **goal_state}
        for i in range(n):
            env_init = {k: v[i] for k, v in merged.items()}
            _apply_callables(world.envs.envs[i].unwrapped, callables, env_init)

    # --- 3. Difundir init/goal dentro de world.infos ------------------------
    shape_prefix = world.infos["pixels"].shape[:2]  # (n_envs, time=1)
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(
                    v[:, None, ...], shape_prefix + v.shape[1:]
                ).copy()
    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}

    # --- 4. Geometría: leída del entorno tras el setup (coincide con success)
    goal_pos = np.full((n, 2), np.nan, dtype=np.float32)
    start_pos = np.full((n, 2), np.nan, dtype=np.float32)
    goal_imgs: list[np.ndarray] = []
    bg_imgs: list[np.ndarray] = []
    for i in range(n):
        env = world.envs.envs[i].unwrapped
        tgt = getattr(env, "target_position", None)
        if tgt is not None:
            goal_pos[i] = np.asarray(tgt, dtype=np.float32).reshape(-1)[:2]
        agt = getattr(env, "agent_position", None)
        if agt is not None:
            start_pos[i] = np.asarray(agt, dtype=np.float32).reshape(-1)[:2]
        if save_images:
            gi = getattr(env, "_target_img", None)
            if gi is not None:
                goal_imgs.append(_to_hwc_uint8(gi))
            try:
                bg_imgs.append(_to_hwc_uint8(env.render()))  # fondo + agente inicial
            except Exception:
                pass

    img_size = float(getattr(world.envs.envs[0].unwrapped, "IMG_SIZE", 224))
    geometry: dict[str, Any] = {
        "goal_pos": goal_pos,        # (n, 2)  coords del entorno (0..IMG, y abajo)
        "start_pos": start_pos,      # (n, 2)
        "img_size": img_size,
    }
    if save_images:
        if len(goal_imgs) == n:
            geometry["goal_image"] = np.stack(goal_imgs)
        if len(bg_imgs) == n:
            geometry["background_image"] = np.stack(bg_imgs)

    has_distance = "distance_to_target" in world.infos
    proprio_key = (
        "proprio" if "proprio" in world.infos
        else ("state" if "state" in world.infos else None)
    )

    # --- 5. Rollout con captura de trazas -----------------------------------
    tr_proprio: list[np.ndarray] = []
    tr_distance: list[np.ndarray] = []
    tr_action: list[np.ndarray] = []
    tr_terminated: list[np.ndarray] = []
    tr_reward: list[np.ndarray] = []
    tr_step_idx: list[np.ndarray] = []

    def on_step(w) -> None:
        w.infos.update(deepcopy(goal_snapshot))  # re-inyectar goal

        if has_distance:
            dist = _drop_time_dim(w.infos["distance_to_target"]).astype(float)
        else:  # fallback genérico (p.ej. PushT: reward = -distancia)
            dist = -np.asarray(w.rewards, dtype=float)
        tr_distance.append(dist.reshape(w.num_envs))

        tr_action.append(
            _drop_time_dim(w.infos["action"]).astype(float).reshape(w.num_envs, -1)
        )
        if proprio_key is not None:
            tr_proprio.append(
                _drop_time_dim(w.infos[proprio_key]).astype(float)
                .reshape(w.num_envs, -1)[:, :2]
            )
        tr_step_idx.append(
            _drop_time_dim(w.infos["step_idx"]).astype(np.int64).reshape(w.num_envs)
        )
        # terminated es True SOLO en el paso exacto de terminación (en 'wait'
        # el env se congela y deja de reportar True después).
        tr_terminated.append(np.asarray(w.terminateds, dtype=bool).copy())
        tr_reward.append(np.asarray(w.rewards, dtype=float).copy())

    world._run(max_steps=eval_budget, mode="wait", on_step=on_step)

    traces = {
        "distance": np.stack(tr_distance),       # (T, n)
        "action": np.stack(tr_action),           # (T, n, action_dim)
        "terminated": np.stack(tr_terminated),   # (T, n)
        "reward": np.stack(tr_reward),           # (T, n)
        "step_idx": np.stack(tr_step_idx),       # (T, n)
    }
    if tr_proprio:
        # posición del agente DESPUÉS de cada paso. La posición ANTES del paso t
        # es start_pos para t=0 y proprio[t-1] para t>0 (útil para anclar flechas).
        traces["proprio"] = np.stack(tr_proprio)  # (T, n, 2)

    per_episode = _compute_per_episode_metrics(traces)
    aggregate = _aggregate(per_episode)

    return {
        "aggregate": aggregate,
        "per_episode": per_episode,
        "traces": traces,
        "geometry": geometry,
    }


def _compute_per_episode_metrics(traces: dict) -> dict:
    """Deriva métricas por episodio a partir de las trazas ``(T, n)``."""
    distance = traces["distance"]      # (T, n)
    action = traces["action"]          # (T, n, action_dim)
    terminated = traces["terminated"]  # (T, n)
    T, n = distance.shape

    success = terminated.any(axis=0)                       # (n,)
    first_true = terminated.argmax(axis=0)                 # 0 si nunca hay True
    time_to_success = np.where(success, first_true, np.nan).astype(float)
    executed = np.where(success, first_true + 1, T).astype(int)

    tgrid = np.arange(T)[:, None]
    run_mask = tgrid < executed[None, :]                   # (T, n)

    dist_run = np.where(run_mask, distance, np.nan)
    min_distance = np.nanmin(dist_run, axis=0)
    distance_auc = np.nansum(dist_run, axis=0)

    last_idx = np.where(success, first_true, executed - 1)
    final_distance = distance[last_idx, np.arange(n)]

    act_mag = np.linalg.norm(action, axis=-1)              # (T, n)
    act_run = np.where(run_mask, act_mag, np.nan)
    mean_action_magnitude = np.nanmean(act_run, axis=0)

    return {
        "success": success,
        "time_to_success": time_to_success,
        "final_distance": final_distance,
        "min_distance": min_distance,
        "distance_auc": distance_auc,
        "mean_action_magnitude": mean_action_magnitude,
        "n_steps_executed": executed,
    }


def _aggregate(per_episode: dict) -> dict:
    success = per_episode["success"]
    n = success.size
    tts = per_episode["time_to_success"]
    any_success = bool(success.any())
    return {
        "success_rate": float(success.mean() * 100.0),
        "num_episodes": int(n),
        "num_success": int(success.sum()),
        "mean_time_to_success": float(np.nanmean(tts)) if any_success else float("nan"),
        "median_time_to_success": float(np.nanmedian(tts)) if any_success else float("nan"),
        "mean_min_distance": float(per_episode["min_distance"].mean()),
        "mean_final_distance": float(per_episode["final_distance"].mean()),
        "mean_distance_auc": float(per_episode["distance_auc"].mean()),
        "mean_action_magnitude": float(per_episode["mean_action_magnitude"].mean()),
    }


def save_traces(path, results: dict) -> None:
    """Guarda trazas + métricas por episodio + geometría en un ``.npz``.

    Para plotear el episodio k luego:
        d = np.load(path)
        bg   = d['geo_background_image'][k]   # imagen de fondo (H, W, 3)
        traj = d['trace_proprio'][:, k, :]    # (T, 2) trayectoria del agente
        acts = d['trace_action'][:, k, :]     # (T, 2) acciones
        goal = d['geo_goal_pos'][k]           # (2,) ubicación del goal
    """
    payload: dict[str, Any] = {}
    payload.update({f"trace_{k}": v for k, v in results["traces"].items()})
    payload.update({f"ep_{k}": v for k, v in results["per_episode"].items()})
    for k, v in results.get("geometry", {}).items():
        if v is not None:
            payload[f"geo_{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)


def format_summary(aggregate: dict) -> str:
    lines = ["==== TRACE METRICS ===="]
    for k, v in aggregate.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"
