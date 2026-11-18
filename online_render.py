"""Render online de frames con stable-envs para ``train.py``.

Con ``data.render.online: true`` el dataset carga filas de estado en vez de
píxeles; este módulo las escribe en ``State`` de stable-envs y dibuja los
frames en el device del batch (GPU). La configuración vive en el YAML de data:

    render:
      online: true
      state_column: state
      env:
        id: PushT
        kwargs: {}
        leaves:
          - {leaf: current.body.agent.position, columns: [0, 2]}
          - {leaf: current.body.block.position, columns: [2, 4]}
          - {leaf: current.body.block.angle, columns: [4, 5]}
        zero: [current.render_segments, current.render_segment_count]

El mapeo ``leaves``/``zero`` es el del ejemplo oficial de LeWM sobre stable-envs
(``examples/lewm/config/pusht.yaml``).
"""
from __future__ import annotations

import torch

import stable_envs as snv
from stable_envs import get_field


class StateRenderer:
    """Escribe filas de estado del dataset en ``State`` y dibuja con ``env.render``."""

    def __init__(self, env_id, leaves, zero=(), *, device="cpu", size=None,
                 env_kwargs=None, seed=0):
        self.env = snv.make(env_id, num_envs=1, device=device, **dict(env_kwargs or {}))
        self.env.reset(seed=seed)
        self.size = size
        self.leaves = [(key, int(start), int(stop)) for key, start, stop in leaves]
        self.zero = list(zero)
        self._template = self.env.get_state()
        self._batch = None

    @property
    def device(self) -> torch.device:
        return self.env.device

    def states(self, rows: torch.Tensor):
        """``(N, D)`` filas -> ``State`` con un entorno por fila."""
        rows = torch.as_tensor(rows, device=self.env.device)
        n = int(rows.shape[0])
        if self._batch is None or self._batch.batch_size[0] != n:
            self._batch = self._template.expand(n).clone()
        state = self._batch
        for key, start, stop in self.leaves:
            field = get_field(state, key)
            field.copy_(rows[:, start:stop].reshape(field.shape).to(field.dtype))
        for key in self.zero:
            get_field(state, key).zero_()
        return state

    @torch.no_grad()
    def __call__(self, rows) -> torch.Tensor:
        """``(*lead, D)`` -> ``(*lead, H, W, 3)`` uint8."""
        rows = torch.as_tensor(rows)
        lead = tuple(rows.shape[:-1])
        kwargs = {} if self.size is None else {"size": self.size}
        frames = self.env.render(self.states(rows.reshape(-1, rows.shape[-1])), **kwargs)
        return frames.reshape(*lead, *frames.shape[-3:])

    def close(self) -> None:
        self.env.close()


class Pixels:
    """``(B, T, D)`` estados -> ``(B, T, 3, H, W)`` frames normalizados (ImageNet).

    El renderer se crea perezosamente en el device de las filas (el batch ya
    está en GPU cuando ``train.py`` llama a este objeto).
    """

    def __init__(self, env_cfg, size, mean, std):
        self.env_cfg = env_cfg
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.renderer = None

    def _build(self, device) -> None:
        env = self.env_cfg
        leaves = [(item["leaf"], *item["columns"]) for item in env["leaves"]]
        self.renderer = StateRenderer(
            env["id"], leaves, list(env.get("zero") or ()),
            device=str(device), size=self.size,
            env_kwargs=dict(env.get("kwargs") or {}),
        )
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    @torch.no_grad()
    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        if self.renderer is None:
            self._build(rows.device)
        b, n = rows.shape[:2]
        x = self.renderer(rows.float()).reshape(b * n, self.size, self.size, 3)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        return ((x - self.mean) / self.std).reshape(b, n, 3, self.size, self.size)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
