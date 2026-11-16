"""Predictor jerárquico: reparte el trabajo por escala.

``emb`` de la jerarquía wavelet es un ``dict[int, Tensor]`` con un latente
``(B, T, N_l, D_l)`` por nivel. ``HierarchicalPredictor`` sólo es un router:

* ``predictors``: ``nn.ModuleDict`` keyed por ``str(level)``. Cada valor es
  cualquier ``nn.Module`` que cumpla la interfaz espacial (``LatentNCAPredictor``,
  ``ViTSpatialPredictor``, o cualquier otro)::

      forward(x, c, grid_hw=None, only_last=False) -> (B, T', N, D)
      rollout(x_ctx, c_seq, grid_hw=None, post=None, persist_hidden=True,
              detach_between=False, history=None) -> (B, M, N, D)

* La acción ``act_emb`` es compartida entre niveles (mismo ``A``).
* ``grid_hw`` es un ``dict[int, tuple]`` (o una secuencia con una grilla por
  nivel, en el orden de ``self.levels``): cada nivel tiene su propia grilla.
* ``post`` en ``rollout`` puede ser ``None``, un único ``Callable`` o un
  ``dict[int, Callable]`` (p. ej. un ``pred_proj`` distinto por nivel).

No asume NCA: no toca canales ocultos ni gates. Si los predictores exponen
``aux``/``aux_rollout``, se agregan en ``self.aux``/``self.aux_rollout`` con
claves ``"<nivel>.<nombre>"`` (diagnóstico); si no, quedan vacíos.
"""

from collections.abc import Mapping
from typing import Callable, Optional, Union

import torch
from torch import nn

PostFn = Callable[[torch.Tensor], torch.Tensor]
PostArg = Optional[Union[PostFn, Mapping]]


class HierarchicalPredictor(nn.Module):
    """Router por escala sobre predictores con interfaz espacial."""

    def __init__(self, predictors: nn.ModuleDict):
        super().__init__()
        assert len(predictors) > 0, "HierarchicalPredictor necesita al menos un nivel"
        self.predictors = predictors
        self.levels = sorted(int(k) for k in predictors.keys())
        for l in self.levels:  # las claves deben ser enteros serializables
            assert str(l) in predictors, f"falta el predictor del nivel {l}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _grid_for(self, grid_hw, level: int):
        """``grid_hw`` puede ser ``dict[int, tuple]`` o una secuencia por nivel."""
        if grid_hw is None:
            return None
        if isinstance(grid_hw, Mapping):
            return grid_hw.get(level)
        return grid_hw[self.levels.index(level)]

    @staticmethod
    def _post_for(post: PostArg, level: int) -> Optional[PostFn]:
        if post is None:
            return None
        if isinstance(post, Mapping):
            return post.get(level)
        return post

    def _collect_aux(self, attr: str) -> dict:
        """Junta el ``aux`` por nivel; agrega escalares cuando todos lo exponen."""
        out: dict = {}
        scalars: dict = {}
        for l in self.levels:
            aux = getattr(self.predictors[str(l)], attr, None) or {}
            for k, v in aux.items():
                out[f"{l}.{k}"] = v
                if torch.is_tensor(v) and v.ndim == 0:
                    scalars.setdefault(k, []).append(v)
        for k, vals in scalars.items():
            if len(vals) == len(self.levels):  # media sobre escalas homogéneas
                out[k] = torch.stack(vals).mean()
        return out

    # ------------------------------------------------------------------ #
    # Forward (teacher forcing)
    # ------------------------------------------------------------------ #
    def forward(self, emb: dict, act_emb: torch.Tensor,
                grid_hw: Optional[Mapping] = None,
                only_last: bool = False) -> dict:
        """``emb``: ``dict[int, (B,T,N_l,D_l)]`` -> mismo dict de predicciones."""
        out = {}
        for l in self.levels:
            out[l] = self.predictors[str(l)](
                emb[l], act_emb,
                grid_hw=self._grid_for(grid_hw, l),
                only_last=only_last,
            )
        self.aux = self._collect_aux("aux")
        return out

    # ------------------------------------------------------------------ #
    # Rollout multi-paso por nivel
    # ------------------------------------------------------------------ #
    def rollout(self, emb_ctx: dict, act_emb_seq: torch.Tensor,
                grid_hw: Optional[Mapping] = None,
                post: PostArg = None,
                persist_hidden: bool = True,
                detach_between: bool = False,
                history: Optional[int] = None) -> dict:
        """Rollout independiente por escala. ``post`` puede ser por nivel."""
        out = {}
        for l in self.levels:
            out[l] = self.predictors[str(l)].rollout(
                emb_ctx[l], act_emb_seq,
                grid_hw=self._grid_for(grid_hw, l),
                post=self._post_for(post, l),
                persist_hidden=persist_hidden,
                detach_between=detach_between,
                history=history,
            )
        self.aux_rollout = self._collect_aux("aux_rollout")
        return out
