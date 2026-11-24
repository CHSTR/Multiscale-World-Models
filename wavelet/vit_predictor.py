"""Predictor ViT espacial: alternativa intercambiable al NCA.

Misma interfaz espacial que ``nca.LatentNCAPredictor``::

    pred  = predictor(x, c, grid_hw=(gh, gw), only_last=False)
    preds = predictor.rollout(x_ctx, c_seq, grid_hw=(gh, gw), ...)

* ``x``: ``(B, T, N, D)`` latentes de contexto (N = gh*gw patch tokens).
* ``c``: ``(B, T, A)`` embeddings de acción.
* ``forward`` devuelve ``(B, T', N, D)``: para cada t predice ``x_{t+1}``
  (teacher forcing, cada t se resuelve en paralelo. Con ``only_last=True``
  devuelve ``(B, 1, N, D)``). El tiempo se pliega en el batch, igual que en el
  NCA.
* ``rollout`` devuelve ``(B, M, N, D)``: M pasos del mundo encadenados desde el
  último frame de ``x_ctx`` con las acciones ``c_seq (B, M, A)``.

Diseño (deliberadamente distinto al NCA, misma interfaz):

* Un paso del mundo = un transformer sobre los N tokens de una grilla. La
  atención es **espacial y no causal**: cada patch ve a todos los demás del
  mismo frame (el NCA propaga información K pasos 3x3; aquí el receptive field
  es global en una sola pasada).
* No hay canales ocultos: el estado es sólo el latente. ``persist_hidden`` se
  acepta por compatibilidad y se ignora (no hay scratchpad que persistir).
* AdaLN-zero con la acción (``module.modulate``): la condición
  ``(B*T, 1, dim)`` se broadcast sobre los N tokens. Igual que en DiT, la última
  capa de cada ``adaLN_modulation`` arranca en cero, así que los bloques son
  identidad al inicio del entrenamiento.
* La velocidad ``x_t - x_{t-1}`` entra como canal extra (``use_velocity``); en
  ``t=0`` es cero. En ``rollout`` sale del frame anterior real o predicho,
  replicando la semántica de ``LatentNCAPredictor.rollout``.
* Determinista en eval: el dropout de ``module.Attention``/``FeedForward`` sólo
  actúa en train, así que el costo del planner (CEM) no tiene ruido.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from module import Attention, FeedForward, modulate


class SpatialBlock(nn.Module):
    """Bloque transformer espacial no causal + AdaLN-zero con la acción.

    Igual que ``module.ConditionalBlock`` pero llamando a la atención con
    ``causal=False`` (aquí el eje de tokens es espacio, no tiempo).

    ``x``: ``(n, N, dim)``; ``c``: ``(n, 1, dim)`` (se broadcast sobre N).
    """

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=False
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ViTSpatialPredictor(nn.Module):
    """Predictor transformer espacial por frame (ver docstring del módulo)."""

    def __init__(
        self,
        *,
        latent_dim: int,
        act_dim: int,
        depth: int = 4,
        heads: int = 6,
        dim_head: int = 64,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_velocity: bool = True,
        max_tokens: int = 4096,
        gated_residual: bool = False,
        gate_init: float = 0.1,
        **_unused,  # tolera kwargs de config del NCA (steps, fire_rate, ...)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.depth = depth
        self.use_velocity = use_velocity
        self.max_tokens = int(max_tokens)
        self.gated_residual = bool(gated_residual)
        if self.gated_residual:
            self.res_gate = nn.Parameter(torch.full((latent_dim,), float(gate_init)))

        dim = heads * dim_head
        self.dim = dim
        mlp_dim = int(mlp_dim) if mlp_dim is not None else 4 * dim

        in_dim = latent_dim * (2 if use_velocity else 1)
        self.input_proj = nn.Linear(in_dim, dim)
        self.act_proj = nn.Linear(act_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_tokens, dim) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SpatialBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = nn.Linear(dim, latent_dim)

        # Mismo contrato de diagnóstico que el NCA (``jepa_spatial`` lee
        # ``aux``/``aux_rollout``). El ViT no tiene gate: quedan vacíos.
        self.aux = {}
        self.aux_rollout = {}

    # ------------------------------------------------------------------ #
    # Piezas compartidas por forward y rollout
    # ------------------------------------------------------------------ #
    def _grid(self, n: int, grid_hw: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """Valida que N calce con la grilla (la atención no la necesita)."""
        if grid_hw is not None:
            gh, gw = int(grid_hw[0]), int(grid_hw[1])
        else:
            import math
            gh = gw = int(math.isqrt(n))
        assert gh * gw == n, f"N={n} no calza con la grilla {gh}x{gw}"
        return gh, gw

    def _check_tokens(self, n: int) -> None:
        assert n <= self.max_tokens, (
            f"N={n} tokens supera max_tokens={self.max_tokens}"
        )

    def _predict_frame(self, x_cur: torch.Tensor, x_prev: torch.Tensor,
                       c: torch.Tensor) -> torch.Tensor:
        """Un frame: ``(n, N, D)`` + ``(n, A)`` -> predicción ``(n, N, D)``."""
        if self.use_velocity:
            inp = torch.cat([x_cur, x_cur - x_prev], dim=-1)
        else:
            inp = x_cur
        n, N, _ = inp.shape
        self._check_tokens(N)

        h = self.input_proj(inp)
        h = h + self.pos_embed[:, :N].to(h.dtype)
        h = self.emb_dropout(h)

        cond = self.act_proj(c)[:, None]  # (n, 1, dim), broadcast sobre N
        for blk in self.blocks:
            h = blk(h, cond)
        out = self.output_proj(self.norm(h))
        if self.gated_residual:
            return x_cur + self.res_gate * out
        return out

    # ------------------------------------------------------------------ #
    # Forward (teacher forcing)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, c: torch.Tensor,
                grid_hw: Optional[Tuple[int, int]] = None,
                only_last: bool = False) -> torch.Tensor:
        B, T, N, D = x.shape
        assert D == self.latent_dim, f"latent_dim={self.latent_dim} pero x tiene D={D}"
        self._grid(N, grid_hw)

        # Velocidad por frame (cero en t=0), igual que en el NCA.
        vel = torch.zeros_like(x)
        if T > 1:
            vel[:, 1:] = x[:, 1:] - x[:, :-1]
        prev = x - vel  # x_{t-1}; en t=0 vale x_t => velocidad cero

        if only_last:
            x, prev, c = x[:, -1:], prev[:, -1:], c[:, -1:]
        Tq = x.size(1)

        # Plegamos el tiempo en el batch: cada t es un problema independiente.
        out = self._predict_frame(
            x.reshape(B * Tq, N, D),
            prev.reshape(B * Tq, N, D),
            c.reshape(B * Tq, -1),
        )
        self.aux = {}
        return out.reshape(B, Tq, N, D)

    # ------------------------------------------------------------------ #
    # Rollout multi-paso (encadena el propio predictor)
    # ------------------------------------------------------------------ #
    def rollout(self, x_ctx: torch.Tensor, c_seq: torch.Tensor,
                grid_hw: Optional[Tuple[int, int]] = None,
                post: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                persist_hidden: bool = True,
                detach_between: bool = False,
                history: Optional[int] = None) -> torch.Tensor:
        """M ``M`` pasos del mundo desde el último frame de ``x_ctx``.

        * ``x_ctx``: ``(B, Tc, N, D)``. Sólo se usan los dos últimos frames
          (latente inicial y velocidad); con ``Tc == 1`` la velocidad es cero.
        * ``c_seq``: ``(B, M, A)``. ``c_seq[:, m]`` produce la predicción m.
        * ``post``: ``(B, N, D) -> (B, N, D)`` aplicada a cada salida antes de
          retroalimentarla (``pred_proj``), igual que en el NCA.
        * ``persist_hidden``: se acepta por compatibilidad con el NCA y se
          ignora; el ViT no tiene canales ocultos que persistir.
        * ``detach_between``: corta el gradiente entre pasos del mundo (TBPTT).
        * ``history``: ventana de contexto. Con 1 la velocidad es siempre cero
          (como teacher forcing con T=1); con > 1 usa el frame anterior, real o
          predicho. Por defecto ``Tc``.
        """
        B, Tc, N, D = x_ctx.shape
        M = c_seq.size(1)
        assert D == self.latent_dim, f"latent_dim={self.latent_dim} pero x tiene D={D}"
        assert M >= 1, "rollout necesita al menos una acción"
        self._grid(N, grid_hw)
        hist = Tc if history is None else int(history)

        x_cur = x_ctx[:, -1]
        x_prev = x_ctx[:, -2] if (Tc > 1 and hist > 1) else x_cur
        outs = []
        for m in range(M):
            out = self._predict_frame(x_cur, x_prev, c_seq[:, m])
            if post is not None:
                out = post(out)
            outs.append(out)
            if m == M - 1:
                break
            nxt = out.detach() if detach_between else out
            x_prev, x_cur = (x_cur, nxt) if hist > 1 else (nxt, nxt)

        self.aux_rollout = {}
        return torch.stack(outs, dim=1)
