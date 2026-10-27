"""Latent Neural Cellular Automaton (NCA) predictor for spatial JEPA.

Adapta el Growing NCA (Mordvintsev et al., Distill 2020) para que actúe como
predictor de un world model sobre la grilla de patch tokens del encoder, en vez
de sobre píxeles.

Cada patch token es una "celda". Su estado tiene dos partes:

    [ latente (D canales) | ocultos (H canales) ]

* La parte latente se inicializa con el token del encoder en t y, tras K pasos
  de la regla local, se lee como la predicción del latente en t+1.
* Los canales ocultos son el "scratchpad" del NCA original: se inicializan con
  una proyección 1x1 de ``[x_t, x_t - x_{t-1}]`` (posición + velocidad local) y
  sirven para que las celdas se coordinen.

Un paso de la regla (idéntico en todas las celdas, pesos compartidos):

    p   = perceive(s)                         # depthwise 3x3: identidad + Sobel x/y
    h   = W1 p
    h   = ReLU(LN(h) * (1 + scale_a) + shift_a)   # AdaLN-zero con la acción
    ds  = W2 h                                # W2 inicializado en cero
    s   = s + ds * m,   m ~ Bernoulli(fire_rate) por celda (train)

Diferencias con el NCA de píxeles:

* No hay máscara de vida (no hay alfa en el latente).
* La percepción parte de identidad + Sobel pero es aprendible por defecto,
  porque en latente "gradiente espacial" no tiene por qué ser la feature útil.
* En eval el update es determinista (mean-field: ``ds * fire_rate``) para que
  el costo del planner (CEM) no tenga ruido entre candidatos.
* K se samplea en ``[k_min, k_max]`` en train: obliga a que la predicción sea
  estable ante pasos extra, es decir, que el target sea un atractor.
* Daño latente opcional (``damage_prob``): borra un círculo de celdas del
  latente de entrada y se pide predecir igual el latente completo. Es el
  análogo latente del entrenamiento de regeneración.
* Gate por celda opcional (``gate=True``): ``ds <- ds * sigmoid(w·h + b)``,
  generalización aprendida de la máscara de vida. Las celdas sin dinámica
  (paredes, fondo) pueden apagar su update. Con ``loss.gate_l1.weight > 0``
  se impone un prior de dinámica dispersa.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def make_perception_kernel(ch: int) -> torch.Tensor:
    """Identidad + Sobel x/y como kernel depthwise ``[3*ch, 1, 3, 3]``.

    Orden de salida: para el canal c, los filtros quedan en ``3c, 3c+1, 3c+2``
    (mismo layout que ``model.py`` del Growing NCA).
    """
    ident = torch.outer(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    dx = torch.outer(torch.tensor([1.0, 2.0, 1.0]), torch.tensor([-1.0, 0.0, 1.0])) / 8.0
    dy = dx.T
    kernel = torch.stack([ident, dx, dy], dim=0)  # (3, 3, 3)
    return kernel.repeat(ch, 1, 1).unsqueeze(1)  # (3*ch, 1, 3, 3)


def make_grid_circle_masks(n: int, h: int, w: int, radius: Tuple[float, float],
                           device=None) -> torch.Tensor:
    """``n`` máscaras circulares binarias sobre la grilla ``(h, w)``; 1 = celda borrada.

    Igual que ``make_circle_masks`` del NCA, con radio en coordenadas [-1, 1].
    """
    x = torch.linspace(-1, 1, w, device=device)[None, None, :]
    y = torch.linspace(-1, 1, h, device=device)[None, :, None]
    center = 2 * torch.rand(n, 2, 1, 1, device=device) - 1
    r = radius[0] + (radius[1] - radius[0]) * torch.rand(n, 1, 1, device=device)
    dx = x - center[:, 0]
    dy = y - center[:, 1]
    return (dx * dx + dy * dy < r * r).float()  # (n, h, w)


class LatentNCAPredictor(nn.Module):
    """Predictor NCA para JEPA espacial.

    Interfaz (la usa ``jepa_spatial.SpatialJEPA``)::

        pred = predictor(x, c, grid_hw=(gh, gw), only_last=False)

    * ``x``: ``(B, T, N, D)`` latentes de contexto (N = gh*gw patch tokens).
    * ``c``: ``(B, T, A)`` embeddings de acción.
    * Devuelve ``(B, T, N, D)``: para cada t, la predicción de x_{t+1}
      (teacher forcing, cada t se resuelve en paralelo). Con ``only_last=True``
      devuelve ``(B, 1, N, D)`` y sólo calcula el último t (rollout).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        act_dim: int,
        hidden_channels: int = 32,
        update_dim: int = 256,
        steps: Sequence[int] = (4, 8),
        eval_steps: Optional[int] = None,
        fire_rate: float = 0.5,
        step_size: float = 1.0,
        learnable_perception: bool = True,
        padding_mode: str = "replicate",
        use_velocity: bool = True,
        damage_prob: float = 0.0,
        damage_radius: Sequence[float] = (0.15, 0.4),
        grad_checkpoint: bool = False,
        gate: bool = False,
        gate_bias_init: float = 2.0,
        **_unused,
    ):
        super().__init__()
        assert padding_mode in ("zeros", "replicate", "reflect", "circular")
        k_min, k_max = int(steps[0]), int(steps[-1])
        assert 1 <= k_min <= k_max, f"steps inválido: {steps}"

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.state_dim = latent_dim + hidden_channels
        self.k_min, self.k_max = k_min, k_max
        self.eval_steps = int(eval_steps) if eval_steps is not None else k_max
        self.fire_rate = fire_rate
        self.step_size = step_size
        self.padding_mode = padding_mode
        self.use_velocity = use_velocity
        self.damage_prob = damage_prob
        self.damage_radius = tuple(damage_radius)
        self.grad_checkpoint = grad_checkpoint

        # Inicialización de los canales ocultos desde [x_t, x_t - x_{t-1}].
        if hidden_channels > 0:
            in_init = latent_dim * (2 if use_velocity else 1)
            self.hidden_init = nn.Linear(in_init, hidden_channels)

        # Percepción depthwise 3x3, inicializada como identidad + Sobel.
        S = self.state_dim
        self.perception = nn.Conv2d(S, 3 * S, 3, padding=0, groups=S, bias=False)
        with torch.no_grad():
            self.perception.weight.copy_(make_perception_kernel(S))
        self.perception.weight.requires_grad_(learnable_perception)

        # Regla de update por celda (equivalente a convs 1x1, en channel-last).
        self.fc1 = nn.Linear(3 * S, update_dim)
        self.norm = nn.LayerNorm(update_dim, elementwise_affine=False, eps=1e-6)
        self.fc2 = nn.Linear(update_dim, S)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # AdaLN-zero con la acción: arranca como modulación identidad.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(act_dim, 2 * update_dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        # Gate por celda: g = sigmoid(w·h + b) multiplica ds. Es función del
        # estado local en cada paso (no un mapa fijo por posición). Arranca
        # uniforme y abierto: sigmoid(gate_bias_init) = 0.88 con el default.
        self.use_gate = gate
        if gate:
            self.gate_head = nn.Linear(update_dim, 1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias_init)

        # Salidas auxiliares del último forward (las lee ``jepa_spatial``):
        #   gate_mean: escalar diferenciable, media de g sobre celdas y pasos.
        #   gate_map:  (B, T, N), media de g sobre los pasos.
        self.aux = {}

    # ------------------------------------------------------------------ #
    # Regla local
    # ------------------------------------------------------------------ #
    def perceive(self, s: torch.Tensor) -> torch.Tensor:
        """``s``: ``(B, S, gh, gw)`` -> ``(B, 3S, gh, gw)``."""
        if self.padding_mode == "zeros":
            s = F.pad(s, (1, 1, 1, 1))
        else:
            s = F.pad(s, (1, 1, 1, 1), mode=self.padding_mode)
        return self.perception(s)

    def nca_step(self, s: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        """Un paso del NCA. ``s``: ``(B, S, gh, gw)``; ``shift/scale``: ``(B, 1, 1, U)``.

        Devuelve ``(s_nuevo, g)`` con ``g`` de shape ``(B, 1, gh, gw)`` (o ``None``
        sin gate).
        """
        p = self.perceive(s).permute(0, 2, 3, 1)  # (B, gh, gw, 3S)
        h = self.norm(self.fc1(p))
        h = F.relu(h * (1 + scale) + shift)
        ds = self.fc2(h).permute(0, 3, 1, 2) * self.step_size  # (B, S, gh, gw)

        g = None
        if self.use_gate:
            g = torch.sigmoid(self.gate_head(h)).permute(0, 3, 1, 2)  # (B, 1, gh, gw)
            ds = ds * g

        if self.training:
            mask = (torch.rand_like(s[:, :1]) <= self.fire_rate).to(ds.dtype)
            return s + ds * mask, g
        # Eval: update esperado (determinista).
        return s + ds * self.fire_rate, g

    def num_steps(self) -> int:
        if not self.training:
            return self.eval_steps
        return int(torch.randint(self.k_min, self.k_max + 1, (1,)).item())

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _grid(self, n: int, grid_hw: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if grid_hw is not None:
            gh, gw = grid_hw
        else:
            gh = gw = int(math.isqrt(n))
        assert gh * gw == n, f"N={n} no calza con la grilla {gh}x{gw}"
        return gh, gw

    def _damage_keep(self, n: int, gh: int, gw: int, ref: torch.Tensor) -> torch.Tensor:
        """Máscara ``(n, gh*gw, 1)`` con 0 en las celdas borradas.

        Sólo una fracción ``damage_prob`` de las muestras recibe daño.
        """
        hit = (torch.rand(n, device=ref.device) < self.damage_prob).to(ref.dtype)
        masks = make_grid_circle_masks(n, gh, gw, self.damage_radius, device=ref.device)
        keep = 1.0 - masks.to(ref.dtype) * hit[:, None, None]
        return keep.reshape(n, gh * gw, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                grid_hw: Optional[Tuple[int, int]] = None,
                only_last: bool = False) -> torch.Tensor:
        B, T, N, D = x.shape
        assert D == self.latent_dim, f"latent_dim={self.latent_dim} pero x tiene D={D}"
        gh, gw = self._grid(N, grid_hw)

        # Velocidad local por celda (cero en t=0, como en teacher forcing).
        vel = torch.zeros_like(x)
        if T > 1:
            vel[:, 1:] = x[:, 1:] - x[:, :-1]

        if only_last:
            x, vel, c = x[:, -1:], vel[:, -1:], c[:, -1:]
        Tq = x.size(1)

        # Plegamos el tiempo en el batch: cada t es un problema independiente.
        x_f = x.reshape(B * Tq, N, D)
        v_f = vel.reshape(B * Tq, N, D)
        c_f = c.reshape(B * Tq, -1)

        # Daño latente: la celda borrada pierde posición y velocidad; el target
        # (fuera de este módulo) sigue siendo el latente completo.
        if self.training and self.damage_prob > 0:
            keep = self._damage_keep(B * Tq, gh, gw, x_f)
            x_f, v_f = x_f * keep, v_f * keep

        lat = x_f.transpose(1, 2).reshape(B * Tq, D, gh, gw)
        parts = [lat]
        if self.hidden_channels > 0:
            init_in = torch.cat([x_f, v_f], dim=-1) if self.use_velocity else x_f
            hid = self.hidden_init(init_in)  # (BT, N, H)
            parts.append(hid.transpose(1, 2).reshape(B * Tq, self.hidden_channels, gh, gw))
        s = torch.cat(parts, dim=1)  # (BT, S, gh, gw)

        shift, scale = self.adaLN(c_f).chunk(2, dim=-1)
        shift, scale = shift[:, None, None], scale[:, None, None]

        gates = []
        for _ in range(self.num_steps()):
            if self.grad_checkpoint and self.training and s.requires_grad:
                s, g = checkpoint(self.nca_step, s, shift, scale, use_reentrant=False)
            else:
                s, g = self.nca_step(s, shift, scale)
            if g is not None:
                gates.append(g)

        self.aux = {}
        if gates:
            gmap = torch.stack(gates, 0).mean(0)  # (BT, 1, gh, gw)
            self.aux["gate_mean"] = gmap.mean()
            self.aux["gate_map"] = gmap.reshape(B, Tq, N)

        out = s[:, :D].reshape(B * Tq, D, N).transpose(1, 2)  # (BT, N, D)
        return out.reshape(B, Tq, N, D)
