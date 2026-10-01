"""Front-end multiescala unificado (transformadas, ablación de bandas e inicialización del patch embedding).

Convención de bandas
--------------------
``BandTransform`` convierte ``x (B, C, H, W)`` en ``bands (B, nb, C, H, W)``, en el MISMO orden que
``starlet_conv4d`` / ``swt_conv4d``: detalles (de fino a grueso) primero y la aproximación al final.
Al aplanar, el canal ``k = banda * C + c`` coincide con el layout que ya usan los encoders.

Transformadas (``kind``)
------------------------
- ``starlet``    : x = c_J + sum_j w_j                        -> nb = J + 1, suma exactamente x.
- ``swt``        : 3 orientaciones (LH, HL, HH) por nivel + LL -> nb = 3J + 1 (o J + 1 con
                   ``merge_orientations``). NO suma x (la síntesis no es la suma de las bandas).
- ``rgb_repeat`` : J + 1 copias de x / (J + 1)                 -> control de capacidad: mismos canales,
                   mismo "suma x", cero estructura multiescala.
- ``random``     : J filtros fijos aleatorios de media cero + residuo x - sum(r_j * x)
                 -> control: descomposición lineal redundante que suma x, sin estructura de escala.
- ``none``       : RGB.

Inicialización del patch embedding (fusión ``replace``)
-------------------------------------------------------
- ``legacy``     : comportamiento original del repo: pesos RGB preentrenados en los 3 primeros canales
                   (que son w_1, el detalle más fino) y la media de los pesos RGB en el resto.
- ``equivalent`` : W_rgb replicado en todas las bandas. Si las bandas suman x y level_weights = 1,
                   conv(W, sum_b band_b) = conv(W, x): en el paso 0 el encoder ES el preentrenado.

Fusión ``residual``
-------------------
tokens = conv_rgb(x) + conv_ms(bandas), con conv_ms inicializada en CERO (como las "zero convs" de
ControlNet): exacto en el paso 0 para cualquier transformada (incluida SWT) y la rama multiescala
solo entra si reduce la pérdida.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .starlet_torch import starlet_conv4d
from .swt_torch import swt_conv4d

TRANSFORMS = ("none", "starlet", "swt", "rgb_repeat", "random")
FUSIONS = ("replace", "residual")
PATCH_INITS = ("legacy", "equivalent")
ABLATION_MODES = ("zero", "mean", "noise")


# --------------------------------------------------------------------------------------
# Metadatos de las bandas
# --------------------------------------------------------------------------------------
def n_bands(kind: str, levels: int, merge_orientations: bool = False) -> int:
    if kind == "none":
        return 1
    if kind in ("starlet", "rgb_repeat", "random"):
        return levels + 1
    if kind == "swt":
        return levels + 1 if merge_orientations else 3 * levels + 1
    raise ValueError(f"transformada desconocida: {kind!r} (opciones: {TRANSFORMS})")


def band_names(kind: str, levels: int, merge_orientations: bool = False) -> List[str]:
    if kind == "none":
        return ["rgb"]
    if kind == "starlet":
        return [f"w{j}" for j in range(1, levels + 1)] + [f"c{levels}"]
    if kind == "swt":
        if merge_orientations:
            return [f"d{j}" for j in range(1, levels + 1)] + [f"LL{levels}"]
        return [f"{o}{j}" for j in range(1, levels + 1) for o in ("LH", "HL", "HH")] + [f"LL{levels}"]
    if kind == "rgb_repeat":
        return [f"copy{j}" for j in range(levels + 1)]
    if kind == "random":
        return [f"r{j}" for j in range(1, levels + 1)] + ["resid"]
    raise ValueError(f"transformada desconocida: {kind!r}")


def sums_to_input(kind: str) -> bool:
    """True si sum_b band_b == x (condición para que ``patch_init=equivalent`` sea exacta)."""
    return kind in ("none", "starlet", "rgb_repeat", "random")


# --------------------------------------------------------------------------------------
# Transformada
# --------------------------------------------------------------------------------------
class BandTransform(nn.Module):
    """x (B, C, H, W) -> bands (B, nb, C, H, W). Sin parámetros entrenables."""

    def __init__(
        self,
        kind: str = "starlet",
        levels: int = 3,
        starlet_filter: str = "b3",
        swt_filter: str = "bior2.2",
        merge_orientations: bool = False,
        random_kernel: int = 9,
        random_seed: int = 0,
    ):
        super().__init__()
        if kind not in TRANSFORMS:
            raise ValueError(f"transformada desconocida: {kind!r} (opciones: {TRANSFORMS})")
        if kind != "none" and levels < 1:
            raise ValueError(f"{kind} requiere levels >= 1")
        self.kind = kind
        self.levels = int(levels)
        self.starlet_filter = starlet_filter
        self.swt_filter = swt_filter
        self.merge_orientations = bool(merge_orientations)
        self.nb = n_bands(kind, self.levels, self.merge_orientations)
        if kind == "random":
            if random_kernel % 2 == 0:
                raise ValueError("random_kernel debe ser impar")
            g = torch.Generator().manual_seed(int(random_seed))
            k = torch.randn(self.levels, random_kernel, random_kernel, generator=g)
            k = k - k.mean(dim=(1, 2), keepdim=True)  # media cero: se comportan como detalles
            k = k / k.flatten(1).norm(dim=1).view(-1, 1, 1)
            # No persistente: se regenera desde la semilla y no cambia el state_dict de nadie.
            self.register_buffer("random_kernels", k, persistent=False)

    @property
    def names(self) -> List[str]:
        return band_names(self.kind, self.levels, self.merge_orientations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = self.levels
        if self.kind == "none":
            return x.unsqueeze(1)
        if self.kind == "starlet":
            return starlet_conv4d(x, L, filter=self.starlet_filter).view(B, L + 1, C, H, W)
        if self.kind == "swt":
            out = swt_conv4d(x, L, filter=self.swt_filter).view(B, 3 * L + 1, C, H, W)
            if self.merge_orientations:
                det = out[:, : 3 * L].reshape(B, L, 3, C, H, W).sum(2)
                out = torch.cat([det, out[:, 3 * L:]], dim=1)
            return out
        if self.kind == "rgb_repeat":
            return (x / (L + 1)).unsqueeze(1).expand(B, L + 1, C, H, W).contiguous()
        # random
        k = self.random_kernels.to(dtype=x.dtype, device=x.device)
        ks = k.shape[-1]
        w = k.repeat_interleave(C, dim=0).unsqueeze(1)  # (L*C, 1, ks, ks): canal j*C + c usa k_j
        xr = F.pad(x.repeat(1, L, 1, 1), (ks // 2,) * 4, mode="reflect")
        r = F.conv2d(xr, w, groups=L * C).view(B, L, C, H, W)
        resid = x - r.sum(1)
        return torch.cat([r, resid.unsqueeze(1)], dim=1)


# --------------------------------------------------------------------------------------
# Ablación de bandas
# --------------------------------------------------------------------------------------
def resolve_band_indices(spec: Union[int, str, Sequence[Union[int, str]]], names: List[str]) -> List[int]:
    """Acepta índices (0-based) o nombres ("w1", "c3", "LH2", ...). También "details" y "approx"."""
    if isinstance(spec, (int, str)):
        spec = [spec]
    out: List[int] = []
    for s in spec:
        if isinstance(s, str) and s == "details":
            out.extend(range(len(names) - 1))
        elif isinstance(s, str) and s == "approx":
            out.append(len(names) - 1)
        elif isinstance(s, str) and not s.lstrip("-").isdigit():
            if s not in names:
                raise ValueError(f"banda {s!r} no existe; disponibles: {names}")
            out.append(names.index(s))
        else:
            i = int(s)
            if not -len(names) <= i < len(names):
                raise ValueError(f"índice de banda {i} fuera de rango para {names}")
            out.append(i % len(names))
    return sorted(set(out))


def ablate_bands(bands: torch.Tensor, idx: Iterable[int], mode: str = "zero",
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """bands (B, nb, C, H, W). Devuelve una copia con las bandas ``idx`` reemplazadas.

    - zero  : la banda desaparece (cambia la imagen efectiva: x - banda).
    - mean  : se reemplaza por su media espacial por imagen y canal (conserva el DC).
    - noise : ruido gaussiano con la media y la desviación por imagen/canal de la banda: destruye la
              información conservando sus estadísticas de primer y segundo orden.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"modo de ablación desconocido: {mode!r} (opciones: {ABLATION_MODES})")
    idx = list(idx)
    if not idx:
        return bands
    keep = torch.ones(bands.shape[1], device=bands.device, dtype=bands.dtype)
    keep[idx] = 0.0
    out = bands * keep.view(1, -1, 1, 1, 1)
    if mode == "zero":
        return out
    sel = bands[:, idx]
    mu = sel.mean(dim=(-2, -1), keepdim=True)
    if mode == "mean":
        fill = mu.expand_as(sel)
    else:
        sd = sel.std(dim=(-2, -1), keepdim=True)
        noise = torch.randn(sel.shape, generator=generator, device="cpu", dtype=torch.float32)
        fill = mu + sd * noise.to(device=bands.device, dtype=bands.dtype)
    add = torch.zeros_like(bands)
    add[:, idx] = fill
    return out + add


# --------------------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------------------
class ResidualMultiscaleProj(nn.Module):
    """Reemplaza ``patch_embed.proj``. Entrada: ``cat([x, bandas_aplanadas])``.

    out = conv_rgb(x) + conv_ms(bandas), con conv_ms inicializada en cero.
    """

    def __init__(self, rgb_conv: nn.Conv2d, ms_in_chans: int):
        super().__init__()
        self.rgb = rgb_conv
        self.ms = nn.Conv2d(
            ms_in_chans, rgb_conv.out_channels, kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride, padding=rgb_conv.padding, bias=False,
        )
        nn.init.zeros_(self.ms.weight)
        self.rgb_chans = rgb_conv.in_channels
        self.last_ratio: Optional[torch.Tensor] = None

    # Atributos estilo Conv2d por si algún código los consulta.
    @property
    def in_channels(self) -> int:
        return self.rgb_chans + self.ms.in_channels

    @property
    def out_channels(self) -> int:
        return self.rgb.out_channels

    @property
    def kernel_size(self):
        return self.rgb.kernel_size

    @property
    def stride(self):
        return self.rgb.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.rgb(x[:, : self.rgb_chans])
        b = self.ms(x[:, self.rgb_chans:])
        if self.training:
            # ||rama multiescala|| / ||rama RGB||: cuánto usa el modelo la multiescala.
            self.last_ratio = (b.detach().norm() / (a.detach().norm() + 1e-8))
        return a + b


class BandAblationMixin:
    """Estado de ablación/escala de bandas para cualquier encoder con front-end multiescala.

    El encoder debe exponer ``band_names`` (lista) y llamar a ``self._apply_band_controls(bands)``
    con ``bands (B, nb, C, H, W)`` antes de aplanar.
    """

    _band_ablation: Optional[dict] = None
    _band_scale: Optional[torch.Tensor] = None

    def set_band_ablation(self, bands, mode: str = "zero", seed: int = 0) -> List[int]:
        idx = resolve_band_indices(bands, list(self.band_names))
        if mode not in ABLATION_MODES:
            raise ValueError(f"modo de ablación desconocido: {mode!r}")
        self._band_ablation = {"idx": idx, "mode": mode,
                               "gen": torch.Generator().manual_seed(int(seed))}
        return idx

    def clear_band_ablation(self) -> None:
        self._band_ablation = None

    def set_band_scale(self, scale: Optional[torch.Tensor]) -> None:
        """Multiplicador por banda (nb,), diferenciable: para Integrated Gradients sobre bandas."""
        if scale is not None and scale.numel() != len(self.band_names):
            raise ValueError(f"band_scale debe tener {len(self.band_names)} elementos")
        self._band_scale = scale

    def _apply_band_controls(self, bands: torch.Tensor) -> torch.Tensor:
        scale = getattr(self, "_band_scale", None)
        if scale is not None:
            bands = bands * scale.to(dtype=bands.dtype, device=bands.device).view(1, -1, 1, 1, 1)
        abl = getattr(self, "_band_ablation", None)
        if abl is not None:
            bands = ablate_bands(bands, abl["idx"], abl["mode"], abl["gen"])
        return bands
