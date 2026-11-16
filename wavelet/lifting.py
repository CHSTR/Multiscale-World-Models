"""Lifting scheme wavelet invertible para LiftWM.

Descomposición piramidal 2D con *lifting scheme* separable (filas y luego
columnas) y reconstrucción perfecta garantizada por estructura:

    split polyphase -> predict -> update

Para cada eje y nivel se parte la señal en muestras pares (``even``) e impares
(``odd``) y se calcula::

    d = odd - P(even)        # detalle
    s = even + U(d)          # aproximación (lifting)

La invertibilidad es *estructural*: el inverso deshace update -> predict ->
merge usando exactamente los mismos filtros, sin importar sus valores. Por eso
los filtros pueden aprenderse (``learnable=True``) sin romper la
reconstrucción perfecta.

Convenciones
------------
- Normalización Haar: la aproximación y el detalle se escalan por ``sqrt(2)``
  de forma que, con ``init="haar"``, la sub-banda LL coincida exactamente con
  el DWT Haar de referencia (``module.WaveletAttention._dwt``)::

      LL = (a + b + c + d) / 2,   LH = (a + b - c - d) / 2, ...
      approx 1D = (a + b) / sqrt(2)
      detail 1D = (a - b) / sqrt(2)

  (por eso ``detail = -d / sqrt(2)`` con ``d = b - a``).
- Orden de sub-bandas en ``Pyramid.details``: ``[LH, HL, HH]`` apiladas en el
  eje de canales (``3C``), con ``details[0]`` el nivel más fino (res/2) y
  ``details[L-1]`` el más grueso.
- Bordes: los filtros son convs depthwise de tamaño 3 con padding
  ``replicate`` (``pad_mode``). El padding se aplica idénticamente en el
  inverso, así que no afecta a la reconstrucción perfecta.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

_SQRT2 = 2.0 ** 0.5

# Coeficientes (kernel de tamaño 3, posición central = muestra actual) para
# cada inicialización del lifting scheme.
_INITS: dict[str, dict[str, list[float]]] = {
    # Haar: P = identidad, U(d) = d/2  ->  s = (a+b)/2, d = b-a
    "haar": {"predict": [0.0, 1.0, 0.0], "update": [0.0, 0.5, 0.0]},
    # CDF 5/3: P(even_i) = (even_i + even_{i+1})/2
    #          U(d_i)    = (d_{i-1} + d_i)/4
    "cdf53": {"predict": [0.0, 0.5, 0.5], "update": [0.25, 0.25, 0.0]},
}


@dataclass
class Pyramid:
    """Pirámide wavelet de ``L`` niveles.

    Attributes:
        approx: ``(B, C, H/2^L, W/2^L)`` — aproximación LL más gruesa.
        details: lista de longitud ``L``; ``details[i]`` tiene forma
            ``(B, 3C, H/2^(i+1), W/2^(i+1))`` con canales ``[LH, HL, HH]``.
            ``details[0]`` es el nivel más fino (res/2).
    """

    approx: torch.Tensor
    details: list[torch.Tensor]


class _Filter(nn.Module):
    """Filtro depthwise 1D de tamaño ``K`` para una orientación.

    El kernel se guarda con forma ``(C, 1, 1, K)`` (eje ancho) y se transpone
    cuando se aplica sobre el eje alto. Si ``learnable`` es ``False`` queda
    como buffer fijo; si es ``True`` es un ``nn.Parameter`` entrenable.
    """

    def __init__(self, channels: int, kernel: list[float], learnable: bool) -> None:
        super().__init__()
        w = torch.tensor(kernel, dtype=torch.float32)
        w = w.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)  # (C, 1, 1, K)
        if learnable:
            self.weight = nn.Parameter(w)
        else:
            self.register_buffer("weight", w)

    def kernel(self, axis: str, dtype: torch.dtype) -> torch.Tensor:
        """Devuelve el kernel con forma ``(C,1,1,K)`` (ancho) o ``(C,1,K,1)`` (alto)."""
        w = self.weight.to(dtype)
        if axis == "h":
            w = w.transpose(-1, -2).contiguous()
        return w


class LiftingScheme(nn.Module):
    """Wavelet lifting scheme separable 2D con reconstrucción perfecta.

    Args:
        levels: número de niveles de descomposición ``L``.
        in_channels: canales de entrada ``C`` (el detalle apila ``3C``).
        init: ``"haar"`` o ``"cdf53"``; inicialización de ``P``/``U``.
        learnable: si ``True`` los filtros son parámetros entrenables
            inicializados a los coeficientes de ``init``.

    La invertibilidad no depende de los valores de los filtros: ``inverse``
    deshace exactamente ``forward`` para cualquier ``P``/``U``.
    """

    def __init__(
        self,
        levels: int,
        in_channels: int = 3,
        init: str = "haar",
        learnable: bool = False,
    ) -> None:
        super().__init__()
        assert levels >= 1, f"levels debe ser >= 1, recibido {levels}"
        assert init in _INITS, f"init debe ser uno de {sorted(_INITS)}, recibido {init!r}"
        self.levels = levels
        self.in_channels = in_channels
        self.init = init
        self.learnable = learnable
        # replicate funciona con cualquier tamaño espacial (reflect no, si es 1).
        self.pad_mode = "replicate"

        pred = _INITS[init]["predict"]
        upd = _INITS[init]["update"]
        # Cuatro filtros por nivel: P/U para ancho y para alto.
        self.predict_w = nn.ModuleList()
        self.update_w = nn.ModuleList()
        self.predict_h = nn.ModuleList()
        self.update_h = nn.ModuleList()
        for _ in range(levels):
            self.predict_w.append(_Filter(in_channels, pred, learnable))
            self.update_w.append(_Filter(in_channels, upd, learnable))
            self.predict_h.append(_Filter(in_channels, pred, learnable))
            self.update_h.append(_Filter(in_channels, upd, learnable))

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, in_channels={self.in_channels}, "
            f"init={self.init!r}, learnable={self.learnable}"
        )

    # ------------------------------------------------------------------ #
    # utilidades 1D
    # ------------------------------------------------------------------ #
    def _conv_axis(self, x: torch.Tensor, w: torch.Tensor, axis: str) -> torch.Tensor:
        """Conv depthwise de tamaño impar ``K`` sobre ancho (``"w"``) o alto (``"h"``)."""
        k = w.shape[-1] if axis == "w" else w.shape[-2]
        left = (k - 1) // 2
        right = k - 1 - left
        if axis == "w":
            xp = F.pad(x, (left, right, 0, 0), mode=self.pad_mode)
        else:
            xp = F.pad(x, (0, 0, left, right), mode=self.pad_mode)
        return F.conv2d(xp, w, groups=x.size(1))

    @staticmethod
    def _split(x: torch.Tensor, axis: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Split polyphase: muestras pares e impares a lo largo de ``axis``."""
        if axis == "w":
            return x[..., 0::2], x[..., 1::2]
        return x[..., 0::2, :], x[..., 1::2, :]

    @staticmethod
    def _merge(even: torch.Tensor, odd: torch.Tensor, axis: str) -> torch.Tensor:
        """Interleave pares/impares (inverso del split)."""
        if axis == "w":
            b, c, h, w2 = even.shape
            out = even.new_empty(b, c, h, w2 * 2)
            out[..., 0::2] = even
            out[..., 1::2] = odd
        else:
            b, c, h2, w = even.shape
            out = even.new_empty(b, c, h2 * 2, w)
            out[..., 0::2, :] = even
            out[..., 1::2, :] = odd
        return out

    def _lift_axis(
        self,
        x: torch.Tensor,
        p: _Filter,
        u: _Filter,
        axis: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un paso de lifting 1D normalizado: ``(approx, detail)``.

        ``approx = (even + U(d)) * sqrt(2)``, ``detail = -d / sqrt(2)``.
        """
        even, odd = self._split(x, axis)
        d = odd - self._conv_axis(even, p.kernel(axis, dtype), axis)
        s = even + self._conv_axis(d, u.kernel(axis, dtype), axis)
        return s * _SQRT2, -d / _SQRT2

    def _unlift_axis(
        self,
        approx: torch.Tensor,
        detail: torch.Tensor,
        p: _Filter,
        u: _Filter,
        axis: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Inverso exacto de :meth:`_lift_axis`: undo update -> undo predict -> merge."""
        s = approx / _SQRT2
        d = -detail * _SQRT2
        even = s - self._conv_axis(d, u.kernel(axis, dtype), axis)
        odd = d + self._conv_axis(even, p.kernel(axis, dtype), axis)
        return self._merge(even, odd, axis)

    # ------------------------------------------------------------------ #
    # un nivel 2D
    # ------------------------------------------------------------------ #
    def _forward_level(
        self, x: torch.Tensor, lv: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un nivel 2D: devuelve ``LL`` y detalles ``[LH, HL, HH]`` (3C canales)."""
        dtype = x.dtype
        # Ancho: filas -> L (paso-bajo) y H_ (paso-alto).
        low_w, high_w = self._lift_axis(
            x, self.predict_w[lv], self.update_w[lv], "w", dtype
        )
        # Alto sobre L -> LL, LH.
        LL, LH = self._lift_axis(
            low_w, self.predict_h[lv], self.update_h[lv], "h", dtype
        )
        # Alto sobre H_ -> HL, HH.
        HL, HH = self._lift_axis(
            high_w, self.predict_h[lv], self.update_h[lv], "h", dtype
        )
        return LL, torch.cat([LH, HL, HH], dim=1)

    def _inverse_level(
        self, LL: torch.Tensor, details: torch.Tensor, lv: int
    ) -> torch.Tensor:
        """Inverso de :meth:`_forward_level`."""
        dtype = LL.dtype
        LH, HL, HH = details.chunk(3, dim=1)
        # Deshace alto.
        low_w = self._unlift_axis(
            LL, LH, self.predict_h[lv], self.update_h[lv], "h", dtype
        )
        high_w = self._unlift_axis(
            HL, HH, self.predict_h[lv], self.update_h[lv], "h", dtype
        )
        # Deshace ancho.
        return self._unlift_axis(
            low_w, high_w, self.predict_w[lv], self.update_w[lv], "w", dtype
        )

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Pyramid:
        """Descompone ``x`` ``(B, C, H, W)`` en una :class:`Pyramid`.

        Args:
            x: tensor ``(B, C, H, W)`` con ``H, W`` divisibles por ``2^levels``.

        Returns:
            Pyramid con ``approx`` (LL más gruesa) y ``details`` (finas->gruesas).
        """
        assert x.dim() == 4, f"x debe ser (B, C, H, W), recibido {tuple(x.shape)}"
        b, c, h, w = x.shape
        assert c == self.in_channels, (
            f"canales {c} != in_channels {self.in_channels}"
        )
        step = 2 ** self.levels
        assert h % step == 0 and w % step == 0, (
            f"H={h} y W={w} deben ser divisibles por 2^levels={step} "
            f"(levels={self.levels}); usa padding o menos niveles."
        )

        # En precisión reducida calculamos en fp32 y solo redondeamos las
        # salidas (patrón estándar de mixed precision); evita acumular error
        # de redondeo en cada paso de lifting.
        out_dtype = x.dtype
        compute = self._compute_dtype(out_dtype)
        approx = x.to(compute)
        details: list[torch.Tensor] = []
        for lv in range(self.levels):
            approx, det = self._forward_level(approx, lv)
            details.append(det.to(out_dtype))
        return Pyramid(approx=approx.to(out_dtype), details=details)

    def inverse(self, p: Pyramid) -> torch.Tensor:
        """Reconstruye ``x`` ``(B, C, H, W)`` desde una :class:`Pyramid`."""
        assert len(p.details) == self.levels, (
            f"la pirámide tiene {len(p.details)} niveles, se esperaban {self.levels}"
        )
        out_dtype = p.approx.dtype
        compute = self._compute_dtype(out_dtype)
        approx = p.approx.to(compute)
        for lv in reversed(range(self.levels)):
            details = p.details[lv].to(compute)
            approx = self._inverse_level(approx, details, lv)
        return approx.to(out_dtype)

    @staticmethod
    def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
        """Precisión interna: fp32 para tipos reducidos (fp16/bf16)."""
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype
