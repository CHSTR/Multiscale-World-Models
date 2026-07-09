"""PyTorch SWT (à trous) no diezmada en GPU."""

from __future__ import annotations

import torch
import torch.nn as nn

# Pares de filtros (análisis): pasa-bajo, pasa-alto
FILTERS_LH: dict[str, tuple[list[float], list[float]]] = {
    "haar": ([0.70710678, 0.70710678],
             [0.70710678, -0.70710678]),
    # CDF 5/3 (bior2.2), simétrico → coherente con el B3 simétrico del starlet
    "bior2.2": ([-0.125, 0.25, 0.75, 0.25, -0.125],
                [-0.5, 1.0, -0.5]),
}


def _dilated(f, dist, device, dtype):
    """Kernel 1D con ceros intercalados (à trous)."""
    k = torch.zeros(1 + (len(f) - 1) * dist, dtype=dtype, device=device)
    k[::dist] = torch.tensor(f, dtype=dtype, device=device)
    return k


def _conv_axis(x, kern, axis, C):
    """Conv 1D separable a lo largo de un eje (2=filas, 3=cols), reflect, mismo tamaño."""
    n = kern.numel()
    pl, pr = (n - 1) // 2, (n - 1) - (n - 1) // 2
    if axis == 2:
        w = kern.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        x = nn.functional.pad(x, (0, 0, pl, pr), mode="reflect")
    else:
        w = kern.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        x = nn.functional.pad(x, (pl, pr, 0, 0), mode="reflect")
    return nn.functional.conv2d(x, w, groups=C)


def swt_conv4d(x: torch.Tensor, levels: int, scale: torch.Tensor | None = None, filter: str = "bior2.2") -> torch.Tensor:
    """SWT no diezmada (à trous) de (B, C, H, W).

    Devuelve (B, C*(3*levels+1), H, W): por nivel, 3 detalles orientados
    (LH, HL, HH) + la aproximación final, apilados en canales.
    Misma interfaz que starlet_conv4d. OJO: NO es aditiva (no la valides con la suma).
    """
    C = x.size(1)
    lo, hi = FILTERS_LH[filter]
    results, approx = [], x

    for lv in range(1, levels + 1):
        dist = 2 ** (lv - 1)
        lo_d = _dilated(lo, dist, x.device, x.dtype)
        hi_d = _dilated(hi, dist, x.device, x.dtype)

        L = _conv_axis(approx, lo_d, 2, C)   # filas pasa-bajo
        H = _conv_axis(approx, hi_d, 2, C)   # filas pasa-alto
        LH = _conv_axis(L, hi_d, 3, C)       # detalle orientado
        HL = _conv_axis(H, lo_d, 3, C)       # detalle orientado
        HH = _conv_axis(H, hi_d, 3, C)       # detalle orientado
        approx = _conv_axis(L, lo_d, 3, C)   # LL → siguiente aproximación

        results.extend([LH, HL, HH])
    results.append(approx)

    out = torch.stack(results, dim=1)        # (B, 3*levels+1, C, H, W)
    if scale is not None:
        out = out * scale.view(1, -1, 1, 1, 1)
    return out.reshape(x.size(0), -1, *x.shape[2:])  # (B, C*(3*levels+1), H, W)


if __name__ == "__main__":
    # ponytail: chequeo mínimo — forma, backward, cuenta de bandas.
    # No chequeamos aditividad (la SWT no la cumple).
    for fname in FILTERS_LH:
        for ch in (3, 1):
            x = torch.randn(2, ch, 64, 64, requires_grad=True)
            c = swt_conv4d(x, levels=4, filter=fname)
            loss = c.pow(2).mean()
            loss.backward()
            expected = ch * (3 * 4 + 1)  # 3*levels+1 bandas por canal
            assert c.shape == (2, expected, 64, 64), f"{fname}: {c.shape} != {(2, expected, 64, 64)}"
            assert x.grad is not None
            print(f"[ok] {fname} C={ch}: {tuple(x.shape)} → {tuple(c.shape)}, grad OK")
    print("OK")