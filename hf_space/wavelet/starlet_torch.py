from __future__ import annotations

import torch
import torch.nn as nn

FILTERS: dict[str, list[float]] = {
    "haar": [0.5, 0.5],
    "b3":   [1/16, 1/4, 3/8, 1/4, 1/16],
}

def reflect_pad2d(x, pl, pr, pt, pb):
    """ Cuando el padding es mayor que el tamaño de la imagen, es necesario aplicar reflect_pad varias veces"""
    while pl > 0 or pr > 0:
        s = x.shape[-1] - 1
        a, b = min(pl, s), min(pr, s)
        if a == 0 and b == 0:
            break
        x = nn.functional.pad(x, (a, b, 0, 0), mode="reflect")
        pl -= a; pr -= b
    while pt > 0 or pb > 0:
        s = x.shape[-2] - 1
        a, b = min(pt, s), min(pb, s)
        if a == 0 and b == 0:
            break
        x = nn.functional.pad(x, (0, 0, a, b), mode="reflect")
        pt -= a; pb -= b
    return x


def starlet_conv4d(
    x: torch.Tensor,
    levels: int,
    scale: torch.Tensor | None = None,
    filter: str = "b3",
) -> torch.Tensor:
    """à trous decomp of (B, C, H, W).

    Returns ``(B, C*(levels+1), H, W)``` — coefficient levels stacked on channels.

    Each level uses a separable scaling-function filter with dilation
    ``2**(level-1)``.  Choose from ``FILTERS`` keys via *filter*.

    If *scale* is provided ``(levels+1,)``, each coefficient level is multiplied
    by its corresponding scalar before concatenation (learnable level weights).
    """
    results = []
    approx = x
    lp = torch.tensor(FILTERS[filter], dtype=x.dtype, device=x.device)

    for lv in range(1, levels + 1):
        dist = 2 ** (lv - 1)
        
        k_len = 1 + (len(lp) - 1) * dist
        kern = torch.zeros(k_len, dtype=x.dtype, device=x.device)
        for i, v in enumerate(lp):
            kern[i * dist] = v

        kh = kern.view(1, 1, -1, 1).repeat(x.size(1), 1, 1, 1)
        kw = kern.view(1, 1, 1, -1).repeat(x.size(1), 1, 1, 1)

        pl = (k_len - 1) // 2
        pr = (k_len - 1) - pl
        # a = nn.functional.pad(approx, (pl, pr, pl, pr))
        # a = nn.functional.pad(approx, (pl, pr, pl, pr), mode="reflect")
        a = reflect_pad2d(approx, pl, pr, pl, pr)
        a = nn.functional.conv2d(a, kh, groups=x.size(1))
        a = nn.functional.conv2d(a, kw, groups=x.size(1))

        # Each level's "approx" is the pre-smoothing, residual = prev_approx - smooth.
        results.append(approx - a)
        approx = a

    results.append(approx)
    out = torch.stack(results, dim=1)  # (B, L+1, C, H, W)
    if scale is not None:
        out = out * scale.view(1, -1, 1, 1, 1)
    return out.reshape(x.size(0), -1, *x.shape[2:])  # (B, C*(L+1), H, W)


if __name__ == "__main__":
    for dtype in [torch.float32, torch.float64]:
        for ch in [3] + ([1] if dtype == torch.float64 else []):
            x = torch.randn(2, ch, 64, 64, dtype=dtype)
            c = starlet_conv4d(x, levels=4)
            recon = c.reshape(2, -1, ch, 64, 64).sum(dim=1)
            err = (recon - x).abs().max().item()
            print(f"[ok] {dtype} shape={x.shape} → {c.shape}, err={err:.2e}")
