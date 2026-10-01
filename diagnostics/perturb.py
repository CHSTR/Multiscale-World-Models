"""Perturbaciones de imagen para evaluar robustez (se aplican en [0, 1], antes de normalizar).

- blur        : desenfoque gaussiano, ``strength`` = sigma en píxeles (quita altas frecuencias).
- noise       : ruido gaussiano aditivo i.i.d., ``strength`` = desviación en [0, 1].
- downsample  : reduce por ``strength`` (p. ej. 2, 4) y vuelve a subir: pérdida de resolución.
- brightness  : multiplica por (1 + strength): cambio de iluminación de baja frecuencia (vive en c_J).
- none        : identidad.

Uso típico (dentro de un ``transforms.Compose`` de torchvision, después de ``ToDtype(scale=True)``):
    ImagePerturbation("blur", 1.5)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PERTURBATIONS = ("none", "blur", "noise", "downsample", "brightness")


def _gaussian_kernel1d(sigma: float, dtype, device) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


class ImagePerturbation:
    def __init__(self, kind: str = "none", strength: float = 0.0, seed: int = 0):
        if kind not in PERTURBATIONS:
            raise ValueError(f"perturbación desconocida: {kind!r} (opciones: {PERTURBATIONS})")
        self.kind = kind
        self.strength = float(strength)
        self.gen = torch.Generator().manual_seed(int(seed))

    def __repr__(self) -> str:
        return f"ImagePerturbation(kind={self.kind!r}, strength={self.strength})"

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) o (B, C, H, W), float en [0, 1]."""
        if self.kind == "none" or self.strength == 0:
            return img
        squeeze = img.dim() == 3
        x = img.unsqueeze(0) if squeeze else img
        x = x.float()
        if self.kind == "blur":
            k = _gaussian_kernel1d(self.strength, x.dtype, x.device)
            C, r = x.shape[1], (k.numel() - 1) // 2
            x = F.pad(x, (r, r, r, r), mode="reflect")
            x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
            x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
        elif self.kind == "noise":
            noise = torch.randn(x.shape, generator=self.gen).to(device=x.device, dtype=x.dtype)
            x = (x + self.strength * noise).clamp(0, 1)
        elif self.kind == "downsample":
            f = float(self.strength)
            if f <= 1:
                return img
            H, W = x.shape[-2:]
            small = F.interpolate(x, size=(max(1, int(H / f)), max(1, int(W / f))), mode="area")
            x = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        elif self.kind == "brightness":
            x = (x * (1.0 + self.strength)).clamp(0, 1)
        x = x.to(img.dtype)
        x = x.squeeze(0) if squeeze else x
        try:  # conserva el tipo tv_tensors.Image dentro de pipelines torchvision v2
            from torchvision import tv_tensors

            if isinstance(img, tv_tensors.TVTensor):
                return tv_tensors.wrap(x, like=img)
        except ImportError:
            pass
        return x
