"""Control de canales para el track desde cero (spec §4, Task 5).

``RGBRepeatEncoder`` es el espejo exacto de ``StarletEncoder`` pero sin
estructura de escala: reemplaza RGB por (L+1) copias de ``x/(L+1)``.

- Mismo número de canales que Starlet(J=L): ``3·(L+1)`` -> misma capacidad
  del patch embedding (aisla "más canales" de "estructura de escala").
- Σ bandas = x (misma propiedad que la descomposición starlet).
- Init "equivalente": el peso RGB del patch embedding se replica en las
  (L+1) copias SIN reescalar, y cada banda entra dividida por (L+1), así
  ``conv(W replicado, bandas) = conv(W, x)``: en el paso 0 el encoder ES
  el ViT plano y cualquier diferencia que aparezca al entrenar es
  atributable a la estructura de escala, no a la inicialización.
"""
import torch
from torch import nn


class RGBRepeatEncoder(nn.Module):
    """Frontend de control: (levels+1) copias de x/(levels+1) sin multiescala."""

    def __init__(self, vit, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.vit = vit

        old_pe = vit.embeddings.patch_embeddings.projection
        new_in_c = old_pe.in_channels * (levels + 1)
        vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            new_in_c,
            old_pe.out_channels,
            kernel_size=old_pe.kernel_size,
            stride=old_pe.stride,
            padding=old_pe.padding,
            bias=old_pe.bias is not None,
        )
        vit.config.num_channels = new_in_c
        vit.embeddings.patch_embeddings.num_channels = new_in_c
        with torch.no_grad():
            new_conv = vit.embeddings.patch_embeddings.projection
            # init equivalente: W_rgb replicada en cada bloque, sin reescalar
            for j in range(levels + 1):
                new_conv.weight[:, j * old_pe.in_channels:(j + 1) * old_pe.in_channels] \
                    .copy_(old_pe.weight)
            if old_pe.bias is not None:
                new_conv.bias.copy_(old_pe.bias)

    def forward(self, pixel_values, **kwargs):
        n = self.levels + 1
        x = pixel_values.repeat(1, n, 1, 1) / n
        return self.vit(x, **kwargs)
