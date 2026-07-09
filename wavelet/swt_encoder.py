"""SWT wrapper: descompone imágenes, apila coeficientes orientados como canales, feed ViT."""

from torch import nn

from .swt_torch import swt_conv4d


class SWTEncoder(nn.Module):
    """Frontend que reemplaza RGB con coeficientes wavelet orientados SWT.

    pixel_values : (B, 3, H, W) → SWT produce (B, 3·(3*levels+1), H, W) →
    proyección 1×1 aprendible comprime a (B, 3·(levels+1), H, W) → ViT ve esa.

    La proyección colapsa las 3 orientaciones por nivel (LH/HL/HH) a una banda,
    manteniendo el mismo canalaje que StarletEncoder para comparar justo.
    """

    def __init__(self, vit, levels: int = 4, filter: str = "bior2.2"):
        super().__init__()
        self.levels = levels
        self.filter = filter
        self.vit = vit

        old_pe = vit.embeddings.patch_embeddings.projection
        C = old_pe.in_channels
        swt_out_c = C * (3 * levels + 1)
        new_in_c = C * (levels + 1)  # mismo canalaje que StarletEncoder

        # Proyección 1×1 aprendible: 3 orientaciones por nivel → 1 banda
        self.channel_proj = nn.Conv2d(swt_out_c, new_in_c, kernel_size=1, bias=False)

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

    def forward(self, pixel_values, **kwargs):
        x = swt_conv4d(pixel_values, self.levels, filter=self.filter)  # (B, 3·(3L+1), H, W)
        x = self.channel_proj(x)                                      # (B, 3·(L+1), H, W)
        return self.vit(x, **kwargs)