import torch
from torch import nn

from .frontend import BandAblationMixin, band_names
from .starlet_torch import starlet_conv4d


class StarletEncoder(BandAblationMixin, nn.Module):
    """Frontend that replaces RGB with (levels+1)×RGB wavelet coefficients.
    """

    def __init__(self, vit, levels: int = 4, filter: str = "b3",
                 learnable_weights: bool = True):
        super().__init__()
        self.levels = levels
        self.filter = filter
        w = torch.ones(levels + 1)
        if learnable_weights:
            self.level_weights = nn.Parameter(w)
        else:
            self.register_buffer("level_weights", w)
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
            # Reusa los pesos RGB en los canales de w_1 (el detalle más fino); el resto queda con la
            # init aleatoria por defecto de nn.Conv2d (no en cero). En un ViT desde cero da igual.
            new_conv.weight[:, :old_pe.in_channels].copy_(old_pe.weight)

    @property
    def band_names(self):
        return band_names("starlet", self.levels)

    def forward(self, pixel_values, **kwargs):
        x = starlet_conv4d(pixel_values, self.levels, scale=self.level_weights, filter=self.filter)
        if getattr(self, "_band_ablation", None) is not None or getattr(self, "_band_scale", None) is not None:
            B, _, H, W = x.shape
            bands = self._apply_band_controls(x.view(B, self.levels + 1, -1, H, W))
            x = bands.flatten(1, 2)
        return self.vit(x, **kwargs)
