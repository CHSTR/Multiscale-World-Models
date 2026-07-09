"""Starlet wrapper: decompose images, stack coefficients as channels, feed ViT."""

import torch
from torch import nn

from .starlet_torch import starlet_conv4d


class StarletEncoder(nn.Module):
    """Frontend that replaces RGB with (levels+1)×RGB wavelet coefficients.

    pixel_values : (B, 3, H, W) ──→ ViT sees (B, 3·(levels+1), H, W).
    The patch-embedding Conv2d is replaced so the channel count matches.

    ``level_weights`` is a scalar per decomposition level (init 1).
    If ``learnable_weights=True`` (default) the model can tune them via backprop.
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

        # Replace the first conv with one that accepts starlet channels
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

        # Copy old weights into the first 3 channels so it starts from a
        # valid ViT-tiny init (ponytail: lazy reuse, works because starlet
        # level sums back to the original image).
        vit.config.num_channels = new_in_c
        vit.embeddings.patch_embeddings.num_channels = new_in_c
        with torch.no_grad():
            new_conv = vit.embeddings.patch_embeddings.projection
            # ponytail: reuse old RGB weights for first-level channels; rest starts zero.
            new_conv.weight[:, :old_pe.in_channels].copy_(old_pe.weight)

    def forward(self, pixel_values, **kwargs):
        x = starlet_conv4d(pixel_values, self.levels, scale=self.level_weights, filter=self.filter)
        return self.vit(x, **kwargs)
