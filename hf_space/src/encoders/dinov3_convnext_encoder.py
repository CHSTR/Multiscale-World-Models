"""DINOv3 ConvNeXt encoder wrapper for the JEPA pipeline.

Wraps the official DINOv3 ConvNeXt backbones (``tiny`` / ``small`` / ``base``,
embed dim 768 / 768 / 1024) loaded through
``src.dinov3.hub.backbones.dinov3_convnext_*``. Pesos: ``ckpt_path`` explícito >
``$DINOV3_CONVNEXT_CKPT`` > descarga por hub oficial (``ckpt_path=null`` =
portable entre máquinas).

Interface (compatible with ``jepa.JEPA.encode``, igual que
``src.encoders.dinov3_encoder.DinoV3Encoder``):

    out = encoder(pixel_values, interpolate_pos_encoding=True)
    emb = out.last_hidden_state[:, 0]      # CLS token -> (B, D)

``last_hidden_state`` is ``(B, 1 + N, D)`` with the CLS token first (storage
tokens follow if present, then patch tokens). ConvNeXt has no register/storage
tokens, so the sequence is ``[CLS, patch...]``.

Multicanal / starlet support (mirror of the ViT encoder): the ConvNeXt stem is
``downsample_layers[0][0]`` (a single ``Conv2d`` with k=4/s=4). With
``in_chans > 3`` (e.g. 15 for a starlet/L=3 frontend) the stem conv is expanded
to the new channel count (RGB weights copied, rest = mean) and made trainable.
LoRA is applied to the block ``pwconv1`` / ``pwconv2`` and everything except
LoRA + norm + stem is frozen (the ``LayerNorm`` custom class is recognized by
the generic freeze because its class name contains "layernorm").

Starlet frontend (igual que el modelo ViT): con ``starlet_levels=L > 0`` el
wrapper aplica ``starlet_conv4d`` a los frames RGB en ``forward`` y deriva
``in_chans = 3*(L+1)`` del int (ignora el ``in_chans`` manual). Con
``starlet_levels=0`` entra RGB puro.
"""

import os
import types
from typing import Optional, Tuple

import torch
from torch import nn

from wavelet.starlet_torch import starlet_conv4d

from src.encoders.vit_lora import (
    apply_lora_to_vit,
    count_total_parameters,
    count_trainable_parameters,
    freeze_except_lora_norm_patch,
)


# Portable default: $DINOV3_CONVNEXT_CKPT si está definido, si no None (= descarga por hub).
DEFAULT_DINOV3_CONVNEXT_CKPT = os.environ.get("DINOV3_CONVNEXT_CKPT") or None

# embed dim per ConvNeXt size.
_SIZE_EMBED_DIM = {"tiny": 768, "small": 768, "base": 1024}


def _expand_convnext_stem(model: nn.Module, in_chans: int) -> nn.Conv2d:
    """Expand the ConvNeXt stem conv to ``in_chans`` input channels.

    The stem is ``downsample_layers[0][0]`` (a single ``Conv2d`` with k=4/s=4).
    The new conv keeps the original ``out_channels`` / kernel / stride and has
    its first ``in_channels`` weights copied from the pretrained RGB weights;
    the remaining channels are filled with the per-output-channel mean
    (``fill="mean"``), mirroring ``expand_patch_embed``. The (new) conv is left
    trainable so the pretrained backbone can consume wavelet coefficient
    channels.
    """
    stem = model.downsample_layers[0]
    conv = stem[0]
    old_in = conv.in_channels
    old_weight = conv.weight.detach()  # (out, in, kH, kW)
    assert old_in > 0, f"cannot expand from {old_in} input channels"
    assert old_in <= in_chans, (
        f"existing stem conv already has {old_in} in_channels, cannot shrink to {in_chans}"
    )

    new_conv = nn.Conv2d(
        in_chans,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :old_in].copy_(old_weight)
        new_conv.weight[:, old_in:].copy_(old_weight.mean(dim=1, keepdim=True))

    # ``stem`` is a live reference from .modules() (a Sequential); replacing an
    # element in-place keeps the model graph intact.
    stem[0] = new_conv
    return new_conv


class DinoV3ConvNeXtEncoder(nn.Module):
    """DINOv3 ConvNeXt (tiny/small/base) encoder with optional LoRA (pwconv) + starlet frontend."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_path: Optional[str] = DEFAULT_DINOV3_CONVNEXT_CKPT,
        size: str = "tiny",
        img_size: int = 224,
        in_chans: int = 3,
        patch_embed_mode: str = "adapt",
        starlet_levels: int = 0,
        starlet_filter: str = "b3",
        starlet_learnable_weights: bool = True,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        lora_targets: Tuple[str, ...] = ("pwconv1", "pwconv2"),
        device: str = "cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.size = size
        self.patch_embed_mode = patch_embed_mode
        # Frontend starlet (igual que wavelet.starlet_encoder.StarletEncoder):
        # el int manda y deriva in_chans = 3*(L+1).
        self.starlet_levels = int(starlet_levels)
        self.starlet_filter = starlet_filter
        if self.starlet_levels > 0:
            in_chans = 3 * (self.starlet_levels + 1)
            w = torch.ones(self.starlet_levels + 1)
            if starlet_learnable_weights:
                self.level_weights = nn.Parameter(w)
            else:
                self.register_buffer("level_weights", w)
        self.in_chans = in_chans

        if size not in _SIZE_EMBED_DIM:
            raise ValueError(
                f"unknown ConvNeXt size '{size}'; expected one of {list(_SIZE_EMBED_DIM)}"
            )

        from src.dinov3.hub.backbones import (
            dinov3_convnext_base,
            dinov3_convnext_small,
            dinov3_convnext_tiny,
        )

        builders = {
            "tiny": dinov3_convnext_tiny,
            "small": dinov3_convnext_small,
            "base": dinov3_convnext_base,
        }
        build = builders[size]

        # Explicit path > $DINOV3_CONVNEXT_CKPT > hub (download). Local paths are
        # loaded strict by the hub; the backbone always builds in_chans=3.
        resolved_ckpt = ckpt_path or os.environ.get("DINOV3_CONVNEXT_CKPT") or None
        if resolved_ckpt is not None:
            if not os.path.isfile(resolved_ckpt):
                raise FileNotFoundError(
                    f"DINOv3 ConvNeXt ckpt no encontrado: {resolved_ckpt}. "
                    "Pasa model.encoder.ckpt_path=<ruta> o exporta "
                    "DINOV3_CONVNEXT_CKPT=<ruta>, o deja ckpt_path=null para descargar por hub."
                )
            self.model = build(weights=resolved_ckpt)
        else:
            self.model = build(pretrained=pretrained)

        # ConvNeXt has no PatchEmbed: the "patch embed" is the stem conv, so the
        # generic expand_patch_embed doesn't apply. Expand the stem instead.
        if in_chans > 3 and patch_embed_mode == "adapt":
            _expand_convnext_stem(self.model, in_chans)

        self.embed_dim = int(self.model.embed_dim)

        # LoRA on the ConvNeXt block pointwise MLPs (pwconv1 / pwconv2).
        self._lora_layers = apply_lora_to_vit(
            self.model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            targets=lora_targets,
            add_convnext=True,
        )
        freeze_except_lora_norm_patch(self.model)

        # The generic freeze only enables LoRA + norm + PatchEmbed modules; for
        # ConvNeXt the "patch embed" is the stem conv, which must stay trainable
        # (its LayerNorms are already enabled by the freeze above).
        for p in self.model.downsample_layers[0].parameters():
            p.requires_grad_(True)

        self.to(device)
        n_storage = getattr(self.model, "n_storage_tokens", 0)
        print(
            f"DinoV3ConvNeXtEncoder({size}, dim={self.embed_dim}, in_chans={in_chans}, "
            f"starlet_levels={self.starlet_levels}, "
            f"n_storage={n_storage}, lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,} params]"
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True,
                prompt: Optional[torch.Tensor] = None, **kwargs):
        """Return an object exposing ``last_hidden_state`` (CLS first)."""
        x = pixel_values
        if self.starlet_levels > 0:
            x = starlet_conv4d(x, self.starlet_levels, scale=self.level_weights, filter=self.starlet_filter)
        # ``is_training=True`` makes the backbone return the feature dict from
        # which we assemble the full (normed) token sequence: CLS, storage
        # tokens (none for ConvNeXt), then patch tokens.
        features = self.model(x, is_training=True)

        cls = features["x_norm_clstoken"]  # (B, D)
        storage = features["x_storage_tokens"]  # (B, n_storage, D)
        patches = features["x_norm_patchtokens"]  # (B, N, D)
        parts = [cls.unsqueeze(1), patches]
        if storage is not None and storage.shape[1] > 0:
            parts.insert(1, storage)
        last_hidden_state = torch.cat(parts, dim=1)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)


def build_dinov3_convnext_encoder(**kwargs) -> DinoV3ConvNeXtEncoder:
    """Convenience factory (e.g. for hydra ``_target_``)."""
    return DinoV3ConvNeXtEncoder(**kwargs)
