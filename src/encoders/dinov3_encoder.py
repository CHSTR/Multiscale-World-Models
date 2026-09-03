"""Small DINOv3 ViT encoder wrapper for the JEPA pipeline.

Replaces the previous ``vit_base`` / hardcoded-path wrapper with a *small*
model (``vit_small`` / patch-16, embed dim 384) loaded through the official hub
backbone ``src.dinov3.hub.backbones.dinov3_vits16``. Pesos: ``ckpt_path``
explícito > ``$DINOV3_CKPT`` > descarga por hub oficial (``ckpt_path=null``
= portable entre máquinas).

Interface (compatible with ``jepa.JEPA.encode``):

    out = encoder(pixel_values, interpolate_pos_encoding=True)
    emb = out.last_hidden_state[:, 0]      # CLS token -> (B, 384)

``last_hidden_state`` is ``(B, 1 + n_storage + N, 384)`` with the CLS token
first (storage tokens follow, then patch tokens).

Multicanal support: with ``in_chans=3`` the original patch embedding is kept.
With ``in_chans > 3`` the ``patch_embed`` conv is expanded to the new channel
count (RGB weights copied, rest = mean) and made trainable. LoRA is applied to
the attention qkv/proj and everything except LoRA + norm + patch_embed (+
storage/register tokens) is frozen.

Starlet frontend (igual que el modelo original): con ``starlet_levels=L > 0``
el wrapper aplica ``starlet_conv4d`` a los frames RGB en ``forward`` y deriva
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
    expand_patch_embed,
    freeze_except_lora_norm_patch,
)


# Portable default: $DINOV3_CKPT si está definido, si no None (= descarga por hub).
DEFAULT_DINOV3_CKPT = os.environ.get("DINOV3_CKPT") or None


class DinoV3Encoder(nn.Module):
    """ViT-small / patch-16 (dim 384) DINOv3 encoder with optional LoRA + multicanal patch embed."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_path: Optional[str] = DEFAULT_DINOV3_CKPT,
        img_size: int = 224,
        in_chans: int = 3,
        patch_embed_mode: str = "adapt",
        starlet_levels: int = 0,
        starlet_filter: str = "b3",
        starlet_learnable_weights: bool = True,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        lora_targets: Tuple[str, ...] = ("qkv", "proj"),
        device: str = "cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = 384
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

        from src.dinov3.hub.backbones import dinov3_vits16

        # NOTE: dinov3_vits16 hardcodes img_size=224 (matching the checkpoint);
        # the ``img_size`` arg is accepted for API parity but the backbone
        # resolution is fixed. Variable input sizes are still handled at runtime
        # by JEPA's ``interpolate_pos_encoding=True``.
        # Resolución: arg explícito > $DINOV3_CKPT > hub oficial.
        # (ckpt_path=None en el yaml = portable entre máquinas.)
        resolved_ckpt = ckpt_path or os.environ.get("DINOV3_CKPT") or None
        if resolved_ckpt is not None:
            if not os.path.isfile(resolved_ckpt):
                raise FileNotFoundError(
                    f"DINOv3 ckpt no encontrado: {resolved_ckpt}. "
                    "Pasa model.encoder.ckpt_path=<ruta> o exporta DINOV3_CKPT=<ruta>, "
                    "o deja ckpt_path=null para descargar por hub."
                )
            self.model = dinov3_vits16(weights=resolved_ckpt)
        else:
            self.model = dinov3_vits16(pretrained=pretrained)

        if in_chans > 3 and patch_embed_mode == "adapt":
            expand_patch_embed(self.model, in_chans, fill="mean")

        self._lora_layers = apply_lora_to_vit(
            self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets
        )
        freeze_except_lora_norm_patch(self.model)

        self.to(device)
        n_storage = getattr(self.model, "n_storage_tokens", 0)
        print(
            f"DinoV3Encoder(vits16, dim={self.embed_dim}, in_chans={in_chans}, "
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
        # tokens, then patch tokens.
        features = self.model(x, prompt=prompt, is_training=True)

        cls = features["x_norm_clstoken"]  # (B, 384)
        storage = features["x_storage_tokens"]  # (B, n_storage, 384)
        patches = features["x_norm_patchtokens"]  # (B, N, 384)
        parts = [cls.unsqueeze(1), patches]
        if storage is not None and storage.shape[1] > 0:
            parts.insert(1, storage)
        last_hidden_state = torch.cat(parts, dim=1)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)


def build_dinov3_encoder(**kwargs) -> DinoV3Encoder:
    """Convenience factory (e.g. for hydra ``_target_``)."""
    return DinoV3Encoder(**kwargs)
