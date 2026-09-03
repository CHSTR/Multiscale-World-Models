"""Small DINOv3 ViT encoder wrapper for the JEPA pipeline.

Replaces the previous ``vit_base`` / hardcoded-path wrapper with a *small*
model (``vit_small`` / patch-16, embed dim 384) loaded through the official hub
backbone ``src.dinov3.hub.backbones.dinov3_vits16``. By default the checkpoint
at ``ckpt_path`` (a local copy of the LVD1689M vit_small/16 weights) is used.

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
"""

import types
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.encoders.vit_lora import (
    apply_lora_to_vit,
    count_total_parameters,
    count_trainable_parameters,
    expand_patch_embed,
    freeze_except_lora_norm_patch,
)


DEFAULT_DINOV3_CKPT = "/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


class DinoV3Encoder(nn.Module):
    """ViT-small / patch-16 (dim 384) DINOv3 encoder with optional LoRA + multicanal patch embed."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_path: Optional[str] = DEFAULT_DINOV3_CKPT,
        img_size: int = 224,
        in_chans: int = 3,
        patch_embed_mode: str = "adapt",
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        lora_targets: Tuple[str, ...] = ("qkv", "proj"),
        device: str = "cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = 384
        self.patch_embed_mode = patch_embed_mode

        from src.dinov3.hub.backbones import dinov3_vits16

        # NOTE: dinov3_vits16 hardcodes img_size=224 (matching the checkpoint);
        # the ``img_size`` arg is accepted for API parity but the backbone
        # resolution is fixed. Variable input sizes are still handled at runtime
        # by JEPA's ``interpolate_pos_encoding=True``.
        if ckpt_path is not None:
            self.model = dinov3_vits16(weights=ckpt_path)
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
            f"n_storage={n_storage}, lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,} params]"
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True,
                prompt: Optional[torch.Tensor] = None, **kwargs):
        """Return an object exposing ``last_hidden_state`` (CLS first)."""
        # ``is_training=True`` makes the backbone return the feature dict from
        # which we assemble the full (normed) token sequence: CLS, storage
        # tokens, then patch tokens.
        features = self.model(pixel_values, prompt=prompt, is_training=True)

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
