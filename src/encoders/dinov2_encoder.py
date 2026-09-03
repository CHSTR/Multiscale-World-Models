"""Small DINOv2 ViT encoder wrapper for the JEPA pipeline.

Replaces the previous ``vit_base`` / hardcoded-path wrapper with a *small*
model (``vit_small`` / patch 14, embed dim 384) loaded through the official
hub backbone ``src.dinov2.hub.backbones.dinov2_vits14``.

Interface (compatible with ``jepa.JEPA.encode``):

    out = encoder(pixel_values, interpolate_pos_encoding=True)
    emb = out.last_hidden_state[:, 0]      # CLS token -> (B, 384)

``last_hidden_state`` is ``(B, 1+N, 384)`` with the CLS token first.

Multicanal support: with ``in_chans=3`` the original patch embedding is kept.
With ``in_chans > 3`` (e.g. 15 for a starlet/L=4 frontend) the ``patch_embed``
conv is expanded to the new channel count (RGB weights copied, rest = mean) and
made trainable, so the pretrained backbone can consume wavelet coefficient
cannels. LoRA is applied to the attention qkv/proj and everything except LoRA +
norm + patch_embed (+ storage/register tokens) is frozen.
"""

import types
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.encoders.vit_lora import (
    apply_lora_to_vit,
    count_total_parameters,
    count_trainable_parameters,
    expand_patch_embed,
    freeze_except_lora_norm_patch,
)


class DinoV2Encoder(nn.Module):
    """ViT-small / patch-14 (dim 384) DINOv2 encoder with optional LoRA + multicanal patch embed."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_path: Optional[str] = None,
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

        from src.dinov2.hub.backbones import dinov2_vits14

        # Build a 3-channel backbone (pretrained weights loaded into the original
        # patch-embedding channels), then expand the patch embed afterwards.
        if ckpt_path is not None:
            self.model = dinov2_vits14(pretrained=False, img_size=img_size)
            state_dict = torch.load(ckpt_path, map_location=device)
            # Official checkpoints are plain state_dicts; some wrappers wrap them
            # under a "model" key -- unwrap in that case.
            if isinstance(state_dict, dict) and "model" in state_dict and len(state_dict) == 1:
                state_dict = state_dict["model"]
            _interp_pos_embed_to_target(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # ckpt_path is None: pull the official hub weights ourselves so we can
            # interpolate their (518-res) pos_embed to img_size before loading.
            self.model = dinov2_vits14(pretrained=False, img_size=img_size)
            from ..dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name

            model_full_name = _make_dinov2_model_name("vit_small", 14)
            url = f"{_DINOV2_BASE_URL}/{_make_dinov2_model_name('vit_small', 14)}/{model_full_name}_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
            _interp_pos_embed_to_target(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=False)

        if in_chans > 3 and patch_embed_mode == "adapt":
            expand_patch_embed(self.model, in_chans, fill="mean")

        self._lora_layers = apply_lora_to_vit(
            self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets
        )
        freeze_except_lora_norm_patch(self.model)

        self.to(device)
        print(
            f"DinoV2Encoder(vits14, dim={self.embed_dim}, in_chans={in_chans}, "
            f"lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,} params]"
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True,
                prompt: Optional[torch.Tensor] = None, **kwargs):
        """Return an object exposing ``last_hidden_state`` ``(B, 1+N, 384)`` (CLS first)."""
        # ``is_training=True`` makes the backbone return the feature dict, from
        # which we assemble the full (normed) token sequence with CLS first.
        features = self.model(pixel_values, prompt=prompt, is_training=True)

        cls = features["x_norm_clstoken"]  # (B, 384)
        patches = features["x_norm_patchtokens"]  # (B, N, 384)
        last_hidden_state = torch.cat([cls.unsqueeze(1), patches], dim=1)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)


def _interp_pos_embed_to_target(state_dict: dict, model: nn.Module) -> None:
    """Interpolate the checkpoint's positional embedding to ``model``'s grid.

    The official DinoV2 vit_small/14 weights are pretrained at 518x518 (1369
    patch tokens), whereas the encoder is normally built at ``img_size=224``
    (256 patch tokens). The forward path already interpolates ``pos_embed`` at
    runtime; this applies the same interpolation *at load time* so the
    pretrained weights actually take effect at the target resolution instead of
    leaving the patch positions zero-initialized. The CLS token is kept as-is.
    """
    key = "pos_embed"
    if key not in state_dict:
        return
    ckpt = state_dict[key]  # (1, N_ckpt, D)
    target_n = model.pos_embed.shape[1]
    ckpt_n = ckpt.shape[1]
    if ckpt_n == target_n:
        return
    cls = ckpt[:, :1]
    patch = ckpt[:, 1:]
    side_c = int(round(ckpt_n ** 0.5))
    patch = patch.reshape(1, side_c, side_c, -1).permute(0, 3, 1, 2)
    side_t = int(round((target_n - 1) ** 0.5))
    patch = F.interpolate(patch, size=(side_t, side_t), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, side_t * side_t, -1)
    state_dict[key] = torch.cat([cls, patch], dim=1)


def build_dinov2_encoder(**kwargs) -> DinoV2Encoder:
    """Convenience factory (e.g. for hydra ``_target_``)."""
    return DinoV2Encoder(**kwargs)
