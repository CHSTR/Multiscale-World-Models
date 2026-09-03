"""Minimal manual LoRA (Low-Rank Adaptation) for ViT attention/MLP linears.

No external dependency (no `peft`). The implementation only rewrites plain
``torch.nn.Linear`` modules in place, copying their existing weights so the
function computed by the module is unchanged when LoRA is first applied
(the rank-B matrix is zero-initialized).

Why r=8 / alpha=16 (defaults)?
------------------------------
* ``r`` (rank) controls how many new trainable degrees of freedom are added per
  linear layer: ``in*r + r*out``. With ``r=8`` on a 384-dim model this is only
  ~6k params per layer, i.e. a tiny fraction of a ViT-small (~49M params).
* ``alpha`` is a scaling factor. The LoRA contribution is multiplied by
  ``alpha/r`` (the standard convention), so ``alpha=16`` => scale ``2.0``.
  Keep ``alpha`` a small multiple of ``r``; you can tune both freely.

The module is agnostic to the ViT implementation: it just looks for ``Linear``
modules named ``qkv``/``proj`` (attention) and -- optionally -- ``fc1``/``fc2``
(MLP), which are the attributes used by both
``src.dinov2.models.vision_transformer`` and
``src.dinov3.models.vision_transformer`` attention blocks.
"""

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Linear):
    """A ``nn.Linear`` with additive low-rank adaptation.

    ``out = W x + b + (alpha/r) * (dropout(x) @ A @ B)``

    ``W``/``b`` are the original (pretrained) weights; ``A``/``B`` are the
    trainable low-rank factors (``A`` random-normal, ``B`` zero-initialized so
    the added delta is initially zero).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__(in_features, out_features, bias)
        self.r = int(r)
        self.alpha = float(alpha)
        self.lora_scale = self.alpha / self.r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Trainable low-rank factors. ``A`` is random-normal, ``B`` stays zero
        # so the LoRA delta starts at 0 (function unchanged at init).
        self.lora_A = nn.Parameter(torch.zeros(in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, out_features))
        nn.init.normal_(self.lora_A, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        lora_delta = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return out + lora_delta * self.lora_scale


def convert_linear_to_lora(
    linear: nn.Linear,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> LoRALinear:
    """Replace an existing ``nn.Linear`` with a LoRALinear keeping its weights."""
    lora = LoRALinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    with torch.no_grad():
        lora.weight.copy_(linear.weight)
        if linear.bias is not None:
            lora.bias.copy_(linear.bias)
    return lora


def _is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def apply_lora_to_vit(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    targets: Iterable[str] = ("qkv", "proj"),
    add_mlp: bool = False,
) -> List[str]:
    """In-place LoRA patch of a ViT.

    Finds attention linear layers (``.qkv`` / ``.proj``) and -- if
    ``add_mlp=True`` -- MLP linears (``.fc1`` / ``.fc2``) and replaces each with
    a :class:`LoRALinear` that keeps the original weights. Works for both
    ``src.dinov2`` and ``src.dinov3`` ViTs because both store attention as a
    module exposing ``qkv`` and ``proj`` and MLP as ``fc1``/``fc2``.

    Returns the dotted names of the layers that were converted.
    """
    targets = list(targets)
    replaced: List[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue

        if "qkv" in targets and _is_linear(getattr(module, "qkv", None)):
            setattr(module, "qkv", convert_linear_to_lora(module.qkv, r, alpha, dropout))
            replaced.append(name + ".qkv")

        if "proj" in targets and _is_linear(getattr(module, "proj", None)):
            setattr(module, "proj", convert_linear_to_lora(module.proj, r, alpha, dropout))
            replaced.append(name + ".proj")

        if add_mlp:
            for sub in ("fc1", "fc2"):
                sub_module = getattr(module, sub, None)
                if _is_linear(sub_module):
                    setattr(module, sub, convert_linear_to_lora(sub_module, r, alpha, dropout))
                    replaced.append(name + "." + sub)
    return replaced


def freeze_except_lora_norm_patch(model: nn.Module) -> None:
    """Freeze every parameter, then unfreeze only:

    * LoRA factors (``LoRALinear`` modules),
    * normalization layers (``LayerNorm`` / ``RMSNorm`` / any module whose class
      name contains "norm"),
    * the patch-embedding module (``PatchEmbed``, including its conv projection),
    * top-level ``storage_tokens`` / ``register_tokens`` if present.

    This keeps the number of trainable parameters tiny (a small % of the
    backbone) while still allowing the model to adapt to extra input channels
    (multicanal / wavelet frontends) through the new patch-embedding conv.
    """
    # 1) freeze everything.
    for param in model.parameters():
        param.requires_grad_(False)

    # 2) enable LoRA factors only (freeze the copied original weight/bias, keep
    #    only the low-rank factors trainable).
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters():
                param.requires_grad_(False)
            for attr in ("lora_A", "lora_B"):
                getattr(module, attr).requires_grad_(True)

    # 3) enable normalization layers.
    for module in model.modules():
        cls = module.__class__.__name__.lower()
        if module is not model and (isinstance(module, (nn.LayerNorm,)) or "rmsnorm" in cls or "layernorm" in cls):
            for param in module.parameters():
                param.requires_grad_(True)

    # 4) enable patch-embedding (and its conv) so it can absorb extra channels.
    for module in model.modules():
        if module.__class__.__name__ == "PatchEmbed":
            for param in module.parameters():
                param.requires_grad_(True)

    # 5) enable storage / register tokens if they exist.
    for attr in ("storage_tokens", "register_tokens"):
        value = getattr(model, attr, None)
        if isinstance(value, nn.Parameter):
            value.requires_grad_(True)


def _find_patch_embed(model: nn.Module):
    """Return (module, dotted_name) of the first patch-embedding module.

    Prefers the DINO-style ``PatchEmbed`` class (exposes ``.proj`` Conv2d), then
    falls back to HF-style ``patch_embeddings`` or the ``patch_embed`` attribute.
    """
    for module in model.modules():
        if module.__class__.__name__ == "PatchEmbed":
            return module
    for attr in ("patch_embeddings", "patch_embed"):
        mod = getattr(model, attr, None)
        if mod is not None:
            return mod
    raise AttributeError("No patch-embedding module found (looked for PatchEmbed / .patch_embed / .patch_embeddings)")


def expand_patch_embed(model: nn.Module, in_chans: int, fill: str = "mean") -> None:
    """Expand a ViT's patch-embedding conv to ``in_chans`` input channels.

    The new conv keeps the original ``out_channels`` / kernel / stride and has
    its first ``in_channels`` weights copied from the pretrained RGB weights;
    the remaining channels are filled with either the per-output-channel mean
    (``fill="mean"``) or zeros (``fill="zeros"``). This mirrors the pattern used
    in ``wavelet/starlet_encoder.py`` so wavelet / starlet frontends can feed a
    DINO encoder unchanged. The (new) conv is left trainable.
    """
    patch_embed = _find_patch_embed(model)
    conv = getattr(patch_embed, "proj")
    old_in = conv.in_channels
    old_weight = conv.weight.detach()  # (out, in, kH, kW)
    assert old_in > 0, f"cannot expand from {old_in} input channels"
    assert old_in <= in_chans, (
        f"existing conv already has {old_in} in_channels, cannot shrink to {in_chans}"
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
        if fill == "mean":
            new_conv.weight[:, old_in:].copy_(old_weight.mean(dim=1, keepdim=True))
        # "zeros" leaves the extra channels at zero (safe: acts as RGB-only path)

    # Replace in-place. ``patch_embed`` is a live reference from .modules().
    setattr(patch_embed, "proj", new_conv)
    if hasattr(patch_embed, "in_chans"):
        patch_embed.in_chans = in_chans
    return new_conv


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
