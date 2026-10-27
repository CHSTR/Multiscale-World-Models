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

Tokens dinámicos (``n_dyn_tokens=k > 0``): se añaden ``k`` tokens aprendibles que
se intercalan entre el CLS y los storage tokens. El orden de secuencia pasa a
ser ``[CLS, dyn_1..k, storage_1..4, patches]``. Con ``k=0`` el comportamiento es
idéntico al actual (``[CLS, storage, patches]``).

La salida es un ``types.SimpleNamespace`` con:

* ``last_hidden_state``: ``(B, 1 + k + n_storage + N, 384)`` con el orden de
  secuencia de arriba.
* ``pooled``: resumen usado por ``jepa.encode`` (ver ``readout``).
* ``cls``: el token CLS (``(B, 384)``).
* ``dyn_tokens``: el ``nn.Parameter`` de tokens dinámicos (o ``None`` si ``k=0``).

``readout`` define cómo se construye ``pooled``:
``cls`` (default) => CLS (con k=0 equivale a ``last_hidden_state[:, 0]``),
``dyn_first`` => primer token dinámico, ``dyn_mean`` => media de los dinámicos,
``dyn_concat`` => concatenación aplanada de los dinámicos (``readout_dim = k*384``).

``dyn_init`` define la inicialización de los tokens dinámicos (sólo con
``k > 0``). ``init_std`` (``None`` por defecto) controla la escala del ruido:
``None`` conserva EXACTAMENTE la escala legacy ``0.02 * base.abs().mean()`` y un
``float`` se usa directamente como ``std`` del ruido (en ``cls_noise`` /
``normal`` / ``random``). ``cls`` y ``zeros`` no usan ruido:

============== ==========================================================
``dyn_init``   inicialización de ``self.dyn_tokens`` ``(1, k, 384)``
============== ==========================================================
``cls``        copia EXACTA del ``cls_token``; ruido forzado a 0 aunque
               ``init_std > 0`` (punto idéntico al baseline).
``cls_noise``  copia del ``cls_token`` + ``N(0, init_std)`` por token.
``zeros``      ceros exactos; con readout sobre los dyn (``dyn_first`` /
               ``dyn_mean`` / ``dyn_concat``) arranca degenerado a propósito:
               el punto de control "sin prior".
``storage``    copia de ``storage_tokens`` + ``N(0, 0.02*sigma)`` (legacy).
``normal``     ``N(0, 0.02)`` (o ``N(0, init_std)`` si se pasa ``init_std``).
============== ==========================================================
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
    freeze_all,
    freeze_except_lora_norm_patch,
)


# Portable default: $DINOV3_CKPT si está definido, si no None (= descarga por hub).
DEFAULT_DINOV3_CKPT = os.environ.get("DINOV3_CKPT") or None

# Validaciones de los args de tokens dinámicos.
READOUT_OPTIONS = ("cls", "dyn_first", "dyn_mean", "dyn_concat")
DYN_INIT_OPTIONS = ("cls", "cls_noise", "zeros", "storage", "normal")
FREEZE_MODE_OPTIONS = ("lora_norm_patch", "all")


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
        n_dyn_tokens: int = 0,
        readout: str = "cls",
        dyn_init: str = "cls",
        init_std: Optional[float] = None,
        freeze_mode: str = "lora_norm_patch",
        device: str = "cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = 384
        self.patch_embed_mode = patch_embed_mode

        # --- Tokens dinámicos / freeze ---
        assert readout in READOUT_OPTIONS, f"readout {readout!r} no válido (opciones: {READOUT_OPTIONS})"
        assert dyn_init in DYN_INIT_OPTIONS, f"dyn_init {dyn_init!r} no válido (opciones: {DYN_INIT_OPTIONS})"
        assert freeze_mode in FREEZE_MODE_OPTIONS, (
            f"freeze_mode {freeze_mode!r} no válido (opciones: {FREEZE_MODE_OPTIONS})"
        )
        self.n_dyn_tokens = int(n_dyn_tokens)
        self.readout = readout
        self.dyn_init = dyn_init
        self.init_std = init_std
        self.freeze_mode = freeze_mode
        # dim de la salida ``pooled``: CLS/readouts por defecto = 384;
        # dyn_concat aplana los k tokens dinámicos => k*384.
        self.readout_dim = self.n_dyn_tokens * self.embed_dim if readout == "dyn_concat" else self.embed_dim
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

        # ``apply_lora_to_vit`` devuelve [] si ``r <= 0`` (backbone sin LoRA).
        self._lora_layers = apply_lora_to_vit(
            self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets
        )

        # Freeze strategy. ``lora_norm_patch`` (default) congela todo salvo LoRA +
        # norms + patch-embed + storage; ``all`` congela el 100% del backbone.
        if freeze_mode == "all":
            freeze_all(self.model)
        else:
            freeze_except_lora_norm_patch(self.model)

        # Tokens dinámicos: se crean *después* del freeze para que queden
        # entrenables por defecto en ``lora_norm_patch``. En ``all`` se congelan
        # explícitamente a continuación.
        if self.n_dyn_tokens > 0:
            self._create_dyn_tokens(device)
            if freeze_mode == "all":
                self.dyn_tokens.requires_grad_(False)

        self.to(device)
        n_storage = getattr(self.model, "n_storage_tokens", 0)
        print(
            f"DinoV3Encoder(vits16, dim={self.embed_dim}, in_chans={in_chans}, "
            f"starlet_levels={self.starlet_levels}, "
            f"n_storage={n_storage}, n_dyn={self.n_dyn_tokens}, readout={self.readout}, "
            f"freeze_mode={self.freeze_mode}, lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,} params]"
        )

    def _create_dyn_tokens(self, device: str) -> None:
        """Crear y inicializar ``self.dyn_tokens`` (``(1, k, 384)``).

        Modos de inicialización (``dyn_init``) con ``k > 0``:

        * ``cls``       -> copia EXACTA del ``cls_token``; ruido forzado a 0
          aunque ``init_std > 0`` (punto idéntico al baseline).
        * ``cls_noise`` -> copia del ``cls_token`` + ruido
          ``N(0, init_std)`` independiente por token (helper ``_dyn_noise``).
        * ``zeros``     -> ``nn.Parameter`` a ceros exactos; con readout sobre
          los dyn arranca degenerado a propósito ("sin prior").
        * ``storage``   -> copia del ``storage_tokens`` + ruido
          ``N(0, 0.02 * sigma)`` (si existe); si no, ``N(0, 0.02)``.
        * ``normal``    -> ``N(0, 0.02)`` (o ``N(0, init_std)``).

        ``init_std`` controla la escala del ruido: ``None`` conserva EXACTAMENTE
        la escala legacy ``0.02 * base.abs().mean()``; un ``float`` se usa como
        ``std`` directo. ``cls`` y ``zeros`` no usan ruido.
        """
        k = self.n_dyn_tokens
        self.dyn_tokens = nn.Parameter(torch.zeros(1, k, self.embed_dim, device=device))

        with torch.no_grad():
            if self.dyn_init == "cls":
                # Exact copy of the CLS register; noise forced to 0 (baseline
                # identical) even when ``init_std > 0``.
                self.dyn_tokens.copy_(self.model.cls_token.detach().expand(1, k, self.embed_dim))
            elif self.dyn_init == "cls_noise":
                base = self.model.cls_token.detach()  # (1, 1, 384)
                self.dyn_tokens.copy_(base.expand(1, k, self.embed_dim))
                self.dyn_tokens.add_(self._dyn_noise(k, base))
            elif self.dyn_init == "zeros":
                # Zero Parameter. With a readout that reads the dynamic tokens
                # (``dyn_first`` / ``dyn_mean`` / ``dyn_concat``) the model starts
                # degenerate on purpose: the "no prior" checkpoint. Nothing to do.
                pass
            elif self.dyn_init == "storage":
                storage = getattr(self.model, "storage_tokens", None)
                if storage is not None:
                    sigma = storage.detach().abs().mean().item()
                    self.dyn_tokens.copy_(storage.expand(1, k, self.embed_dim))
                    noise = torch.randn(1, k, self.embed_dim, device=self.dyn_tokens.device) * (
                        0.02 * sigma
                    )
                    self.dyn_tokens.add_(noise)
                else:
                    nn.init.normal_(self.dyn_tokens, std=0.02)
            else:  # "normal"
                self.dyn_tokens.data.normal_(
                    0, self.init_std if self.init_std is not None else 0.02
                )

    def _dyn_noise(self, k: int, base: torch.Tensor) -> torch.Tensor:
        """Gaussian noise ``N(0, std)`` per dynamic token (used by ``cls_noise``).

        ``std`` = ``init_std`` when passed, else the legacy
        ``0.02 * base.abs().mean()`` (so ``init_std=None`` is unchanged).
        """
        std = self.init_std if self.init_std is not None else 0.02 * base.abs().mean().item()
        return torch.randn(1, k, self.embed_dim, device=self.dyn_tokens.device) * std

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True,
                prompt: Optional[torch.Tensor] = None, **kwargs):
        """Return a ``SimpleNamespace`` with ``last_hidden_state``, ``pooled``,
        ``cls`` and ``dyn_tokens``.

        The ``prompt`` kwarg is kept for API parity; the dynamic tokens are always
        fed internally via ``self.dyn_tokens``, so callers do not pass it. The
        backbone expects a per-crop *list* of prompt tensors, so we wrap the
        single prompt tensor in a length-1 list -- ``prompt_list[0].shape[1]``
        then equals the number of dynamic tokens ``k``.

        Sequence order is ``[CLS, dyn_1..k, storage_1..4, patches]``; with
        ``n_dyn_tokens=0`` this collapses to ``[CLS, storage, patches]`` (the
        previous behavior).
        """
        x = pixel_values
        if self.starlet_levels > 0:
            x = starlet_conv4d(x, self.starlet_levels, scale=self.level_weights, filter=self.starlet_filter)

        dyn = getattr(self, "dyn_tokens", None)
        # ``is_training=True`` makes the backbone return the feature dict from
        # which we assemble the full (normed) token sequence.
        features = self.model(x, prompt=[dyn] if dyn is not None else None, is_training=True)

        cls = features["x_norm_clstoken"]            # (B, 384)
        storage = features["x_storage_tokens"]       # (B, n_storage, 384)
        patches = features["x_norm_patchtokens"]     # (B, N, 384)
        dyn_toks = features["x_prompt_tokens"] if dyn is not None else None  # (B, k, 384)

        # Sequence order: [CLS, dyn_1..k, storage_1..4, patches]. With k=0 this
        # collapses to [CLS, storage, patches] (identical to the old behavior).
        parts = [cls.unsqueeze(1)]
        if dyn_toks is not None and dyn_toks.shape[1] > 0:
            parts.append(dyn_toks)
        if storage is not None and storage.shape[1] > 0:
            parts.append(storage)
        parts.append(patches)
        last_hidden_state = torch.cat(parts, dim=1)

        # ``pooled``: summary used by ``jepa.encode``, chosen by ``self.readout``.
        if self.readout == "cls":
            # CLS (with k=0 this equals ``last_hidden_state[:, 0]``).
            pooled = cls
        elif self.readout == "dyn_first":
            assert dyn_toks is not None and dyn_toks.shape[1] > 0, (
                "readout='dyn_first' needs n_dyn_tokens > 0"
            )
            pooled = dyn_toks[:, 0]
        elif self.readout == "dyn_mean":
            assert dyn_toks is not None and dyn_toks.shape[1] > 0, (
                "readout='dyn_mean' needs n_dyn_tokens > 0"
            )
            pooled = dyn_toks.mean(dim=1)
        elif self.readout == "dyn_concat":
            assert dyn_toks is not None and dyn_toks.shape[1] > 0, (
                "readout='dyn_concat' needs n_dyn_tokens > 0"
            )
            pooled = dyn_toks.reshape(cls.size(0), -1)
        else:  # pragma: no cover - guarded by __init__ assert
            raise ValueError(f"unknown readout {self.readout!r}")

        return types.SimpleNamespace(
            last_hidden_state=last_hidden_state,
            pooled=pooled,
            dyn_tokens=dyn,
            cls=cls,
        )


def build_dinov3_encoder(**kwargs) -> DinoV3Encoder:
    """Convenience factory (e.g. for hydra ``_target_``)."""
    return DinoV3Encoder(**kwargs)
