"""Small DINOv3 ViT encoder wrapper for the JEPA pipeline.

Replaces the previous ``vit_base`` / hardcoded-path wrapper with a *small*
model (``vit_small`` / patch-16, embed dim 384) loaded through the official hub
backbone ``src.dinov3.hub.backbones.dinov3_vits16``. Pesos: ``ckpt_path``
explícito > ``$DINOV3_CKPT`` > descarga por hub oficial (``ckpt_path=null``
= portable entre máquinas).

Interface (compatible with ``jepa.JEPA.encode``):

    out = encoder(pixel_values, interpolate_pos_encoding=True)
    emb = out.pooled                       # == tok_in  (BT, 384)

``jepa.encode`` reads ``out.pooled`` and falls back to
``out.last_hidden_state[:, 0]`` when the encoder does not expose ``pooled``.

Direct-predictor contract (unified dynamic prompt tokens)
---------------------------------------------------------
The encoder owns a *single* learnable ``prompt_tokens`` parameter of shape
``(1, n_in_tokens + n_tgt_tokens, 384)``. The backbone prepends its own CLS
token and concatenates ``prompt_tokens`` right after it (concatenation via
``expand`` -- never in-place -- so the shared ``nn.Parameter`` is never mutated),
so the sequence seen by the transformer is::

    [CLS, prompt_in_1..n_in, prompt_tgt_1..n_tgt, storage_1..4, patches]

In ``forward`` the raw prompt tokens are read from the backbone feature dict::

    features = self.model(x, prompt=self.prompt_tokens, is_training=True)
    tok      = features["x_prompt_tokens"]        # (BT, n_in+n_tgt, 384)
    tok_in   = tok[:, 0]                           # (BT, 384)  -- the "input" token
    tok_tgt  = tok[:, 1:]                          # (BT, n_tgt, 384)
    if tie_tgt_to_in: tok_tgt = tok_in.unsqueeze(1)

The returned ``types.SimpleNamespace`` exposes:

* ``last_hidden_state``: full sequence ``[CLS, prompt_tokens, storage, patches]``.
* ``cls``: the backbone CLS token ``(BT, 384)``.
* ``tok_in``: first prompt token ``(BT, 384)``.
* ``tok_tgt``: remaining prompt tokens ``(BT, n_tgt, 384)`` (or ``tok_in``
  repeated when ``tie_tgt_to_in=True``).
* ``pooled``: == ``tok_in`` (this is what ``jepa.encode`` reads -- the original
  JEPA path keeps working unchanged).

``readout_dim`` is always 384 (the pooled summary is a single 384-d token).

Defaults are ``n_in_tokens=1`` and ``n_tgt_tokens=1`` (a 2-token prompt); the
total prompt length is ``n_in_tokens + n_tgt_tokens``.

``tok_init`` initializes *every* prompt token:

============ =========================================================
``tok_init`` inicialización de ``self.prompt_tokens`` ``(1, n_prompt, 384)``
============ =========================================================
``cls``      copia EXACTA del ``cls_token`` del backbone; ruido forzado a 0.
``cls_noise`` copia del ``cls_token`` + ``N(0, init_std)`` por token.
``normal``   ``N(0, 0.02)`` (o ``N(0, init_std)`` si se pasa ``init_std``).
============ =========================================================

``freeze_mode`` congela el backbone: ``lora_norm_patch`` (default) congela todo
salvo LoRA + norms + patch-embed (+ storage tokens); ``all`` congela el 100%.

Compatibilidad con la API vieja
-------------------------------
El argumento legacy ``n_dyn_tokens=k`` (número de tokens dinámicos que seguían al
CLS) sigue aceptado y se mapea a la nueva configuración de prompt como
``n_in_tokens=1`` + ``n_tgt_tokens=k-1`` (``k<=0`` => ``n_tgt_tokens=0``, i.e. un
sólo token de prompt). Esto mantiene los configs existentes (p. ej.
``dinov3_frozen.yaml`` con ``n_dyn_tokens=0``) funcionando sin tocar el yaml.

Los nombres viejos ``readout`` / ``dyn_init`` / ``dyn_tokens`` **ya no** existen:
usa ``tok_init`` / ``prompt_tokens``.
"""

import os
import types
from typing import Optional

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

# Inicialización de los prompt tokens.
TOK_INIT_OPTIONS = ("cls", "cls_noise", "normal")
FREEZE_MODE_OPTIONS = ("lora_norm_patch", "all")


class DinoV3Encoder(nn.Module):
    """ViT-small / patch-16 (dim 384) DINOv3 encoder with LoRA + multicanal patch embed.

    Contrato direct-predictor: un sólo ``nn.Parameter prompt_tokens`` de forma
    ``(1, n_in_tokens + n_tgt_tokens, 384)``; la salida expone ``tok_in`` /
    ``tok_tgt`` / ``pooled == tok_in``.
    """

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
        lora_targets = ("qkv", "proj"),
        # --- Contrato direct-predictor (API unificada de prompt) ---
        n_in_tokens: int = 1,
        n_tgt_tokens: int = 1,
        tok_init: str = "cls",
        tie_tgt_to_in: bool = False,
        # --- Compat con la vieja API de tokens dinámicos ---
        n_dyn_tokens: Optional[int] = None,
        init_std: Optional[float] = None,
        freeze_mode: str = "lora_norm_patch",
        device: str = "cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = 384
        self.patch_embed_mode = patch_embed_mode

        assert tok_init in TOK_INIT_OPTIONS, (
            f"tok_init {tok_init!r} no válido (opciones: {TOK_INIT_OPTIONS})"
        )
        assert freeze_mode in FREEZE_MODE_OPTIONS, (
            f"freeze_mode {freeze_mode!r} no válido (opciones: {FREEZE_MODE_OPTIONS})"
        )
        assert int(n_in_tokens) >= 1, "n_in_tokens debe ser >= 1 (tok_in = tok[:,0])"
        assert int(n_tgt_tokens) >= 0, "n_tgt_tokens debe ser >= 0"

        # --- Compat legacy: n_dyn_tokens=k => prompt tokens tras el CLS ---
        # Nueva API: 1 token de input + (k - 1) tokens de target.
        if n_dyn_tokens is not None:
            n_in_tokens = 1
            n_tgt_tokens = max(int(n_dyn_tokens) - 1, 0)

        self.n_in_tokens = int(n_in_tokens)
        self.n_tgt_tokens = int(n_tgt_tokens)
        self.n_prompt_tokens = self.n_in_tokens + self.n_tgt_tokens
        self.tok_init = tok_init
        self.init_std = init_std
        self.tie_tgt_to_in = bool(tie_tgt_to_in)
        self.freeze_mode = freeze_mode
        # El pooled es un sólo token 384-d => readout_dim fijo.
        self.readout_dim = self.embed_dim

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

        # Prompt tokens: se crean *después* del freeze para que queden
        # entrenables por defecto en ``lora_norm_patch``. En ``all`` se congelan
        # explícitamente a continuación. ``_init_prompt_tokens`` crea el
        # ``nn.Parameter`` y lo inicializa.
        self._init_prompt_tokens()
        if freeze_mode == "all":
            self.prompt_tokens.requires_grad_(False)

        self.to(device)
        n_storage = getattr(self.model, "n_storage_tokens", 0)
        print(
            f"DinoV3Encoder(vits16, dim={self.embed_dim}, in_chans={in_chans}, "
            f"starlet_levels={self.starlet_levels}, "
            f"n_storage={n_storage}, n_prompt={self.n_prompt_tokens} "
            f"(n_in={self.n_in_tokens}, n_tgt={self.n_tgt_tokens}), tok_init={self.tok_init}, "
            f"tie_tgt_to_in={self.tie_tgt_to_in}, readout_dim={self.readout_dim}, "
            f"freeze_mode={self.freeze_mode}, lora_layers={len(self._lora_layers)}) "
            f"[trainable={count_trainable_parameters(self):,}/{count_total_parameters(self):,} params]"
        )

    def _init_prompt_tokens(self) -> None:
        """Inicializar ``self.prompt_tokens`` (``(1, n_prompt, 384)``).

        Modos (``tok_init``):

        * ``cls``       -> copia EXACTA del ``cls_token`` del backbone; ruido 0.
        * ``cls_noise`` -> copia + ``N(0, init_std)`` por token (helper
          ``_prompt_noise``).
        * ``normal``    -> ``N(0, 0.02)`` (o ``N(0, init_std)``).

        ``init_std`` controla la escala del ruido: ``None`` conserva EXACTAMENTE
        la escala legacy ``0.02 * base.abs().mean()``; un ``float`` se usa como
        ``std`` directo.
        """
        k = self.n_prompt_tokens
        device = self.model.cls_token.device
        self.prompt_tokens = nn.Parameter(torch.zeros(1, k, self.embed_dim, device=device))
        with torch.no_grad():
            if self.tok_init == "cls":
                # Exact copy of the CLS register; noise forced to 0 even when
                # ``init_std > 0`` (baseline identical).
                self.prompt_tokens.copy_(
                    self.model.cls_token.detach().expand(1, k, self.embed_dim)
                )
            elif self.tok_init == "cls_noise":
                base = self.model.cls_token.detach()  # (1, 1, 384)
                self.prompt_tokens.copy_(base.expand(1, k, self.embed_dim))
                self.prompt_tokens.add_(self._prompt_noise(k, base))
            else:  # "normal"
                self.prompt_tokens.data.normal_(
                    0, self.init_std if self.init_std is not None else 0.02
                )

    def _prompt_noise(self, k: int, base: torch.Tensor) -> torch.Tensor:
        """Gaussian noise ``N(0, std)`` per prompt token (used by ``cls_noise``).

        ``std`` = ``init_std`` when passed, else the legacy
        ``0.02 * base.abs().mean()`` (so ``init_std=None`` is unchanged).
        """
        std = self.init_std if self.init_std is not None else 0.02 * base.abs().mean().item()
        return torch.randn(1, k, self.embed_dim, device=self.prompt_tokens.device) * std

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
        prompt: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Return a ``SimpleNamespace`` with ``last_hidden_state``, ``cls``,
        ``tok_in``, ``tok_tgt`` and ``pooled`` (== ``tok_in``).

        The ``prompt`` kwarg is kept for API parity; the prompt tokens are always
        fed internally via ``self.prompt_tokens`` so callers do not pass it.
        """
        x = pixel_values
        if self.starlet_levels > 0:
            x = starlet_conv4d(x, self.starlet_levels, scale=self.level_weights, filter=self.starlet_filter)

        # ``is_training=True`` makes the backbone return the feature dict from
        # which we assemble the token sequence. The single prompt tensor is
        # shared across the (single) input; ``n_prompt`` is derived from its
        # ``shape[1]`` inside the backbone.
        features = self.model(x, prompt=self.prompt_tokens, is_training=True)

        cls = features["x_norm_clstoken"]            # (BT, 384)
        storage = features["x_storage_tokens"]       # (BT, n_storage, 384)
        patches = features["x_norm_patchtokens"]     # (BT, N, 384)
        tok = features["x_prompt_tokens"]            # (BT, n_prompt, 384)

        # ``tok_in`` = primer prompt token (el "input"); ``tok_tgt`` = el resto.
        tok_in = tok[:, 0]                           # (BT, 384)
        if self.tie_tgt_to_in:
            tok_tgt = tok_in.unsqueeze(1)            # (BT, 1, 384) == tok_in broadcast
        else:
            tok_tgt = tok[:, 1:]                     # (BT, n_tgt, 384)

        # Secuencia completa: [CLS, prompt_tokens, storage, patches].
        parts = [cls.unsqueeze(1)]
        if tok.shape[1] > 0:
            parts.append(tok)
        if storage is not None and storage.shape[1] > 0:
            parts.append(storage)
        parts.append(patches)
        last_hidden_state = torch.cat(parts, dim=1)

        # ``pooled``: resumen que lee ``jepa.encode`` (== tok_in).
        pooled = tok_in

        return types.SimpleNamespace(
            last_hidden_state=last_hidden_state,
            cls=cls,
            tok_in=tok_in,
            tok_tgt=tok_tgt,
            pooled=pooled,
        )


def build_dinov3_encoder(**kwargs) -> DinoV3Encoder:
    """Convenience factory (e.g. for hydra ``_target_``)."""
    return DinoV3Encoder(**kwargs)
