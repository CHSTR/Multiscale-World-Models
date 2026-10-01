"""Front-end multiescala configurable para los wrappers DINOv2/DINOv3 + LoRA.

Se usa como mixin en ``DinoV2Encoder`` y ``DinoV3Encoder``. Con los argumentos por defecto
(``frontend="auto"``, ``fusion="replace"``, ``patch_init="legacy"``) reproduce EXACTAMENTE el
comportamiento y las claves del ``state_dict`` previos, así que los checkpoints existentes cargan
sin cambios.

Argumentos nuevos (todos opcionales):
    frontend               "auto" | "none" | "starlet" | "swt" | "rgb_repeat" | "random"
                           ("auto" = starlet si starlet_levels > 0, si no RGB: el comportamiento previo)
    frontend_levels        J; por defecto starlet_levels
    swt_filter             filtro SWT ("bior2.2" | "haar")
    swt_merge_orientations suma LH+HL+HH por nivel (ablación de orientación)
    random_kernel / random_seed   control de filtros aleatorios
    fusion                 "replace" (bandas reemplazan RGB) | "residual" (RGB + rama multiescala en cero)
    patch_init             "legacy" (original) | "equivalent" (W_rgb replicado: exacto en el paso 0)
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from src.encoders.vit_lora import _find_patch_embed, expand_patch_embed
from wavelet.frontend import (
    FUSIONS,
    PATCH_INITS,
    BandAblationMixin,
    BandTransform,
    ResidualMultiscaleProj,
    band_names,
    n_bands,
    sums_to_input,
)


class MultiscaleFrontendMixin(BandAblationMixin):
    """Requiere que la subclase sea un ``nn.Module`` y que guarde el backbone en ``self.model``."""

    def _init_frontend(
        self,
        *,
        frontend: str,
        frontend_levels,
        starlet_levels: int,
        starlet_filter: str,
        starlet_learnable_weights: bool,
        swt_filter: str,
        swt_merge_orientations: bool,
        random_kernel: int,
        random_seed: int,
        fusion: str,
        patch_init: str,
        in_chans: int,
    ) -> int:
        """Configura la transformada y devuelve los canales que recibirá el patch embedding."""
        starlet_levels = int(starlet_levels)
        if frontend == "auto":
            frontend = "starlet" if starlet_levels > 0 else "none"
        levels = int(frontend_levels) if frontend_levels is not None else starlet_levels
        if fusion not in FUSIONS:
            raise ValueError(f"fusion desconocida: {fusion!r} (opciones: {FUSIONS})")
        if patch_init not in PATCH_INITS:
            raise ValueError(f"patch_init desconocido: {patch_init!r} (opciones: {PATCH_INITS})")
        if frontend not in ("none",) and fusion == "replace" and patch_init == "equivalent" \
                and not sums_to_input(frontend):
            raise ValueError(
                f"patch_init=equivalent requiere bandas que sumen la imagen; {frontend} no cumple. "
                "Usa fusion=residual."
            )

        self.frontend_kind = frontend
        self.frontend_levels = levels if frontend != "none" else 0
        self.fusion = fusion
        self.patch_init = patch_init
        # Compatibilidad con código que lee estos atributos (smoke test, visualizaciones).
        self.starlet_levels = levels if frontend == "starlet" else 0
        self.starlet_filter = starlet_filter

        if frontend == "none":
            self.band_transform = None
            self._band_names = ["rgb"]
            return in_chans  # modo multicanal manual previo (in_chans > 3) o RGB

        self.band_transform = BandTransform(
            kind=frontend, levels=levels, starlet_filter=starlet_filter, swt_filter=swt_filter,
            merge_orientations=swt_merge_orientations, random_kernel=random_kernel,
            random_seed=random_seed,
        )
        nb = n_bands(frontend, levels, swt_merge_orientations)
        self._band_names = band_names(frontend, levels, swt_merge_orientations)
        w = torch.ones(nb)
        if starlet_learnable_weights:
            self.level_weights = nn.Parameter(w)
        else:
            self.register_buffer("level_weights", w)
        ms_chans = 3 * nb
        return ms_chans if fusion == "replace" else 3 + ms_chans

    def _setup_patch_embed(self, in_chans: int) -> None:
        """Adapta el patch embedding del backbone preentrenado (llamar antes de aplicar LoRA)."""
        if self.frontend_kind == "none":
            if in_chans > 3 and getattr(self, "patch_embed_mode", "adapt") == "adapt":
                expand_patch_embed(self.model, in_chans, fill="mean")
            return
        if self.fusion == "replace":
            if self.patch_init == "legacy":
                expand_patch_embed(self.model, in_chans, fill="mean")
            else:
                expand_patch_embed(self.model, in_chans, fill="repeat", copy_bias=True)
            return
        patch_embed = _find_patch_embed(self.model)
        rgb_conv = patch_embed.proj
        patch_embed.proj = ResidualMultiscaleProj(rgb_conv, in_chans - rgb_conv.in_channels)
        if hasattr(patch_embed, "in_chans"):
            patch_embed.in_chans = in_chans

    def _frontend_input(self, x: torch.Tensor) -> torch.Tensor:
        """RGB (B, 3, H, W) -> tensor que consume el patch embedding."""
        if self.frontend_kind == "none":
            if getattr(self, "_band_ablation", None) is None and getattr(self, "_band_scale", None) is None:
                return x
            return self._apply_band_controls(x.unsqueeze(1)).squeeze(1)
        bands = self.band_transform(x) * self.level_weights.view(1, -1, 1, 1, 1)
        bands = self._apply_band_controls(bands)
        flat = bands.flatten(1, 2)
        if self.fusion == "residual":
            return torch.cat([x, flat], dim=1)
        return flat

    @property
    def band_names(self) -> List[str]:
        return list(self._band_names)

    def frontend_stats(self) -> Dict[str, torch.Tensor]:
        """Escalares para loguear durante el entrenamiento."""
        stats: Dict[str, torch.Tensor] = {}
        if self.frontend_kind != "none":
            for name, w in zip(self._band_names, self.level_weights.detach()):
                stats[f"band_weight/{name}"] = w
        proj = getattr(_find_patch_embed(self.model), "proj", None)
        ratio = getattr(proj, "last_ratio", None)
        if ratio is not None:
            stats["ms_branch_ratio"] = ratio
        return stats
