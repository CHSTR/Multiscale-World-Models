"""Utilidades compartidas por los diagnósticos: carga de modelo, datos y embeddings.

Las importaciones de ``stable_worldmodel`` / ``stable_pretraining`` son perezosas para que el
núcleo numérico (y los tests) no dependan de ellas.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------
def load_world_model(policy: str, device: str = "cuda"):
    """Carga un checkpoint con ``swm.wm.utils.load_pretrained`` (relativo a <cache>/checkpoints)."""
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def encode_pixels(model, pixels: torch.Tensor, return_cls: bool = False):
    """(B, T, C, H, W) -> emb (B, T, D) igual que ``JEPA.encode`` (CLS -> projector)."""
    b = pixels.size(0)
    flat = rearrange(pixels.float(), "b t ... -> (b t) ...")
    out = model.encoder(flat, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    emb = rearrange(model.projector(cls), "(b t) d -> b t d", b=b)
    if return_cls:
        return emb, rearrange(cls, "(b t) d -> b t d", b=b)
    return emb


def pred_loss_per_sample(model, ctx_emb: torch.Tensor, act_emb: torch.Tensor,
                         tgt_emb: torch.Tensor) -> torch.Tensor:
    """Pérdida de predicción de LeWM por muestra (mismo cálculo que ``lejepa_forward``)."""
    pred = model.predict(ctx_emb, act_emb)
    return (pred - tgt_emb).pow(2).mean(dim=(1, 2))


# --------------------------------------------------------------------------------------
# Importancia de bandas (núcleo testeable)
# --------------------------------------------------------------------------------------
def band_importance_batch(model, batch: Dict[str, torch.Tensor], ctx_len: int, n_preds: int,
                          modes: Sequence[str] = ("zero", "noise"), ig_steps: int = 8,
                          extra_groups: Optional[Dict[str, List[str]]] = None,
                          seed: int = 0) -> Dict[str, object]:
    """Importancia de cada banda del front-end para la pérdida de PREDICCIÓN.

    Solo se perturba el CONTEXTO; el target se codifica limpio (si se ablacionara también el target,
    quitar información lo haría más fácil de predecir y la "importancia" saldría negativa).

    Devuelve, por banda y modo: Δpérdida por muestra y deriva del embedding (1 - cos) del contexto.
    Además, Integrated Gradients sobre un multiplicador por banda en el camino 0 -> 1:
    sum_b IG_b ~= L(todas las bandas) - L(ninguna banda) (completitud).
    """
    enc = model.encoder
    if not hasattr(enc, "set_band_ablation"):
        raise ValueError(f"{type(enc).__name__} no tiene front-end multiescala con ablación de bandas")
    names = list(enc.band_names)
    nb = len(names)
    pixels = batch["pixels"]
    action = torch.nan_to_num(batch["action"], 0.0)
    ctx_px, tgt_px = pixels[:, :ctx_len], pixels[:, n_preds:]

    enc.clear_band_ablation()
    enc.set_band_scale(None)
    with torch.no_grad():
        tgt = encode_pixels(model, tgt_px)
        act = model.action_encoder(action)[:, :ctx_len]
        ctx = encode_pixels(model, ctx_px)
        base = pred_loss_per_sample(model, ctx, act, tgt)

        groups: Dict[str, List[str]] = {n: [n] for n in names}
        if nb > 1:
            groups["details"] = names[:-1]
        groups.update(extra_groups or {})

        ablations = {}
        for mode in modes:
            for gname, members in groups.items():
                enc.set_band_ablation(members, mode=mode, seed=seed)
                ctx_b = encode_pixels(model, ctx_px)
                loss_b = pred_loss_per_sample(model, ctx_b, act, tgt)
                drift = 1 - F.cosine_similarity(ctx_b, ctx, dim=-1).mean(dim=1)
                ablations[(gname, mode)] = {"delta": (loss_b - base).cpu(), "drift": drift.cpu()}
        enc.clear_band_ablation()

    # Integrated Gradients sobre el multiplicador por banda.
    ig = torch.zeros(nb)
    if ig_steps > 0:
        for k in range(ig_steps):
            alpha = (k + 0.5) / ig_steps
            s = torch.full((nb,), alpha, device=pixels.device, requires_grad=True)
            enc.set_band_scale(s)
            with torch.enable_grad():
                loss = pred_loss_per_sample(model, encode_pixels(model, ctx_px), act, tgt).mean()
                (g,) = torch.autograd.grad(loss, s)
            ig += g.detach().cpu() / ig_steps
        with torch.no_grad():
            enc.set_band_scale(torch.zeros(nb, device=pixels.device))
            loss_none = pred_loss_per_sample(model, encode_pixels(model, ctx_px), act, tgt).mean().cpu()
        enc.set_band_scale(None)
    else:
        loss_none = torch.tensor(float("nan"))

    return {
        "names": names,
        "base": base.cpu(),
        "ablations": ablations,
        "ig": ig,
        "loss_all": base.mean().cpu(),
        "loss_none": loss_none,
    }


# --------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------
def build_val_loader(cfg, batch_size: Optional[int] = None, max_items: Optional[int] = None):
    """Replica el dataset y el split de validación de ``train.py`` (misma semilla => mismo split)."""
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import stable_worldmodel.data.formats.hdf5  # noqa: F401  (registra HDF5)
    from omegaconf import OmegaConf, open_dict

    from utils import get_column_normalizer, get_img_preprocessor

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset_cfg.pop("fraction", None)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split],
                                       generator=rnd_gen)
    if max_items is not None and max_items < len(val_set):
        val_set = torch.utils.data.Subset(val_set, range(max_items))
    loader_cfg = dict(cfg.loader)
    if batch_size is not None:
        loader_cfg["batch_size"] = batch_size
    loader_cfg.pop("persistent_workers", None)
    loader_cfg.pop("prefetch_factor", None)
    return torch.utils.data.DataLoader(val_set, shuffle=False, drop_last=False, **loader_cfg)


def frame_transform(img_size: int):
    """Transformación de un frame crudo del dataset (igual que en evaluación)."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
