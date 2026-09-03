import argparse
import json
import math
import os
import sys

import hydra.utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from scipy.ndimage import zoom as zoom_attn
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_pretraining.backbone.utils import vit_hf
from wavelet.starlet_torch import starlet_conv4d
from wavelet.swt_torch import swt_conv4d
from wavelet.wave_vit_encoder import WaveViTEncoder


def _load_hf_encoder(enc_cfg, ckpt_path):
    is_starlet = enc_cfg.get("_target_", "").endswith("StarletEncoder")
    is_swt = enc_cfg.get("_target_", "").endswith("SWTEncoder")
    has_vit_sub = is_starlet or is_swt
    vit_cfg = enc_cfg["vit"] if has_vit_sub else enc_cfg

    model = vit_hf(
        size=vit_cfg["size"],
        patch_size=vit_cfg["patch_size"],
        image_size=vit_cfg.get("image_size", 224),
        pretrained=False,
        use_mask_token=vit_cfg.get("use_mask_token", False),
        attn_implementation="eager",
    )

    if has_vit_sub:
        levels = enc_cfg.get("levels", 4)
        old_pe = model.embeddings.patch_embeddings.projection
        new_in_c = old_pe.in_channels * (levels + 1)
        model.embeddings.patch_embeddings.projection = nn.Conv2d(
            new_in_c, old_pe.out_channels,
            kernel_size=old_pe.kernel_size, stride=old_pe.stride,
            padding=old_pe.padding, bias=old_pe.bias is not None,
        )
        model.config.num_channels = new_in_c
        model.embeddings.patch_embeddings.num_channels = new_in_c

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if is_swt:
        prefix = "encoder.vit."
        mapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        model.load_state_dict(mapped, strict=False)

        proj_w = ckpt.get("encoder.channel_proj.weight")
        proj_b = ckpt.get("encoder.channel_proj.bias")
        if proj_w is not None:
            model._swt_channel_proj = nn.Conv2d(
                proj_w.shape[1], proj_w.shape[0], 1,
                bias=proj_b is not None,
            )
            model._swt_channel_proj.weight.data = proj_w
            if proj_b is not None:
                model._swt_channel_proj.bias.data = proj_b
        model._encoder_is_swt = True
        model._swt_levels = enc_cfg.get("levels", 4)
        model._swt_filter = enc_cfg.get("filter", "bior2.2")
        kind = "SWT"
    elif is_starlet:
        prefix = "encoder.vit."
        mapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        model.load_state_dict(mapped, strict=False)
        kind = "starlet"
    else:
        prefix = "encoder."
        mapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        model.load_state_dict(mapped, strict=False)
        kind = "plain"

    print(f"Keys cargados: {len(mapped)} (HF {kind})")
    model.eval()
    return model


def _load_wave_encoder(enc_cfg, ckpt_path):
    # Instanciar WaveViTEncoder desde config.json (ignorando _target_)
    kwargs = {k: v for k, v in enc_cfg.items() if k != "_target_"}
    model = WaveViTEncoder(**kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "encoder."
    mapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Keys cargados: {len(mapped)} (WaveViTEncoder), "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def load_encoder(ckpt_path: str):
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    enc_cfg = cfg["encoder"]
    target = enc_cfg.get("_target_", "")
    if "WaveViTEncoder" in target:
        model = _load_wave_encoder(enc_cfg, ckpt_path)
        model._encoder_kind = "wave"
    elif "DinoV2Encoder" in target or "DinoV3Encoder" in target:
        model = _load_dino_encoder(enc_cfg, ckpt_path)
        model._encoder_kind = "dino"
    else:
        model = _load_hf_encoder(enc_cfg, ckpt_path)
        model._encoder_kind = "hf"
    return model


# ---- DINO (v2/v3 + LoRA + starlet interno): carga y captura de atencion ----
# El wrapper (DinoV2Encoder/DinoV3Encoder) ya aplica starlet_conv4d en forward
# cuando starlet_levels>0, asi que siempre se le da RGB. El ckpt guarda el
# state_dict del JEPA -> se quita el prefijo "encoder.".
def _load_dino_encoder(enc_cfg, ckpt_path):
    target = enc_cfg.get("_target_", "")
    cls = hydra.utils.get_class(target)
    kwargs = {k: v for k, v in enc_cfg.items() if k != "_target_"}
    # Sin pesos oficiales: construimos la arquitectura y cargamos el ckpt.
    # (pretrained=False evita descargas; ckpt_path=null no dispara el hub.)
    kwargs["pretrained"] = False
    kwargs["ckpt_path"] = None
    model = cls(**kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "encoder."
    mapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Keys cargados: {len(mapped)} (DINO {target.rsplit('.', 1)[-1]}), "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing: {missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    model.eval()
    model._dino_target = target
    model._is_dinov3 = "DinoV3Encoder" in target
    return model


def _effective_linear_weight(m):
    """W efectivo de un Linear, sumando el delta LoRA si existe (eval).

    En forward el delta es (x @ A @ B) con A:(in,r), B:(r,out); para sumarlo
    al weight (out,in) hay que transponerlo.
    """
    w = m.weight.detach()
    if hasattr(m, "lora_A"):
        w = w + (m.lora_A.detach() @ m.lora_B.detach()).T * float(m.lora_scale)
    return w


def _iter_dino_attn(backbone):
    """Yieldea los modulos de atencion de cada bloque, en orden."""
    for blk in backbone.blocks:
        attn = getattr(blk, "attn", None)
        if attn is None:
            # BlockChunk u otro contenedor: baja un nivel.
            for sub in blk.children():
                attn = getattr(sub, "attn", None)
                if attn is not None:
                    yield attn
        else:
            yield attn


def dino_forward_features(wrapper, frame_np, dev):
    """Forward RGB + captura por hooks.

    Returns:
        attns: lista por capa de (h, N, N) numpy (solo DINOv2; [] en v3).
        hidden_last: (1+N, D) numpy del ultimo layer (CLS primero).
        psize: lado del grid de parches. n_patches: N-1 (v2) o N-1-storage (v3).
    """
    backbone = wrapper.model
    attn_mods = list(_iter_dino_attn(backbone))
    saved = {}
    hooks = []
    for i, mod in enumerate(attn_mods):
        hooks.append(mod.register_forward_hook(
            lambda m, inp, out, _i=i: saved.__setitem__(_i, inp[0].detach())))
    try:
        x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = wrapper(x)
    finally:
        for h in hooks:
            h.remove()

    hidden_last = out.last_hidden_state[0].cpu().numpy()  # (1+N, D)
    is_v3 = getattr(wrapper, "_is_dinov3", False)
    n_storage = 0
    attns = []
    if not is_v3:
        for i, mod in enumerate(attn_mods):
            x_in = saved[i].float()
            w_eff = _effective_linear_weight(mod.qkv).float()
            b = mod.qkv.bias.float() if mod.qkv.bias is not None else None
            B, N, _ = x_in.shape
            h = mod.num_heads
            d = w_eff.shape[0] // 3 // h
            qkv = F.linear(x_in, w_eff, b).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k = qkv[0] * mod.scale, qkv[1]
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # (B, h, N, N)
            attns.append(attn[0].cpu().numpy())
    else:
        n_storage = int(getattr(backbone, "n_storage_tokens", 0) or 0)

    n_patches = hidden_last.shape[0] - 1 - n_storage
    psize = int(math.sqrt(n_patches))
    assert psize * psize == n_patches, f"grid no cuadrado: N={n_patches}"
    return attns, hidden_last, psize, n_storage


def _blend(frame_np, norm):
    cmap = plt.get_cmap("magma_r")(norm)[:, :, :3]
    img_f = frame_np.astype(np.float64) / 255.0
    blended = np.clip(img_f * 0.35 + cmap * 0.65, 0, 1)
    return (blended * 255).astype(np.uint8)


def _upsample_grid(g, h):
    norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
    big = zoom_attn(norm, h / g.shape[0], order=1)
    return np.clip(big, 0, 1)


def dino_attn_map(attns, psize, h, layer=-1, head=-1, aggregation="capa"):
    """Heatmap CLS->parches desde attention explicita (DINOv2)."""
    if aggregation == "global":
        raw = np.stack([a[:, 0, 1:].mean(axis=0) for a in attns]).mean(axis=0)
    elif aggregation == "head":
        raw = np.stack([a[head, 0, 1:] for a in attns]).mean(axis=0)
    else:
        lid = layer if layer >= 0 else len(attns) - 1
        a = attns[lid][:, 0, 1:]
        raw = a[head] if head >= 0 else a.mean(axis=0)
    return raw.reshape(psize, psize)


def dino_heads_maps(attns, psize, layer=-1, aggregate_layers=False):
    """Lista de heatmaps por cabeza (DINOv2)."""
    if aggregate_layers:
        raw = np.stack([a[:, 0, 1:] for a in attns]).mean(axis=0)  # (h, Np)
    else:
        lid = layer if layer >= 0 else len(attns) - 1
        raw = attns[lid][:, 0, 1:]  # (h, Np)
    return [g.reshape(psize, psize) for g in raw]


def dino_cosine_map(hidden_last, psize, n_storage=0):
    """Similitud coseno CLS->parches del ultimo layer (v2 y v3)."""
    cls_vec = hidden_last[0]
    patches = hidden_last[1 + n_storage:]
    sim = (patches / (np.linalg.norm(patches, axis=-1, keepdims=True) + 1e-8)) @ (
        cls_vec / (np.linalg.norm(cls_vec) + 1e-8))
    return sim.reshape(psize, psize)


def generate_gif_dino(wrapper, frames, output_path, fps=6, layer=-1, head=-1,
                      aggregation="capa", dev="cpu"):
    h = frames[0].shape[0]
    frame_images = []
    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        attns, _, psize, _ = dino_forward_features(wrapper, frm, dev)
        raw = dino_attn_map(attns, psize, h, layer=layer, head=head, aggregation=aggregation)
        heatmap = _blend(frm, _upsample_grid(raw, h))
        frame_images.append(make_frame_image(frm, heatmap))
    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_images[0].save(output_path, save_all=True, append_images=frame_images[1:],
                         duration=500, loop=0, optimize=False)
    print(f"GIF guardado: {output_path}")


def generate_heads_gif_dino(wrapper, frames, output_path, fps=6, layer=-1,
                            goal_np=None, aggregate_layers=False, dev="cpu"):
    frame_images = []
    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        attns, _, psize, _ = dino_forward_features(wrapper, frm, dev)
        raw_maps = dino_heads_maps(attns, psize, layer=layer, aggregate_layers=aggregate_layers)
        h = frm.shape[0]
        maps = [(plt.get_cmap("magma_r")(_upsample_grid(g, h))[:, :, :3] * 255).astype(np.uint8)
                for g in raw_maps]
        frame_images.append(make_heads_frame_image(frm, maps, goal_np))
    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_images[0].save(output_path, save_all=True, append_images=frame_images[1:],
                         duration=500, loop=0, optimize=False)
    print(f"GIF guardado: {output_path}")


def generate_cosine_gif_dino(wrapper, frames, output_path, fps=6, dev="cpu"):
    """GIF coseno CLS->parches (vale para v2 y v3; sin attention explicita)."""
    h_orig = frames[0].shape[0]
    pil_frames = []
    for t, frame_np in enumerate(frames):
        print(f"Frame {t + 1}/{len(frames)}")
        _, hidden_last, psize, n_storage = dino_forward_features(wrapper, frame_np, dev)
        raw = dino_cosine_map(hidden_last, psize, n_storage)
        # reutiliza el render con colorbar del path HF
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        big = zoom_attn(norm, h_orig / psize, order=1)
        cmap_arr = plt.get_cmap("RdYlBu_r")(big)[:, :, :3]
        img_f = frame_np.astype(np.float64) / 255.0
        blended = np.clip(img_f * 0.35 + cmap_arr * 0.65, 0, 1)
        import io
        fig, ax = plt.subplots(figsize=(0.5, h_orig / 72), dpi=72)
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap("RdYlBu_r"),
                                              orientation="vertical", ticks=[0, 0.5, 1])
        cb.set_ticklabels(["Bajo", "Medio", "Alto"], fontsize=8)
        cb.ax.yaxis.set_tick_params(pad=6)
        cb.set_label("Similitud coseno CLS→parches", fontsize=7, labelpad=10)
        fig.subplots_adjust(left=0.25, right=0.5, top=0.95, bottom=0.05)
        fig.canvas.draw()
        cbar = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        pad = np.zeros((h_orig, 20, 3), dtype=np.uint8) + 255
        full = np.concatenate([(blended * 255).astype(np.uint8), pad, cbar], axis=1)
        pil_frames.append(Image.fromarray(full))
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                       duration=1000 // fps, loop=0)


def cosine_sim(cls_vec, patch_tokens):
    cls_norm = cls_vec / (cls_vec.norm() + 1e-8)
    patches_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-8)
    return (cls_norm * patches_norm).sum(dim=-1)


# ---- WaveViTEncoder: atencion recibida por token wavelet ----
# Cada bloque produce (B, h, N=256, Np=64). Promedio sobre N (queries) -> (h, Np).
# Reshape a 8x8, upsample bicubic a 224x224, blend con frame.
def _wave_grid_size(psize_q, psize_kv):
    return psize_q, psize_kv, psize_kv  # H_q, W_q, H_kv=W_kv (cuadrados)


def attn_map_for_frame_wave(encoder, frame_np, psize_q, psize_kv, h, layer=-1, head=-1, aggregation="capa"):
    x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(next(encoder.parameters()).device)
    with torch.no_grad():
        out = encoder(x, output_attentions=True)
    n_layers = len(out.attentions)
    lid = layer if layer >= 0 else n_layers - 1
    Hkv = Wkv = psize_kv  # ponytail: grid wavelet comprimido cuadrado
    if aggregation == "global":
        raw = torch.stack([att[0].mean(dim=1) for att in out.attentions])  # (n_layers, h, Np)
        raw = raw.mean(dim=(0, 1)).cpu().numpy()
    elif aggregation == "head":
        raw = torch.stack([att[0, head].mean(dim=1) for att in out.attentions])  # (n_layers, Np)
        raw = raw.mean(dim=0).cpu().numpy()
    else:
        layer_attn = out.attentions[lid][0]  # (h, N, Np)
        if head >= 0:
            raw = layer_attn[head].mean(dim=0).cpu().numpy()  # (Np,)
        else:
            raw = layer_attn.mean(dim=(0, 1)).cpu().numpy()
    raw_rs = raw.reshape(Hkv, Wkv)
    norm = (raw_rs - raw_rs.min()) / (raw_rs.max() - raw_rs.min() + 1e-8)
    big = F.interpolate(torch.from_numpy(norm).float()[None, None],
                        size=(h, h), mode="bicubic", align_corners=False)[0, 0].numpy()
    big = np.clip(big, 0, 1)
    cmap = plt.get_cmap("magma_r")(big)[:, :, :3]
    img_f = frame_np.astype(np.float64) / 255.0
    blended = np.clip(img_f * 0.35 + cmap * 0.65, 0, 1)
    return (blended * 255).astype(np.uint8)


def attn_heads_for_frame_wave(encoder, frame_np, psize_q, psize_kv, h, layer=-1, aggregate_layers=False):
    x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(next(encoder.parameters()).device)
    with torch.no_grad():
        out = encoder(x, output_attentions=True)
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    Hkv = Wkv = psize_kv
    if aggregate_layers:
        raw = torch.stack([att[0].mean(dim=1) for att in out.attentions])  # (n_layers, h, Np)
        raw = raw.mean(dim=0).cpu().numpy()
    else:
        lid = layer if layer >= 0 else n_layers - 1
        raw = out.attentions[lid][0].mean(dim=1).cpu().numpy()  # (h, Np)
    maps = []
    for hid in range(n_heads):
        g = raw[hid].reshape(Hkv, Wkv)
        norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
        big = F.interpolate(torch.from_numpy(norm).float()[None, None],
                            size=(h, h), mode="bicubic", align_corners=False)[0, 0].numpy()
        big = np.clip(big, 0, 1)
        cmap = plt.get_cmap("magma_r")(big)[:, :, :3]
        maps.append((cmap * 255).astype(np.uint8))
    return maps


def attn_map_for_frame(encoder, frame_np, psize, h, layer=-1, head=-1, aggregation="capa"):
    """Dado un frame numpy (H,W,3), devuelve heatmap blended.

    Args:
        layer: index de capa (-1 = ultima).
        head: index de cabeza (-1 = promedio de todas).
        aggregation: "capa" (una capa), "head" (una cabeza en todas capas),
                     "global" (promedio de todas capas y todas cabezas).
    """
    x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(next(encoder.parameters()).device)

    in_c = encoder.embeddings.patch_embeddings.projection.in_channels
    if getattr(encoder, '_encoder_is_swt', False):
        x = swt_conv4d(x, encoder._swt_levels, filter=encoder._swt_filter)
        x = encoder._swt_channel_proj(x)
    elif in_c != 3:
        levels = in_c // 3 - 1
        x = starlet_conv4d(x, levels)

    with torch.no_grad():
        out = encoder(x, output_attentions=True, output_hidden_states=True)

    n_layers = len(out.attentions)
    lid = layer if layer >= 0 else n_layers - 1

    if aggregation == "global":
        raw = torch.stack([att[0, :, 0, 1:] for att in out.attentions])
        raw = raw.mean(dim=(0, 1)).cpu().numpy()
    elif aggregation == "head":
        raw = torch.stack([att[0, head, 0, 1:] for att in out.attentions])
        raw = raw.mean(dim=0).cpu().numpy()
    else:
        raw = out.attentions[lid][0, :, 0, 1:]
        if head >= 0:
            raw = raw[head]
        else:
            raw = raw.mean(dim=0)
        raw = raw.cpu().numpy()
    raw = raw.reshape(psize, psize)

    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    big = zoom_attn(norm, h / psize, order=1)

    cmap = plt.get_cmap("magma_r")(big)[:, :, :3]
    img_f = frame_np.astype(np.float64) / 255.0
    blended = np.clip(img_f * 0.35 + cmap * 0.65, 0, 1)
    return (blended * 255).astype(np.uint8)


def cosine_map_for_frame(encoder, frame_np, psize, h, layer=-1):
    """Similitud coseno entre CLS y patches de la ultima capa."""
    x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(next(encoder.parameters()).device)
    in_c = encoder.embeddings.patch_embeddings.projection.in_channels
    if getattr(encoder, '_encoder_is_swt', False):
        x = swt_conv4d(x, encoder._swt_levels, filter=encoder._swt_filter)
        x = encoder._swt_channel_proj(x)
    elif in_c != 3:
        x = starlet_conv4d(x, in_c // 3 - 1)
    with torch.no_grad():
        out = encoder(x, output_attentions=False, output_hidden_states=True)
    hs = out.hidden_states[layer]  # (1, N+1, D)
    cls_vec = hs[0, 0]             # (D,)
    patches = hs[0, 1:]            # (N, D)
    sim = cosine_sim(cls_vec, patches).cpu().numpy()  # (N,)
    raw = sim.reshape(psize, psize)
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    big = zoom_attn(norm, h / psize, order=1)
    cmap_arr = plt.get_cmap("RdYlBu_r")(big)[:, :, :3]
    img_f = frame_np.astype(np.float64) / 255.0
    blended = np.clip(img_f * 0.35 + cmap_arr * 0.65, 0, 1)

    # barra de color vertical
    fig, ax = plt.subplots(figsize=(0.5, h / 72), dpi=72)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap("RdYlBu_r"),
                                          orientation="vertical",
                                          ticks=[0, 0.5, 1])
    cb.set_ticklabels(["Bajo", "Medio", "Alto"], fontsize=8)
    cb.ax.yaxis.set_tick_params(pad=6)
    cb.set_label("Similitud coseno CLS→parches", fontsize=7, labelpad=10)
    fig.subplots_adjust(left=0.25, right=0.5, top=0.95, bottom=0.05)
    fig.canvas.draw()
    cbar = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    pad = np.zeros((h, 20, 3), dtype=np.uint8) + 255
    full = np.concatenate([(blended * 255).astype(np.uint8), pad, cbar], axis=1)
    return full


def generate_cosine_gif(encoder, frames, psize, output_path, fps=6, layer=-1):
    """GIF de similitud coseno CLS→parches sobre frames."""
    h_orig = frames[0].shape[0]
    pil_frames = []
    for t, frame_np in enumerate(frames):
        blended = cosine_map_for_frame(encoder, frame_np, psize, h_orig, layer=layer)
        pil_frames.append(Image.fromarray(blended))
        print(f"Frame {t + 1}/{len(frames)}")
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                        duration=1000 // fps, loop=0)


def load_frames(n_frames=30, start_idx=0, dataset="tworoom", frames_path=None):
    """Cargar frames contiguos del dataset (tworoom | pusht) con num_steps=1.

    Orden de resolucion: --frames-path explicito > legacy repo/dataset/datasets/
    (si existe) > loader oficial con cache (STABLEWM_HOME/LOCAL_DATASET_DIR).
    """
    import stable_worldmodel
    import stable_worldmodel.data.formats.hdf5
    names = {"tworoom": "tworoom.h5", "pusht": "pusht_expert_train.lance"}
    name = names.get(dataset, dataset)
    if frames_path is not None:
        src = frames_path
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy = os.path.join(repo, "dataset", "datasets", name)
        src = legacy if os.path.exists(legacy) else name
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", os.environ.get("STABLEWM_HOME"))
    swm_dset = stable_worldmodel.data.load_dataset(
        name=src, cache_dir=cache_dir, keys_to_load=["pixels"], num_steps=1, frameskip=1,
    )
    if start_idx + n_frames > len(swm_dset):
        raise ValueError(f"start={start_idx} + nframes={n_frames} excede len(dataset)={len(swm_dset)}")
    frames = []
    for i in range(n_frames):
        arr = swm_dset[start_idx + i]["pixels"]  # (1, C, H, W)
        frame = arr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frames.append(frame)
    return frames


def _colorbar(total_width, height=35):
    fig, ax = plt.subplots(figsize=(total_width / 72, height / 72), dpi=72)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap="magma_r", norm=norm, orientation="horizontal",
        ticks=[0, 1],
    )
    cb.set_ticklabels(["0\nbaja", "1\nalta"], fontsize=7)
    cb.set_label("atencion CLS→parches", fontsize=8)
    fig.subplots_adjust(bottom=0.35, top=0.85, left=0.05, right=0.95)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    out = rgba[:, :, :3].astype(np.uint8)
    plt.close(fig)
    return out


def make_frame_image(frame_np, heatmap_np):
    """Juntar lado a lado: imagen original | heatmap, con barra de color abajo."""
    h, w = frame_np.shape[:2]
    canvas = np.zeros((h, 2 * w + 10, 3), dtype=np.uint8)
    canvas[:, :w] = frame_np
    canvas[:, w + 10:] = heatmap_np

    cbar = _colorbar(canvas.shape[1])
    full = np.concatenate([canvas, cbar], axis=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(full)
    plt.axis("off")
    import io
    buf_io = io.BytesIO()
    plt.savefig(buf_io, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
    buf_io.seek(0)
    img = Image.open(buf_io)
    plt.close()
    return img


def attn_heads_for_frame(encoder, frame_np, psize, h, layer=-1, aggregate_layers=False):
    """Return list of heatmaps, one per head.

    Si aggregate_layers=True, promedia todas las capas por cabeza.
    Sino, usa la capa especificada (layer=-1 = ultima).
    """
    x = ToTensor()(Image.fromarray(frame_np)).unsqueeze(0).to(next(encoder.parameters()).device)

    in_c = encoder.embeddings.patch_embeddings.projection.in_channels
    if getattr(encoder, '_encoder_is_swt', False):
        x = swt_conv4d(x, encoder._swt_levels, filter=encoder._swt_filter)
        x = encoder._swt_channel_proj(x)
    elif in_c != 3:
        levels = in_c // 3 - 1
        x = starlet_conv4d(x, levels)

    with torch.no_grad():
        out = encoder(x, output_attentions=True, output_hidden_states=True)

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]

    if aggregate_layers:
        raw = torch.stack([att[0, :, 0, 1:] for att in out.attentions])
        raw = raw.mean(dim=0).cpu().numpy()
    else:
        lid = layer if layer >= 0 else n_layers - 1
        raw = out.attentions[lid][0, :, 0, 1:].cpu().numpy()

    maps = []
    for hid in range(n_heads):
        g = raw[hid].reshape(psize, psize)
        norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
        big = zoom_attn(norm, h / psize, order=1)
        cmap = plt.get_cmap("magma_r")(big)[:, :, :3]
        maps.append((cmap * 255).astype(np.uint8))
    return maps


def make_heads_frame_image(frame_np, head_maps, goal_np=None):
    """Layout: original | h1 | h2 | h3 | goal, con colorbar abajo."""
    h, w = frame_np.shape[:2]
    gap = 5
    n_heads = len(head_maps)
    total_w = w * (1 + n_heads) + gap * n_heads
    if goal_np is not None:
        total_w += w + gap

    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    x = 0
    canvas[:, x:x + w] = frame_np
    x += w + gap
    for m in head_maps:
        canvas[:, x:x + w] = m
        x += w + gap
    if goal_np is not None:
        goal_resized = Image.fromarray(goal_np).resize((w, h), Image.BILINEAR)
        canvas[:, x:x + w] = np.asarray(goal_resized)
        x += w + gap

    cbar = _colorbar(canvas.shape[1])
    full = np.concatenate([canvas, cbar], axis=0)

    plt.figure(figsize=(14, 6))
    plt.imshow(full)
    plt.axis("off")
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


def generate_heads_gif(encoder, frames, psize, output_path, fps=6, layer=-1, goal_np=None, aggregate_layers=False):
    """GIF con original | h1 | h2 | h3 | goal."""
    h = frames[0].shape[0]
    frame_images = []

    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        maps = attn_heads_for_frame(encoder, frm, psize, h, layer=layer, aggregate_layers=aggregate_layers)
        img = make_heads_frame_image(frm, maps, goal_np)
        frame_images.append(img)

    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_images[0].save(
        output_path,
        save_all=True,
        append_images=frame_images[1:],
        duration=500,
        loop=0,
        optimize=False,
    )
    print(f"GIF guardado: {output_path}")


def generate_gif(encoder, frames, psize, output_path, fps=6, layer=-1, head=-1, aggregation="capa"):
    h = frames[0].shape[0]
    frame_images = []

    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        heatmap = attn_map_for_frame(encoder, frm, psize, h, layer=layer, head=head, aggregation=aggregation)
        img = make_frame_image(frm, heatmap)
        frame_images.append(img)

    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_images[0].save(
        output_path,
        save_all=True,
        append_images=frame_images[1:],
        duration=500,
        loop=0,
        optimize=False,
    )
    print(f"GIF guardado: {output_path}")


def generate_all_layers_gif(encoder, frames, psize, output_dir, fps=6):
    """Generar un GIF por cada capa, promediando sobre todas las cabezas."""
    n_layers = encoder.config.num_hidden_layers
    os.makedirs(output_dir, exist_ok=True)

    for lid in range(n_layers):
        out_path = os.path.join(output_dir, f"layer_{lid}.gif")
        print(f"\n=== Capa {lid}/{n_layers - 1} ===")
        generate_gif(encoder, frames, psize, out_path, fps=fps, layer=lid, head=-1)


def generate_all_heads_gif(encoder, frames, psize, output_dir, fps=6, layer=-1):
    """Generar un GIF por cada cabeza en la capa dada."""
    n_heads = 3
    os.makedirs(output_dir, exist_ok=True)

    for hid in range(n_heads):
        out_path = os.path.join(output_dir, f"head_{hid}.gif")
        print(f"\n=== Cabeza {hid}/{n_heads - 1} (capa {layer}) ===")
        generate_gif(encoder, frames, psize, out_path, fps=fps, layer=layer, head=hid)


def generate_gif_wave(encoder, frames, psize_q, psize_kv, output_path, fps=6, layer=-1, head=-1, aggregation="capa"):
    h = frames[0].shape[0]
    frame_images = []
    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        heatmap = attn_map_for_frame_wave(encoder, frm, psize_q, psize_kv, h, layer=layer, head=head, aggregation=aggregation)
        img = make_frame_image(frm, heatmap)
        frame_images.append(img)
    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_images[0].save(output_path, save_all=True, append_images=frame_images[1:],
                         duration=500, loop=0, optimize=False)
    print(f"GIF guardado: {output_path}")


def generate_heads_gif_wave(encoder, frames, psize_q, psize_kv, output_path, fps=6, layer=-1, goal_np=None, aggregate_layers=False):
    h = frames[0].shape[0]
    frame_images = []
    for i, frm in enumerate(frames):
        print(f"Frame {i + 1}/{len(frames)}")
        maps = attn_heads_for_frame_wave(encoder, frm, psize_q, psize_kv, h, layer=layer, aggregate_layers=aggregate_layers)
        img = make_heads_frame_image(frm, maps, goal_np)
        frame_images.append(img)
    print("Armado GIF...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_images[0].save(output_path, save_all=True, append_images=frame_images[1:],
                         duration=500, loop=0, optimize=False)
    print(f"GIF guardado: {output_path}")


def generate_all_layers_gif_wave(encoder, frames, psize_q, psize_kv, output_dir, fps=6):
    n_layers = len(encoder.blocks)
    os.makedirs(output_dir, exist_ok=True)
    for lid in range(n_layers):
        out_path = os.path.join(output_dir, f"layer_{lid}.gif")
        print(f"\n=== Capa {lid}/{n_layers - 1} ===")
        generate_gif_wave(encoder, frames, psize_q, psize_kv, out_path, fps=fps, layer=lid, head=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="dataset/checkpoints/lewm/weights_epoch_10.pt")
    parser.add_argument("--nframes", type=int, default=30)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--layer", type=int, default=-1, help="Capa (-1 = ultima)")
    parser.add_argument("--head", type=int, default=-1, help="Cabeza (-1 = promedio de todas)")
    parser.add_argument("--mode", default="capa", choices=["capa", "head", "global", "heads", "cosine"],
                        help='"capa" (default): heatmap por capa.\n'
                             '"head": una cabeza promediada sobre 12 capas.\n'
                             '"global": promedio de 12 capas x 3 cabezas.\n'
                             '"heads": original | h1 | h2 | h3 | goal.\n'
                             '"cosine": similitud coseno CLS→parches.')
    parser.add_argument("--goal-offset", type=int, default=30,
                        help="Frames hacia adelante para el panel goal (mode=heads)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directorio donde guardar los resultados")
    parser.add_argument("--aggregate-layers", action="store_true",
                        help="Promediar atencion sobre todas las capas (mode=heads)")
    parser.add_argument("--dataset", default="tworoom", choices=["tworoom", "pusht"],
                        help="Dataset de origen de los frames")
    parser.add_argument("--frames-path", default=None,
                        help="Ruta local al dataset (por defecto: resolucion por cache)")
    args = parser.parse_args()

    if "LOCAL_DATASET_DIR" not in os.environ and "STABLEWM_HOME" not in os.environ:
        os.environ["LOCAL_DATASET_DIR"] = os.path.expanduser("~/.stable_worldmodel")

    print("Cargando modelo...")
    encoder = load_encoder(args.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(dev)
    print(f"Dispositivo: {dev}")

    is_wave = getattr(encoder, "_encoder_kind", "hf") == "wave"
    is_dino = getattr(encoder, "_encoder_kind", "") == "dino"
    is_dinov3 = is_dino and getattr(encoder, "_is_dinov3", False)

    if is_dino:
        # DINO: psize dinamico por forward (grid sqrt de parches); nada que precalcular.
        psize, n_layers = None, len(list(encoder.model.blocks))
        print(f"DINO ({'v3' if is_dinov3 else 'v2'}). layers={n_layers} (psize dinamico)")

    if is_wave:
        # ponytail: N_q = (224/14)^2 = 256, N_kv = N_q/4 = 64 -> grids 16x16 / 8x8
        psize_q = encoder.img_size // encoder.patch_size if hasattr(encoder, "img_size") else 16
        psize_kv = psize_q // 2
        psize = psize_q
        n_layers = len(encoder.blocks)
        n_heads = encoder.num_heads
        print(f"WaveViTEncoder. Patch grid Q: {psize_q}x{psize_q}, KV (wavelet): {psize_kv}x{psize_kv}, "
              f"layers={n_layers}, heads={n_heads}")
    elif not is_dino:
        # ponytail: psize desde config en vez de dummy forward
        psize = encoder.config.image_size // encoder.config.patch_size
        n_layers = encoder.config.num_hidden_layers
        print(f"HF ViT. Patch grid: {psize}x{psize}, layers={n_layers}")

    print("Cargando frames...")
    frames = load_frames(n_frames=args.nframes, start_idx=args.start,
                         dataset=args.dataset, frames_path=args.frames_path)
    print(f"{len(frames)} frames cargados")

    os.makedirs(args.output_dir, exist_ok=True)

    if is_dino:
        if is_dinov3 and args.mode != "cosine":
            raise ValueError("DINOv3 solo soporta mode=cosine (RoPE impide recomputar "
                             "attention exacta); usa --mode cosine.")
        if args.mode == "cosine":
            generate_cosine_gif_dino(encoder, frames, f"{args.output_dir}/cosine.gif", dev=dev)
        elif args.mode == "heads":
            goal_np = frames[-1]
            layer_tag = f"layer{args.layer}" if not args.aggregate_layers else "agglayers"
            generate_heads_gif_dino(encoder, frames,
                                    f"{args.output_dir}/heads_{layer_tag}.gif",
                                    layer=args.layer, goal_np=goal_np,
                                    aggregate_layers=args.aggregate_layers, dev=dev)
        else:
            aggregation = {"capa": "capa", "head": "head", "global": "global"}[args.mode]
            generate_gif_dino(encoder, frames, f"{args.output_dir}/attention_gif.gif",
                              layer=args.layer, head=args.head,
                              aggregation=aggregation, dev=dev)
        return

    if args.mode == "cosine":
        lid = args.layer if args.layer >= 0 else n_layers - 1
        generate_cosine_gif(encoder, frames, psize,
                            f"{args.output_dir}/cosine_layer{lid}.gif",
                            layer=lid)
    elif is_wave:
        if args.mode == "heads":
            goal_np = frames[-1]
            layer_tag = f"layer{args.layer}" if not args.aggregate_layers else "agglayers"
            generate_heads_gif_wave(encoder, frames, psize_q, psize_kv,
                                    f"{args.output_dir}/heads_{layer_tag}.gif",
                                    layer=args.layer, goal_np=goal_np,
                                    aggregate_layers=args.aggregate_layers)
        elif args.layer == -1 and args.head == -1:
            generate_all_layers_gif_wave(encoder, frames, psize_q, psize_kv, f"{args.output_dir}/by_layer")
            generate_gif_wave(encoder, frames, psize_q, psize_kv, f"{args.output_dir}/attention_gif.gif")
        elif args.head == -1:
            generate_gif_wave(encoder, frames, psize_q, psize_kv, f"{args.output_dir}/layer_{args.layer}.gif", layer=args.layer)
        elif args.layer >= 0:
            generate_gif_wave(encoder, frames, psize_q, psize_kv,
                              f"{args.output_dir}/head_{args.head}_layer{args.layer}.gif",
                              layer=args.layer, head=args.head)
        else:
            os.makedirs(f"{args.output_dir}/by_head", exist_ok=True)
            for lid in range(n_layers):
                out_path = f"{args.output_dir}/by_head/head_{args.head}_layer{lid}.gif"
                print(f"\n=== Cabeza {args.head}, capa {lid}/{n_layers - 1} ===")
                generate_gif_wave(encoder, frames, psize_q, psize_kv, out_path, layer=lid, head=args.head)
    else:
        if args.mode == "heads":
            goal_np = frames[-1]
            layer_tag = f"layer{args.layer}" if not args.aggregate_layers else "agglayers"
            generate_heads_gif(encoder, frames, psize,
                              f"{args.output_dir}/heads_{layer_tag}.gif",
                              layer=args.layer, goal_np=goal_np,
                              aggregate_layers=args.aggregate_layers)
        elif args.layer == -1 and args.head == -1:
            generate_all_layers_gif(encoder, frames, psize, f"{args.output_dir}/by_layer")
            generate_gif(encoder, frames, psize, f"{args.output_dir}/attention_gif.gif")
        elif args.head == -1:
            generate_gif(encoder, frames, psize, f"{args.output_dir}/layer_{args.layer}.gif", layer=args.layer)
        elif args.layer >= 0:
            generate_gif(encoder, frames, psize,
                         f"{args.output_dir}/head_{args.head}_layer{args.layer}.gif",
                         layer=args.layer, head=args.head)
        else:
            os.makedirs(f"{args.output_dir}/by_head", exist_ok=True)
            for lid in range(n_layers):
                out_path = f"{args.output_dir}/by_head/head_{args.head}_layer{lid}.gif"
                print(f"\n=== Cabeza {args.head}, capa {lid}/{n_layers - 1} ===")
                generate_gif(encoder, frames, psize, out_path, layer=lid, head=args.head)


if __name__ == "__main__":
    main()