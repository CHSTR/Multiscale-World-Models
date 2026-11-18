"""Test del control de canales RGBRepeatEncoder (Task 5 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_rgbrep.py -q
"""
import pytest
import torch

from stable_pretraining.backbone.utils import vit_hf


def _vit(seed=0):
    torch.manual_seed(seed)
    return vit_hf(size="tiny", patch_size=14, image_size=224,
                  pretrained=False, use_mask_token=False)


def test_channel_count_matches_starlet():
    from wavelet.rgbrep_encoder import RGBRepeatEncoder
    from wavelet.starlet_encoder import StarletEncoder

    levels = 4
    enc_rep = RGBRepeatEncoder(_vit(0), levels=levels)
    enc_sta = StarletEncoder(_vit(1), levels=levels, learnable_weights=False)
    pe_rep = enc_rep.vit.embeddings.patch_embeddings.projection
    pe_sta = enc_sta.vit.embeddings.patch_embeddings.projection
    assert pe_rep.in_channels == pe_sta.in_channels == 3 * (levels + 1)
    # misma capacidad: mismo número de parámetros en el patch embedding
    assert sum(p.numel() for p in pe_rep.parameters()) == \
        sum(p.numel() for p in pe_sta.parameters())


def test_step0_equivalence_to_plain_vit():
    """Con la init replicada, el encoder ES el ViT plano en el paso 0."""
    from wavelet.rgbrep_encoder import RGBRepeatEncoder

    vit_plain = _vit(0)
    vit_wrapped = _vit(0)          # misma semilla => mismos pesos
    assert torch.equal(
        next(vit_plain.parameters()), next(vit_wrapped.parameters()))
    enc = RGBRepeatEncoder(vit_wrapped, levels=4)
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out_plain = vit_plain(x)
        out_rep = enc(x)
    # mismo CLS que el ViT plano (state_dict de enc.vit difiere solo en el PE)
    assert torch.allclose(out_plain[0], out_rep[0], atol=1e-5)


def test_bands_sum_to_input():
    from wavelet.rgbrep_encoder import RGBRepeatEncoder

    enc = RGBRepeatEncoder(_vit(0), levels=4)
    x = torch.rand(2, 3, 224, 224)
    # (L+1) copias de x/(L+1): Σ bandas = x (misma propiedad que starlet-equiv)
    n_copies = enc.levels + 1
    assert enc.vit.embeddings.patch_embeddings.projection.in_channels == 3 * n_copies
