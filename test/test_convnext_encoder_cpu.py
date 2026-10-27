"""CPU smoke test for DinoV3ConvNeXtEncoder (pretrained=False -> no network).

Verifies the JEPA-compatible interface, embed dims, LoRA/freeze accounting,
starlet frontend and stem expansion for the tiny/small/base ConvNeXt backbones.
"""
import re
import types as _types

import torch

from src.encoders.dinov3_convnext_encoder import DinoV3ConvNeXtEncoder
from src.encoders.vit_lora import LoRALinear

# pwconv base weights must be frozen (only LoRA factors train)
PAT_PWC_BASE = re.compile(r"\.pwconv[12]\.(weight|bias)$")


def test_sizes_and_forward():
    x = torch.randn(2, 3, 224, 224)
    for size, dim in [("tiny", 768), ("small", 768), ("base", 1024)]:
        enc = DinoV3ConvNeXtEncoder(pretrained=False, size=size, device="cpu")
        assert enc.embed_dim == dim, (size, enc.embed_dim, dim)
        assert enc.in_chans == 3
        out = enc(x, interpolate_pos_encoding=True)
        assert isinstance(out, _types.SimpleNamespace)
        # 224 -> 56 (stem /4) -> 28 -> 14 -> 7 (three /2 stages) = 49 patches + 1 CLS
        n_patches = (224 // 32) ** 2
        lhs = out.last_hidden_state
        assert lhs.shape == (2, 1 + n_patches, dim), lhs.shape
        # CLS first: column 0 must equal the backbone CLS feature.
        feats = enc.model(x, is_training=True)
        cls = feats["x_norm_clstoken"]
        assert torch.allclose(lhs[:, 0], cls, atol=1e-5), "CLS not first"
        assert lhs.shape[1] == 1 + feats["x_norm_patchtokens"].shape[1]
        assert torch.isfinite(lhs).all()
        total = sum(p.numel() for p in enc.parameters())
        trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"[ok] {size}: dim={dim}, lhs={tuple(lhs.shape)}, "
              f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
        assert 0 < trainable < total * 0.05, "trainable fraction too large"


# allowed trainable patterns (matched on owning module class + structure):
#  - stem conv:  downsample_layers.0.0.<weight,bias>
#  - norm:       owning module is a LayerNorm subclass
#  - lora factor: owning module is a LoRALinear
from src.encoders.vit_lora import LoRALinear


def _owning_module(model, name):
    parts = name.split(".")[:-1]  # drop the leaf weight/bias/lora_A/.._B
    mod = model
    for p in parts:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def test_trainable_only_lora_norm_stem():
    enc = DinoV3ConvNeXtEncoder(pretrained=False, size="tiny", device="cpu")
    model = enc.model
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]

    stem_conv_key = "downsample_layers.0.0.weight"
    assert any(n == stem_conv_key for n in trainable), "stem conv not trainable"

    for n in trainable:
        mod = _owning_module(model, n)
        cls = mod.__class__.__name__
        is_stem = n.split(".")[:3] == ["downsample_layers", "0", "0"]
        is_norm = cls.startswith("LayerNorm")
        is_lora = isinstance(mod, LoRALinear)
        assert is_stem or is_norm or is_lora, f"unexpected trainable module for {n}: {cls}"

    # pwconv base weights must be frozen (only LoRA factors train)
    for n, p in model.named_parameters():
        if PAT_PWC_BASE.search(n):
            assert not p.requires_grad, f"pwconv base weight must be frozen: {n}"
    print(f"[ok] trainable params={len(trainable)} groups = LoRA + norms + stem only")


def test_starlet_frontend():
    enc = DinoV3ConvNeXtEncoder(pretrained=False, size="tiny", starlet_levels=3,
                                device="cpu")
    assert enc.starlet_levels == 3
    assert enc.in_chans == 12, enc.in_chans
    assert isinstance(enc.level_weights, torch.nn.Parameter)
    assert enc.level_weights.requires_grad
    assert enc.level_weights.shape == (4,)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    n_patches = (224 // 32) ** 2
    assert out.last_hidden_state.shape == (2, 1 + n_patches, 768), out.last_hidden_state.shape
    assert torch.isfinite(out.last_hidden_state).all()
    print("[ok] starlet_levels=3 -> in_chans=12, level_weights trainable, forward OK")


def test_in_chans_stem_expansion():
    enc = DinoV3ConvNeXtEncoder(pretrained=False, size="tiny", in_chans=15, device="cpu")
    assert enc.in_chans == 15
    stem_conv = enc.model.downsample_layers[0][0]
    assert stem_conv.in_channels == 15, stem_conv.in_channels
    assert stem_conv.weight.requires_grad, "expanded stem conv must be trainable"
    x = torch.randn(2, 15, 224, 224)
    out = enc(x)
    n_patches = (224 // 32) ** 2
    assert out.last_hidden_state.shape == (2, 1 + n_patches, 768)
    assert torch.isfinite(out.last_hidden_state).all()
    print("[ok] in_chans=15 -> stem conv expanded to 15ch, trainable, forward OK")


if __name__ == "__main__":
    test_sizes_and_forward()
    test_trainable_only_lora_norm_stem()
    test_starlet_frontend()
    test_in_chans_stem_expansion()
    print("\nALL CPU CHECKS PASSED")
