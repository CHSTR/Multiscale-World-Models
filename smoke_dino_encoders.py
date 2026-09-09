"""Smoke test for the DINOv2/v3 LoRA model configs.

For each of the 4 model yamls we compose the full train config (base: lewm.yaml,
which provides img_size / embed_dim / history_size etc.), instantiate the encoder
via hydra, and run a forward pass. Starlet configs receive RGB (3ch) and apply
starlet_conv4d internally (starlet_levels => 3*(L+1) channels).

We also exercise the *dynamic token* feature (``n_dyn_tokens=k>0``): for the
cls / dyn_first / dyn_mean / dyn_concat readouts we assert the token-sequence
shape, the pooled dim and the off-by-one CLS placement. We also exercise the
``dyn_init`` modes: ``cls`` (exact copy of the CLS register, zero noise),
``cls_noise`` (copy + ``N(0, init_std)``) and ``zeros`` (exact zeros before the
forward). No training happens. (action_encoder.input_dim is a placeholder filled
by train.py, so we instantiate the encoder only, as required.)
"""
import os
import warnings

warnings.filterwarnings("ignore")

import hydra
from hydra import compose, initialize
import hydra.utils
import torch

# Local copy of the official vit_small/16 weights (avoids a hub download).
DINOV3_CKPT = os.environ.get(
    "DINOV3_CKPT", "/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

# (config model name, input channels, expected internal channels)
CONFIGS = [
    ("dinov2_lora", 3, 3),
    ("dinov3_lora", 3, 3),
    ("dinov2_starlet_lora", 3, 12),  # starlet_levels=3 => 3*4
    ("dinov3_starlet_lora", 3, 12),
]

# (n_dyn_tokens, readout) for the dynamic-token smoke.
DYN_CONFIGS = [
    (4, "cls"),
    (4, "dyn_first"),
    (4, "dyn_mean"),
    (4, "dyn_concat"),
]

PATCH_SIZE = 16  # matches dinov3_vits16 (img_size=224 => 14x14=196 patches)
N_STORAGE = 4  # dinov3_vits16 hardcodes n_storage_tokens=4


def build_encoder(model_name, extra_overrides=None, local_ckpt=False):
    overrides = [f"model={model_name}"]
    # Only DINOv3 configs use the local vit_small/16 checkpoint; the dinov2
    # configs fall back to their own hub checkpoint.
    if local_ckpt:
        overrides.append(f"model.encoder.ckpt_path={DINOV3_CKPT}")
    if extra_overrides:
        overrides += extra_overrides
    with initialize(version_base=None, config_path="config/train"):
        cfg = compose(config_name="lewm", overrides=overrides)
    enc_cfg = hydra.utils.instantiate(cfg.model.encoder)
    return cfg, enc_cfg


def main():
    img_size = None

    # ------------------------------------------------------------------ base
    for model_name, input_ch, internal_ch in CONFIGS:
        print(f"\n===== {model_name} (input={input_ch}ch, internal={internal_ch}ch) =====")
        cfg, enc = build_encoder(model_name, local_ckpt=("dinov3" in model_name))
        img_size = cfg.img_size
        print(f"encoder: {enc.__class__.__module__}.{enc.__class__.__name__} "
              f"dim={enc.embed_dim}, in_chans={enc.in_chans}, "
              f"starlet_levels={getattr(enc, 'starlet_levels', 0)}")
        assert enc.in_chans == internal_ch, f"in_chans {enc.in_chans} != {internal_ch}"

        x = torch.randn(2, input_ch, cfg.img_size, cfg.img_size)
        out = enc(x, interpolate_pos_encoding=True)
        lhs = out.last_hidden_state
        emb = lhs[:, 0]
        assert torch.isfinite(lhs).all()
        print(f"  forward {input_ch}ch: last_hidden_state={tuple(lhs.shape)}  emb={tuple(emb.shape)}")

        # level_weights entrenables cuando hay starlet (igual que el original)
        if getattr(enc, "starlet_levels", 0) > 0:
            assert isinstance(enc.level_weights, torch.nn.Parameter) and enc.level_weights.requires_grad
            assert enc.level_weights.shape == (enc.starlet_levels + 1,)
            print(f"  level_weights: {enc.level_weights.detach().tolist()}")

        # k=0 (dinov3 only): readout=cls => pooled == CLS == last_hidden_state[:, 0]
        # (this is the path jepa.encode now uses). dinov2 doesn't expose .pooled.
        if getattr(enc, "n_dyn_tokens", 1) == 0 and hasattr(out, "pooled"):
            assert torch.allclose(out.pooled, out.last_hidden_state[:, 0], atol=1e-6)
            assert out.cls.shape[0] == out.pooled.shape[0]

        trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        total = sum(p.numel() for p in enc.parameters())
        print(f"  total params: {total:,}  trainable params: {trainable:,}")

    # -------------------------------------------------- dynamic tokens (k>0)
    assert img_size is not None
    N_patches = (img_size // PATCH_SIZE) ** 2  # 196
    for k, readout in DYN_CONFIGS:
        print(f"\n===== dinov3_lora + n_dyn_tokens={k} readout={readout} =====")
        cfg, enc = build_encoder(
            "dinov3_lora",
            [
                f"+model.encoder.n_dyn_tokens={k}",
                f"+model.encoder.readout={readout}",
            ],
            local_ckpt=True,
        )
        assert enc.n_dyn_tokens == k
        assert enc.readout == readout
        readout_dim = k * 384 if readout == "dyn_concat" else 384
        assert enc.readout_dim == readout_dim, (enc.readout_dim, readout_dim)

        x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
        out = enc(x, interpolate_pos_encoding=True)
        lhs = out.last_hidden_state
        assert torch.isfinite(lhs).all()

        # --- token sequence shape (CLS + k dyn + 4 storage + patches) ---
        assert lhs.shape == (2, 1 + k + N_STORAGE + N_patches, 384), tuple(lhs.shape)
        print(f"  last_hidden_state={tuple(lhs.shape)}  pooled={tuple(out.pooled.shape)}  "
              f"dyn_tokens={tuple(enc.dyn_tokens.shape)}  readout_dim={enc.readout_dim}")

        # --- off-by-one CLS: CLS at index 0, first dyn at index 1 ---
        assert torch.allclose(out.cls, lhs[:, 0], atol=1e-6), "CLS must be at index 0"
        if readout == "dyn_first":
            assert torch.allclose(out.pooled, lhs[:, 1], atol=1e-4), (
                "dyn_first must read the token right after CLS (index 1)"
            )
        elif readout == "dyn_mean":
            assert torch.allclose(out.pooled, lhs[:, 1:1 + k].mean(dim=1), atol=1e-4)
        elif readout == "dyn_concat":
            assert torch.allclose(out.pooled, lhs[:, 1:1 + k].reshape(2, -1), atol=1e-4)
            assert out.pooled.shape == (2, k * 384)

    # ------------------------------------------ dyn_init modes: zeros / cls
    # ``cls`` now forces zero noise (exact copy of the CLS register), so we
    # verify the exact-equality cases that the smoke now covers.
    print("\n===== dyn_init modes: zeros / cls / cls_noise =====")
    k_init = 4

    # zeros: dyn_tokens are exactly zero BEFORE any forward pass
    cfg_z, enc_z = build_encoder(
        "dinov3_lora",
        [f"+model.encoder.n_dyn_tokens={k_init}", "+model.encoder.dyn_init=zeros"],
        local_ckpt=True,
    )
    assert enc_z.dyn_init == "zeros" and enc_z.dyn_tokens.shape == (1, k_init, 384)
    assert enc_z.dyn_tokens.abs().max().item() == 0.0, (
        "zeros init: dyn_tokens must be exactly zero before the forward"
    )
    print(f"  zeros : max|dyn_tokens| = {enc_z.dyn_tokens.abs().max().item():.2e}  (==0, no prior)")

    # cls: dyn_tokens are an EXACT copy of the source cls_token (zero noise)
    cfg_c, enc_c = build_encoder(
        "dinov3_lora",
        [f"+model.encoder.n_dyn_tokens={k_init}", "+model.encoder.dyn_init=cls"],
        local_ckpt=True,
    )
    assert enc_c.dyn_init == "cls"
    raw_cls = enc_c.model.cls_token.detach()  # (1, 1, 384)
    expected = raw_cls[0].expand(k_init, 384)
    assert torch.allclose(enc_c.dyn_tokens[0], expected, atol=1e-4), (
        "cls init: dyn_tokens must equal the source CLS token (atol 1e-4)"
    )
    # pooled==cls equivalence for the zero-noise cls point
    x_c = torch.randn(2, 3, cfg_c.img_size, cfg_c.img_size)
    out_c = enc_c(x_c, interpolate_pos_encoding=True)
    assert torch.allclose(out_c.cls, out_c.last_hidden_state[:, 0], atol=1e-6)
    assert torch.allclose(out_c.pooled, out_c.last_hidden_state[:, 0], atol=1e-6)
    print(f"  cls   : max|dyn_tokens - raw CLS| = "
          f"{(enc_c.dyn_tokens[0] - expected).abs().max().item():.2e}  (==0, baseline)")

    # cls_noise with an explicit init_std: copy + N(0, init_std) per token
    cfg_n, enc_n = build_encoder(
        "dinov3_lora",
        [
            f"+model.encoder.n_dyn_tokens={k_init}",
            "+model.encoder.dyn_init=cls_noise",
            "+model.encoder.init_std=0.5",
        ],
        local_ckpt=True,
    )
    assert enc_n.dyn_init == "cls_noise" and enc_n.init_std == 0.5
    raw_cls_n = enc_n.model.cls_token.detach()
    diff = (enc_n.dyn_tokens[0] - raw_cls_n[0].expand(k_init, 384)).abs().max().item()
    assert diff > 1e-4, "cls_noise: tokens must differ from raw CLS (noise added at init)"
    print(f"  cls_n : max|dyn_tokens - raw CLS| = {diff:.2e}  (nonzero, init_std=0.5)")

    print("\n===== ALL SMOKE CHECKS PASSED =====")


if __name__ == "__main__":
    main()
