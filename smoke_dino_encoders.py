"""Smoke test for the DINOv3 LoRA model configs.

For each of the 4 model yamls we compose the full train config (base: lewm.yaml,
which provides img_size / embed_dim / history_size etc.), instantiate the encoder
via hydra, and run a forward pass. Starlet configs receive RGB (3ch) and apply
starlet_conv4d internally (starlet_levels => 3*(L+1) channels).

The core of this smoke is the **direct-predictor contract (P0)** for the DINOv3
encoder: it owns a single ``prompt_tokens`` parameter of shape
``(1, n_in_tokens + n_tgt_tokens, 384)`` and the forward exposes

    tok_in  : (BT, 384)     -- first prompt token
    tok_tgt : (BT, n_tgt, 384)
    pooled  : == tok_in     -- what jepa.encode reads
    tie     : with tie_tgt_to_in=True, tok_tgt == tok_in.unsqueeze(1)

We also exercise ``tok_init`` (``cls`` / ``cls_noise`` / ``normal``) and the
legacy ``n_dyn_tokens`` compat alias. No training happens. (action_encoder.input_dim
is a placeholder filled by train.py, so we instantiate the encoder only.)
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
        assert torch.isfinite(lhs).all()
        print(f"  forward {input_ch}ch: last_hidden_state={tuple(lhs.shape)}")

        # level_weights entrenables cuando hay starlet (igual que el original)
        if getattr(enc, "starlet_levels", 0) > 0:
            assert isinstance(enc.level_weights, torch.nn.Parameter) and enc.level_weights.requires_grad
            assert enc.level_weights.shape == (enc.starlet_levels + 1,)
            print(f"  level_weights: {enc.level_weights.detach().tolist()}")

        # --- DINOv3 contract (P0): tok_in / tok_tgt / pooled == tok_in ---
        # dinov2 does not expose the prompt-token API; JEPA falls back to the CLS.
        if getattr(out, "tok_in", None) is not None:
            assert tuple(out.tok_in.shape) == (2, 384), tuple(out.tok_in.shape)
            assert tuple(out.pooled.shape) == (2, 384), tuple(out.pooled.shape)
            assert torch.allclose(out.pooled, out.tok_in, atol=1e-5), "pooled must == tok_in"
            n_tgt = enc.n_tgt_tokens
            assert tuple(out.tok_tgt.shape) == (2, n_tgt, 384), tuple(out.tok_tgt.shape)
            print(f"  P0: tok_in={tuple(out.tok_in.shape)}  tok_tgt={tuple(out.tok_tgt.shape)}  "
                  f"pooled==tok_in  readout_dim={enc.readout_dim}")
        else:
            emb = getattr(out, "pooled", None)
            if emb is None:
                emb = out.last_hidden_state[:, 0]
            assert emb.shape == (2, 384)
            print(f"  dinov2 fallback: emb={tuple(emb.shape)}")

        # JEPA compat: the pooled summary is always 384-d (the projector consumes it).
        if getattr(enc, "readout_dim", None) is not None:
            assert enc.readout_dim == 384

        trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        total = sum(p.numel() for p in enc.parameters())
        print(f"  total params: {total:,}  trainable params: {trainable:,}")

    assert img_size is not None
    N_patches = (img_size // PATCH_SIZE) ** 2  # 196

    # --------------------------------------------------- direct-predictor P0
    print("\n===== dinov3_lora + contract P0 (default n_in=1, n_tgt=1) =====")
    cfg_p, enc_p = build_encoder("dinov3_lora", local_ckpt=True)
    assert enc_p.n_in_tokens == 1 and enc_p.n_tgt_tokens == 1
    assert enc_p.prompt_tokens.shape == (1, 2, 384), enc_p.prompt_tokens.shape
    x = torch.randn(2, 3, cfg_p.img_size, cfg_p.img_size)
    out = enc_p(x, interpolate_pos_encoding=True)
    # sequence: [CLS(1) + prompt(2) + storage(4) + patches]
    assert out.last_hidden_state.shape == (2, 1 + 2 + N_STORAGE + N_patches, 384), tuple(out.last_hidden_state.shape)
    assert out.tok_in.shape == (2, 384)
    assert out.tok_tgt.shape == (2, 1, 384)          # n_tgt = 1
    assert torch.allclose(out.pooled, out.tok_in, atol=1e-5)
    print(f"  default : tok_in={tuple(out.tok_in.shape)}  tok_tgt={tuple(out.tok_tgt.shape)}  "
          f"last_hidden_state={tuple(out.last_hidden_state.shape)}")

    # n_tgt > 1
    print("\n===== dinov3_lora + n_tgt=3 =====")
    cfg_t, enc_t = build_encoder(
        "dinov3_lora",
        ["+model.encoder.n_tgt_tokens=3"],
        local_ckpt=True,
    )
    assert enc_t.n_tgt_tokens == 3
    assert enc_t.prompt_tokens.shape == (1, 4, 384), enc_t.prompt_tokens.shape  # n_in(1)+n_tgt(3)
    out_t = enc_t(torch.randn(2, 3, cfg_t.img_size, cfg_t.img_size), interpolate_pos_encoding=True)
    assert out_t.tok_tgt.shape == (2, 3, 384), tuple(out_t.tok_tgt.shape)
    assert out_t.last_hidden_state.shape == (2, 1 + 4 + N_STORAGE + N_patches, 384), tuple(out_t.last_hidden_state.shape)
    assert torch.allclose(out_t.pooled, out_t.tok_in, atol=1e-5)
    print(f"  n_tgt=3 : tok_tgt={tuple(out_t.tok_tgt.shape)}  prompt_tokens={tuple(enc_t.prompt_tokens.shape)}")

    # tie_tgt_to_in => tok_tgt == tok_in.unsqueeze(1)
    print("\n===== dinov3_lora + tie_tgt_to_in=True =====")
    cfg_b, enc_b = build_encoder(
        "dinov3_lora",
        ["+model.encoder.tie_tgt_to_in=true"],
        local_ckpt=True,
    )
    assert enc_b.tie_tgt_to_in is True
    out_b = enc_b(torch.randn(2, 3, cfg_b.img_size, cfg_b.img_size), interpolate_pos_encoding=True)
    assert out_b.tok_tgt.shape == (2, 1, 384)
    assert torch.equal(out_b.tok_tgt, out_b.tok_in.unsqueeze(1)), "tie: tok_tgt must == tok_in.unsqueeze(1)"
    assert torch.allclose(out_b.pooled, out_b.tok_in, atol=1e-5)
    print("  tie     : tok_tgt == tok_in.unsqueeze(1)  (torch.equal OK)")

    # --------------------------------------------------- tok_init modes
    print("\n===== tok_init modes: cls / cls_noise / normal =====")
    # cls: prompt_tokens are an EXACT copy of the source cls_token (zero noise)
    cfg_c, enc_c = build_encoder(
        "dinov3_lora",
        ["+model.encoder.tok_init=cls"],
        local_ckpt=True,
    )
    raw_cls = enc_c.model.cls_token.detach()  # (1, 1, 384)
    assert torch.allclose(enc_c.prompt_tokens[0], raw_cls[0].expand(2, 384), atol=1e-4), "cls: == raw CLS"
    print(f"  cls     : max|prompt_tokens - raw CLS| = "
          f"{(enc_c.prompt_tokens[0] - raw_cls[0].expand(2, 384)).abs().max().item():.2e}  (==0)")

    # cls_noise with an explicit init_std: copy + N(0, init_std) per token
    cfg_n, enc_n = build_encoder(
        "dinov3_lora",
        ["+model.encoder.tok_init=cls_noise", "+model.encoder.init_std=0.5"],
        local_ckpt=True,
    )
    diff = (enc_n.prompt_tokens[0] - raw_cls[0].expand(2, 384)).abs().max().item()
    assert diff > 1e-4, "cls_noise: tokens must differ from raw CLS (noise added at init)"
    print(f"  cls_n   : max|prompt_tokens - raw CLS| = {diff:.2e}  (nonzero, init_std=0.5)")

    # normal: not equal to the CLS copy (and not all identical to cls)
    cfg_m, enc_m = build_encoder(
        "dinov3_lora",
        ["+model.encoder.tok_init=normal"],
        local_ckpt=True,
    )
    assert not torch.allclose(enc_m.prompt_tokens[0], raw_cls[0].expand(2, 384), atol=1e-3), "normal: != raw CLS"
    print(f"  normal  : prompt_tokens differ from raw CLS  (max diff = "
          f"{(enc_m.prompt_tokens[0] - raw_cls[0].expand(2, 384)).abs().max().item():.2e})")

    # --------------------------------------------- n_dyn_tokens compat alias
    print("\n===== n_dyn_tokens compat alias (legacy configs) =====")
    # n_dyn_tokens=0  (as dinov3_frozen.yaml) => 1 input + 0 target => prompt(1,1,384)
    cfg_z, enc_z = build_encoder(
        "dinov3_lora",
        ["+model.encoder.n_dyn_tokens=0"],
        local_ckpt=True,
    )
    assert enc_z.n_in_tokens == 1 and enc_z.n_tgt_tokens == 0
    assert enc_z.prompt_tokens.shape == (1, 1, 384), enc_z.prompt_tokens.shape
    out_z = enc_z(torch.randn(2, 3, cfg_z.img_size, cfg_z.img_size), interpolate_pos_encoding=True)
    assert out_z.tok_in.shape == (2, 384)
    assert out_z.pooled.shape == (2, 384)
    assert torch.allclose(out_z.pooled, out_z.tok_in, atol=1e-5)
    print(f"  n_dyn=0 : prompt_tokens={tuple(enc_z.prompt_tokens.shape)}  tok_in={tuple(out_z.tok_in.shape)}")

    # n_dyn_tokens=4  => 1 input + 3 target => prompt(1,4,384)
    cfg_k, enc_k = build_encoder(
        "dinov3_lora",
        ["+model.encoder.n_dyn_tokens=4"],
        local_ckpt=True,
    )
    assert enc_k.n_in_tokens == 1 and enc_k.n_tgt_tokens == 3
    assert enc_k.prompt_tokens.shape == (1, 4, 384), enc_k.prompt_tokens.shape
    print(f"  n_dyn=4 : prompt_tokens={tuple(enc_k.prompt_tokens.shape)}  (n_in=1, n_tgt=3)")

    print("\n===== ALL SMOKE CHECKS PASSED =====")


if __name__ == "__main__":
    main()
