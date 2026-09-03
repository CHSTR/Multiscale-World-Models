"""Smoke test for the DINOv2/v3 LoRA model configs.

For each of the 4 model yamls we compose the full train config (base: lewm.yaml,
which provides img_size / embed_dim / history_size etc.), instantiate the encoder
via hydra, and run a forward pass on the encoder with the matching channel count.
No training happens. (action_encoder.input_dim is a placeholder filled by
train.py, so we instantiate the encoder only, as required.)
"""
import warnings
warnings.filterwarnings("ignore")

import hydra
from hydra import compose, initialize
import hydra.utils
import torch

# (config model name, configured in_chans)
CONFIGS = [
    ("dinov2_lora", 3),
    ("dinov3_lora", 3),
    ("dinov2_starlet_lora", 15),
    ("dinov3_starlet_lora", 15),
]


def build_encoder(model_name, in_chans):
    with initialize(version_base=None, config_path="config/train"):
        cfg = compose(config_name="lewm", overrides=[f"model={model_name}"])
    enc_cfg = hydra.utils.instantiate(cfg.model.encoder)
    # override in_chans if requested (rebuilds patch_embed expansion)
    if in_chans != enc_cfg.in_chans:
        from src.encoders.vit_lora import expand_patch_embed, freeze_except_lora_norm_patch

        expand_patch_embed(enc_cfg.model, in_chans, fill="mean")
        freeze_except_lora_norm_patch(enc_cfg.model)
    return cfg, enc_cfg


def main():
    for model_name, native_ch in CONFIGS:
        print(f"\n===== {model_name} (native in_chans={native_ch}) =====")
        cfg, enc = build_encoder(model_name, native_ch)
        print(f"encoder: {enc.__class__.__module__}.{enc.__class__.__name__} "
              f"dim={enc.embed_dim}, in_chans={enc.in_chans}")

        x = torch.randn(2, native_ch, cfg.img_size, cfg.img_size)
        out = enc(x, interpolate_pos_encoding=True)
        lhs = out.last_hidden_state
        emb = lhs[:, 0]
        print(f"  forward {native_ch}ch: last_hidden_state={tuple(lhs.shape)}  emb={tuple(emb.shape)}")

        # multicanal check: expand patch_embed to the OTHER channel count.
        # (expand only grows channels, so for the 15ch starlet configs we just
        # confirm the native 15ch forward above.)
        other_ch = 15 if native_ch == 3 else None
        if other_ch is not None:
            from src.encoders.vit_lora import expand_patch_embed, freeze_except_lora_norm_patch

            expand_patch_embed(enc.model, other_ch, fill="mean")
            freeze_except_lora_norm_patch(enc.model)
            enc.in_chans = other_ch
            x2 = torch.randn(2, other_ch, cfg.img_size, cfg.img_size)
            out2 = enc(x2, interpolate_pos_encoding=True)
            print(f"  multicanal forward {other_ch}ch (expanded): {tuple(out2.last_hidden_state.shape)} OK")

        trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        total = sum(p.numel() for p in enc.parameters())
        print(f"  total params: {total:,}  trainable params: {trainable:,}")


if __name__ == "__main__":
    main()
