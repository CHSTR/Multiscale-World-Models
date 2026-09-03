"""Smoke test for the DINOv2/v3 LoRA model configs.

For each of the 4 model yamls we compose the full train config (base: lewm.yaml,
which provides img_size / embed_dim / history_size etc.), instantiate the encoder
via hydra, and run a forward pass. Starlet configs receive RGB (3ch) and apply
starlet_conv4d internally (starlet_levels => 3*(L+1) channels).
No training happens. (action_encoder.input_dim is a placeholder filled by
train.py, so we instantiate the encoder only, as required.)
"""
import warnings
warnings.filterwarnings("ignore")

import hydra
from hydra import compose, initialize
import hydra.utils
import torch

# (config model name, input channels, expected internal channels)
CONFIGS = [
    ("dinov2_lora", 3, 3),
    ("dinov3_lora", 3, 3),
    ("dinov2_starlet_lora", 3, 12),  # starlet_levels=3 => 3*4
    ("dinov3_starlet_lora", 3, 12),
]


def build_encoder(model_name):
    with initialize(version_base=None, config_path="config/train"):
        cfg = compose(config_name="lewm", overrides=[f"model={model_name}"])
    enc_cfg = hydra.utils.instantiate(cfg.model.encoder)
    return cfg, enc_cfg


def main():
    for model_name, input_ch, internal_ch in CONFIGS:
        print(f"\n===== {model_name} (input={input_ch}ch, internal={internal_ch}ch) =====")
        cfg, enc = build_encoder(model_name)
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

        trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        total = sum(p.numel() for p in enc.parameters())
        print(f"  total params: {total:,}  trainable params: {trainable:,}")


if __name__ == "__main__":
    main()
