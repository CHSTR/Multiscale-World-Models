"""Smoke test for the direct / asymmetric-predictor JEPA (Fase B).

Verifies exactly the two reviewer points for the `direct-predictor` work:

  a. MaskedPredictor.mask_token.shape == (1, 1, input_dim) and std ~= 0.02.
  b. k_idx == [0,0,0,1,2,3,4] for ctx=3, H=5 (K>=5).
  c. AsymJEPA.rollout(...) returns predicted_emb of shape (B, S, T, D) with a
     single vectorised predict() call (NO autoregressive loop).
  d. With tie_tgt_to_in=True, num_preds=1 and SHARED weights (projector_in
     == projector_tgt == projector of the original JEPA, same encoder /
     predictor / action_encoder), asym_forward produces the SAME pred_loss and
     sigreg_loss as lejepa_forward.
  e. predictor.num_frames >= history_size + num_preds - 1 is required so that
     rollout/predict with n_future>0 does not crash on pos_embedding.

No training, no DINO checkpoint, no external config. Everything is built
directly from the local modules (module.py, jepa.py, jepa_asym.py).
"""
import torch
import torch.nn as nn
from einops import rearrange
from types import SimpleNamespace

from module import MaskedPredictor, Embedder, MLP
from jepa import JEPA
from jepa_asym import AsymJEPA

DEV = "cpu"


# ------------------------------------------------------------------ helpers
class DummyEncoder(nn.Module):
    """Minimal encoder exposing the JEPA contract (tok_in / tok_tgt / pooled).

    pixels: (B, T, C, H, W) -> encoder -> SimpleNamespace(tok_in, tok_tgt, pooled)
    tok_in : (BT, embed_dim)
    tok_tgt: (BT, K, embed_dim)   (K = n_tgt_tokens; with tie this is 1)
    pooled == tok_in               (jepa.encode reads this)
    """

    def __init__(self, img_h=8, img_w=8, in_ch=3, embed_dim=64, n_tgt_tokens=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tgt_tokens = n_tgt_tokens
        self.proj = nn.Linear(in_ch * img_h * img_w, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        b, c, h, w = pixels.shape
        tok = self.proj(pixels.reshape(b, c * h * w))  # (BT, embed_dim)
        if self.n_tgt_tokens > 0:
            tok_tgt = tok.unsqueeze(1).expand(-1, self.n_tgt_tokens, -1).clone()
        else:
            tok_tgt = torch.zeros(tok.shape[0], 0, self.embed_dim, device=DEV)
        return SimpleNamespace(tok_in=tok, tok_tgt=tok_tgt, pooled=tok)


class Ctx(SimpleNamespace):
    """Fake spt.Module `self` for running train.lejepa_forward / asym_forward."""

    def __init__(self, model, sigreg_weight=0.09, num_proj=256):
        super().__init__()
        self.model = model
        self.sigreg = SIGReg(num_proj=num_proj)
        self.sigreg_weight = sigreg_weight
        self.log_dict = lambda *a, **k: None


def SIGReg(num_proj=256):
    from module import SIGReg as _SIGReg
    return _SIGReg(knots=17, num_proj=num_proj)


def build_model(embed_dim=64, K=1, action_dim=4, num_frames=7, depth=2,
                heads=8, dim_head=32, mlp_dim=128):
    """Shared components so that JEPA and AsymJEPA have byte-identical weights."""
    torch.manual_seed(0)
    encoder = DummyEncoder(embed_dim=embed_dim, n_tgt_tokens=K)
    predictor = MaskedPredictor(
        num_frames=num_frames, input_dim=embed_dim,
        hidden_dim=embed_dim, output_dim=embed_dim,
        depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    # single shared projector (projector_in == projector_tgt == projector)
    projector = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=mlp_dim)
    pred_proj = nn.Identity()
    return encoder, predictor, action_encoder, projector, pred_proj


def forward_config(history_size, num_preds):
    return SimpleNamespace(
        history_size=history_size,
        num_preds=num_preds,
        loss=SimpleNamespace(sigreg=SimpleNamespace(weight=0.09)),
    )


# ------------------------------------------------------------------ checks
def check_a_mask_token():
    print("== [a] MaskedPredictor.mask_token ==")
    p = MaskedPredictor(num_frames=3, input_dim=64, hidden_dim=64,
                        output_dim=64, depth=2, heads=8, dim_head=32, mlp_dim=128)
    assert p.mask_token.shape == (1, 1, 64), p.mask_token.shape
    std = p.mask_token.std().item()
    assert abs(std - 0.02) < 0.005, f"mask std {std:.4f} != 0.02"
    print(f"  mask_token.shape={tuple(p.mask_token.shape)}  std={std:.4f}  OK")


def check_b_k_idx():
    print("== [b] k_idx for ctx=3, H=5 (K>=5) ==")
    ctx, H, K = 3, 5, 5
    # exact formula from train.asym_forward (line 82-83)
    k_idx = (torch.arange(ctx + H - 1) - (ctx - 1) + 1).clamp(min=1) - 1
    k_idx = k_idx.clamp(max=K - 1)
    assert k_idx.tolist() == [0, 0, 0, 1, 2, 3, 4], k_idx.tolist()
    print(f"  k_idx={k_idx.tolist()}  OK")


def check_c_rollout_vectorised():
    print("== [c] AsymJEPA.rollout -> (B,S,T,D), no AR loop ==")
    embed_dim, K, action_dim, num_preds = 64, 1, 4, 5
    # documented minimum: num_frames >= history_size + num_preds - 1 = 7
    num_frames = 3 + num_preds - 1
    encoder, predictor, ae, proj, pred_proj = build_model(
        embed_dim=embed_dim, K=K, action_dim=action_dim,
        num_frames=num_frames,
    )
    asym = AsymJEPA(encoder=encoder, predictor=predictor, action_encoder=ae,
                    projector_in=proj, projector_tgt=proj, pred_proj=pred_proj,
                    num_preds=num_preds)

    B, S, H_ctx, T = 2, 3, 3, 5  # H_ctx<=num_frames, T<=num_frames
    pixels = torch.randn(B, S, H_ctx, 3, 8, 8)
    action_seq = torch.rand(B, S, T, action_dim)  # arbitrary planning actions

    # count how many times the (vectorised) predict() is invoked
    calls = {"n": 0}
    _orig = asym.predict

    def counting(x_ctx, act_emb, n_future=0):
        calls["n"] += 1
        return _orig(x_ctx, act_emb, n_future=n_future)

    asym.predict = counting
    info = asym.rollout({"pixels": pixels, "action_sequence": action_seq},
                        action_seq, history_size=H_ctx)
    assert calls["n"] == 1, f"rollout called predict {calls['n']} times (want 1)"
    pe = info["predicted_emb"]
    assert pe.shape == (B, S, T, embed_dim), tuple(pe.shape)
    assert torch.isfinite(pe).all()
    print(f"  predicted_emb.shape={tuple(pe.shape)}  predict_calls={calls['n']}  OK")


def check_e_num_frames():
    print("== [e] predictor.num_frames >= history_size + num_preds - 1 ==")
    embed_dim, action_dim = 64, 4
    # SUFFICIENT num_frames (documented minimum) -> rollout/predict works.
    encoder, predictor, ae, proj, pred_proj = build_model(
        embed_dim=embed_dim, action_dim=action_dim,
        num_frames=3 + 5 - 1,  # history_size + num_preds - 1
    )
    asym = AsymJEPA(encoder=encoder, predictor=predictor, action_encoder=ae,
                    projector_in=proj, projector_tgt=proj, pred_proj=pred_proj,
                    num_preds=5)
    nf = 3 + 5 - 1  # history_size + num_preds - 1 (documented minimum)
    B, S, H_ctx, T = 2, 2, 3, 5
    info = asym.rollout({"pixels": torch.randn(B, S, H_ctx, 3, 8, 8),
                         "action_sequence": torch.rand(B, S, T, action_dim)},
                        torch.rand(B, S, T, action_dim), history_size=H_ctx)
    assert info["predicted_emb"].shape == (B, S, T, embed_dim)
    print(f"  num_frames={nf} (min {nf}), rollout OK")

    # NEGATIVE: too-small num_frames (== history_size, H>=2) crashes on the
    # pos_embedding add, exactly as reported by the reviewer.
    enc2, pred2, ae2, proj2, pp2 = build_model(
        embed_dim=embed_dim, action_dim=action_dim, num_frames=3)
    asym2 = AsymJEPA(encoder=enc2, predictor=pred2, action_encoder=ae2,
                     projector_in=proj2, projector_tgt=proj2, pred_proj=pp2,
                     num_preds=5)
    crashed = False
    try:
        asym2.predict(torch.randn(2, 3, embed_dim), torch.randn(2, 5, embed_dim),
                      n_future=1)  # needs T=4 > num_frames=3
    except (RuntimeError, IndexError):
        crashed = True
    assert crashed, "expected a crash with num_frames < history_size+num_preds-1"
    print("  num_frames=3 with n_future=1 -> crashes as documented  OK")


def check_d_loss_equivalence():
    print("== [d] asym_forward == lejepa_forward (tie, num_preds=1, shared wts) ==")
    from module import SIGReg as _SIGReg
    from train import lejepa_forward, asym_forward

    embed_dim, action_dim = 64, 4
    # K=1 so that emb_tgt[:,:,0,:] == emb_in (tie_tgt_to_in => single target)
    encoder, predictor, ae, proj, pred_proj = build_model(
        embed_dim=embed_dim, K=1, action_dim=action_dim, num_frames=7)

    # JEPA (lejepa path) and AsymJEPA (asym path) SHARE every submodule.
    jepa = JEPA(encoder=encoder, predictor=predictor, action_encoder=ae,
                projector=proj, pred_proj=pred_proj)
    asym = AsymJEPA(encoder=encoder, predictor=predictor, action_encoder=ae,
                    projector_in=proj, projector_tgt=proj, pred_proj=pred_proj,
                    num_preds=1)

    # same batch for both
    B, T = 2, 4  # T = ctx + num_preds so both target slices coincide
    batch = {
        "pixels": torch.randn(B, T, 3, 8, 8),
        "action": torch.randn(B, T, action_dim),
    }

    cfg_le = forward_config(history_size=3, num_preds=1)
    cfg_as = forward_config(history_size=3, num_preds=1)
    ctx_le = Ctx(jepa)
    ctx_as = Ctx(asym)

    # same RNG state before each call -> identical sigreg random projections
    torch.manual_seed(1234)
    out_le = lejepa_forward(ctx_le, batch, "train", cfg_le)
    torch.manual_seed(1234)
    out_as = asym_forward(ctx_as, batch, "train", cfg_as)

    pl_le = out_le["pred_loss"].item()
    pl_as = out_as["pred_loss"].item()
    sl_le = out_le["sigreg_loss"].item()
    sl_as = out_as["sigreg_loss"].item()

    print(f"  pred_loss  lejepa={pl_le:.8f}  asym={pl_as:.8f}")
    print(f"  sigreg_loss lejepa={sl_le:.8f}  asym={sl_as:.8f}")
    assert torch.isfinite(out_le["pred_loss"]) and torch.isfinite(out_as["pred_loss"])
    assert torch.allclose(out_le["pred_loss"], out_as["pred_loss"], atol=1e-6), \
        "pred_loss mismatch"
    assert torch.allclose(out_le["sigreg_loss"], out_as["sigreg_loss"], atol=1e-6), \
        "sigreg_loss mismatch"
    # also the combined loss should match
    assert torch.allclose(out_le["loss"], out_as["loss"], atol=1e-6)
    print("  pred_loss and sigreg_loss identical  OK")


def main():
    check_a_mask_token()
    check_b_k_idx()
    check_c_rollout_vectorised()
    check_e_num_frames()
    check_d_loss_equivalence()
    print("\n===== ALL ASYM SMOKE CHECKS PASSED =====")


if __name__ == "__main__":
    main()
