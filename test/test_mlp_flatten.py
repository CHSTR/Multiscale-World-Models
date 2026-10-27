"""Verify module.MLP.forward flatten/restore (review direct-predictor point #2).

- flatten 2D around net + restore leading dims
- architecture (self.net) intact
- bit-identical in 2D (no-op view for contiguous input)
- correct reshape for 3D (projector_tgt case) with BatchNorm1d

Runs on CPU only.
"""
import torch
import torch.nn as nn
from module import MLP


def build():
    torch.manual_seed(0)
    return MLP(input_dim=384, output_dim=192, hidden_dim=2048,
               norm_fn=lambda d: nn.BatchNorm1d(d))


def test_bit_identical_2d():
    m = build()
    m.eval()
    x = torch.randn(7, 384)
    y_direct = m.net(x)                 # original architecture, direct
    y_new = m(x)                        # new flatten/restore path
    assert y_new.shape == (7, 192), y_new.shape
    assert torch.equal(y_direct, y_new), "2D output NOT bit-identical"
    print("test_bit_identical_2d OK (bit-identical, shape (7,192))")


def test_contiguous_and_noncontiguous_2d():
    m = build()
    m.eval()
    x = torch.randn(5, 384)
    # --- contiguous 2D: the case all real usages use => bit-identical ---
    assert torch.equal(m(x), m.net(x)), "contiguous 2D path differs"
    # --- non-contiguous 2D: the NEW path must equal the ORIGINAL architecture
    #     applied to the SAME (non-contiguous) tensor. (BatchNorm1d itself
    #     differs between contiguous/non-contiguous layouts by ~1e-7, but that is
    #     inherent to the original `net(x)` too, not to the flatten.) ---
    x_nc = x.t().contiguous().t()       # non-contiguous view of x
    assert torch.equal(m(x_nc), m.net(x_nc)), \
        "non-contiguous 2D path must equal original net(x_nc)"
    print("test_contiguous_and_noncontiguous_2d OK")


def test_3d_reshape_and_match():
    m = build()
    m.eval()
    # projector_tgt case: (BT, K, D) with BatchNorm1d inside -> must flatten.
    x = torch.randn(6, 3, 384)
    y = m(x)
    assert y.shape == (6, 3, 192), y.shape
    # must equal running net on the flattened (BT*K, D) then reshaped.
    y_flat = m.net(x.reshape(-1, 384)).reshape(6, 3, 192)
    assert torch.equal(y, y_flat), "3D reshape inconsistent"
    print("test_3d_reshape_and_match OK (shape (6,3,192), matches flattened net)")


def test_params_unchanged():
    m = build()
    n_before = sum(p.numel() for p in m.net.parameters())
    # architecture intact: only the documented Linear + BatchNorm + GELU seq.
    assert len(m.net) == 4, len(m.net)
    assert isinstance(m.net[0], nn.Linear)
    assert isinstance(m.net[1], nn.BatchNorm1d)
    assert isinstance(m.net[2], nn.GELU)
    assert isinstance(m.net[3], nn.Linear)
    n_after = sum(p.numel() for p in m.net.parameters())
    assert n_before == n_after
    print("test_params_unchanged OK (net architecture intact)")


if __name__ == "__main__":
    test_bit_identical_2d()
    test_contiguous_and_noncontiguous_2d()
    test_3d_reshape_and_match()
    test_params_unchanged()
    print("\n===== MLP FLATTEN/RESTORE CHECKS PASSED =====")
