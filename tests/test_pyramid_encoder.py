import torch

from wavelet.pyramid_encoder import PyramidEncoder, PyramidOutput


def test_output_scales():
    enc = PyramidEncoder(levels=3, dims=[192] * 4, stem_blocks=1)
    out = enc.forward(torch.randn(2, 3, 224, 224))
    assert isinstance(out, PyramidOutput) and len(out.scales) == 4
    assert out.scales[0].level == 0 and out.scales[3].level == 3
    for s in out.scales:
        gh, gw = s.grid_hw
        assert s.tokens.shape == (2, gh * gw, s.tokens.shape[-1])
    assert out.scales[0].grid_hw[0] < out.scales[3].grid_hw[0]


def test_grid_sizes_stem_patch4():
    enc = PyramidEncoder(levels=3, stem_blocks=0, stem_patch=4)
    out = enc.forward(torch.randn(1, 3, 224, 224))
    assert [s.grid_hw for s in out.scales] == [(7, 7), (7, 7), (14, 14), (28, 28)]


def test_dims_default():
    enc = PyramidEncoder(levels=2, stem_blocks=1)
    out = enc.forward(torch.randn(1, 3, 224, 224))
    assert all(s.tokens.shape[-1] == 192 for s in out.scales)


def test_grad_flow():
    enc = PyramidEncoder(levels=2, learnable_lifting=True, stem_blocks=1)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = enc.forward(x)
    sum(s.tokens.sum() for s in out.scales).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
