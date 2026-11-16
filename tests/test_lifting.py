import pytest
import torch

from wavelet.lifting import LiftingScheme, Pyramid


def test_shapes():
    m = LiftingScheme(levels=3, in_channels=3)
    p = m.forward(torch.randn(2, 3, 224, 224))
    assert p.approx.shape == (2, 3, 28, 28)
    assert len(p.details) == 3
    assert p.details[0].shape == (2, 9, 112, 112)
    assert p.details[2].shape == (2, 9, 28, 28)


def test_reconstruction_fp32():
    m = LiftingScheme(levels=3, learnable=True)
    x = torch.randn(2, 3, 64, 64)
    assert (m.inverse(m.forward(x)) - x).abs().max().item() < 1e-4


def test_reconstruction_bf16():
    m = LiftingScheme(levels=3, learnable=True)
    x = torch.randn(2, 3, 64, 64, dtype=torch.bfloat16)
    assert (m.inverse(m.forward(x)) - x).abs().max().item() < 5e-2


def test_haar_init_matches_reference():
    m = LiftingScheme(levels=1, init="haar", learnable=False)
    x = torch.randn(1, 3, 32, 32)
    p = m.forward(x)
    from module import WaveletAttention

    ref = WaveletAttention(dim=3, num_heads=1)
    LL, LH, HL, HH = ref._dwt(x)
    assert torch.allclose(p.approx, LL, atol=1e-5)


def test_learnable_grads():
    m = LiftingScheme(levels=2, learnable=True)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    m.inverse(m.forward(x)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_non_divisible_raises():
    m = LiftingScheme(levels=3)
    with pytest.raises(AssertionError):
        m.forward(torch.randn(1, 3, 30, 30))


def test_reconstruction_random_filters():
    """Invertibilidad estructural: válida aunque los filtros no sean los del init."""
    m = LiftingScheme(levels=2, learnable=True)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    x = torch.randn(2, 3, 32, 32)
    err = (m.inverse(m.forward(x)) - x).abs().max().item()
    assert err < 1e-4


def test_structural_invertibility_float64():
    """Con filtros aleatorios, en float64 el error solo refleja condicionamiento."""
    m = LiftingScheme(levels=3, learnable=True)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p))
    x = torch.randn(1, 3, 64, 64, dtype=torch.float64)
    assert (m.inverse(m.forward(x)) - x).abs().max().item() < 1e-8


def test_cdf53_roundtrip_and_init():
    m = LiftingScheme(levels=2, init="cdf53", learnable=False)
    x = torch.randn(1, 3, 32, 32)
    assert (m.inverse(m.forward(x)) - x).abs().max().item() < 1e-4
