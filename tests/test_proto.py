"""Proto JEPA frame-level — Task 1: PrototypeHead + sinkhorn (TDD RED)."""

import torch


def test_prototype_head_shapes():
    from module import PrototypeHead

    m = PrototypeHead(num_prototypes=64, dim=192, tau=0.1)
    h = torch.randn(4, 3, 192)
    logits, pi, z = m(h)
    assert logits.shape == (4, 3, 64)
    assert pi.shape == (4, 3, 64)
    assert z.shape == (4, 3, 192)
    assert torch.allclose(pi.sum(-1), torch.ones(4, 3), atol=1e-5)


def test_sinkhorn_rows_sum_to_one():
    from module import sinkhorn

    q = sinkhorn(torch.randn(8, 64) / 0.1, eps=0.05, iters=3)
    assert q.shape == (8, 64)
    assert torch.isfinite(q).all()
    assert torch.allclose(q.sum(-1), torch.ones(8), atol=1e-3)
