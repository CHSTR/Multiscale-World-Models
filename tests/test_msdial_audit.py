"""Test de la auditoría espectral de msdial (Task 3 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_audit.py -q
"""
import numpy as np
import pytest


def test_audit_balanced_coarse_fraction():
    from swm_msdial.audit_band_energy import audit

    # dial fino (k=3), balanced: la fuga a bandas gruesas debe ser
    # despreciable (spec §1.1: 0.000 en 224; en 96 px verificamos < 0.02)
    f = audit(style="balanced", k=3, img_size=96)
    assert f.shape == (6,)
    assert abs(f.sum() - 1.0) < 1e-6
    assert f[4] + f[5] < 0.02, f
    # y el wedge (con momento dipolar) filtra más a lo grueso que el balanced
    f_w = audit(style="wedge", k=1, img_size=96)
    f_b = audit(style="balanced", k=1, img_size=96)
    assert f_w[4] + f_w[5] > f_b[4] + f_b[5]
