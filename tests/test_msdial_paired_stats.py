"""Tests de la estadística pareada (Task 6 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_paired_stats.py -q
"""
import numpy as np
import pytest


def test_paired_stats_known_counts():
    """b=9 (solo gana A), c=0 (solo gana B): Δpp=+9 y McNemar exacto p<0.05."""
    from swm_msdial.paired_stats import paired

    rng = np.random.default_rng(0)
    n = 100
    a = np.ones(n, bool)
    b = np.ones(n, bool)
    a[91:] = False            # A pierde 9
    # B pierde los mismos 9 MÁS otros 9 (solo gana B en ninguno => c=0)
    b[82:] = False
    res = paired(a, b, rng=rng)
    assert res["delta_pp"] == pytest.approx(9.0)
    assert res["b_only_a"] == 9 and res["c_only_b"] == 0
    # A gana 9 episodios que B pierde (b=9), B nunca gana solo (c=0)
    b_only_a = int((a & ~b).sum())
    c_only_b = int((~a & b).sum())
    assert b_only_a == 9 and c_only_b == 0
    assert res["mcnemar_p"] < 0.05


def test_paired_stats_identical():
    from swm_msdial.paired_stats import paired

    rng = np.random.default_rng(1)
    a = rng.random(100) > 0.5
    res = paired(a, a.copy(), rng=rng)
    assert res["delta_pp"] == 0.0
    assert res["b_only_a"] == 0 and res["c_only_b"] == 0
    assert res["mcnemar_p"] == 1.0
    # el IC bootstrap contiene al 0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_holm_correction():
    from swm_msdial.paired_stats import holm

    p = [0.01, 0.04, 0.03]
    q = holm(p)
    # ordenados: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04, con running max
    assert q[0] == 0.03 and q[1] == 0.06 and q[2] == 0.06
    assert all(0 <= x <= 1 for x in q)
