"""Tests del colector H5 de MultiScaleDial (Task 2 del plan msdial).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_collect.py -q
Genera H5 chicos en /tmp/opencode (no toca ~/.stable_worldmodel).
"""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TMP = Path("/tmp/opencode/msdial_t2")
TRAIN = TMP / "msdial_train_k2_balanced.h5"
VAL = TMP / "msdial_val_k2_balanced.h5"

# Esquema exacto (clon de tworoom.h5) que fija el plan (Task 2, Interfaces)
# train: 4 eps × 10 steps = 40 filas (val va en archivo aparte, 2 eps = 20)
# --levels 2 = UN dial en el nivel k=2 -> K=1: action 2+1=3, state 3, obs 4
SCHEMA = {
    "pixels": ((40, 224, 224, 3), "uint8"),
    "action": ((40, 3), "float32"),
    "raw_state": ((40, 3), "float64"),                # [p_x, p_y, θ] crudo
    "observation": ((40, 4), "float64"),           # [p, sinθ, cosθ]
    "proprio": ((40, 4), "float64"),
    "ep_idx": ((40,), "int32"),
    "step_idx": ((40,), "int64"),
    "ep_len": ((4,), "int32"),
    "ep_offset": ((4,), "int64"),
    "id": ((40,), "int64"),
    "reward": ((40,), "float64"),
}


def _collect(train_eps=4, val_eps=2, seed=0):
    TMP.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "swm_msdial" / "collect.py"),
        "--levels", "2", "--style", "balanced",
        "--episodes", str(train_eps), "--val-episodes", str(val_eps),
        "--steps", "10", "--seed", str(seed),
        "--out-dir", str(TMP),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.fixture(scope="module", autouse=True)
def collected():
    _collect()
    yield


def test_schema_exacta():
    with h5py.File(TRAIN) as f:
        assert set(f.keys()) == set(SCHEMA), sorted(f.keys())
        for k, (shape, dtype) in SCHEMA.items():
            assert f[k].shape == shape, (k, f[k].shape, shape)
            assert f[k].dtype.name == dtype, (k, f[k].dtype, dtype)
            if k == "reward":
                assert np.all(f[k][()] == 0.0)


def test_episodios_contiguos():
    with h5py.File(TRAIN) as f:
        ep = f["ep_idx"][()]
        step = f["step_idx"][()]
        lens = f["ep_len"][()]
        offs = f["ep_offset"][()]
        n_eps = len(lens)
        assert sorted(set(ep.tolist())) == list(range(n_eps))
        for i in range(n_eps):
            idx = np.where(ep == i)[0]
            assert len(idx) == 10
            assert np.all(np.diff(idx) == 1)
            assert np.array_equal(step[idx], np.arange(10))
            assert lens[i] == 10 and offs[i] == idx[0]


def test_acciones_rango():
    with h5py.File(TRAIN) as f:
        a = f["action"][()]
        assert a.shape[1] == 3
        assert a.min() >= -1.0 and a.max() <= 1.0


def test_train_val_disjuntos():
    with h5py.File(TRAIN) as ft, h5py.File(VAL) as fv:
        assert ft["raw_state"].shape == (40, 3) and fv["raw_state"].shape == (20, 3)
        # episodios de val: seeds con offset 10_000_000 -> estados distintos
        st = ft["raw_state"][()]
        sv = fv["raw_state"][()]
        cen_t = st.reshape(4, 10, 3).mean(axis=1)
        cen_v = sv.reshape(2, 10, 3).mean(axis=1)
        for c in cen_v:
            assert np.min(np.linalg.norm(cen_t - c, axis=1)) > 1e-3
