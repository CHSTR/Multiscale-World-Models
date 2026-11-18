"""Tests del generador de sets de eval fijos de msdial (Task 4 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_eval_set.py -q
"""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin  # registra el filtro blosc para LEER pixels comprimidos
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TMP = Path("/tmp/opencode/msdial_t4")

GOAL_OFFSET = 25


def _gen():
    TMP.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "swm_msdial" / "gen_eval_set.py"),
        "--levels", "2", "--style", "balanced",
        "--episodes", "4", "--seed", "0", "--out-dir", str(TMP),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.fixture(scope="module", autouse=True)
def generated():
    _gen()
    yield


def _meta():
    return json.loads((TMP / "eval_meta.json").read_text())


def _load(mode):
    f = TMP / f"msdial_eval_k2_balanced_{mode}.h5"
    assert f.exists()
    return f


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def test_eval_rows_count():
    for mode in ("coarse", "fine", "both"):
        with h5py.File(_load(mode)) as f:
            n_eps = f["ep_len"].shape[0]
            assert f["pixels"].shape == (n_eps * (GOAL_OFFSET + 1), 224, 224, 3)
            assert np.all(f["ep_len"][()] == GOAL_OFFSET + 1)


def test_eval_rows_start_goal():
    meta = _meta()
    for mode in ("coarse", "fine", "both"):
        with h5py.File(_load(mode)) as f:
            st = f["raw_state"][()].reshape(-1, GOAL_OFFSET + 1, 3)
            eps = meta["sets"][f"msdial_eval_k2_balanced_{mode}.h5"]
            for i, ep in enumerate(eps):
                assert np.allclose(st[i, 0], ep["start"], atol=1e-9)
                assert np.allclose(st[i, -1], ep["goal"], atol=1e-9)


def test_eval_mode_semantics():
    meta = _meta()
    for mode in ("coarse", "fine", "both"):
        eps = meta["sets"][f"msdial_eval_k2_balanced_{mode}.h5"]
        for ep in eps:
            s, g = np.array(ep["start"]), np.array(ep["goal"])
            dpos = np.linalg.norm(g[:2] - s[:2])
            dth = np.abs(wrap(g[2] - s[2]))
            if mode == "coarse":
                assert dpos > 0.3 and dth < 1e-9
            elif mode == "fine":
                assert dpos < 1e-9 and np.pi / 2 <= dth <= np.pi
            else:
                assert dpos > 0.3 and np.pi / 2 <= dth <= np.pi


def test_meta_merge_across_levels(tmp_path):
    """Generar otro nivel en el mismo dir no debe borrar las entradas previas."""
    import swm_msdial.gen_eval_set as g

    g.gen_eval_set((2,), "balanced", 2, seed=0, out_dir=tmp_path, modes=("fine",))
    g.gen_eval_set((3,), "balanced", 2, seed=0, out_dir=tmp_path, modes=("fine",))
    meta = json.loads((tmp_path / "eval_meta.json").read_text())
    assert "msdial_eval_k2_balanced_fine.h5" in meta["sets"]
    assert "msdial_eval_k3_balanced_fine.h5" in meta["sets"]


def test_eval_goal_pixels_render():
    # los pixels de la fila 25 se renderizaron EN el estado goal
    with h5py.File(_load("fine")) as f_fine, h5py.File(_load("coarse")) as f_co:
        for f in (f_fine, f_co):
            px = f["pixels"][()].reshape(-1, GOAL_OFFSET + 1, 224, 224, 3)
            # filler estático: filas 1..24 iguales a la 0 (no contaminan extremos)
            assert np.array_equal(px[0, 1], px[0, 0]) and np.array_equal(px[0, 24], px[0, 0])
            # el goal difiere del inicio (el render realmente cambió)
            assert not np.array_equal(px[0, -1], px[0, 0])
