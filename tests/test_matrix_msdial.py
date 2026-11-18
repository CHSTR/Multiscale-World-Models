"""Test del generador de matriz E1 (Task 5 del plan).

Correr: ./.venv_hf/bin/python -m pytest tests/test_matrix_msdial.py -q
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_matrix_counts_and_names():
    from experiments.matrix_msdial import ARMS, DATASETS_E1, emit

    assert ARMS == ["lewm", "lewm_s8", "lewm_rgbrep", "starlet", "swt",
                    "wave_vit", "convnext_ms"]
    assert len(DATASETS_E1) == 4  # k1, k2, k3 balanced + k2 wedge
    cmds = emit(arms=ARMS, datasets=DATASETS_E1, seeds=[0, 1, 2],
                stages=["train", "eval"])
    train = [c for c in cmds if "train.py" in c]
    ev = [c for c in cmds if "eval_reward.py" in c]
    assert len(train) == 7 * 4 * 3      # 84 train
    assert len(ev) == len(train) * 3   # 3 modos por run
    assert all("output_model_name=msdial_k" in c and "__s" in c for c in train)
    assert all("trainer.max_epochs=2" in c for c in train)
    assert all("--config-name=msdial" in c for c in ev)
    assert all("world.levels=" in c and "world.dial_style=" in c for c in ev)
    # el piloto (k1 lewm s0) ya está entrenado: si el checkpoint existe,
    # el comando se marca para saltarse
    assert any("skip_if_exists" in c or "msdial_k1_balanced__lewm__s0" in c
               for c in train)


def test_cli_emit(tmp_path):
    out = tmp_path / "run_E1.sh"
    r = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "matrix_msdial.py"),
         "--emit", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1000:]
    text = out.read_text()
    assert text.count("train.py") == 84
    assert "#!/bin/bash" in text
