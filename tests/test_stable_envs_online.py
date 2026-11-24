"""Tests for lightweight stable-envs online rendering, eval_snv presets, and Colab portability."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def test_eval_snv_no_hardcoded_local_paths():
    for rel in (
        "eval_snv.py",
        "config/train/data/pusht_online.yaml",
        "config/train/data/tworoom_online.yaml",
        "scripts/make_states_only_h5.py",
        "scripts/run_ab_slow_svreg.sh",
        "notebooks/colab_slow_svreg_ab.ipynb",
    ):
        txt = (ROOT / rel).read_text(encoding="utf-8")
        assert "/home/chr/" not in txt, f"Found hardcoded local path in {rel}"


def test_eval_snv_presets_and_parser():
    import eval_snv

    assert set(eval_snv.PRESETS.keys()) >= {"pusht", "tworoom"}
    for name in ("pusht", "tworoom"):
        env_cfg, eval_cfg = eval_snv.get_preset_configs(name)
        assert env_cfg["id"] in ("PushT", "TwoRoom")
        assert len(env_cfg["leaves"]) >= 1
        assert "observation_leaf" in eval_cfg
        assert "reset_options" in eval_cfg
        assert "keys_to_cache" in eval_cfg

    args = eval_snv.build_parser().parse_args(["--env", "tworoom", "--policy", "random", "--num-eval", "2"])
    assert args.env == "tworoom"
    assert args.policy == "random"
    assert args.num_eval == 2


def test_run_ab_launcher_dry_run_online_envs():
    for ds in ("pusht_online", "tworoom_online"):
        res = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/run_ab_slow_svreg.sh"),
                "--dataset",
                ds,
                "--model",
                "lewm",
                "--loss",
                "slow_svreg",
                "--epochs",
                "2",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"data={ds}" in res.stdout
        assert "eval_snv.py" in res.stdout
        assert "+experiment=slow_svreg" in res.stdout


def test_colab_notebook_valid_json_and_python_syntax():
    import ast

    nb = json.loads((ROOT / "notebooks/colab_slow_svreg_ab.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8
    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # Replace IPython bang commands (including backslash continuations) with `pass`
        cleaned_lines = []
        in_bang = False
        for line in src.splitlines():
            if in_bang:
                in_bang = line.rstrip().endswith("\\")
                continue
            if line.lstrip().startswith("!"):
                indent = line[: len(line) - len(line.lstrip())]
                cleaned_lines.append(f"{indent}pass")
                in_bang = line.rstrip().endswith("\\")
            else:
                cleaned_lines.append(line)
        ast.parse("\n".join(cleaned_lines), filename=f"cell_{idx}")


def test_online_render_and_collect_and_eval_snv_tworoom(tmp_path):
    pytest.importorskip("stable_envs")
    from online_render import Pixels
    from scripts.make_states_only_h5 import collect_states, strip_pixels

    h5_path = tmp_path / "tworoom_state.h5"
    info = collect_states("tworoom", h5_path, episodes=4, steps=15, seed=0, device="cpu")
    assert info["rows"] == 60
    assert h5_path.is_file()

    # Test strip_pixels helper produces an equivalent state-only file
    stripped = tmp_path / "tworoom_stripped.h5"
    sinfo = strip_pixels(h5_path, stripped, episodes=3)
    assert sinfo["rows"] == 45
    assert "proprio" in sinfo["columns"]

    # Test online_render.Pixels on TwoRoom
    px = Pixels(
        {
            "id": "TwoRoom",
            "kwargs": {},
            "leaves": [{"leaf": "agent/position", "columns": [0, 2]}],
            "zero": [],
        },
        size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    out = px(torch.full((2, 3, 2), 100.0))
    assert out.shape == (2, 3, 3, 224, 224)
    assert torch.isfinite(out).all()
    px.close()

    # End-to-end eval_snv.py smoke with random policy on TwoRoom
    out_jsonl = tmp_path / "results.jsonl"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval_snv.py"),
            "--env",
            "tworoom",
            "--data",
            str(h5_path),
            "--policy",
            "random",
            "--num-eval",
            "2",
            "--goal-offset",
            "5",
            "--budget",
            "5",
            "--device",
            "cpu",
            "--env-device",
            "cpu",
            "--out",
            str(out_jsonl),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["env"] == "tworoom"
    assert len(row["successes"]) == 2
    assert row["episode_successes"] == row["successes"]
    assert 0.0 <= row["success_rate"] <= 100.0


def test_online_render_and_eval_snv_pusht(tmp_path):
    pytest.importorskip("stable_envs")
    from scripts.make_states_only_h5 import collect_states

    h5_path = tmp_path / "pusht_state.h5"
    info = collect_states("pusht", h5_path, episodes=3, steps=15, seed=0, device="cpu")
    assert info["rows"] == 45

    out_jsonl = tmp_path / "pusht_results.jsonl"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval_snv.py"),
            "--env",
            "pusht",
            "--data",
            str(h5_path),
            "--policy",
            "random",
            "--num-eval",
            "2",
            "--goal-offset",
            "5",
            "--budget",
            "5",
            "--device",
            "cpu",
            "--env-device",
            "cpu",
            "--out",
            str(out_jsonl),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["env"] == "pusht"
    assert len(row["successes"]) == 2
