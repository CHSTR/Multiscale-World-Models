"""Generador de la matriz E1 de MultiScaleDial (spec §7.6, Task 5).

Armas del track desde cero × datasets {k1,k2,k3 balanced, k2 wedge} × 3 seeds
= 84 runs de entrenamiento (2 epochs) + 3 evals (modos coarse/fine/both) por
run. El run del piloto (`msdial_k1_balanced__lewm__s0`) comparte nombre con
la matriz: si el checkpoint ya existe, el script emitido lo salta.

Uso:
  ./.venv_hf/bin/python experiments/matrix_msdial.py --emit run_E1.sh
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PY = "/home/chr/le-wm/.venv/bin/python"
CKPT_DIR = "~/.stable_worldmodel/checkpoints"

ARMS = [
    "lewm",          # baseline ViT-S/14
    "lewm_s8",       # control de resolución (ViT-S/8, "resolución barata")
    "lewm_rgbrep",   # control de canales (mismos canales que starlet, sin escala)
    "starlet",       # multiescala wavelet
    "swt",           # multiescala con orientaciones
    "wave_vit",      # DWT dentro de la atención
    "convnext_ms",   # multiescala jerárquica nativa (sin wavelets)
]

DATASETS_E1 = [
    ("k1", "balanced"),
    ("k2", "balanced"),
    ("k3", "balanced"),
    ("k2", "wedge"),
]

MODES = ("coarse", "fine", "both")


def _levels(k: str) -> str:
    return f"'[{k[1]}]'"


def ckpt_path(run: str) -> str:
    return f"{CKPT_DIR}/{run}/weights_epoch_2.pt"


def emit(arms=ARMS, datasets=DATASETS_E1, seeds=(0, 1, 2),
         stages=("train", "eval")) -> list[str]:
    cmds = []
    for (k, style) in datasets:
        for arm in arms:
            for seed in seeds:
                run = f"msdial_{k}_{style}__{arm}__s{seed}"
                ckpt = ckpt_path(run)
                if "train" in stages:
                    cmds.append(
                        f"{PY} train.py data=msdial model={arm} "
                        f"data.dataset.name=msdial/msdial_train_{k}_{style}.h5 "
                        f"trainer.max_epochs=2 output_model_name={run} "
                        f"seed={seed}"
                    )
                if "eval" in stages:
                    for mode in MODES:
                        cmds.append(
                            f"{PY} viz/eval_reward.py policy={ckpt} "
                            f"eval.dataset_name=msdial/msdial_eval_{k}_{style}_{mode}.h5 "
                            f"world.levels={_levels(k)} world.dial_style={style} "
                            f"--config-name=msdial"
                        )
    return cmds


def to_bash(cmds: list[str]) -> str:
    lines = ["#!/bin/bash", "# Matriz E1 MultiScaleDial (generado por matrix_msdial.py)", "set -e"]
    for c in cmds:
        if "train.py" in c:
            run = c.split("output_model_name=")[1].split()[0]
            lines.append(f'CKPT={ckpt_path(run)}')
            lines.append('if [ ! -f "$CKPT" ]; then')
            lines.append(f"  {c}")
            lines.append("fi")
        else:
            lines.append(c)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default="run_E1.sh")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    cmds = emit(arms=a.arms, seeds=tuple(a.seeds))
    out = Path(a.emit)
    out.write_text(to_bash(cmds))
    n_train = sum(1 for c in cmds if "train.py" in c)
    n_eval = len(cmds) - n_train
    print(f"{n_train} train + {n_eval} eval -> {out}")


if __name__ == "__main__":
    main()
