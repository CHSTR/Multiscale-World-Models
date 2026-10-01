"""Genera los comandos de la matriz de experimentos multiescala (ver EXPERIMENTOS.md).

No ejecuta nada por defecto: imprime (o escribe con --emit) un script bash con un comando por línea,
agrupado por etapa, para lanzarlo en Vast.ai o repartirlo entre máquinas.

Ejemplos:
    # Diagnóstico de Cube (lo más urgente): init legacy vs equivalent vs residual, 3 semillas
    python experiments/matrix.py --envs cube --encoders dinov3 --groups core --stages train eval stats

    # Ablación de bandas y probes sobre checkpoints que YA existen (sin entrenar)
    python experiments/matrix.py --envs pusht --groups scratch --stages ablate bands probe --seeds 0

    # Matriz completa con controles de capacidad y barrido de J
    python experiments/matrix.py --groups core controls jsweep scratch --stages all --emit run_all.sh

Convención de nombres: ``<env>__<variante>__s<seed>`` (= output_model_name del checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavelet.frontend import band_names  # noqa: E402

ENVS: Dict[str, Dict[str, str]] = {
    "pusht": {"data": "pusht", "eval": "pusht"},
    "tworoom": {"data": "tworoom", "eval": "tworoom"},
    "cube": {"data": "ogb", "eval": "cube"},
}


@dataclass
class Variant:
    name: str
    model: str
    overrides: List[str] = field(default_factory=list)
    kind: str = "none"          # transformada (para saber qué bandas ablacionar)
    levels: int = 0
    merge: bool = False

    def ablation_groups(self) -> List[List[str]]:
        """Grupos de bandas a ablacionar: cada nivel (orientaciones juntas en SWT), la aproximación
        y todos los detalles."""
        if self.kind == "none":
            return []
        names = band_names(self.kind, self.levels, self.merge)
        if self.kind == "swt" and not self.merge:
            groups = [[f"{o}{j}" for o in ("LH", "HL", "HH")] for j in range(1, self.levels + 1)]
            groups.append([names[-1]])
        else:
            groups = [[n] for n in names]
        groups.append(["details"])
        return groups


def dino_variants(enc: str, group: str) -> List[Variant]:
    J = 3
    if group == "core":
        return [
            Variant(f"{enc}-rgb", f"{enc}_lora"),
            Variant(f"{enc}-starlet-legacy", f"{enc}_starlet_lora", [], "starlet", J),
            Variant(f"{enc}-starlet-equiv", f"{enc}_starlet_lora",
                    ["model.encoder.patch_init=equivalent"], "starlet", J),
            Variant(f"{enc}-starlet-resid", f"{enc}_starlet_lora",
                    ["model.encoder.fusion=residual"], "starlet", J),
            Variant(f"{enc}-swt-resid", f"{enc}_swt_lora", [], "swt", J),
        ]
    if group == "controls":
        return [
            Variant(f"{enc}-rgbrep-equiv", f"{enc}_lora",
                    ["model.encoder.frontend=rgb_repeat", f"model.encoder.frontend_levels={J}",
                     "model.encoder.patch_init=equivalent"], "rgb_repeat", J),
            Variant(f"{enc}-random-equiv", f"{enc}_lora",
                    ["model.encoder.frontend=random", f"model.encoder.frontend_levels={J}",
                     "model.encoder.patch_init=equivalent"], "random", J),
            Variant(f"{enc}-swtmerged-resid", f"{enc}_swt_lora",
                    ["model.encoder.swt_merge_orientations=true"], "swt", J, merge=True),
        ]
    if group == "jsweep":
        return [Variant(f"{enc}-starlet-equiv-J{j}", f"{enc}_starlet_lora",
                        ["model.encoder.patch_init=equivalent", f"model.encoder.starlet_levels={j}"],
                        "starlet", j) for j in (1, 2, 3, 4)]
    return []


def scratch_variants(group: str) -> List[Variant]:
    if group == "scratch":
        return [
            Variant("vit", "lewm"),
            Variant("starlet3", "starlet", ["model.encoder.levels=3"], "starlet", 3),
            Variant("swt4", "swt", ["model.encoder.levels=4"], "swt", 4),
        ]
    if group == "scratch-jsweep":
        out = []
        for j in (1, 2, 3, 4, 5):
            out.append(Variant(f"starlet{j}", "starlet", [f"model.encoder.levels={j}"], "starlet", j))
            out.append(Variant(f"swt{j}", "swt", [f"model.encoder.levels={j}"], "swt", j))
        return out
    return []


ROBUST = [("blur", 1.0), ("blur", 2.0), ("noise", 0.05), ("noise", 0.1),
          ("downsample", 2), ("downsample", 4), ("brightness", 0.3), ("brightness", -0.3)]
HARD = [("long-goal", ["eval.goal_offset_steps=50", "eval.eval_budget=100"]),
        ("few-samples", ["solver.num_samples=100"])]
STAGES = ["train", "eval", "ablate", "robust", "hard", "bands", "probe", "rollout", "stats"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS))
    ap.add_argument("--encoders", nargs="+", default=["dinov3", "dinov2"], choices=["dinov3", "dinov2"])
    ap.add_argument("--groups", nargs="+", default=["core"],
                    choices=["core", "controls", "jsweep", "scratch", "scratch-jsweep"])
    ap.add_argument("--stages", nargs="+", default=["train", "eval", "stats"], choices=STAGES + ["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="semillas de entrenamiento")
    ap.add_argument("--eval-seed", type=int, default=42, help="mismo seed => mismos episodios (pareado)")
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--epoch", type=int, default=10, help="checkpoint weights_epoch_<E>.pt a evaluar")
    ap.add_argument("--ablation-modes", nargs="+", default=["noise", "zero"])
    ap.add_argument("--extra-train", nargs="*", default=[], help="overrides extra para train.py")
    ap.add_argument("--emit", help="escribe un script bash en vez de imprimir")
    args = ap.parse_args(argv)
    stages = STAGES if "all" in args.stages else args.stages

    variants: List[Variant] = []
    for g in args.groups:
        for enc in args.encoders:
            variants += dino_variants(enc, g)
        variants += scratch_variants(g)
    seen, uniq = set(), []
    for v in variants:
        if v.name not in seen:
            seen.add(v.name)
            uniq.append(v)
    variants = uniq

    out = ["#!/usr/bin/env bash", "# Generado por experiments/matrix.py: " + " ".join(sys.argv[1:]),
           "set -euo pipefail", 'CACHE="${STABLEWM_HOME:?exporta STABLEWM_HOME (raíz de checkpoints y resultados)}"', ""]

    def ckpt(run: str) -> str:
        return f"{run}/weights_epoch_{args.epoch}.pt"

    def eval_cmd(env: str, run: str, extra: List[str]) -> str:
        return " ".join([f"python viz/eval_reward.py --config-name={ENVS[env]['eval']}",
                         f"policy={ckpt(run)}", f"seed={args.eval_seed}",
                         f"eval.num_eval={args.num_eval}"] + extra)

    for stage in stages:
        out.append(f"# ===== {stage} =====")
        for env in args.envs:
            for v in variants:
                for s in args.seeds:
                    run = f"{env}__{v.name}__s{s}"
                    if stage == "train":
                        out.append(" ".join([f"python train.py data={ENVS[env]['data']} model={v.model}",
                                             f"seed={s} output_model_name={run}"] + v.overrides + args.extra_train))
                    elif stage == "eval":
                        out.append(eval_cmd(env, run, []))
                    elif stage == "ablate":
                        for grp in v.ablation_groups():
                            for mode in args.ablation_modes:
                                bands = "[" + ",".join(grp) + "]"
                                out.append(eval_cmd(env, run, [f"'ablation.bands={bands}'", f"ablation.mode={mode}"]))
                    elif stage == "robust":
                        for kind, strength in ROBUST:
                            out.append(eval_cmd(env, run, [f"perturb.kind={kind}", f"perturb.strength={strength}"]))
                    elif stage == "hard":
                        for tag, extra in HARD:
                            out.append(eval_cmd(env, run, extra + [f"output.tag={tag}"]))
                    elif stage == "bands" and v.kind != "none":
                        # seed={s}: el split de validación de train.py depende de cfg.seed
                        out.append(f"python -m diagnostics.band_importance data={ENVS[env]['data']} "
                                   f"+policy={ckpt(run)} seed={s} loader.batch_size=16 +diag.max_batches=30")
                    elif stage == "rollout":
                        out.append(f"python -m diagnostics.rollout_error data={ENVS[env]['data']} "
                                   f"+policy={ckpt(run)} seed={s} data.dataset.num_steps=13 "
                                   f"+rollout.horizon=10 +rollout.max_batches=20")
                    elif stage == "probe":
                        out.append(f"python -m diagnostics.linear_probe data={ENVS[env]['data']} "
                                   f"+policy={ckpt(run)} seed={s} +probe.max_frames=20000")
            if stage == "stats":
                by_enc: Dict[str, List[Variant]] = {}
                for v in variants:
                    by_enc.setdefault(v.name.split("-")[0] if "-" in v.name else "scratch", []).append(v)
                for enc, vs in by_enc.items():
                    ref = next((v for v in vs if v.name.endswith("-rgb") or v.name == "vit"), vs[0])
                    specs = []
                    for v in [ref] + [x for x in vs if x is not ref]:
                        pattern = (f"$CACHE/{env}__{v.name}__s*/traces_weights_epoch_{args.epoch}_clean_"
                                   f"seed{args.eval_seed}.npz")
                        specs.append(f'--run "{v.name}={pattern}"')
                    out.append(f"python -m diagnostics.paired_stats {' '.join(specs)} --ref {ref.name} "
                               f"--out reports/{env}_{enc}.md")
        out.append("")

    text = "\n".join(out)
    if args.emit:
        Path(args.emit).write_text(text + "\n")
        n = sum(1 for line in out if line.startswith("python"))
        print(f"{n} comandos -> {args.emit}")
    else:
        print(text)


if __name__ == "__main__":
    main()
