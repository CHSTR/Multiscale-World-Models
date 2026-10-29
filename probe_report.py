"""Agrega los CSV de probe_train.py y genera tabla markdown + heatmap.

Uso:
  python probe_report.py outputs/probe/*.csv \
      --out-csv assets/probe_results.csv \
      --out-md  assets/probe_results.md \
      --out-fig assets/probe_heatmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["checkpoint", "source", "probe", "group"]


def aggregate(csv_paths: list[str]) -> pd.DataFrame:
    """Une varios CSV y deja el ultimo run por (checkpoint, source, probe, group)."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if "run" in df.columns:
        df = df.sort_values("run")
    df = df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    df["model"] = df["checkpoint"].str.split("/").str[0]
    df["model"] = df["model"].replace(
        {
            "dinov3_s_lora_cube": "DINOv3",
            "dinov2_lora_cube": "DINOv2",
            "starlet_L3_cube": "Starlet-3L",
        }
    )
    df["setting"] = df["model"] + " | " + df["source"] + " | " + df["probe"]
    return df


def write_markdown(df: pd.DataFrame, path: str | Path) -> None:
    cols = ["group", "model", "source", "probe", "dim", "mse_z", "mse_raw", "pearson"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "group": "Quantity",
            "model": "Model",
            "source": "Features",
            "probe": "Probe",
            "dim": "Dim",
            "mse_z": "MSE (z)",
            "mse_raw": "MSE (raw)",
            "pearson": "Pearson r",
        }
    )
    out["MSE (z)"] = out["MSE (z)"].map(lambda x: f"{x:.3f}")
    out["MSE (raw)"] = out["MSE (raw)"].map(lambda x: f"{x:.4f}")
    out["Pearson r"] = out["Pearson r"].map(lambda x: f"{x:.3f}")
    out = out.sort_values(["Quantity", "Model", "Features", "Probe"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(out.to_markdown(index=False) + "\n")


def _draw_heatmap(pivot, path, usetex: bool) -> None:
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(1.05 * len(pivot.columns) + 3.2, 0.55 * len(pivot) + 2.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace(" | ", "\n") for c in pivot.columns], fontsize=7.5, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    if usetex:
        labels = [r"\texttt{" + g.replace("_", r"\_") + "}" for g in pivot.index]
        title = r"\textbf{Pearson $r$ of physical quantities from frozen latents (Cube)}"
        cblabel = r"Pearson $r$"
    else:
        labels = [g.replace("_", " ") for g in pivot.index]
        title = "Pearson r of physical quantities from frozen latents (Cube)"
        cblabel = "Pearson r"
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cblabel)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_heatmap(df: pd.DataFrame, path: str | Path) -> None:
    pivot = df.pivot_table(index="group", columns="setting", values="pearson")
    order = [
        "joint_pos", "joint_vel", "effector_pos", "effector_yaw",
        "gripper", "block_pos", "block_quat", "block_yaw",
    ]
    pivot = pivot.reindex([g for g in order if g in pivot.index])
    try:
        _draw_heatmap(pivot, path, usetex=True)
    except RuntimeError:
        # dvipng ausente en este entorno: fallback sin LaTeX
        _draw_heatmap(pivot, path, usetex=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reporte de probes de Cube")
    p.add_argument("csv", nargs="+")
    p.add_argument("--out-csv", default="assets/probe_results.csv")
    p.add_argument("--out-md", default="assets/probe_results.md")
    p.add_argument("--out-fig", default="assets/probe_heatmap.png")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = aggregate(args.csv)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_markdown(df, args.out_md)
    write_heatmap(df, args.out_fig)
    print(f"[report] {len(df)} filas -> {args.out_csv}, {args.out_md}, {args.out_fig}")


if __name__ == "__main__":
    main()
