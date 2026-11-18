"""Auditoría espectral de MultiScaleDial (port de la referencia, spec §1.1).

Fracción de la energía dependiente de θ_k que cae en cada banda
(descomposición tipo Laplaciana con gaussianas σ = 1, 2, 4, 8, 16 px):

  B0 = I - G1 | B1 = G1 - G2 | B2 = G2 - G4 | B3 = G4 - G8 | B4 = G8 - G16 | B5 = G16

E_j = Σ_px Var_θ[B_j(I_θ)], promediado sobre offsets subpíxel. Sin ruido de
sensor. Lo que importa para un ViT /16 es cuánta señal de θ queda en B4+B5
(escalas > medio patch).

Uso:
  ./.venv_hf/bin/python swm_msdial/audit_band_energy.py --out reports/msdial_band_audit.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm_msdial.env import MultiScaleDialEnv  # noqa: E402

SIG = [1, 2, 4, 8, 16]
NAMES = ["B0(<1)", "B1(1-2)", "B2(2-4)", "B3(4-8)", "B4(8-16)", "B5(>16)"]


def bands(img):
    G = [img] + [gaussian_filter(img, s) for s in SIG]
    return [G[i] - G[i + 1] for i in range(len(SIG))] + [G[-1]]


def audit(style: str, k: int, img_size: int = 224, n_offsets: int = 4,
          n_theta: int = 48) -> np.ndarray:
    """Fracciones de energía por banda (B0..B5) del dial de nivel `k`."""
    rng = np.random.default_rng(0)
    env = MultiScaleDialEnv(levels=(k,), sensor_noise=0.0, dial_style=style,
                            img_size=img_size)
    env.reset(seed=0)
    E = np.zeros(len(NAMES))
    for _ in range(n_offsets):
        off = rng.uniform(0, 1 / env.H, 2)
        stack = []
        for th in np.linspace(-np.pi, np.pi, n_theta, endpoint=False):
            env.set_state([0.5 + off[0], 0.5 + off[1], th])
            stack.append(bands(env.render()[..., 0].astype(np.float64) / 255))
        stack = np.array(stack)  # (θ, banda, H, W)
        E += stack.var(axis=0).sum(axis=(1, 2))
    return E / E.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/msdial_band_audit.md")
    args = ap.parse_args()

    lines = [
        "# Auditoría espectral MultiScaleDial (spec §1.1)",
        "",
        f"Bandas: {' | '.join(NAMES)}; `coarse(B4+B5)` = energía de θ en escalas > medio patch.",
        "",
        "| estilo | k | " + " | ".join(NAMES) + " | coarse(B4+B5) |",
        "|---|---|" + "---|" * (len(NAMES) + 1),
    ]
    for style in ("wedge", "balanced"):
        for k in (1, 2, 3, 4):
            f = audit(style=style, k=k)
            lines.append(
                f"| {style} | {k} | " + " | ".join(f"{v:.4f}" for v in f)
                + f" | **{f[4] + f[5]:.4f}** |")
            print(f"{style:>9} k={k} coarse(B4+B5)={f[4] + f[5]:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
