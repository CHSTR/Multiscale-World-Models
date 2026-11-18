"""SPIKE (throwaway) — prototipo visual del mundo MSWorld.

NO es el generador final: solo produce el grid de ejemplo para decidir si el
mundo "se ve" multiescala antes de implementar el pipeline completo
(diseño: docs/superpowers/specs/2026-11-18-msworld-dataset-design.md).

Produce outputs/msworld_spike/msworld_spike.png con:
  (A) episodio `both` en 4 pasos (agente marcado con círculo blanco)
  (B) los 4 mundos lado a lado (both / coarse-only / fine-only / cluttered)
  (C) pirámide lifting de un frame `both` (LL + |detalle| por nivel)
  (D) energía de movimiento por banda: agente solo vs bloque solo
      (preview de la puerta de validez del diseño)

Uso: /home/chr/le-wm/.venv/bin/python data_gen/spike_msworld.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from wavelet.lifting import LiftingScheme

S, SC = 224, 2          # resolución final y supermuestreo
EPS = 92                # pasos por episodio (paridad con el diseño)

# paleta
BG = (240, 240, 235)
WALL = (60, 60, 60)
AGENT = (200, 30, 30)
BLOCK = (30, 60, 200)
CLUT = (185, 185, 178)

# ------------------------------------------------------------------ mundo
def make_walls():
    """3 salas (izquierda + derecha-arriba + derecha-abajo), paredes 12 px."""
    return [
        (144, 0, 156, 96),      # pared vertical, hueco puerta y:96..128
        (144, 128, 156, 224),
        (144, 100, 170, 112),   # pared horizontal der., hueco puerta x:170..202
        (202, 100, 224, 112),
    ]

ROOMS = {"izq": (72, 112), "der_sup": (184, 56), "der_inf": (184, 168)}
DOORS = {"v": (150, 112), "h": (186, 106)}
BLOCK_BOUNDS = (32, 32, 144 - 32, 224 - 32)   # el bloque vive en la sala izq.

def make_clutter(rng, n=12):
    out = []
    for _ in range(n):
        w = rng.uniform(16, 64)
        x, y = rng.uniform(0, S - 8), rng.uniform(0, S - 8)
        out.append((x, y, w, w * rng.uniform(0.5, 1.2)))
    return out

def in_wall(x, y, walls, margin=4):
    return any(x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin
               for (x0, y0, x1, y1) in walls)

# ------------------------------------------------------------------ render
def render(state, walls, clutter, agent_size, draw_block=True):
    img = Image.new("RGB", (S * SC, S * SC), BG)
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in walls:
        d.rectangle([x0 * SC, y0 * SC, x1 * SC, y1 * SC], fill=WALL)
    for (x, y, w, h) in clutter:
        d.ellipse([x * SC, y * SC, (x + w) * SC, (y + h) * SC], fill=CLUT)
    if draw_block and state["block"] is not None:
        # Bordes BLANDOS (blur de la máscara): un bloque con bordes nítidos
        # se mueve "en sus bordes" (frecuencia alta) aunque sea enorme; con
        # bordes blandos su dinámica vive en banda gruesa, como exige el diseño.
        bx, by, bs = state["block"]
        mask = Image.new("L", (S * SC, S * SC), 0)
        ImageDraw.Draw(mask).rectangle(
            [(bx - bs / 2) * SC, (by - bs / 2) * SC,
             (bx + bs / 2) * SC, (by + bs / 2) * SC], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(14))
        img.paste(Image.new("RGB", mask.size, BLOCK), (0, 0), mask)
    ax, ay, ath = state["agent"]
    a = agent_size
    tip = np.array([ax + a * np.cos(ath), ay + a * np.sin(ath)])
    left = np.array([ax + a * np.cos(ath + 2.5), ay + a * np.sin(ath + 2.5)])
    right = np.array([ax + a * np.cos(ath - 2.5), ay + a * np.sin(ath - 2.5)])
    d.polygon([tuple(tip * SC), tuple(left * SC), tuple(right * SC)], fill=AGENT)
    return np.asarray(img.resize((S, S), Image.LANCZOS), dtype=np.uint8)

# ------------------------------------------------------------------ sim
def simulate(variant, rng, n_steps=EPS, freeze_agent=False, freeze_block=False):
    walls = [] if variant == "fine-only" else make_walls()
    clutter = make_clutter(rng) if variant == "cluttered" else []
    agent_size = 48 if variant == "coarse-only" else 6
    has_block = variant != "fine-only"

    def sample_wp():
        if walls:
            pts = list(ROOMS.values()) + list(DOORS.values())
        else:
            pts = [(rng.uniform(30, S - 30), rng.uniform(30, S - 30))]
        x, y = pts[rng.integers(len(pts))]
        return (np.clip(x + rng.normal(0, 15), 8, S - 8),
                np.clip(y + rng.normal(0, 15), 8, S - 8))

    agent = np.array([40.0, 60.0])
    heading = 0.0
    wp = sample_wp()
    bvel = np.array([1.6, 1.2])
    block = np.array([80.0, 170.0]) if has_block else None
    frames, apos, bpos = [], [], []
    for t in range(n_steps):
        if not freeze_agent:
            v = wp - agent
            dist = np.linalg.norm(v)
            if dist < 5 or t % 18 == 0:
                wp = sample_wp()
            step = v / max(dist, 1e-6) * 4.0
            cand = agent + step
            if not (8 < cand[0] < S - 8 and 8 < cand[1] < S - 8) or in_wall(*cand, walls):
                wp = sample_wp()          # spike: sin pathfinding, re-elige
            else:
                agent = cand
                heading = np.arctan2(step[1], step[0])
        if block is not None and not freeze_block:
            block = block + bvel
            x0, y0, x1, y1 = BLOCK_BOUNDS
            if not x0 < block[0] < x1: bvel[0] *= -1; block[0] += bvel[0]
            if not y0 < block[1] < y1: bvel[1] *= -1; block[1] += bvel[1]
        state = {"agent": (agent[0], agent[1], heading),
                 "block": (block[0], block[1], 64.0) if block is not None else None}
        frames.append(render(state, walls, clutter, agent_size, draw_block=has_block))
        apos.append(agent.copy()); bpos.append(block.copy() if block is not None else None)
    return frames, apos, bpos, walls

# ------------------------------------------------------------------ paneles
def fig_show(ax, img, title, marker=None):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if marker is not None:
        x, y = marker
        ax.add_patch(plt.Circle((x, y), 12, color="white", fill=False, lw=1.2))

def band_motion_energy(frames):
    """Energía por banda del Δ temporal (LiftingScheme, L=3)."""
    lift = LiftingScheme(levels=3, in_channels=3)
    x = torch.tensor(np.stack(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    diffs = (x[1:] - x[:-1]).unsqueeze(0)          # (1, T-1, 3, 224, 224)
    b, t = diffs.shape[:2]
    flat = diffs.reshape(b * t, *diffs.shape[2:])
    p = lift(flat)
    # Haar ortonormal => Parseval: usar SUMAS por banda (las medias castigan
    # injustamente a las bandas finas por su mayor soporte espacial).
    E = [float((p.details[i] ** 2).sum()) for i in range(3)]        # fino→grueso
    E.append(float((p.approx ** 2).sum()))                         # LL (gruesa)
    tot = sum(E)
    return np.array(E[::-1]) / tot   # orden: approx, L3, L2, L1(fina)

def main():
    out = "outputs/msworld_spike"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    # (A) episodio `both`
    frames, apos, bpos, _ = simulate("both", rng)
    ts = [0, 30, 61, 91]

    # (B) los 4 mundos en t=45
    variants = ["both", "coarse-only", "fine-only", "cluttered"]
    vframes = {}
    for v in variants:
        vf, _, _, _ = simulate(v, np.random.default_rng(1))
        vframes[v] = vf[45]

    # (C) pirámide lifting de un frame both
    img_t = frames[61] / 255.0
    lift = LiftingScheme(levels=3, in_channels=3)
    p = lift(torch.tensor(img_t).permute(2, 0, 1).unsqueeze(0))
    det = [d[0].abs().mean(0).numpy() for d in p.details]           # fino→grueso
    ll = p.approx[0].mean(0).numpy()

    # (D) energía de movimiento por banda
    fa, _, _, _ = simulate("both", np.random.default_rng(2), n_steps=30,
                           freeze_block=True)                        # solo agente
    fb, _, _, _ = simulate("both", np.random.default_rng(3), n_steps=30,
                           freeze_agent=True)                        # solo bloque
    ea, eb = band_motion_energy(fa), band_motion_energy(fb)

    # ---------------------------------------------------------------- fig
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.15)

    rowA = gs[0, :].subgridspec(1, 4)
    for i, t in enumerate(ts):
        ax = fig.add_subplot(rowA[0, i])
        fig_show(ax, frames[t], f"(A) both · t={t}", marker=apos[t])

    rowB = gs[1, :].subgridspec(1, 4)
    for i, v in enumerate(variants):
        ax = fig.add_subplot(rowB[0, i])
        fig_show(ax, vframes[v], f"(B) {v} · t=45")

    ax = fig.add_subplot(gs[2, 0]); fig_show(ax, frames[61], "(C) original t=61", marker=apos[61])
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(ll, cmap="gray"); ax.set_title("LL approx (gruesa, 28px)", fontsize=9); ax.axis("off")
    for j, (lvl, name) in enumerate([(2, "detalle L3 (grueso)"), (1, "detalle L2 (medio)"),
                                     (0, "detalle L1 (fino)")]):
        ax = fig.add_subplot(gs[2, 2 + j])
        ax.imshow(det[lvl], cmap="magma"); ax.set_title(f"|{name}|", fontsize=9); ax.axis("off")

    ax = fig.add_subplot(gs[3, :])
    labels = ["approx/LL\n(muy gruesa)", "detalle L3\n(gruesa)", "detalle L2\n(media)", "detalle L1\n(fina)"]
    xpos = np.arange(4)
    ax.bar(xpos - 0.2, ea, 0.38, label="solo agente se mueve (fino)", color="#c81e1e")
    ax.bar(xpos + 0.2, eb, 0.38, label="solo bloque se mueve (grueso)", color="#1e3cc8")
    ax.set_xticks(xpos, labels, fontsize=8)
    ax.set_ylabel("fracción de energía de movimiento")
    ax.set_title("(D) puerta de validez (preview): la energía del agente debe caer en bandas finas y la del bloque en gruesas", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("MSWorld — spike visual (throwaway): mundo, variantes, pirámide y energía por banda", fontsize=13)
    path = os.path.join(out, "msworld_spike.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("spike ->", path)
    print("energía agente (approx,L3,L2,L1):", np.round(ea, 3))
    print("energía bloque (approx,L3,L2,L1):", np.round(eb, 3))

if __name__ == "__main__":
    main()
