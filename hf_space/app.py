"""LeWM interactivo — Space de Hugging Face (ZeroGPU).

Anclas pre-cargadas de pushT (latentes ya codificados con el encoder DinoV3+LoRA).
Eliges una ancla y tus acciones (pad 2D) mueven el mundo imaginado. El encoder no
vuelve a intervenir: cada accion genera un paso latente con el predictor y se
decodifica a pixeles. El horizonte fiable son ~5 pasos.

Notas ZeroGPU:
- El modelo se coloca en CUDA a nivel de modulo (emulado en arranque, real en
  el worker GPU) y las funciones ligadas a la UI van decoradas con @spaces.GPU.
- Todo lo que cruza al worker debe ser serializable: el plot de energia se
  devuelve como imagen numpy (no como Figure de matplotlib).
"""
import json
import numpy as np
import torch
from pathlib import Path

import gradio as gr
import spaces
import stable_worldmodel as swm

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
ANCHORS = HERE / "anchors"
# En ZeroGPU torch.cuda.is_available() es True (emulado) en arranque; el
# dispositivo real se adjunta dentro de las funciones decoradas con @spaces.GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUDGET = 5          # horizonte fiable, en pasos latentes
N_CONTEXT = 3
HISTORY = 3
N_SUB = 5
A_DIM = 2
CHANCE = 2.0        # nivel de azar en el espacio latente (SIGReg)

DIRS = {
    "↖": (-1, -1), "↑": (0, -1), "↗": (1, -1),
    "←": (-1, 0),  "•": (0, 0),  "→": (1, 0),
    "↙": (-1, 1),  "↓": (0, 1),  "↘": (1, 1),
}


# --------------------------------------------------------------------------- #
#  Modelo y decoder
# --------------------------------------------------------------------------- #


def load_model():
    model = swm.wm.utils.load_pretrained(
        "pusht/weights_epoch_10.pt", cache_dir=str(WEIGHTS)
    )
    model = model.to(DEVICE).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


class CLSDecoder(torch.nn.Module):
    """Decoder debil por cross-attention de un unico token (el CLS proyectado)."""

    def __init__(self, cls_dim=384, img_size=224, patch_size=16, dim=256,
                 heads=8, depth=3):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3
        self.queries = torch.nn.Parameter(torch.zeros(1, self.num_patches, dim))
        torch.nn.init.normal_(self.queries, std=0.02)
        self.cls_proj = torch.nn.Sequential(
            torch.nn.Linear(cls_dim, dim), torch.nn.LayerNorm(dim), torch.nn.GELU()
        )
        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleDict({
                "cross_attn": torch.nn.MultiheadAttention(dim, heads, batch_first=True),
                "norm1": torch.nn.LayerNorm(dim),
                "mlp": torch.nn.Sequential(
                    torch.nn.Linear(dim, dim * 4), torch.nn.GELU(),
                    torch.nn.Linear(dim * 4, dim),
                ),
                "norm2": torch.nn.LayerNorm(dim),
            }))
        self.to_pixels = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), torch.nn.Linear(dim, patch_dim)
        )
        self.patch_size = patch_size

    def forward(self, x):
        B = x.size(0)
        P = self.num_patches
        kv = self.cls_proj(x).unsqueeze(1)
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            attn_out = layer["cross_attn"](q, kv, kv)[0]
            q = layer["norm1"](q + attn_out)
            q = layer["norm2"](q + layer["mlp"](q))
        patches = self.to_pixels(q).reshape(B, P, self.patch_size, self.patch_size, 3)
        H = W = int(self.num_patches ** 0.5)
        patches = patches.reshape(B, H, W, self.patch_size, self.patch_size, 3)
        img = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            B, 3, H * self.patch_size, W * self.patch_size
        )
        return img


def load_decoder():
    ckpt = torch.load(WEIGHTS / "checkpoints" / "pusht" / "decoder.pt", map_location="cpu")
    if ckpt.get("source") != "emb":
        raise ValueError(f"decoder source={ckpt.get('source')!r}; necesita source='emb'")
    decoder = CLSDecoder(
        cls_dim=ckpt["cls_dim"], img_size=ckpt["img_size"], patch_size=ckpt["patch_size"],
        dim=ckpt["dim"], heads=ckpt["heads"], depth=ckpt["depth"],
    )
    decoder.load_state_dict(ckpt["state_dict"])
    return decoder.to(DEVICE).eval().requires_grad_(False)


def load_action_stats():
    d = np.load(WEIGHTS / "checkpoints" / "pusht" / "action_stats.npz")
    mean = torch.from_numpy(d["mean"]).float().view(1, -1)
    std = torch.from_numpy(d["std"]).float().view(1, -1)
    return mean, std


def load_anchors():
    """Devuelve lista de dicts con la metadata y rutas de cada ancla."""
    anchors = []
    for adir in sorted(ANCHORS.glob("a*"), key=lambda p: int(p.name[1:])):
        if not adir.is_dir():
            continue
        meta = json.loads((adir / "meta.json").read_text())
        meta["dir"] = adir
        meta["label"] = f"ancla {meta['index']}  ·  fila {meta['start_row']}"
        anchors.append(meta)
    return anchors


# --------------------------------------------------------------------------- #
#  Helpers latentes (identicos a demo_app.py)
# --------------------------------------------------------------------------- #


def make_action_block(action_vec, magnitude, act_mean, act_std):
    n_sub = N_SUB
    raw = torch.tensor(list(action_vec), dtype=torch.float32) * magnitude
    raw = raw.repeat(n_sub).reshape(1, n_sub, A_DIM)
    return ((raw - act_mean) / act_std).reshape(1, n_sub * A_DIM)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_display(emb_vec):
    with torch.no_grad():
        img = CTX.decoder(emb_vec.unsqueeze(0)).float()
    img = (img * IMAGENET_STD.to(DEVICE) + IMAGENET_MEAN.to(DEVICE)).clamp(0, 1)
    return (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def metrics(emb):
    z = emb[0]
    n_pred = z.size(0) - N_CONTEXT
    energy = z[-1].pow(2).mean().item()
    straight = float("nan")
    if z.size(0) >= 3:
        v = torch.nn.functional.normalize(z[1:] - z[:-1], dim=-1)
        straight = (v[-1] * v[-2]).sum().item()
    drift = (z[-1] - z[N_CONTEXT - 1]).pow(2).mean().item()
    return {"n_pred": n_pred, "energy": energy, "straight": straight, "drift": drift}


def status_md(m):
    n, budget = m["n_pred"], BUDGET
    if n == 0:
        badge, note = "🟢 anclado", "estado real codificado por el encoder"
    elif n <= budget:
        badge, note = "🟢 fiable", f"dentro del horizonte de planificacion (H={budget})"
    elif n <= 2 * budget:
        badge, note = "🟡 degradado", "fuera del regimen entrenado; el error compone"
    else:
        badge, note = "🔴 a la deriva", "plausible pero probablemente ya no fiel"
    energy_flag = "" if 0.5 < m["energy"] < 2.0 else "  ⚠ fuera de la gaussiana"
    straight = "—" if np.isnan(m["straight"]) else f"{m['straight']:+.2f}"
    return (
        f"### {badge}\n"
        f"**{n}** pasos imaginados desde el ancla ({n * N_SUB} pasos de entorno) — {note}\n\n"
        f"| indicador | valor | esperado |\n|---|---|---|\n"
        f"| `|z|²/D` | {m['energy']:.2f}{energy_flag} | ≈ 1.0 (SIGReg) |\n"
        f"| rectitud | {straight} | alto ⇒ no responde a la accion |\n"
        f"| deriva desde el ancla | {m['drift']:.2f} | ~2.0 ⇒ nivel de azar |\n"
    )


def _advance(emb, acts, action_vec, magnitude):
    a = make_action_block(action_vec, magnitude, CTX.act_mean, CTX.act_std).to(DEVICE)
    acts = torch.cat([acts, a.unsqueeze(0)], dim=1)
    with torch.no_grad():
        act_emb = CTX.model.action_encoder(acts[:, : emb.size(1)])
        pred = CTX.model.predict(emb[:, -HISTORY:], act_emb[:, -HISTORY:])[:, -1:]
    return torch.cat([emb, pred], dim=1), acts


def _plot(history):
    """Renderiza el plot de energia a imagen numpy (serializable en ZeroGPU)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    if history:
        ax.plot(range(1, len(history) + 1), history, marker="o", color="tab:red")
    ax.axhline(1.0, ls=":", color="0.4", lw=1)
    ax.axvspan(0, BUDGET, color="tab:green", alpha=0.10)
    ax.set_xlabel("pasos imaginados")
    ax.set_ylabel("|z|²/D")
    ax.set_ylim(0, max(2.5, max(history) * 1.2 if history else 2.5))
    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# --------------------------------------------------------------------------- #
#  Callbacks (ligados a la UI -> decorados con @spaces.GPU para ZeroGPU)
# --------------------------------------------------------------------------- #


@spaces.GPU(duration=30)
def anchor(label):
    meta = CTX.anchors[[a["label"] for a in CTX.anchors].index(label)]
    # En CPU para que el retorno cruce el worker ZeroGPU (el padre no tiene GPU).
    emb = torch.load(meta["dir"] / "emb.pt", map_location="cpu")   # (1, n_ctx, D)
    acts = torch.load(meta["dir"] / "acts.pt", map_location="cpu")  # (n_ctx-1, A)
    emb_d, acts_d = emb.to(DEVICE), acts.to(DEVICE)
    m = metrics(emb_d)
    return (
        preview_for(meta),
        {"emb": emb, "acts": acts.unsqueeze(0)},
        to_display(emb_d[0, -1]),
        status_md(m),
        [],
        _plot([]),
    )


def step(state, action_vec, magnitude, history):
    if state is None:
        return state, None, "Pulsa **Anclar** primero.", history, _plot(history or [])
    emb_d = state["emb"].to(DEVICE)
    acts_d = state["acts"].to(DEVICE)
    emb_d, acts_d = _advance(emb_d, acts_d, action_vec, magnitude)
    m = metrics(emb_d)
    history = (history or []) + [m["energy"]]
    img = to_display(emb_d[0, -1])
    return (
        {"emb": emb_d.cpu(), "acts": acts_d.cpu()},
        img, status_md(m), history, _plot(history),
    )


@spaces.GPU(duration=30)
def step_pad(state, direction, magnitude, history):
    return step(state, _full_action(DIRS[direction]), magnitude, history)


@spaces.GPU(duration=15)
def undo(state, history):
    if state is None or state["emb"].size(1) <= N_CONTEXT:
        return state, gr.update(), gr.update(), history, gr.update()
    emb = state["emb"][:, :-1]
    acts = state["acts"][:, :-1]
    emb_d = emb.to(DEVICE)
    m = metrics(emb_d)
    history = (history or [])[:-1]
    return (
        {"emb": emb, "acts": acts},
        to_display(emb_d[0, -1]), status_md(m),
        history, _plot(history),
    )


def _full_action(vec):
    a = [0.0] * A_DIM
    for i, v in enumerate(vec[:A_DIM]):
        a[i] = float(v)
    return a


CYCLES = {
    "ida y vuelta (horizontal)": lambda n: ["→"] * n + ["←"] * n,
    "ida y vuelta (vertical)": lambda n: ["↓"] * n + ["↑"] * n,
    "cuadrado": lambda n: ["→"] * n + ["↓"] * n + ["←"] * n + ["↑"] * n,
    "control: solo ida": lambda n: ["→"] * (2 * n),
}


def cycle_duration(state, pattern, n, magnitude):
    return min(60, int(15 + 0.75 * int(n)))


@spaces.GPU(duration=cycle_duration)
def run_cycle(state, pattern, n, magnitude):
    if state is None:
        return None, "Pulsa **Anclar** primero."
    emb = state["emb"].to(DEVICE)
    acts = state["acts"].to(DEVICE)
    z0 = emb[0, -1].clone()
    seq = CYCLES[pattern](int(n))
    dists, frames = [], []
    keep = max(1, len(seq) // 6)
    for i, d in enumerate(seq):
        emb, acts = _advance(emb, acts, _full_action(DIRS[d]), magnitude)
        dists.append((emb[0, -1] - z0).pow(2).mean().item())
        if i % keep == 0 or i == len(seq) - 1:
            frames.append((to_display(emb[0, -1]), f"{i + 1}: {d}"))
    chance = CHANCE
    closure = dists[-1] / chance
    excursion = max(dists) / chance
    ratio = closure / max(excursion, 1e-8)
    if excursion < 0.05:
        verdict = "⚠️ el modelo apenas se movio — sube la magnitud o N."
    elif pattern.startswith("control"):
        verdict = "Control: no deberia cerrar. Si cierra, hay un atractor."
    elif ratio < 0.15:
        verdict = "✅ cierra — la dinamica es reversible con poca perdida."
    elif ratio < 0.40:
        verdict = "🟡 cierra parcialmente — con deriva acumulada."
    else:
        verdict = "🔴 no cierra — la deriva domina."
    md = (
        f"### {pattern} · N={int(n)} · {len(seq)} pasos\n"
        f"| | valor (1.0 = azar) |\n|---|---|\n"
        f"| cierre `|z_fin − z_ini|²` | **{closure:.3f}** |\n"
        f"| excursion maxima | {excursion:.3f} |\n"
        f"| cierre / excursion | {ratio:.3f} |\n\n"
        f"{verdict}"
    )
    return frames, md


# --------------------------------------------------------------------------- #
#  Carga global + UI
# --------------------------------------------------------------------------- #


class Ctx:
    model = None
    decoder = None
    anchors = []
    act_mean = None
    act_std = None


CTX = Ctx()


def preview_for(meta):
    """Devuelve los 3 frames de contexto de una ancla (para la UI)."""
    return [str(meta["dir"] / f) for f in meta["frames"]]


def build_ui():
    with gr.Blocks(title="LeWM interactivo · pusht") as demo:
        gr.Markdown(
            "# LeWM interactivo\n"
            "Elige una **ancla** de pushT (3 frames reales codificados), y a partir de "
            "ahi **tus acciones** mueven el mundo imaginado. El encoder no vuelve a "
            "intervenir. El horizonte fiable son ~5 pasos."
        )

        state = gr.State(None)
        history = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                anchor_dd = gr.Dropdown(
                    [a["label"] for a in CTX.anchors], label="ancla",
                    value=CTX.anchors[0]["label"],
                )
                ctx_gallery = gr.Gallery(
                    label="contexto (3 frames reales)", columns=3, height=220,
                    value=preview_for(CTX.anchors[0]),
                )
                btn_anchor = gr.Button("⚓ anclar", variant="primary")

            with gr.Column(scale=2):
                imag_img = gr.Image(label="estado imaginado", height=360)
                magnitude = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                      label="magnitud (≈ 1 std de accion)")
                btn_undo = gr.Button("↩ deshacer paso")
                gr.Markdown("**Pad 2D** — accion = direccion × magnitud")
                pad = []
                for row in (["↖", "↑", "↗"], ["←", "•", "→"], ["↙", "↓", "↘"]):
                    with gr.Row():
                        for d in row:
                            pad.append(gr.Button(d, scale=1))

            with gr.Column(scale=1):
                status = gr.Markdown()
                plot = gr.Image(label="energia latente")

        outs = [state, imag_img, status, history, plot]
        # Seleccionar un ancla también la activa. Antes este evento solo
        # cambiaba el preview y el estado imaginado quedaba en el ancla inicial.
        anchor_dd.change(anchor, anchor_dd, [ctx_gallery] + outs)
        btn_anchor.click(anchor, anchor_dd, [ctx_gallery] + outs)
        for b in pad:
            b.click(step_pad, [state, gr.State(b.value), magnitude, history], outs)
        btn_undo.click(undo, [state, history], outs)

        gr.Markdown("---\n## Test de ciclo\n"
                    "Ejecuta una secuencia cerrada y mide si vuelves al mismo latente.")
        with gr.Row():
            cyc_pattern = gr.Dropdown(list(CYCLES), value="ida y vuelta (horizontal)", label="patron")
            cyc_n = gr.Slider(2, 120, value=20, step=1, label="N (pasos por tramo)")
            btn_cycle = gr.Button("▶ ejecutar ciclo", variant="primary")
        with gr.Row():
            cyc_gallery = gr.Gallery(label="trayectoria del ciclo", columns=7, height=180)
            cyc_result = gr.Markdown()
        btn_cycle.click(run_cycle, [state, cyc_pattern, cyc_n, magnitude],
                        [cyc_gallery, cyc_result])

    return demo


def main():
    CTX.model = load_model()
    CTX.decoder = load_decoder()
    CTX.act_mean, CTX.act_std = load_action_stats()
    CTX.anchors = load_anchors()
    if not CTX.anchors:
        raise RuntimeError("No hay anclas en /anchors")
    print(f"[demo] device={DEVICE} {len(CTX.anchors)} anclas, decoder listo")
    build_ui().launch()


if __name__ == "__main__":
    main()
