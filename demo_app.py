"""Demo interactiva: tu das las acciones, LeWM imagina el resto.

Se ancla en un frame REAL del dataset (n_context observaciones codificadas), y a
partir de ahi cada accion que pulsas genera UN paso latente con el predictor,
decodificado a pixeles. El encoder no vuelve a intervenir: todo lo que ves
despues del ancla es imaginado.

Realismo sobre el horizonte
---------------------------
El predictor se entrena teacher-forced (Eq. 1): nunca vio sus propias salidas.
En TwoRoom el error latente cruza el nivel de azar hacia el paso 8-10, y el
horizonte de planificacion del paper es H=5. Asi que la demo es honesta sobre
esto: muestra tres indicadores de deriva y colorea el frame segun cuanto te has
alejado del ancla. Pasados ~5 pasos, lo que ves es plausible pero ya no fiel.

Indicadores
-----------
- pasos desde el ancla: presupuesto conocido (~5).
- |z|^2/D: SIGReg empuja los embeddings a N(0,I), asi que esto deberia valer ~1.
  Alejarse mucho de 1 significa que el latente salio de la region entrenada.
  Es la misma señal que sostiene la evaluacion de sorpresa (Fig. 8).
- rectitud: coseno entre velocidades latentes consecutivas. Valores muy altos
  indican que el rollout va "en linea recta" y ya no responde a tus acciones
  (el temporal collapse que describe App. H).

Uso
---
    python demo_app.py \
        --ckpt models_server/checkpoints/dinov3_s_lora/weights_epoch_10.pt \
        --decoder outputs/decoder_dinov3_tworoom/decoder.pt \
        --config models_server/checkpoints/dinov3_s_lora/config.yaml

    # con pusht (el pad de 2D vale para fuerzas x/y):
    python demo_app.py \
        --ckpt models_server/checkpoints/dinov3_s_lora_pusht/weights_epoch_10.pt \
        --decoder outputs/decoder_dinov3_pusht/decoder.pt \
        --data pusht_expert_train.h5
"""

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registra el formato HDF5

from decoder_train import build_dataset, denormalize, encode_frames, load_world_model
from decoder_rollout import collect_sequence, episode_span, load_decoder
from utils import ZScoreNormalizer


# --------------------------------------------------------------------------- #
#  Contexto global (cargado una vez al arrancar)
# --------------------------------------------------------------------------- #


class Ctx:
    model = None
    decoder = None
    dataset = None
    act_scaler = None
    device = None
    n_context = 3
    history_size = 3
    n_sub = 5          # sub-acciones por bloque (frameskip)
    a_dim = 2          # dimension de la accion cruda
    budget = 5         # horizonte fiable, en pasos latentes
    chance = 2.0       # MSE medio entre embeddings reales no emparejados


CTX = Ctx()


DIRS = {
    "↖": (-1, -1), "↑": (0, -1), "↗": (1, -1),
    "←": (-1, 0),  "•": (0, 0),  "→": (1, 0),
    "↙": (-1, 1),  "↓": (0, 1),  "↘": (1, 1),
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def make_action_block(vec2, magnitude):
    """Construye el bloque de accion normalizado que espera el action_encoder.

    La accion cruda es de a_dim (2 en TwoRoom) y el dataset la entrega agrupada
    en n_sub sub-pasos (frameskip), es decir n_sub*a_dim = 10. Las estadisticas
    z-score son de la accion SIN agrupar, asi que se normaliza por sub-accion.
    """
    a_dim, n_sub = CTX.a_dim, CTX.n_sub
    raw = torch.tensor(list(vec2[:a_dim]), dtype=torch.float32) * magnitude
    raw = raw.repeat(n_sub).reshape(1, n_sub, a_dim)
    return CTX.act_scaler(raw).reshape(1, n_sub * a_dim)


def to_display(emb_vec):
    """Decodifica un embedding (D,) a una imagen HxWx3 uint8."""
    with torch.no_grad():
        img = CTX.decoder(emb_vec.unsqueeze(0)).float()
    img = denormalize(img)[0].cpu().numpy().transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


def metrics(emb):
    """emb: (1, T, D) -> dict de indicadores de deriva."""
    z = emb[0]
    n_pred = z.size(0) - CTX.n_context
    energy = z[-1].pow(2).mean().item()          # ~1 bajo N(0,I)

    straight = float("nan")
    if z.size(0) >= 3:
        v = z[1:] - z[:-1]
        v = torch.nn.functional.normalize(v, dim=-1)
        straight = (v[-1] * v[-2]).sum().item()

    drift = (z[-1] - z[CTX.n_context - 1]).pow(2).mean().item()
    return {"n_pred": n_pred, "energy": energy, "straight": straight, "drift": drift}


def status_md(m):
    n, budget = m["n_pred"], CTX.budget
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
        f"**{n}** pasos imaginados desde el ancla ({n * CTX.n_sub} pasos de entorno) — {note}\n\n"
        f"| indicador | valor | esperado |\n|---|---|---|\n"
        f"| `|z|²/D` | {m['energy']:.2f}{energy_flag} | ≈ 1.0 (SIGReg) |\n"
        f"| rectitud | {straight} | alto ⇒ no responde a la accion |\n"
        f"| deriva desde el ancla | {m['drift']:.2f} | ~2.0 ⇒ nivel de azar |\n"
    )


# --------------------------------------------------------------------------- #
#  Callbacks
# --------------------------------------------------------------------------- #


def anchor(start_idx):
    """Codifica n_context frames reales y arranca el estado."""
    start_idx = int(start_idx)
    pixels, actions = collect_sequence(
        CTX.dataset, start_idx, CTX.n_context, 1, CTX.act_scaler
    )
    with torch.no_grad():
        emb = encode_frames(CTX.model, pixels.to(CTX.device), source="emb").unsqueeze(0)

    acts = actions[: CTX.n_context - 1].unsqueeze(0).to(CTX.device)
    state = {"emb": emb, "acts": acts}

    real = denormalize(pixels[-1:].to(CTX.device))[0].cpu().numpy().transpose(1, 2, 0)
    real = (real * 255).astype(np.uint8)

    m = metrics(emb)
    return state, real, to_display(emb[0, -1]), status_md(m), [], _plot([])


def random_anchor():
    n = len(CTX.dataset) - CTX.n_context - 1
    while True:
        i = int(np.random.randint(0, n))
        _, hi = episode_span(CTX.dataset, i)
        if i + CTX.n_context <= hi:
            return i


def _advance(emb, acts, direction, magnitude):
    """Un paso latente. Devuelve (emb, acts) actualizados."""
    a = make_action_block(DIRS[direction], magnitude).to(CTX.device)
    acts = torch.cat([acts, a.unsqueeze(0)], dim=1)   # len(acts) == len(emb)
    HS = CTX.history_size
    with torch.no_grad():
        act_emb = CTX.model.action_encoder(acts[:, : emb.size(1)])
        pred = CTX.model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
    return torch.cat([emb, pred], dim=1), acts


def step(state, direction, magnitude, history):
    """Aplica UNA accion y avanza un paso latente imaginado."""
    if state is None:
        return state, None, "Pulsa **Anclar** primero.", history, _plot(history)

    emb, acts = _advance(state["emb"], state["acts"], direction, magnitude)
    state = {"emb": emb, "acts": acts}
    m = metrics(emb)
    history = (history or []) + [m["energy"]]
    return state, to_display(emb[0, -1]), status_md(m), history, _plot(history)


def undo(state, history):
    if state is None or state["emb"].size(1) <= CTX.n_context:
        return state, gr.update(), gr.update(), history, gr.update()
    state = {"emb": state["emb"][:, :-1], "acts": state["acts"][:, :-1]}
    history = (history or [])[:-1]
    m = metrics(state["emb"])
    return state, to_display(state["emb"][0, -1]), status_md(m), history, _plot(history)


def _plot(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    if history:
        ax.plot(range(1, len(history) + 1), history, marker="o", color="tab:red")
    ax.axhline(1.0, ls=":", color="0.4", lw=1)
    ax.axvspan(0, CTX.budget, color="tab:green", alpha=0.10)
    ax.set_xlabel("pasos imaginados")
    ax.set_ylabel("|z|²/D")
    ax.set_ylim(0, max(2.5, max(history) * 1.2 if history else 2.5))
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Test de ciclo
# --------------------------------------------------------------------------- #

CYCLES = {
    "ida y vuelta (horizontal)": lambda n: ["→"] * n + ["←"] * n,
    "ida y vuelta (vertical)": lambda n: ["↓"] * n + ["↑"] * n,
    "cuadrado": lambda n: ["→"] * n + ["↓"] * n + ["←"] * n + ["↑"] * n,
    "control: solo ida": lambda n: ["→"] * (2 * n),
}


def run_cycle(state, pattern, n, magnitude):
    """Ejecuta una secuencia cerrada y mide el error de cierre.

    Si la dinamica es fiel, volver sobre tus pasos deberia devolverte al mismo
    latente. Se reportan dos numeros, y el segundo es imprescindible:

      - cierre:    |z_final - z_inicial|^2, normalizado por el nivel de azar.
      - excursion: la distancia MAXIMA alcanzada durante el ciclo.

    Un cierre bajo solo significa algo si la excursion fue alta. Si ambos son
    bajos, el modelo simplemente no se movio. Y el patron "control: solo ida"
    NO deberia cerrar: si cierra, hay un atractor y el resultado es espurio.
    """
    if state is None:
        return None, "Pulsa **Anclar** primero."

    emb, acts = state["emb"], state["acts"]
    z0 = emb[0, -1].clone()
    seq = CYCLES[pattern](int(n))

    dists, frames = [], []
    keep = max(1, len(seq) // 6)
    for i, d in enumerate(seq):
        emb, acts = _advance(emb, acts, d, magnitude)
        dists.append((emb[0, -1] - z0).pow(2).mean().item())
        if i % keep == 0 or i == len(seq) - 1:
            frames.append((to_display(emb[0, -1]), f"{i + 1}: {d}"))

    chance = CTX.chance
    closure = dists[-1] / chance
    excursion = max(dists) / chance
    ratio = closure / max(excursion, 1e-8)

    if excursion < 0.05:
        verdict = ("⚠️ **el modelo apenas se movio** — excursion casi nula. "
                   "Sube la magnitud o N; el cierre no significa nada asi.")
    elif pattern.startswith("control"):
        verdict = ("Este es el **control**: no deberia cerrar. Si el cierre es bajo "
                   "y parecido al de 'ida y vuelta', hay un atractor y el resultado "
                   "del ciclo es espurio.")
    elif ratio < 0.15:
        verdict = "✅ **cierra** — la dinamica es reversible con poca perdida."
    elif ratio < 0.40:
        verdict = "🟡 **cierra parcialmente** — vuelve a la vecindad, con deriva acumulada."
    else:
        verdict = "🔴 **no cierra** — la deriva domina sobre el desplazamiento."

    md = (
        f"### {pattern} · N={int(n)} · {len(seq)} pasos\n"
        f"| | valor (1.0 = azar) |\n|---|---|\n"
        f"| cierre `|z_fin − z_ini|²` | **{closure:.3f}** |\n"
        f"| excursion maxima | {excursion:.3f} |\n"
        f"| cierre / excursion | {ratio:.3f} |\n\n"
        f"{verdict}\n\n"
        f"<sub>Repitelo desde dos anclas bien separadas: si ambos ciclos cierran cerca de "
        f"su PROPIO origen, la reversibilidad es real. Si acaban en el mismo sitio, es un atractor.</sub>"
    )
    return frames, md


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #


def build_ui():
    with gr.Blocks(title="LeWM interactivo") as demo:
        gr.Markdown(
            "# LeWM interactivo\n"
            "Ancla en un frame real, y a partir de ahi **tus acciones** mueven el mundo "
            "imaginado. El encoder no vuelve a intervenir. El horizonte fiable son "
            f"~{CTX.budget} pasos: mas alla, lo que ves sigue siendo plausible pero deja de ser fiel."
        )

        state = gr.State(None)
        history = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                start = gr.Number(value=0, precision=0, label="fila del dataset")
                with gr.Row():
                    btn_rand = gr.Button("🎲 aleatoria")
                    btn_anchor = gr.Button("⚓ anclar", variant="primary")
                real_img = gr.Image(label="frame real (ancla)", height=240)

            with gr.Column(scale=2):
                imag_img = gr.Image(label="estado imaginado", height=360)
                magnitude = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="magnitud de la accion")
                pad = []
                for row in (["↖", "↑", "↗"], ["←", "•", "→"], ["↙", "↓", "↘"]):
                    with gr.Row():
                        for d in row:
                            pad.append(gr.Button(d, scale=1))
                btn_undo = gr.Button("↩ deshacer paso")

            with gr.Column(scale=1):
                status = gr.Markdown()
                plot = gr.Plot(label="energia latente")

        outs = [state, imag_img, status, history, plot]
        btn_rand.click(lambda: random_anchor(), outputs=start)
        btn_anchor.click(anchor, [start], [state, real_img, imag_img, status, history, plot])
        for b in pad:
            b.click(step, [state, gr.State(b.value), magnitude, history], outs)
        btn_undo.click(undo, [state, history], outs)

        gr.Markdown("---\n## Test de ciclo\n"
                    "Ejecuta una secuencia cerrada desde el estado actual y mide si vuelves "
                    "al mismo latente. **No modifica** el estado interactivo. Corre siempre el "
                    "patron *control* tambien: no deberia cerrar.")
        with gr.Row():
            cyc_pattern = gr.Dropdown(list(CYCLES), value="ida y vuelta (horizontal)", label="patron")
            cyc_n = gr.Slider(2, 120, value=20, step=1, label="N (pasos por tramo)")
            btn_cycle = gr.Button("▶ ejecutar ciclo", variant="primary")
        with gr.Row():
            cyc_gallery = gr.Gallery(label="trayectoria imaginada", columns=7, height=180)
            cyc_result = gr.Markdown()

        btn_cycle.click(run_cycle, [state, cyc_pattern, cyc_n, magnitude],
                        [cyc_gallery, cyc_result])

    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--decoder", required=True, help="entrenado con --source emb")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None, help="nombre literal del dataset (default: del config.yaml)")
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--budget", type=int, default=5, help="horizonte fiable en pasos latentes")
    p.add_argument("--share", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cache_dir = Path(swm.data.utils.get_cache_dir())
    cfg_path = Path(args.config) if args.config else (cache_dir / args.ckpt).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    if args.data is not None:
        with open_dict(cfg):
            cfg.data.dataset.name = args.data

    CTX.device = torch.device(args.device)
    CTX.model = load_world_model(args.ckpt, CTX.device)
    CTX.decoder = load_decoder(args.decoder, CTX.device)
    CTX.dataset = build_dataset(cfg)
    CTX.history_size = cfg.history_size
    CTX.n_context = args.context or cfg.history_size
    CTX.n_sub = cfg.data.dataset.frameskip
    CTX.budget = args.budget

    col = torch.from_numpy(np.array(CTX.dataset.get_col_data("action")))
    col = col[~torch.isnan(col).any(dim=1)]
    CTX.a_dim = col.size(-1)
    CTX.act_scaler = ZScoreNormalizer(
        col.mean(0, keepdim=True).clone(), col.std(0, keepdim=True).clone()
    )

    # Calibra el nivel de azar: MSE medio entre embeddings reales NO emparejados.
    # Da unidades interpretables al test de ciclo (1.0 = tan lejos como dos
    # estados cualesquiera). Bajo SIGReg deberia salir cerca de 2.0.
    n_col = len(CTX.dataset)
    idx = np.random.default_rng(0).integers(0, n_col, size=64)
    with torch.no_grad():
        px = torch.stack([CTX.dataset[int(i)]["pixels"][0] for i in idx]).to(CTX.device)
        z = encode_frames(CTX.model, px, source="emb")
    CTX.chance = (z - z[torch.randperm(z.size(0), device=z.device)]).pow(2).mean().item()

    print(f"[demo] a_dim={CTX.a_dim} n_sub={CTX.n_sub} bloque={CTX.a_dim * CTX.n_sub} "
          f"history_size={CTX.history_size} context={CTX.n_context} "
          f"nivel_azar={CTX.chance:.3f}")

    build_ui().launch(share=args.share)


if __name__ == "__main__":
    main()
