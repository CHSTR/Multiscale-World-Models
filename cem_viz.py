"""Visualizador de CEM sobre el entorno: rutas candidatas dibujadas encima del frame.

Reimplementa el Cross-Entropy Method del Alg. 2 del paper, pero avanzando
ITERACION A ITERACION bajo tu control, y proyectando los planes sobre la imagen.

Dos proyecciones, y la diferencia importa
-----------------------------------------
1. RUTA GEOMETRICA (siempre disponible). Se obtiene de las acciones, sin tocar
   el world model. En Push-T la accion ya ES la posicion objetivo del agente en
   el canvas, asi que el plan son coordenadas directas. En TwoRoom la accion es
   un delta, asi que se integra desde la posicion inicial con una escala
   calibrada contra el dataset (regresion Δpos ~ k·Σacciones). Esto es lo que
   las acciones harian EN EL ENTORNO REAL.

2. RUTA IMAGINADA (opcional, --probe). Un probe lineal emb -> posicion 2D,
   entrenado sobre `proprio`, aplicado a los latentes del rollout. Esto es donde
   el MODELO cree que va a acabar. Es el mismo probe de la Tab. 3 del paper, que
   en TwoRoom alcanza r = 0.996 con regresion lineal.

Superponer las dos separa las hipotesis: si divergen, el error esta en el
predictor; si coinciden y CEM aun asi falla, el problema es el coste terminal de
la Eq. 4, que solo compara el estado final contra el goal.

Uso
---
    python cem_viz.py --ckpt <ckpt> --config <config.yaml> --probe
"""

import argparse
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401

from decoder_train import build_dataset, denormalize, encode_frames, load_world_model
from decoder_rollout import collect_sequence, episode_column, load_decoder
from utils import ZScoreNormalizer


class Ctx:
    model = decoder = dataset = act_scaler = device = None
    n_context = history_size = 3
    n_sub, a_dim = 5, 2
    chance = 2.0
    img_size = 224
    probe = None                # (W, b) para emb -> posicion 2D
    action_is_position = False
    scale = 1.0                 # unidades de posicion por unidad de accion
    pos_lo = np.zeros(2)
    pos_hi = np.ones(2)


CTX = Ctx()


# --------------------------------------------------------------------------- #
#  CEM (Alg. 2), paso a paso
# --------------------------------------------------------------------------- #


@torch.no_grad()
def rollout_batch(emb0, acts0, plans, return_all=False):
    """plans: (N, H, A) normalizados -> (N, D) terminal, o (N, H, D) si return_all."""
    N, H, _ = plans.shape
    HS = CTX.history_size
    emb = emb0.expand(N, -1, -1).clone()
    acts = acts0.expand(N, -1, -1).clone()

    for h in range(H):
        acts = torch.cat([acts, plans[:, h : h + 1]], dim=1)
        act_emb = CTX.model.action_encoder(acts[:, : emb.size(1)])
        pred = CTX.model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)

    return emb[:, -H:] if return_all else emb[:, -1]


@torch.no_grad()
def cem_iteration(sess):
    """Una sola iteracion de CEM. Muta la sesion y devuelve el registro."""
    A = CTX.n_sub * CTX.a_dim
    H = sess["horizon"]

    eps = torch.randn(sess["n_samples"], H, A, generator=sess["gen"])
    plans = (sess["mu"] + sess["sigma"] * eps).to(CTX.device)

    z_end = rollout_batch(sess["emb0"], sess["acts0"], plans)
    cost = (z_end - sess["z_goal"]).pow(2).mean(dim=-1)          # Eq. 4

    order = cost.argsort()
    elite_idx = order[: sess["n_elites"]]
    elites = plans[elite_idx].cpu()

    rec = {
        "it": len(sess["trace"]),
        "plans": plans.cpu().numpy(),
        "cost": cost.cpu().numpy(),
        "elite_idx": elite_idx.cpu().numpy(),
        "mu": sess["mu"].clone().numpy(),
        "sigma": sess["sigma"].clone().numpy(),
        "best_plan": plans[order[0]].cpu(),
    }
    sess["trace"].append(rec)

    sess["mu"] = elites.mean(dim=0)
    sess["sigma"] = elites.std(dim=0).clamp_min(1e-3)
    return rec


# --------------------------------------------------------------------------- #
#  Proyeccion de planes al plano de la imagen
# --------------------------------------------------------------------------- #


def unnorm(plan):
    """(H, n_sub*a_dim) normalizado -> (H*n_sub, a_dim) crudo."""
    p = torch.as_tensor(plan, dtype=torch.float32).reshape(-1, CTX.n_sub, CTX.a_dim)
    raw = p * CTX.act_scaler.std + CTX.act_scaler.mean
    return raw.reshape(-1, CTX.a_dim).numpy()


def pos_to_px(pos):
    """Coordenadas del entorno -> pixeles de la imagen mostrada."""
    pos = np.atleast_2d(pos)
    rng = np.maximum(CTX.pos_hi - CTX.pos_lo, 1e-8)
    return (pos - CTX.pos_lo) / rng * (CTX.img_size - 1)


def geometric_path(plan, pos0):
    """Ruta que las acciones producirian en el entorno real, en pixeles."""
    raw = unnorm(plan)[:, :2]
    if CTX.action_is_position:
        pts = raw                                              # Push-T
    else:
        pts = pos0[:2] + np.cumsum(raw, axis=0) * CTX.scale    # TwoRoom
    return pos_to_px(np.vstack([pos0[:2][None, :], pts]))


def imagined_path(embs):
    """Ruta que el MODELO cree, via probe lineal. embs: (H, D) -> pixeles."""
    if CTX.probe is None:
        return None
    W, b = CTX.probe
    return pos_to_px(embs.cpu().numpy() @ W + b)


# --------------------------------------------------------------------------- #
#  Figuras
# --------------------------------------------------------------------------- #


def fig_overlay(sess, rec, show_all, show_imagined):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(sess["start_img"])

    pos0, elite = sess["pos0"], set(rec["elite_idx"].tolist())
    if show_all:
        for i in range(len(rec["plans"])):
            if i in elite:
                continue
            p = geometric_path(rec["plans"][i], pos0)
            ax.plot(p[:, 0], p[:, 1], color="white", lw=0.5, alpha=0.18, zorder=2)
    for i in rec["elite_idx"]:
        p = geometric_path(rec["plans"][i], pos0)
        ax.plot(p[:, 0], p[:, 1], color="tab:cyan", lw=0.9, alpha=0.55, zorder=3)

    p = geometric_path(rec["best_plan"].numpy(), pos0)
    ax.plot(p[:, 0], p[:, 1], color="tab:red", lw=2.8, zorder=5, label="mejor plan (real)")
    ax.scatter(p[0, 0], p[0, 1], c="lime", s=70, ec="k", zorder=6, marker="s")
    ax.scatter(p[-1, 0], p[-1, 1], c="tab:red", s=70, ec="k", zorder=6)

    if show_imagined and CTX.probe is not None:
        embs = rollout_batch(sess["emb0"], sess["acts0"],
                             rec["best_plan"].unsqueeze(0).to(CTX.device),
                             return_all=True)[0]
        q = np.vstack([pos_to_px(pos0[:2]), imagined_path(embs)])
        ax.plot(q[:, 0], q[:, 1], color="yellow", lw=2.2, ls="--", zorder=5,
                label="mejor plan (imaginado)")

    gp = pos_to_px(sess["goal_pos"][:2])[0]
    ax.scatter(gp[0], gp[1], c="magenta", s=190, marker="*", ec="k", zorder=7, label="goal")

    ax.set_xlim(0, CTX.img_size); ax.set_ylim(CTX.img_size, 0); ax.axis("off")
    ax.set_title(f"iteracion {rec['it'] + 1}")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    plt.tight_layout()
    return fig


def fig_cost(rec):
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    c = rec["cost"] / CTX.chance
    thr = c[rec["elite_idx"]].max()
    ax.hist(c, bins=45, color="0.7")
    ax.axvline(thr, color="tab:blue", lw=1.6, label=f"umbral elite {thr:.2f}")
    ax.axvline(1.0, color="0.3", ls=":", lw=1, label="azar")
    ax.set_xlabel("C(ẑ_H) / azar"); ax.set_ylabel("candidatos")
    ax.legend(fontsize=7); plt.tight_layout()
    return fig


def fig_conv(trace):
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    its = np.arange(1, len(trace) + 1)
    ax.plot(its, [t["cost"].mean() / CTX.chance for t in trace], color="0.6", label="media")
    ax.plot(its, [t["cost"][t["elite_idx"]].mean() / CTX.chance for t in trace],
            color="tab:blue", label="elites")
    ax.plot(its, [t["cost"].min() / CTX.chance for t in trace], color="tab:red", label="mejor")
    ax.axhline(1.0, color="0.3", ls=":", lw=1)
    ax.set_xlabel("iteracion"); ax.set_ylabel("coste / azar"); ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(its, [t["sigma"].mean() for t in trace], color="tab:green", ls="--", lw=1)
    ax2.set_ylabel("σ", color="tab:green"); ax2.tick_params(axis="y", labelcolor="tab:green")
    plt.tight_layout()
    return fig


def diagnosis(trace):
    rec = trace[-1]
    c = rec["cost"] / CTX.chance
    elite = c[rec["elite_idx"]]
    spread, gap = float(c.std()), float(c.mean() - elite.mean())
    best0, best1 = trace[0]["cost"].min() / CTX.chance, rec["cost"].min() / CTX.chance

    if spread < 0.03:
        v = ("🔴 **el coste no discrimina.** Las elites son practicamente ruido. El cuello "
             "de botella es el COSTE TERMINAL de la Eq. 4: muchos planes distintos acaban "
             "a distancia parecida del goal, y el paisaje no guia la busqueda.")
    elif gap < 0.05:
        v = "🟡 **discriminacion debil.** Las elites apenas baten a la media."
    elif best1 > 0.8 * best0:
        v = "🟡 **converge sin mejorar.** σ encoge hacia un optimo local."
    else:
        v = "🟢 **el planificador funciona.** El coste discrimina y el mejor plan mejora."

    return (f"### iteracion {len(trace)}\n| | valor |\n|---|---|\n"
            f"| mejor coste (÷azar) | {best1:.3f} (inicial {best0:.3f}) |\n"
            f"| media elites | {elite.mean():.3f} |\n"
            f"| media global | {c.mean():.3f} |\n"
            f"| dispersion | {spread:.3f} |\n"
            f"| σ media | {rec['sigma'].mean():.3f} |\n\n{v}")


# --------------------------------------------------------------------------- #
#  Callbacks
# --------------------------------------------------------------------------- #


def _frame_and_pos(idx, n):
    """px[-1] es el frame 0 de la fila idx+n-1, asi que proprio se toma de ahi."""
    px, acts = collect_sequence(CTX.dataset, int(idx), n, 1, CTX.act_scaler)
    pro = np.asarray(CTX.dataset[int(idx) + n - 1]["proprio"], dtype=np.float32)
    pos = (pro[0] if pro.ndim > 1 else pro).reshape(-1)
    img = denormalize(px[-1:].to(CTX.device))[0].cpu().numpy().transpose(1, 2, 0) * 255
    return px, acts, pos, img.astype(np.uint8)


def preview(start_idx, goal_off):
    """Muestra inicio y goal SIN ejecutar CEM."""
    start_idx, goal_off = int(start_idx), int(goal_off)
    _, _, _, s_img = _frame_and_pos(start_idx, CTX.n_context)
    _, _, _, g_img = _frame_and_pos(start_idx + CTX.n_context - 1 + goal_off, 1)
    return s_img, g_img, None, "Pulsa **iniciar CEM** cuando te guste el par."


def start_cem(start_idx, goal_off, horizon, n_samples, n_elites, sigma0, seed):
    start_idx, goal_off = int(start_idx), int(goal_off)
    px, acts, pos0, s_img = _frame_and_pos(start_idx, CTX.n_context)
    gpx, _, goal_pos, g_img = _frame_and_pos(start_idx + CTX.n_context - 1 + goal_off, 1)

    with torch.no_grad():
        emb0 = encode_frames(CTX.model, px.to(CTX.device), source="emb").unsqueeze(0)
        z_goal = encode_frames(CTX.model, gpx.to(CTX.device), source="emb")[0]

    H, A = int(horizon), CTX.n_sub * CTX.a_dim
    sess = {
        "emb0": emb0, "acts0": acts[: CTX.n_context - 1].unsqueeze(0).to(CTX.device),
        "z_goal": z_goal, "horizon": H,
        "n_samples": int(n_samples), "n_elites": int(n_elites),
        "mu": torch.zeros(H, A), "sigma": torch.full((H, A), float(sigma0)),
        "gen": torch.Generator(device="cpu").manual_seed(int(seed)),
        "trace": [], "pos0": pos0, "goal_pos": goal_pos, "start_img": s_img,
    }
    cem_iteration(sess)
    return (sess, s_img, g_img, *render(sess, True, True))


def advance(sess, k, show_all, show_imag):
    if sess is None:
        return sess, None, None, None, "Pulsa **iniciar CEM** primero."
    for _ in range(int(k)):
        cem_iteration(sess)
    return (sess, *render(sess, show_all, show_imag))


def render(sess, show_all, show_imag):
    rec = sess["trace"][-1]
    return (fig_overlay(sess, rec, show_all, show_imag), fig_cost(rec),
            fig_conv(sess["trace"]), diagnosis(sess["trace"]))


def random_start():
    """Indice de clip valido con margen para el goal. Se acota por len(dataset),
    que cuenta clips y es menor que la columna de episodios."""
    ep = CTX.dataset.get_col_data(episode_column(CTX.dataset))
    hi = min(len(CTX.dataset), len(ep)) - 60
    while True:
        i = int(np.random.randint(0, max(hi, 1)))
        if len(set(ep[i : i + 40].tolist())) == 1:
            return i


# --------------------------------------------------------------------------- #
#  Calibracion
# --------------------------------------------------------------------------- #


def _sample_rows(n, seed):
    """Indices validos de CLIP. len(dataset) cuenta clips, no filas crudas:
    cada episodio pierde la cola al construir sub-trayectorias, asi que
    get_col_data() es mas largo que el dataset y no se puede indexar cruzado."""
    return np.random.default_rng(seed).integers(0, len(CTX.dataset), size=n)


def calibrate(n=1200):
    """Decide si la accion es POSICION o DELTA, y calibra la escala en el segundo caso.

    Usa frames CONSECUTIVOS DE LA MISMA FILA: cada clip vive dentro de un unico
    episodio, asi que no hay que comprobar limites ni alinear indices.
    """
    dpos, sact, a0, allpos, ref = [], [], [], [], []
    for i in _sample_rows(n, 0):
        row = CTX.dataset[int(i)]
        pro = np.asarray(row["proprio"], dtype=np.float32)
        act = np.nan_to_num(np.asarray(row["action"], dtype=np.float32))
        if pro.ndim < 2 or len(pro) < 2:
            continue
        a = act[0].reshape(CTX.n_sub, CTX.a_dim)
        dpos.append(pro[1][: CTX.a_dim] - pro[0][: CTX.a_dim])
        sact.append(a.sum(0))
        a0.append(a[0])
        ref.append(pro[0][0])
        allpos.append(pro[:, : CTX.a_dim])

    dpos, sact, a0 = np.array(dpos), np.array(sact), np.array(a0)
    allpos = np.concatenate(allpos, axis=0)
    CTX.pos_lo, CTX.pos_hi = np.nanmin(allpos, 0)[:2], np.nanmax(allpos, 0)[:2]

    # ¿la accion correlaciona con la POSICION absoluta (Push-T) o con el DELTA (TwoRoom)?
    r_pos = abs(np.corrcoef(a0[:, 0], np.array(ref))[0, 1])
    denom = (sact * sact).sum()
    k = float((dpos * sact).sum() / denom) if denom > 1e-8 else 1.0
    r2 = 1.0 - (dpos - k * sact).var() / max(dpos.var(), 1e-8)

    CTX.action_is_position = r_pos > 0.9 and r2 < 0.3
    CTX.scale = k
    modo = "POSICION (Push-T)" if CTX.action_is_position else "DELTA (TwoRoom)"
    print(f"[cem] calibracion: k={k:.4f} R²(delta)={r2:.3f} corr(accion,posicion)={r_pos:.3f} -> {modo}")
    print(f"[cem] rango de posicion: {CTX.pos_lo} .. {CTX.pos_hi}")
    if not CTX.action_is_position and r2 < 0.3:
        print("[cem] ⚠ ajuste delta pobre; la ruta geometrica puede no ser fiable.")


def fit_probe(n=2000):
    """Probe lineal emb -> posicion 2D (minimos cuadrados), como en la Tab. 3.

    pixels y proprio salen de la MISMA fila, asi que estan alineados por
    construccion.
    """
    idx = _sample_rows(n, 1)
    px, Y = [], []
    for i in idx:
        row = CTX.dataset[int(i)]
        pro = np.asarray(row["proprio"], dtype=np.float32)
        px.append(row["pixels"][0])
        Y.append((pro[0] if pro.ndim > 1 else pro)[:2])

    px = torch.stack(px).to(CTX.device)
    with torch.no_grad():
        Z = torch.cat([encode_frames(CTX.model, px[i : i + 128], source="emb")
                       for i in range(0, len(px), 128)]).cpu().numpy()
    Y = np.array(Y, dtype=np.float32)

    ok = ~np.isnan(Y).any(1)
    Z, Y = Z[ok], Y[ok]
    sol, *_ = np.linalg.lstsq(np.hstack([Z, np.ones((len(Z), 1))]), Y, rcond=None)
    W, b = sol[:-1], sol[-1]
    r = np.corrcoef((Z @ W + b).ravel(), Y.ravel())[0, 1]
    print(f"[cem] probe lineal emb->posicion: r={r:.4f} (Tab. 3 reporta 0.996 en TwoRoom)")
    CTX.probe = (W, b)


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #


def build_ui():
    with gr.Blocks(title="CEM sobre el entorno") as demo:
        gr.Markdown(
            "# CEM paso a paso\n"
            "Elige un par inicio/goal, mira si te gusta, y **avanza CEM iteracion a "
            "iteracion**. Las rutas se dibujan sobre el frame: en rojo el mejor plan segun "
            "las acciones, en amarillo discontinuo donde el *modelo* cree que acabara."
        )
        sess = gr.State(None)

        with gr.Row():
            start = gr.Number(value=0, precision=0, label="fila inicial", scale=1)
            btn_rand = gr.Button("🎲", scale=0)
            goal_off = gr.Slider(1, 125, value=5, step=1, label="goal (pasos latentes adelante)")
            btn_prev = gr.Button("👁 previsualizar")
        with gr.Row():
            img_start = gr.Image(label="inicio", height=220)
            img_goal = gr.Image(label="goal", height=220)

        with gr.Accordion("parametros de CEM (Alg. 2)", open=False):
            with gr.Row():
                horizon = gr.Slider(1, 15, value=5, step=1, label="horizonte H")
                n_samples = gr.Slider(50, 600, value=300, step=50, label="candidatos")
                n_elites = gr.Slider(5, 100, value=30, step=5, label="elites")
                sigma0 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="σ inicial")
                seed = gr.Number(value=0, precision=0, label="seed")

        with gr.Row():
            btn_start = gr.Button("▶ iniciar CEM", variant="primary")
            btn_1 = gr.Button("+1 iteracion")
            btn_5 = gr.Button("+5")
            btn_25 = gr.Button("+25")
        with gr.Row():
            show_all = gr.Checkbox(True, label="mostrar candidatos no-elite")
            show_imag = gr.Checkbox(True, label="mostrar ruta imaginada (probe)")

        with gr.Row():
            p_over = gr.Plot(label="planes sobre el entorno")
            with gr.Column():
                p_cost = gr.Plot(label="costes")
                p_conv = gr.Plot(label="convergencia")
            diag = gr.Markdown()

        outs = [p_over, p_cost, p_conv, diag]
        btn_rand.click(lambda: random_start(), outputs=start)
        btn_prev.click(preview, [start, goal_off], [img_start, img_goal, p_over, diag])
        btn_start.click(start_cem,
                        [start, goal_off, horizon, n_samples, n_elites, sigma0, seed],
                        [sess, img_start, img_goal, *outs])
        for b, k in ((btn_1, 1), (btn_5, 5), (btn_25, 25)):
            b.click(advance, [sess, gr.State(k), show_all, show_imag], [sess, *outs])
        for c in (show_all, show_imag):
            c.change(lambda s, a, i: render(s, a, i) if s else (None, None, None, ""),
                     [sess, show_all, show_imag], outs)

    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--decoder", default=None)
    p.add_argument("--probe", action="store_true", help="ajusta el probe lineal emb->posicion")
    p.add_argument("--share", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cache = Path(swm.data.utils.get_cache_dir())
    cfg = OmegaConf.load(Path(args.config) if args.config
                         else (cache / args.ckpt).parent / "config.yaml")

    CTX.device = torch.device(args.device)
    CTX.model = load_world_model(args.ckpt, CTX.device)
    CTX.decoder = load_decoder(args.decoder, CTX.device) if args.decoder else None
    CTX.dataset = build_dataset(cfg)
    CTX.history_size = CTX.n_context = cfg.history_size
    CTX.n_sub = cfg.data.dataset.frameskip
    CTX.img_size = cfg.img_size

    col = torch.from_numpy(np.array(CTX.dataset.get_col_data("action")))
    col = col[~torch.isnan(col).any(dim=1)]
    CTX.a_dim = col.size(-1)
    CTX.act_scaler = ZScoreNormalizer(col.mean(0, keepdim=True).clone(),
                                      col.std(0, keepdim=True).clone())

    idx = np.random.default_rng(0).integers(0, len(CTX.dataset), size=64)
    with torch.no_grad():
        z = encode_frames(CTX.model,
                          torch.stack([CTX.dataset[int(i)]["pixels"][0] for i in idx]).to(CTX.device),
                          source="emb")
    CTX.chance = (z - z[torch.randperm(z.size(0), device=z.device)]).pow(2).mean().item()
    print(f"[cem] a_dim={CTX.a_dim} n_sub={CTX.n_sub} nivel_azar={CTX.chance:.3f}")

    calibrate()
    if args.probe:
        fit_probe()

    build_ui().launch(share=args.share)


if __name__ == "__main__":
    main()
