"""Smoke test (CPU, sin descargas) para el NCA latente + JEPA espacial.

Usa un encoder dummy con el mismo formato de salida que DinoV3Encoder
(``[CLS, storage x4, patches]``, 384-d), así que no necesita pesos de DINO.

    python smoke_nca.py
"""
import types

import torch
from torch import nn

from jepa_spatial import SpatialJEPA
from module import MLP, SIGReg, Embedder, sigreg_input
from nca import LatentNCAPredictor

torch.manual_seed(0)

IMG, PATCH, ENC_DIM, D, A_RAW = 112, 16, 384, 64, 10  # grilla 7x7
GH = GW = IMG // PATCH
N = GH * GW


class DummyEncoder(nn.Module):
    """Patchify conv + 1 CLS + 4 storage tokens, como DinoV3Encoder."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, ENC_DIM, PATCH, stride=PATCH)
        self.prefix = nn.Parameter(torch.randn(1, 5, ENC_DIM))

    def forward(self, x, interpolate_pos_encoding=True):
        p = self.patch(x).flatten(2).transpose(1, 2)
        pre = self.prefix.expand(x.size(0), -1, -1)
        return types.SimpleNamespace(last_hidden_state=torch.cat([pre, p], 1))


def build(idm=None, **nca_kw):
    norm = lambda d: nn.BatchNorm1d(d)  # noqa: E731
    return SpatialJEPA(
        encoder=DummyEncoder(),
        predictor=LatentNCAPredictor(latent_dim=D, act_dim=D, **nca_kw),
        action_encoder=Embedder(input_dim=A_RAW, emb_dim=D),
        projector=MLP(ENC_DIM, 128, D, norm_fn=norm),
        pred_proj=MLP(D, 128, D, norm_fn=norm),
        patch_size=PATCH,
        inverse_dynamics=idm,
    )


def check(name, cond):
    assert cond, name
    print(f"  ok  {name}")


# 1) Forward/backward de entrenamiento (igual a lejepa_forward)
print("[train step]")
model = build(steps=(2, 4), damage_prob=0.5).train()
sigreg = SIGReg(knots=17, num_proj=64)
B, T, HS = 4, 4, 3
batch = {"pixels": torch.randn(B, T, 3, IMG, IMG), "action": torch.randn(B, T, A_RAW)}
out = model.encode(batch)
emb = out["emb"]
check("emb es (B,T,N,D)", emb.shape == (B, T, N, D))
check("grid_hw inferida", model.grid_hw == (GH, GW))
pred = model.predict(emb[:, :HS], out["act_emb"][:, :HS])
check("pred es (B,HS,N,D)", pred.shape == (B, HS, N, D))
sg_in = sigreg_input(emb, max_tokens=B)
check("sigreg_input submuestrea tokens", sg_in.shape == (T, B, D))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
# fc2 y AdaLN parten en cero: el gradiente "abre" una capa por update
# (paso 0: fc2, paso 1: AdaLN, paso 2: action encoder). Inofensivo en un run real.
for it in range(3):
    opt.zero_grad()
    out = model.encode(dict(batch))
    pred = model.predict(out["emb"][:, :HS], out["act_emb"][:, :HS])
    loss = (pred - out["emb"][:, 1:]).pow(2).mean() + 0.09 * sigreg(sigreg_input(out["emb"], B))
    loss.backward()
    if it < 2:
        opt.step()
check("loss finita", torch.isfinite(loss).item())
for mod in ["encoder", "projector", "action_encoder", "predictor", "pred_proj"]:
    g = [p.grad for p in getattr(model, mod).parameters() if p.requires_grad]
    check(f"grad llega a {mod}", any(x is not None and x.abs().sum() > 0 for x in g))

# 2) sigreg_input no cambia el caso global (B,T,D)
e3 = torch.randn(B, T, D)
check("sigreg_input 3D == transpose", torch.equal(sigreg_input(e3, 7), e3.transpose(0, 1)))

# 3) Init en cero => el NCA es la identidad
print("[nca]")
nca = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(3, 3)).eval()
x, c = torch.randn(2, 3, N, D), torch.randn(2, 3, D)
check("init cero => identidad", torch.allclose(nca(x, c), x))

# 4) Eval determinista y only_last consistente
with torch.no_grad():
    for p in nca.fc2.parameters():
        p.normal_(0, 0.05)
    y1, y2 = nca(x, c), nca(x, c)
    check("eval determinista", torch.equal(y1, y2))
    check("only_last == último de full", torch.allclose(nca(x, c, only_last=True), y1[:, -1:], atol=1e-6))

# 5) Train estocástico; daño cambia la salida
nca.train()
check("train estocástico", not torch.equal(nca(x, c), nca(x, c)))
nca.damage_prob = 1.0
torch.manual_seed(1); dmg = nca(x, c)
nca.damage_prob = 0.0
torch.manual_seed(1); clean = nca(x, c)
check("daño latente altera la predicción", not torch.allclose(dmg, clean))

# 6) La acción importa (tras perturbar AdaLN)
nca.eval()
with torch.no_grad():
    nca.adaLN[-1].weight.normal_(0, 0.1)
    check("condicionado en la acción", not torch.allclose(nca(x, c), nca(x, -c)))

# 7) Grad checkpoint da el mismo gradiente
print("[grad checkpoint]")
for gc in (False, True):
    m = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(3, 3), grad_checkpoint=gc)
    torch.manual_seed(0)
    m.load_state_dict(nca.state_dict())
    xx = x.clone().requires_grad_(True)
    torch.manual_seed(2)
    m(xx, c).pow(2).sum().backward()
    if not gc:
        ref = xx.grad.clone()
check("checkpoint == sin checkpoint", torch.allclose(xx.grad, ref, atol=1e-5))

# 8) Planning: rollout + get_cost -> (B, S)
print("[planning]")
model.eval()
for H in (1, 3):
    Bp, S, Tp = 2, 5, H + 4
    info = {
        "pixels": torch.randn(Bp, S, H, 3, IMG, IMG),
        "goal": torch.randn(Bp, S, H, 3, IMG, IMG),
        "action": torch.randn(Bp, S, H, A_RAW),
    }
    with torch.no_grad():
        cost = model.get_cost(info, torch.randn(Bp, S, Tp, A_RAW))
    check(f"get_cost (B,S) con historia {H}", cost.shape == (Bp, S))
    check(f"rollout (B,S,T+1,N,D) con historia {H}",
          info["predicted_emb"].shape == (Bp, S, Tp + 1, N, D))

# 9) Gate por celda
print("[gate]")
gnca = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(3, 3), gate=True).eval()
y = gnca(x, c)
check("gate: init cero => identidad", torch.allclose(y, x))
check("gate: aux gate_map (B,T,N)", gnca.aux["gate_map"].shape == (2, 3, N))
check("gate: arranca abierto (sigmoid(2))",
      abs(gnca.aux["gate_mean"].item() - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5)
with torch.no_grad():
    gnca.fc2.weight.normal_(0, 0.05)
    check("gate abierto => cambia la salida", not torch.allclose(gnca(x, c), x))
    gnca.gate_head.bias.fill_(-30.0)
    check("gate cerrado => identidad aunque fc2 != 0", torch.allclose(gnca(x, c), x, atol=1e-6))
    gnca.gate_head.bias.fill_(2.0)
    gnca.gate_head.weight.normal_(0, 0.5)
    gnca(x, c)
    check("gate depende del estado local (no uniforme)", gnca.aux["gate_map"].std() > 1e-3)
check("sin gate => aux vacío", (nca(x, c) is not None) and nca.aux == {})

for gc in (False, True):
    m = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(3, 3), gate=True, grad_checkpoint=gc).train()
    m.load_state_dict(gnca.state_dict())
    xx = x.clone().requires_grad_(True)
    torch.manual_seed(3)
    (m(xx, c).pow(2).sum() + m.aux["gate_mean"]).backward()
    if not gc:
        ref = xx.grad.clone()
check("gate + checkpoint == sin checkpoint", torch.allclose(xx.grad, ref, atol=1e-5))

# 10) Dinámica inversa + spatial_losses (como en lejepa_forward)
print("[idm + spatial_losses]")
from types import SimpleNamespace  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

from jepa_spatial import InverseDynamicsHead, spatial_losses  # noqa: E402

idm = InverseDynamicsHead(latent_dim=D, action_dim=A_RAW, hidden_dim=32)
h0, h1 = torch.randn(2, 3, N, D), torch.randn(2, 3, N, D)
for pool, gmap in [("attn", None), ("mean", None), ("gate", torch.rand(2, 3, N))]:
    idm.pool = pool
    check(f"idm pool={pool} -> (B,T,A)", idm(h0, h1, gmap).shape == (2, 3, A_RAW))
idm.pool = "gate"
onehot = torch.zeros(2, 3, N)
onehot[..., 5] = 1.0
tok = idm.token_mlp(torch.cat([h0, h1 - h0], -1))[..., 5, :]
check("pool=gate con gate one-hot lee ese token", torch.allclose(idm(h0, h1, onehot), idm.head(tok), atol=1e-5))
idm.pool = "auto"

cfg = OmegaConf.create({"loss": {"gate_l1": {"weight": 0.01}, "idm": {"weight": 1.0}},
                        "diag": {"moving_frac": 0.1}})
dummy = SimpleNamespace(predictor=SimpleNamespace(aux={}))
e = torch.randn(2, 4, N, D)
_, logs = spatial_losses(dummy, {}, e, e[:, :3], e[:, 1:], cfg)
check("pred = copiar => moving_ratio == 1", abs(logs["moving_ratio"].item() - 1.0) < 1e-5)
check("copy_loss sin gate ni idm => sin pérdidas extra", "idm_loss" not in logs and "gate_mean" not in logs)

full = build(idm=InverseDynamicsHead(latent_dim=D, action_dim=A_RAW, hidden_dim=32),
             steps=(2, 3), gate=True).train()
opt = torch.optim.SGD(full.parameters(), lr=0.05)
for it in range(3):
    opt.zero_grad()
    b = {"pixels": torch.randn(B, T, 3, IMG, IMG), "action": torch.randn(B, T, A_RAW)}
    out = full.encode(b)
    emb = out["emb"]
    pred = full.predict(emb[:, :HS], out["act_emb"][:, :HS])
    extra, logs = spatial_losses(full, b, emb, pred, emb[:, 1:], cfg)
    loss = (pred - emb[:, 1:]).pow(2).mean() + 0.09 * sigreg(sigreg_input(emb, B)) + extra
    loss.backward()
    if it < 2:
        opt.step()
want = {"copy_loss", "pred_loss_moving", "copy_loss_moving", "moving_ratio", "gate_mean",
        "gate_moving_ratio", "gate_l1_loss", "idm_loss"}
check("logs completos", want <= set(logs))
check("logs finitos", all(torch.isfinite(v).all() for v in logs.values()))
check("logs sin grafo", all(not v.requires_grad for v in logs.values()))
for name, params in [("encoder (vía idm)", full.encoder.parameters()),
                     ("inverse_dynamics", full.inverse_dynamics.parameters()),
                     ("gate_head", full.predictor.gate_head.parameters())]:
    check(f"grad llega a {name}", any(p.grad is not None and p.grad.abs().sum() > 0 for p in params))

# idm aislada: el gradiente de idm_loss llega al gate (pool=gate)
opt.zero_grad()
out = full.encode(b)
full.predict(out["emb"][:, :HS], out["act_emb"][:, :HS])
gm = full.predictor.aux["gate_map"]
F_idm = full.inverse_dynamics(out["emb"][:, :HS], out["emb"][:, 1:HS + 1], gm)
torch.nn.functional.mse_loss(F_idm, b["action"][:, :HS]).backward()
check("idm_loss empuja al gate", full.predictor.gate_head.weight.grad.abs().sum() > 0)

# planning sigue funcionando con gate + idm
full.eval()
info = {"pixels": torch.randn(2, 3, 3, 3, IMG, IMG), "goal": torch.randn(2, 3, 3, 3, IMG, IMG),
        "action": torch.randn(2, 3, 3, A_RAW)}
with torch.no_grad():
    check("get_cost con gate + idm", full.get_cost(info, torch.randn(2, 3, 6, A_RAW)).shape == (2, 3))

# 10a) gate_scope: "latent" apaga sólo el latente, los ocultos siguen
print("[gate_scope]")
for scope in ("all", "latent"):
    m = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(1, 1), gate=True, gate_scope=scope).eval()
    with torch.no_grad():
        m.fc2.weight.normal_(0, 0.05)
        m.gate_head.bias.fill_(-30.0)  # gate cerrado
        s0 = torch.randn(2, m.state_dim, GH, GW)
        sh, sc = m._modulation(c[:, 0])
        s1, _ = m.nca_step(s0, sh, sc, 0)
    check(f"gate_scope={scope}: latente congelado", torch.allclose(s1[:, :D], s0[:, :D], atol=1e-6))
    moved = not torch.allclose(s1[:, D:], s0[:, D:], atol=1e-6)
    check(f"gate_scope={scope}: ocultos {'se actualizan' if scope == 'latent' else 'congelados'}",
          moved == (scope == "latent"))

# 10b) eval_mode="fixed_mask": determinista, compartida por el batch, != mean-field
print("[eval_mode]")
fm = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(4, 4), eval_mode="fixed_mask").eval()
mf = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(4, 4), eval_mode="mean_field").eval()
with torch.no_grad():
    fm.fc2.weight.normal_(0, 0.05)
    mf.load_state_dict(fm.state_dict())
    a1, a2 = fm(x, c), fm(x, c)
    check("fixed_mask determinista", torch.equal(a1, a2))
    check("fixed_mask != mean_field", not torch.allclose(a1, mf(x, c)))
    check("fixed_mask: muestra sola == dentro del batch", torch.allclose(fm(x[1:], c[1:]), a1[1:], atol=1e-6))
    msk = fm._eval_mask(torch.zeros(1, 1, 64, 64), 0)
    check("fixed_mask: fracción ~ fire_rate", abs(msk.mean().item() - fm.fire_rate) < 0.1)
    check("fixed_mask: only_last == último de full", torch.allclose(fm(x, c, only_last=True), a1[:, -1:], atol=1e-6))

# 10c) rollout(): sin persistir ocultos == encadenar forward(only_last)
print("[rollout nca]")
for mode in ("mean_field", "fixed_mask"):
    r = LatentNCAPredictor(latent_dim=D, act_dim=D, steps=(3, 3), gate=True, eval_mode=mode).eval()
    with torch.no_grad():
        r.fc2.weight.normal_(0, 0.05)
        r.hidden_init.weight.normal_(0, 0.2)
        r.adaLN[-1].weight.normal_(0, 0.1)
        cs = torch.randn(2, 4, D)
        for hs in (1, 3):
            emb_c = x[:, -hs:]
            chain = [emb_c[:, i] for i in range(hs)]
            for m in range(cs.size(1)):
                ctx_m = torch.stack(chain[-hs:], 1)
                cm = cs[:, m:m + 1].expand(-1, ctx_m.size(1), -1)
                chain.append(r(ctx_m, cm, only_last=True)[:, 0])
            ref_chain = torch.stack(chain[hs:], 1)
            got = r.rollout(emb_c, cs, persist_hidden=False)
            check(f"rollout {mode} hist={hs}: sin persistir == encadenado",
                  torch.allclose(got, ref_chain, atol=1e-5))
        per = r.rollout(x, cs, persist_hidden=True)
        nop = r.rollout(x, cs, persist_hidden=False)
        check(f"rollout {mode}: primer paso igual con y sin persistir", torch.allclose(per[:, 0], nop[:, 0], atol=1e-6))
        check(f"rollout {mode}: persistir ocultos cambia los pasos siguientes", not torch.allclose(per[:, 1:], nop[:, 1:]))
        check(f"rollout {mode}: aux_rollout gate_map (B,M,N)", r.aux_rollout["gate_map"].shape == (2, 4, N))
        check(f"rollout {mode}: no pisa aux del forward", r.aux.get("gate_map") is None or r.aux["gate_map"].shape[1] != 4)

# 10d) SpatialJEPA.stateful_rollout: sin persistir == bucle de planning original
print("[stateful planning]")
sj = build(steps=(3, 3), gate=True).eval()
with torch.no_grad():
    sj.predictor.fc2.weight.normal_(0, 0.05)
    sj.predictor.adaLN[-1].weight.normal_(0, 0.1)
    for H in (1, 3):
        info = {"pixels": torch.randn(2, 3, H, 3, IMG, IMG), "goal": torch.randn(2, 3, H, 3, IMG, IMG),
                "action": torch.randn(2, 3, H, A_RAW)}
        cand = torch.randn(2, 3, H + 4, A_RAW)
        sj.stateful_rollout = False
        c_loop = sj.get_cost(dict(info), cand.clone())
        sj.stateful_rollout, sj.rollout_persist_hidden = True, False
        c_state = sj.get_cost(dict(info), cand.clone())
        check(f"stateful (sin persistir) == bucle original, historia {H}", torch.allclose(c_loop, c_state, rtol=1e-4, atol=1e-3))
        sj.rollout_persist_hidden = True
        out_info = dict(info)
        c_pers = sj.get_cost(out_info, cand.clone())
        check(f"stateful (persistiendo) -> (B,S) y rollout (B,S,T+1,N,D), historia {H}",
              c_pers.shape == (2, 3) and out_info["predicted_emb"].shape == (2, 3, H + 5, N, D))
    sj.stateful_rollout = False

# 10e) Estandarización por canal de los tokens (sin projector)
print("[token stats]")
import os, tempfile, warnings  # noqa: E401,E402

tmp = tempfile.mkdtemp()
stats_path = os.path.join(tmp, "stats.pt")
mu, sd = torch.randn(ENC_DIM), torch.rand(ENC_DIM) + 0.5
torch.save({"mean": mu, "std": sd}, stats_path)


def build_raw(path, **kw):
    return SpatialJEPA(
        encoder=DummyEncoder(),
        predictor=LatentNCAPredictor(latent_dim=ENC_DIM, act_dim=D, steps=(2, 2), gate=True),
        action_encoder=Embedder(input_dim=A_RAW, emb_dim=D),
        projector=None, pred_proj=None, patch_size=PATCH,
        token_stats_path=path, token_dim=ENC_DIM, **kw,
    )


raw = build_raw(stats_path).eval()
plain = build_raw(None).eval()
plain.encoder.load_state_dict(raw.encoder.state_dict())
px = torch.randn(2, 2, 3, IMG, IMG)
with torch.no_grad():
    e_norm = raw.encode({"pixels": px})["emb"]
    e_raw = plain.encode({"pixels": px})["emb"]
check("sin projector: emb = tokens del encoder", e_raw.shape == (2, 2, N, ENC_DIM))
check("stats: emb = (tokens - mean) / std", torch.allclose(e_norm, (e_raw - mu) / sd, atol=1e-5))
check("stats: buffers en el state_dict", {"token_mean", "token_std"} <= set(raw.state_dict()))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    reloaded = build_raw(os.path.join(tmp, "no_existe.pt"))
reloaded.load_state_dict(raw.state_dict())
check("stats: recarga sin archivo vía state_dict", torch.equal(reloaded.token_mean, mu))

# 10f) idm_pred: el gradiente llega al predictor, no a la cabeza IDM
print("[idm_pred + rollout loss]")
cfg_ip = OmegaConf.create({"history_size": HS, "loss": {"idm": {"weight": 0.0}, "idm_pred": {"weight": 1.0}},
                           "diag": {"moving_frac": 0.1}})
mi = build(idm=InverseDynamicsHead(latent_dim=D, action_dim=A_RAW, hidden_dim=32), steps=(2, 2), gate=True).train()
b = {"pixels": torch.randn(B, T, 3, IMG, IMG), "action": torch.randn(B, T, A_RAW)}
out = mi.encode(b)
pred = mi.predict(out["emb"][:, :HS], out["act_emb"][:, :HS])
extra, logs = spatial_losses(mi, b, out["emb"], pred, out["emb"][:, 1:HS + 1], cfg_ip, act_emb=out["act_emb"])
extra.backward()
check("idm_pred_loss logueada", "idm_pred_loss" in logs)
check("idm_pred: grad llega al predictor (fc2)", mi.predictor.fc2.weight.grad.abs().sum() > 0)
check("idm_pred: cabeza IDM sin gradiente",
      all(p.grad is None or p.grad.abs().sum() == 0 for p in mi.inverse_dynamics.parameters()))

# 10g) loss.rollout
for steps_r in (2, 3):
    cfg_r = OmegaConf.create({"history_size": HS, "loss": {"rollout": {"weight": 1.0, "steps": steps_r}},
                              "diag": {"moving_frac": 0.1}})
    mr = build(steps=(2, 2), gate=True).train()
    mr.zero_grad()
    out = mr.encode(dict(b))
    pred = mr.predict(out["emb"][:, :HS], out["act_emb"][:, :HS])
    extra, logs = spatial_losses(mr, b, out["emb"], pred, out["emb"][:, 1:HS + 1], cfg_r, act_emb=out["act_emb"])
    extra.backward()
    check(f"rollout steps={steps_r}: pérdida finita", torch.isfinite(logs["rollout_loss"]).item())
    check(f"rollout steps={steps_r}: grad llega al predictor", mr.predictor.fc2.weight.grad.abs().sum() > 0)
    check(f"rollout steps={steps_r}: no pisa el gate_map del forward",
          mr.predictor.aux["gate_map"].shape == (B, HS, N))
try:
    cfg_bad = OmegaConf.create({"history_size": HS, "loss": {"rollout": {"weight": 1.0, "steps": T}}})
    spatial_losses(mr, b, out["emb"], pred, out["emb"][:, 1:HS + 1], cfg_bad, act_emb=out["act_emb"])
    check("rollout steps > T-1 falla con mensaje claro", False)
except AssertionError as e:
    check("rollout steps > T-1 falla con mensaje claro", "num_steps" in str(e))

# 11) Probe lineal (sin swm)
print("[probe]")
import numpy as np  # noqa: E402

from viz.probe_latent import _ridge_fit_predict, episode_split, expand_angles, ridge_probe  # noqa: E402

g = torch.Generator().manual_seed(0)
Xa, Ya = torch.randn(30, 50, generator=g).double(), torch.randn(30, 2, generator=g).double()
Xe = torch.randn(7, 50, generator=g).double()
primal = Xe @ torch.linalg.solve(Xa.T @ Xa + 3.0 * torch.eye(50, dtype=torch.double), Xa.T @ Ya)
check("ridge dual == primal", torch.allclose(_ridge_fit_predict(Xa, Ya, Xe, 3.0), primal, atol=1e-8))

split = torch.as_tensor(np.repeat([0, 1, 2], [420, 90, 90]))
for d in (40, 1500):  # forma primal y dual
    # features con baja dimensión intrínseca (como tokens reales): X = Z A + ruido
    Z = torch.randn(600, 5, generator=g)
    X = Z @ torch.randn(5, d, generator=g) + 0.1 * torch.randn(600, d, generator=g)
    Yl = Z[:, :3] @ torch.randn(3, 2, generator=g) + 0.05 * torch.randn(600, 2, generator=g)
    check(f"probe d={d}: señal lineal => R² alto", ridge_probe(X, Yl, split)["r2_test_mean"] > 0.9)
    Yn = torch.randn(600, 2, generator=g)
    check(f"probe d={d}: ruido => R² ~ 0", ridge_probe(X, Yn, split)["r2_test_mean"] < 0.1)

off = np.array([0, 50, 120, 160, 300, 330, 400, 470, 520, 600])
ln = np.diff(np.append(off, 650))
rows, sp = episode_split(off, ln, 200, seed=0)
ep_of = np.searchsorted(off, rows, side="right") - 1
check("split por episodio sin fugas",
      all(len(set(ep_of[sp == a]) & set(ep_of[sp == b])) == 0 for a, b in [(0, 1), (0, 2), (1, 2)]))
check("split sin filas repetidas", len(np.unique(rows)) == len(rows))
Ya, nm = expand_angles(torch.tensor([[1.0, 2.0, 0.5]]), {2})
check("expand_angles -> sin/cos", nm == ["0", "1", "2_sin", "2_cos"] and Ya.shape == (1, 4))

print("smoke_nca: todo OK")
