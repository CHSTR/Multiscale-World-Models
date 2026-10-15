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


def build(**nca_kw):
    norm = lambda d: nn.BatchNorm1d(d)  # noqa: E731
    return SpatialJEPA(
        encoder=DummyEncoder(),
        predictor=LatentNCAPredictor(latent_dim=D, act_dim=D, **nca_kw),
        action_encoder=Embedder(input_dim=A_RAW, emb_dim=D),
        projector=MLP(ENC_DIM, 128, D, norm_fn=norm),
        pred_proj=MLP(D, 128, D, norm_fn=norm),
        patch_size=PATCH,
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

print("smoke_nca: todo OK")
