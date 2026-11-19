"""Tests del replay priorizado + red de alucinación H (rama per-hnet).

Todo corre en CPU con un JEPA diminuto (encoder falso). ``train.py`` se importa
con stubs de stable_pretraining / stable_worldmodel si no están instalados (sólo
se usa ``lejepa_forward``, que no los toca).
"""
import copy
import importlib
import os
import sys
import types
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from module import MLP, ARPredictor, Embedder, SIGReg
from prioritized_replay import (
    HallucinationNet,
    IndexedDataset,
    PrioritizedReplayCallback,
    PrioritizedSampler,
    PriorityBuffer,
    ReplayConfig,
    ReplayState,
    hnet_loss,
    importance_weights,
    resolve_index,
    rollout_predictions,
    spearman,
    split_time,
    valid_windows,
)

HS, K, D, A = 3, 5, 8, 4


# ---------------------------------------------------------------------- #
# Utilidades
# ---------------------------------------------------------------------- #
def _import_train():
    """Importa train.py; si faltan spt/swm (CI sin el stack) usa stubs vacíos.

    Los stubs sólo existen durante el import (después se quitan de sys.modules),
    así no contaminan a otros tests que importen los paquetes reales.
    """
    if "train" in sys.modules:
        return sys.modules["train"]
    stubbed = []

    def stub(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            stubbed.append(name)
        return sys.modules[name]

    try:
        importlib.import_module("stable_pretraining")
    except Exception:
        stub("stable_pretraining").data = stub("stable_pretraining.data")
    try:
        importlib.import_module("stable_worldmodel.data.formats.hdf5")
    except Exception:
        for name in ("stable_worldmodel", "stable_worldmodel.data",
                     "stable_worldmodel.data.formats", "stable_worldmodel.data.formats.hdf5"):
            stub(name)
    try:
        return importlib.import_module("train")
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


class FakeEncoder(nn.Module):
    def __init__(self, hw=8, d=16):
        super().__init__()
        self.lin = nn.Linear(3 * hw * hw, d)

    def forward(self, x, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.lin(x.flatten(1))[:, None, :])


def tiny_jepa(seed=0):
    from jepa import JEPA
    torch.manual_seed(seed)
    return JEPA(
        encoder=FakeEncoder(),
        predictor=ARPredictor(num_frames=HS, depth=1, heads=2, mlp_dim=16, input_dim=D,
                              hidden_dim=D, output_dim=D, dim_head=4, dropout=0.1),
        action_encoder=Embedder(input_dim=A, emb_dim=D),
        projector=MLP(16, 32, D, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(D, 32, D, norm_fn=nn.BatchNorm1d),
    )


def make_batch(B=6, T=HS + K, seed=1, prob=None):
    g = torch.Generator().manual_seed(seed)
    b = {
        "pixels": torch.randn(B, T, 3, 8, 8, generator=g),
        "action": torch.randn(B, T, A, generator=g),
        "idx": torch.arange(B),
        "sample_prob": torch.full((B,), float("nan"), dtype=torch.float64) if prob is None
        else prob.double(),
    }
    return b


def train_cfg():
    return OmegaConf.create({
        "history_size": HS, "num_preds": 1,
        "loss": {"sigreg": {"weight": 0.09, "kwargs": {"knots": 17, "num_proj": 64}}},
    })


class FakeModule:
    def __init__(self, model):
        self.model = model
        self.sigreg = SIGReg(knots=17, num_proj=64)
        self.logged = {}

    def log_dict(self, d, **kw):
        self.logged.update(d)


def rcfg(**kw):
    base = dict(enabled=True, horizon=K, warmup_epochs=0.0, chunk_batches=1, log_every=1,
                dump_topk=4)
    base.update(kw)
    c = ReplayConfig(**base)
    c.validate()
    return c


# ---------------------------------------------------------------------- #
# Configuración
# ---------------------------------------------------------------------- #
def test_config_from_hydra_presets():
    from hydra import compose, initialize_config_dir
    # stable_pretraining registra el resolver `eval` al importarse; hacerlo acá
    # primero evita el ValueError por doble registro cuando otro test lo importa.
    try:
        import stable_pretraining  # noqa: F401
    except Exception:
        pass
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(config_dir=os.path.abspath("config/train"), version_base=None):
        cfg = compose(config_name="lewm", overrides=["replay=per_focal", "data=ogb",
                                                     "model=dinov3_lora"])
    c = ReplayConfig.from_cfg(cfg.replay)
    assert c.enabled and c.sampler == "prioritized" and c.beta_start == 0.0
    assert c.hnet_input == "state_action" and c.horizon == 5
    assert cfg.data.dataset.num_steps == cfg.history_size + c.horizon
    with initialize_config_dir(config_dir=os.path.abspath("config/train"), version_base=None):
        off = compose(config_name="lewm", overrides=["data=ogb"])
    assert off.replay.enabled is False and off.data.dataset.num_steps == 4


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ReplayConfig.from_cfg({"enabled": True, "samplr": "uniform"})
    with pytest.raises(ValueError):
        ReplayConfig.from_cfg({"enabled": True, "sampler": "greedy"})
    with pytest.raises(ValueError):
        ReplayConfig.from_cfg({"enabled": True, "priority": "surprise", "hnet": {"enabled": False}})


# ---------------------------------------------------------------------- #
# Buffer, dataset y sampler
# ---------------------------------------------------------------------- #
def test_buffer_update_duplicates_and_roundtrip():
    buf = PriorityBuffer(10, K)
    idx = torch.tensor([1, 1, 4])
    buf.update(idx, torch.tensor([2.0, 3.0, 5.0]), torch.rand(3, K),
               torch.tensor([True, True, False]), step=7)
    assert buf.visits[1] == 2 and buf.visits[4] == 1 and buf.visits[0] == 0
    assert bool(buf.valid[1]) and not bool(buf.valid[4])
    assert buf.last_step[1] == 7
    sd = buf.state_dict()
    other = PriorityBuffer(10, K)
    other.load_state_dict(sd)
    for k in PriorityBuffer._FIELDS:
        assert torch.equal(getattr(other, k), getattr(buf, k))
    with pytest.raises(ValueError):
        PriorityBuffer(11, K).load_state_dict(sd)


def test_sampler_uniform_during_warmup():
    st = ReplayState(rcfg(sampler="prioritized", warmup_epochs=1.0), n=20)
    s = PrioritizedSampler(st, num_samples=20, batch_size=4)
    st.buffer.priority[:] = torch.arange(20).float()
    st.buffer.visits[:] = 1
    st.step = s.warmup_steps - 1
    P = s.probabilities()
    assert torch.allclose(P, torch.full((20,), 1 / 20, dtype=torch.float64))
    st.step = s.warmup_steps
    assert not torch.allclose(s.probabilities(), P)


def test_sampler_prioritized_formula_invalid_and_unvisited():
    cfg = rcfg(sampler="prioritized", alpha=0.5, eps_uniform=0.2, clip_quantile=1.0)
    st = ReplayState(cfg, n=6)
    s = PrioritizedSampler(st, num_samples=6, batch_size=2)
    buf = st.buffer
    buf.priority[:] = torch.tensor([1.0, 4.0, 9.0, 16.0, 0.0, 0.0])
    buf.visits[:4] = 1                           # 4 y 5 no visitadas
    buf.valid[3] = False                         # inválida: P = 0
    P = s.probabilities()
    q = 9.0                                      # cuantil 1.0 de las visitadas válidas
    p = torch.tensor([1.0, 4.0, 9.0, 0.0, q, q], dtype=torch.float64)
    w = p.sqrt() * torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.float64)
    uni = torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.float64) / 5
    expected = 0.8 * w / w.sum() + 0.2 * uni
    assert torch.allclose(P, expected)
    assert P[3] == 0
    assert P[4] == P[2]                          # no visitada = prioridad de recorte


def test_sampler_draws_match_probabilities():
    cfg = rcfg(sampler="prioritized", alpha=1.0, eps_uniform=0.0, clip_quantile=1.0)
    st = ReplayState(cfg, n=5)
    st.buffer.priority[:] = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0])
    st.buffer.visits[:] = 1
    n = 40_000
    s = PrioritizedSampler(st, num_samples=n, batch_size=100,
                           generator=torch.Generator().manual_seed(0))
    draws = list(iter(s))
    idx = torch.tensor([d[0] for d in draws])
    freq = torch.bincount(idx, minlength=5).double() / n
    P = s.probabilities()
    assert torch.allclose(freq, P, atol=0.01)
    assert all(abs(p - float(P[i])) < 1e-12 for i, p in draws[:50])   # (i, P_i) exacto


def test_sampler_shuffled_same_distribution_other_indices():
    cfg = rcfg(sampler="shuffled", alpha=1.0, eps_uniform=0.0, clip_quantile=1.0, seed=3)
    st = ReplayState(cfg, n=50)
    st.buffer.priority[:] = torch.arange(1, 51).float()
    st.buffer.visits[:] = 1
    s = PrioritizedSampler(st, num_samples=50, batch_size=5)
    P_shuf = s.probabilities()
    st.cfg = rcfg(sampler="prioritized", alpha=1.0, eps_uniform=0.0, clip_quantile=1.0)
    P_prio = s.probabilities()
    assert torch.allclose(P_shuf.sort().values, P_prio.sort().values)
    assert not torch.allclose(P_shuf, P_prio)


class DictDS(torch.utils.data.Dataset):
    def __len__(self):
        return 30

    def __getitem__(self, i):
        return {"x": torch.tensor(float(i))}


def test_indexed_dataset_dataloader_workers():
    sub = torch.utils.data.Subset(DictDS(), list(range(10, 30)))
    ds = IndexedDataset(sub)
    st = ReplayState(rcfg(sampler="uniform"), n=len(ds))
    s = PrioritizedSampler(st, num_samples=len(ds), batch_size=4)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, sampler=s, num_workers=2, drop_last=True)
    for b in dl:
        assert torch.equal(b["x"], (b["idx"] + 10).float())
        assert torch.allclose(b["sample_prob"], torch.full((4,), 1 / 20, dtype=torch.float64))
    assert resolve_index(ds, 3) == 13
    assert torch.isnan(ds[2]["sample_prob"])      # clave entera (val)
    # Fetch por lotes y proxy de atributos (set_pl_trainer, gpu_transform, ...).
    items = ds.__getitems__([(0, 0.5), (5, 0.25)])
    assert [int(it["idx"]) for it in items] == [0, 5] and float(items[1]["x"]) == 15.0
    sub.custom_attr = "ok"
    assert ds.custom_attr == "ok"
    import pickle
    ds2 = pickle.loads(pickle.dumps(ds))
    assert float(ds2[(4, 0.1)]["x"]) == 14.0


# ---------------------------------------------------------------------- #
# Pesos IS y SIGReg ponderado
# ---------------------------------------------------------------------- #
def test_importance_weights():
    p = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    assert importance_weights(p, 0.0) is None
    w = importance_weights(p, 1.0)
    assert torch.allclose(w.mean(), torch.tensor(1.0))
    assert torch.allclose(w / w[0], torch.tensor([1.0, 0.5, 0.25]))
    w5 = importance_weights(p, 0.5)
    assert torch.allclose(w5 / w5[0], torch.tensor([1.0, 0.5 ** 0.5, 0.25 ** 0.5]))


def test_sigreg_weighted_matches_unweighted_and_duplicates():
    sig = SIGReg(knots=17, num_proj=32)
    x = torch.randn(2, 6, 5)                       # (T, B, D)
    torch.manual_seed(0); a = sig(x)
    torch.manual_seed(0); b = sig(x, weights=torch.ones(6))
    assert torch.allclose(a, b, atol=1e-6)
    # Peso 2 en la muestra 0 == duplicarla (el estadístico escala con n).
    x_dup = torch.cat([x, x[:, :1]], dim=1)
    w = torch.tensor([2.0, 1, 1, 1, 1, 1])
    torch.manual_seed(1); c = sig(x, weights=w) / 6
    torch.manual_seed(1); d = sig(x_dup) / 7
    assert torch.allclose(c, d, atol=1e-6)


# ---------------------------------------------------------------------- #
# Etiquetas de K horizontes
# ---------------------------------------------------------------------- #
def test_valid_windows_and_split_time():
    a = torch.randn(3, HS + K, A)
    a[0, -1] = float("nan")                        # última acción: no se usa
    a[1, HS] = float("nan")                        # usada por el rollout
    v = valid_windows(a, HS + K - 1)
    assert v.tolist() == [True, False, True]
    b = {"pixels": torch.zeros(3, HS + K, 3, 8, 8), "action": a, "idx": torch.arange(3),
         "batch_idx": 0}
    s = split_time(b, HS + 1)
    assert s["pixels"].shape[1] == HS + 1 and s["action"].shape[1] == HS + 1
    assert s["idx"] is b["idx"] and s["batch_idx"] == 0


def test_rollout_labels_match_jepa_rollout():
    """Las etiquetas usan exactamente el rollout que usa CEM (JEPA.rollout)."""
    model = tiny_jepa().eval()
    B = 4
    pix = torch.randn(B, 1, HS, 3, 8, 8)
    acts = torch.randn(B, 1, HS + K - 1, A)
    info = model.rollout({"pixels": pix.clone()}, acts.clone(), history_size=HS)
    ref = info["predicted_emb"][:, 0, HS:]                       # (B, K, D)
    with torch.no_grad():
        z = model.encode({"pixels": pix[:, 0], "action": acts[:, 0, :HS]})["emb"]
        a_emb = model.action_encoder(acts[:, 0])
        got = rollout_predictions(model, z, a_emb, K, HS)
    assert torch.allclose(got, ref, atol=1e-5)


def test_forward_main_loss_equals_baseline_and_fills_labels():
    train = _import_train()
    model = tiny_jepa().eval()
    fm = FakeModule(model)
    cfg = train_cfg()
    full = make_batch()
    short = {k: (v[:, :HS + 1] if torch.is_tensor(v) and v.ndim >= 2 else v)
             for k, v in full.items()}
    torch.manual_seed(0)
    base = train.lejepa_forward(fm, dict(short), "fit", cfg)
    st = ReplayState(rcfg(sampler="uniform"), n=6)
    torch.manual_seed(0)
    rep = train.lejepa_forward(fm, dict(full), "fit", cfg, replay=st)
    assert torch.allclose(base["loss"], rep["loss"], atol=1e-6)
    assert torch.allclose(base["pred_loss"], rep["pred_loss"], atol=1e-6)
    last = st.last
    assert last["err"].shape == (6, K) and torch.isfinite(last["err"]).all()
    assert last["z_ctx"].shape == (6, HS, D) and last["act_k"].shape == (6, K, D)
    # e_1 == error teacher-forced de la última posición de contexto.
    with torch.no_grad():
        enc = model.encode({"pixels": full["pixels"][:, :HS + 1],
                            "action": full["action"][:, :HS + 1]})
        tf = model.predict(enc["emb"][:, :HS], enc["act_emb"][:, :HS])[:, -1]
        e1 = (tf - enc["emb"][:, HS]).pow(2).mean(-1)
    assert torch.allclose(last["err"][:, 0], e1, atol=1e-5)
    assert "fit/replay_err_k1" in fm.logged


def test_forward_is_weights_and_frozen_bn_stats():
    train = _import_train()
    cfg = train_cfg()
    prob = torch.tensor([0.05, 0.1, 0.1, 0.2, 0.25, 0.3])
    full = make_batch(prob=prob)
    # Pesos IS en pred_loss (modelo en eval: determinista).
    model = tiny_jepa().eval()
    fm = FakeModule(model)
    st = ReplayState(rcfg(sampler="prioritized"), n=6)
    st.beta = 1.0
    out = train.lejepa_forward(fm, dict(full), "fit", cfg, replay=st)
    with torch.no_grad():
        enc = model.encode({"pixels": full["pixels"][:, :HS + 1],
                            "action": full["action"][:, :HS + 1]})
        pred = model.predict(enc["emb"][:, :HS], enc["act_emb"][:, :HS])
        per = (pred - enc["emb"][:, 1:HS + 1]).pow(2).mean(dim=(1, 2))
    w = 1.0 / prob.double()
    expected = float((w * per.double()).sum() / w.sum())
    assert abs(float(out["pred_loss"]) - expected) < 1e-5
    # En train, la pasada de etiquetado no toca las running stats de BatchNorm.
    m1 = tiny_jepa().train()
    m2 = copy.deepcopy(m1)
    torch.manual_seed(5)
    train.lejepa_forward(FakeModule(m1), dict(full), "fit", cfg,
                         replay=ReplayState(rcfg(sampler="uniform"), n=6))
    short = {k: (v[:, :HS + 1] if torch.is_tensor(v) and v.ndim >= 2 else v)
             for k, v in full.items()}
    torch.manual_seed(5)
    train.lejepa_forward(FakeModule(m2), dict(short), "fit", cfg)
    for (n1, b1), (_, b2) in zip(m1.named_buffers(), m2.named_buffers()):
        assert torch.allclose(b1.float(), b2.float(), atol=1e-6), n1


def test_forward_rejects_short_windows():
    train = _import_train()
    model = tiny_jepa().eval()
    st = ReplayState(rcfg(sampler="uniform"), n=6)
    with pytest.raises(ValueError, match="num_steps"):
        train.lejepa_forward(FakeModule(model), make_batch(T=HS + 2), "fit", train_cfg(),
                             replay=st)


# ---------------------------------------------------------------------- #
# H
# ---------------------------------------------------------------------- #
def test_hnet_learns_synthetic_error_ranking():
    torch.manual_seed(0)
    net = HallucinationNet(D, D, HS, K, mode="state_action", hidden=64, depth=2)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    u = torch.randn(HS * D)

    def sample(n):
        z = torch.randn(n, HS, D)
        a = torch.randn(n, K, D)
        score = (z.flatten(1) @ u / (HS * D) ** 0.5)[:, None] + 0.3 * a[:, :, 0]
        err = torch.exp(score + 0.1 * torch.randn(n, K))
        return z, a, err

    for _ in range(400):
        z, a, err = sample(128)
        loss, _, _ = hnet_loss(net(z, a), err, None)
        opt.zero_grad(); loss.backward(); opt.step()
    z, a, err = sample(512)
    with torch.no_grad():
        lp = net(z, a)
    assert spearman(lp[:, 0], err[:, 0]) > 0.8


def test_callback_updates_buffer_trains_hnet_and_logs():
    st = ReplayState(rcfg(sampler="prioritized"), n=10)
    cb = PrioritizedReplayCallback(st, z_dim=D, a_dim=D, history_size=HS)
    logged = {}
    pl_module = SimpleNamespace(device=torch.device("cpu"),
                                log_dict=lambda d, **kw: logged.update(d))
    trainer = SimpleNamespace(global_step=0, estimated_stepping_batches=100,
                              sanity_checking=False, current_epoch=0)
    cb.on_fit_start(trainer, pl_module)
    before = [p.clone() for p in cb.hnet.parameters()]
    idx = torch.tensor([0, 3, 3, 7])
    st.last = {"stage": "fit", "idx": idx, "valid": torch.tensor([True, True, True, False]),
               "err": torch.rand(4, K) + 0.1, "z_ctx": torch.randn(4, HS, D),
               "act_k": torch.randn(4, K, D), "pred": torch.randn(4, K, D)}
    batch = {"sample_prob": torch.tensor([0.1, 0.2, 0.2, 0.1], dtype=torch.float64)}
    cb.on_train_batch_start(trainer, pl_module, batch, 0)
    cb.on_train_batch_end(trainer, pl_module, None, batch, 0)
    buf = st.buffer
    assert st.last is None
    assert buf.visits[3] == 2 and buf.visits[0] == 1 and buf.visits[1] == 0
    assert not bool(buf.valid[7]) and float(buf.priority[7]) == 0.0
    assert float(buf.priority[0]) > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, cb.hnet.parameters()))
    assert "hnet/loss" in logged and "replay/beta" in logged and "replay/frac_invalid" in logged
    # Estado del callback (resume): ida y vuelta.
    sd = cb.state_dict()
    st2 = ReplayState(rcfg(sampler="prioritized"), n=10)
    cb2 = PrioritizedReplayCallback(st2, z_dim=D, a_dim=D, history_size=HS)
    cb2.load_state_dict(sd)
    cb2.on_fit_start(trainer, pl_module)
    assert torch.equal(st2.buffer.priority, buf.priority)
    assert torch.allclose(st2.err_ema, st.err_ema)
    for a, b in zip(cb2.hnet.parameters(), cb.hnet.parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------- #
# Extremo a extremo con Lightning (fit + val + checkpoint + resume)
# ---------------------------------------------------------------------- #
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, n=48, T=HS + K, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.pixels = torch.randn(n, T, 3, 8, 8, generator=g)
        self.action = torch.randn(n, T, A, generator=g)
        self.action[::7, -1] = float("nan")          # borde inocuo (última acción)
        self.action[3::11, 2] = float("nan")         # borde que invalida la ventana

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, i):
        return {"pixels": self.pixels[i], "action": self.action[i]}


def _lightning_run(tmp_path, max_epochs, ckpt=None):
    pl = pytest.importorskip("lightning")
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    train = _import_train()

    class MiniModule(pl.LightningModule):
        """Imita spt.Module: optimización manual y forward(batch, stage)."""

        def __init__(self, model, cfg, replay):
            super().__init__()
            self.model = model
            self.sigreg = SIGReg(knots=17, num_proj=64)
            self.automatic_optimization = False
            self._fwd = partial(train.lejepa_forward, cfg=cfg, replay=replay)

        def forward(self, batch, stage):
            return self._fwd(self, batch, stage)

        def training_step(self, batch, batch_idx):
            state = self(batch, stage="fit")
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(state["loss"])
            opt.step()
            return state

        def validation_step(self, batch, batch_idx):
            return self(batch, stage="validate")

        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    full = WindowDataset()
    tr = IndexedDataset(torch.utils.data.Subset(full, list(range(36))))
    va = IndexedDataset(torch.utils.data.Subset(full, list(range(36, 48))))
    st = ReplayState(rcfg(sampler="prioritized", warmup_epochs=0.5, chunk_batches=2), n=len(tr))
    sampler = PrioritizedSampler(st, num_samples=len(tr), batch_size=6,
                                 generator=torch.Generator().manual_seed(0))
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=6, sampler=sampler, drop_last=True,
                                        num_workers=2, persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=6)
    cb = PrioritizedReplayCallback(st, train_dataset=tr, run_dir=tmp_path, z_dim=D, a_dim=D,
                                   history_size=HS)
    ckpt_cb = ModelCheckpoint(dirpath=tmp_path / "ckpt", save_last=True)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1,
                         logger=CSVLogger(tmp_path, name="logs"), callbacks=[cb, ckpt_cb],
                         num_sanity_val_steps=1, log_every_n_steps=1,
                         enable_progress_bar=False, enable_model_summary=False)
    torch.manual_seed(0)
    trainer.fit(MiniModule(tiny_jepa(), train_cfg(), st), dl_tr, dl_va, ckpt_path=ckpt)
    return st, cb, trainer


def test_lightning_fit_val_checkpoint_resume(tmp_path):
    import csv
    st, cb, trainer = _lightning_run(tmp_path, max_epochs=2)
    out = tmp_path / "replay"
    for name in ("hnet_epoch_1.pt", "hnet_epoch_2.pt", "topk_epoch_2.csv", "topk_epoch_2.png",
                 "calibration_epoch_2.csv", "buffer_final.pt"):
        assert (out / name).exists(), name
    buf = st.buffer
    assert buf.visited().float().mean() > 0.5
    bad = torch.zeros(36, dtype=torch.bool)
    bad[3::11] = True                               # índices inválidos del dataset
    assert not buf.valid[bad & buf.visited()].any()
    assert buf.valid[~bad & buf.visited()].all()
    metrics_files = list((tmp_path / "logs").rglob("metrics.csv"))
    header = next(csv.reader(open(metrics_files[0])))
    for key in ("hnet/val_spearman", "hnet/loss", "replay/ess", "fit/replay_err_k1"):
        assert key in header, key

    # El checkpoint guarda el buffer y H; un resume los restaura y continúa.
    last = tmp_path / "ckpt" / "last.ckpt"
    sd = torch.load(last, weights_only=False)["callbacks"]["PrioritizedReplay"]
    assert torch.equal(sd["replay"]["buffer"]["priority"], buf.priority)
    st2, cb2, trainer2 = _lightning_run(tmp_path, max_epochs=3, ckpt=str(last))
    assert trainer2.current_epoch == 3
    assert int(st2.buffer.visits.sum()) > int(buf.visits.sum())
