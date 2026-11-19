"""Replay priorizado por sorpresa + red de alucinación H (F1, rama ``per-hnet``).

Diseño (ver ``docs/replay_hnet.md``):

* **Unidad de prioridad**: la ventana del dataset (índice del ``Subset`` de
  entrenamiento). Cada ventana trae ``history_size + K`` frames: el camino
  principal (pred_loss + SIGReg) usa sólo los primeros ``history_size +
  num_preds``, igual que el baseline; el resto alimenta las etiquetas.
* **Etiquetas de K horizontes**: rollout autoregresivo de K pasos desde el
  contexto, con el mismo procedimiento que ``JEPA.rollout`` (el que usa CEM):
  ``e_k = ||ẑ_{HS-1+k} - z_{HS-1+k}||² / D``  (k = 1..K).
* **Buffer** (memoria "episódica"): tabla por índice de ventana con la prioridad,
  la última curva de error medida, antigüedad, visitas y validez.
* **H** (memoria "semántica"): MLP que predice ``log e_{1..K}`` a partir del
  contexto (y opcionalmente acciones / predicciones). Se entrena en un callback
  con su propio optimizador y entradas detached: cero gradiente y cero
  interferencia (clipping, scheduler) con el world model.
* **Sampler**: ``P(i) = (1-ε)·p_i^α / Σp^α + ε/N_valid`` sobre ventanas válidas,
  uniforme durante el warm-up. Pesos IS ``w_i ∝ P(i)^-β`` autonormalizados en el
  batch para pred_loss (β configurable: 0 = focal) y siempre β=1 para SIGReg
  (ECF ponderada), de modo que la isotropía vale bajo la distribución real.

Todo lo de este módulo es torch puro (sin stable_pretraining / swm), así que se
testea en CPU (``tests/test_prioritized_replay.py``).
"""
from __future__ import annotations

import csv
import math
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

try:  # el callback sólo existe si lightning está instalado (siempre en train)
    from lightning.pytorch.callbacks import Callback
except Exception:  # pragma: no cover - entornos sin lightning
    Callback = object

_SAMPLERS = ("uniform", "prioritized", "shuffled")
_PRIORITIES = ("error", "surprise", "progress")
_HNET_INPUTS = ("state", "state_action", "state_action_pred")


# ---------------------------------------------------------------------------- #
# Configuración
# ---------------------------------------------------------------------------- #
@dataclass
class ReplayConfig:
    enabled: bool = False
    sampler: str = "prioritized"      # uniform | prioritized | shuffled (control)
    priority: str = "error"           # error | surprise | progress
    horizon: int = 5                  # K (= horizonte del planificador)
    alpha: float = 0.6
    beta_start: float = 0.4           # IS de pred_loss; beta_start=beta_end=0 -> focal
    beta_end: float = 1.0
    beta_anneal_frac: float = 1.0     # fracción del entrenamiento para anillar beta
    eps_uniform: float = 0.1          # mezcla uniforme (evita inanición)
    clip_quantile: float = 0.99       # recorte de prioridades (y prioridad de no visitadas)
    warmup_epochs: float = 1.0        # muestreo uniforme al inicio
    ema_decay: float = 0.99           # media móvil del error por horizonte
    chunk_batches: int = 32           # el sampler recalcula P cada chunk_batches batches
    log_every: int = 50
    dump_topk: int = 16
    seed: int = 0
    # H
    hnet_enabled: bool = True
    hnet_input: str = "state_action"  # state | state_action | state_action_pred
    hnet_hidden: int = 512
    hnet_depth: int = 2
    hnet_lr: float = 1e-3
    hnet_weight_decay: float = 1e-4
    hnet_rank_weight: float = 1.0
    hnet_rank_margin: float = 0.05    # margen en log-error para formar pares
    hnet_huber_delta: float = 1.0

    @classmethod
    def from_cfg(cls, cfg) -> "ReplayConfig":
        """Desde un DictConfig/dict con el bloque ``replay`` (sub-bloque ``hnet``)."""
        try:
            from omegaconf import OmegaConf
            if not isinstance(cfg, dict):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        cfg = dict(cfg or {})
        hnet = dict(cfg.pop("hnet", None) or {})
        flat = {**cfg, **{f"hnet_{k}": v for k, v in hnet.items()}}
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(flat) - names)
        if unknown:
            raise ValueError(f"claves de replay desconocidas: {unknown}")
        out = cls(**flat)
        out.validate()
        return out

    def validate(self) -> None:
        if self.sampler not in _SAMPLERS:
            raise ValueError(f"replay.sampler='{self.sampler}' no válido; usa {_SAMPLERS}")
        if self.priority not in _PRIORITIES:
            raise ValueError(f"replay.priority='{self.priority}' no válido; usa {_PRIORITIES}")
        if self.hnet_input not in _HNET_INPUTS:
            raise ValueError(f"replay.hnet.input='{self.hnet_input}' no válido; usa {_HNET_INPUTS}")
        if self.horizon < 1:
            raise ValueError("replay.horizon debe ser >= 1")
        if self.priority == "surprise" and not self.hnet_enabled:
            raise ValueError("replay.priority=surprise necesita replay.hnet.enabled=true")
        if not 0.0 <= self.eps_uniform <= 1.0:
            raise ValueError("replay.eps_uniform debe estar en [0, 1]")


# ---------------------------------------------------------------------------- #
# Buffer (memoria episódica) y estado compartido
# ---------------------------------------------------------------------------- #
class PriorityBuffer:
    """Tabla por índice de ventana (CPU). ~(4 + 2K + 4 + 4 + 1) bytes por ventana."""

    _FIELDS = ("priority", "err", "last_step", "visits", "valid")

    def __init__(self, n: int, horizon: int):
        self.n = int(n)
        self.horizon = int(horizon)
        self.priority = torch.zeros(self.n, dtype=torch.float32)
        self.err = torch.zeros(self.n, self.horizon, dtype=torch.float16)
        self.last_step = torch.full((self.n,), -1, dtype=torch.int32)
        self.visits = torch.zeros(self.n, dtype=torch.int32)
        self.valid = torch.ones(self.n, dtype=torch.bool)

    def update(self, idx: torch.Tensor, priority: torch.Tensor, err: torch.Tensor,
               valid: torch.Tensor, step: int) -> None:
        idx = idx.detach().long().cpu()
        self.priority[idx] = priority.detach().float().cpu()
        self.err[idx] = err.detach().float().cpu().clamp(max=6e4).half()
        self.last_step[idx] = int(step)
        # index_add_: con muestreo con reemplazo un batch puede repetir índices.
        self.visits.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
        self.valid[idx] &= valid.detach().bool().cpu()

    def visited(self) -> torch.Tensor:
        return self.visits > 0

    def state_dict(self) -> dict:
        return {k: getattr(self, k).clone() for k in self._FIELDS}

    def load_state_dict(self, sd: dict) -> None:
        for k in self._FIELDS:
            src = sd[k]
            dst = getattr(self, k)
            if src.shape != dst.shape:
                raise ValueError(f"buffer.{k}: forma {tuple(src.shape)} != {tuple(dst.shape)} "
                                 "(¿cambió el dataset, el split o replay.horizon?)")
            dst.copy_(src)


class ReplayState:
    """Objeto compartido por sampler (proceso principal), forward y callback."""

    def __init__(self, cfg: ReplayConfig, n: int):
        self.cfg = cfg
        self.buffer = PriorityBuffer(n, cfg.horizon)
        self.step = 0                     # global_step (lo actualiza el callback)
        self.beta = float(cfg.beta_start)
        self.err_ema: Optional[torch.Tensor] = None   # (K,) media móvil del error
        self.last: Optional[dict] = None              # salida del último forward
        self.sampler_stats: dict = {}

    def state_dict(self) -> dict:
        return {"buffer": self.buffer.state_dict(), "step": self.step,
                "err_ema": None if self.err_ema is None else self.err_ema.clone()}

    def load_state_dict(self, sd: dict) -> None:
        self.buffer.load_state_dict(sd["buffer"])
        self.step = int(sd.get("step", 0))
        ema = sd.get("err_ema", None)
        self.err_ema = None if ema is None else ema.clone().float()


# ---------------------------------------------------------------------------- #
# Dataset y sampler
# ---------------------------------------------------------------------------- #
class IndexedDataset(torch.utils.data.Dataset):
    """Agrega ``idx`` y ``sample_prob`` a cada ventana.

    El sampler entrega claves ``(i, P_i)``: así la probabilidad con que se
    muestreó la ventana viaja con el batch y los pesos IS son exactos aunque el
    buffer cambie entre el muestreo y el paso de entrenamiento (prefetch).
    Con claves enteras (val, sampler secuencial) ``sample_prob = NaN``.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _split_key(key):
        if isinstance(key, (tuple, list)):
            i, p = key
        else:
            i, p = key, float("nan")
        return int(i), float(p)

    @staticmethod
    def _tag(item, i: int, p: float) -> dict:
        item = dict(item)
        item["idx"] = torch.tensor(i, dtype=torch.long)
        item["sample_prob"] = torch.tensor(p, dtype=torch.float64)
        return item

    def __getitem__(self, key):
        i, p = self._split_key(key)
        return self._tag(self.base[i], i, p)

    def __getitems__(self, keys) -> list:
        """Fetch por lotes (``torch.utils.data._utils.fetch``): conserva el
        ``__getitems__`` del dataset envuelto (lecturas HDF5 por lote)."""
        pairs = [self._split_key(k) for k in keys]
        ids = [i for i, _ in pairs]
        getitems = getattr(self.base, "__getitems__", None)
        items = getitems(ids) if callable(getitems) else [self.base[i] for i in ids]
        return [self._tag(item, i, p) for item, (i, p) in zip(items, pairs)]

    def __getattr__(self, name):
        # Proxy transparente (set_pl_trainer, gpu_transform, column_names, ...),
        # igual que ``spt.data.Subset``. Sin dunders ni ``base`` (pickle/copy).
        if name == "base" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.base, name)


def resolve_index(ds, i: int) -> int:
    """Índice en el dataset raíz atravesando ``IndexedDataset``/``Subset`` anidados."""
    while True:
        if isinstance(ds, IndexedDataset):
            ds = ds.base
            continue
        indices = getattr(ds, "indices", None)
        inner = getattr(ds, "dataset", None)
        if indices is None or inner is None:
            return int(i)
        i = int(indices[i])
        ds = inner


class PrioritizedSampler(torch.utils.data.Sampler):
    """Muestreo con reemplazo según el buffer; entrega claves ``(i, P_i)``."""

    def __init__(self, state: ReplayState, num_samples: int, batch_size: int,
                 generator: Optional[torch.Generator] = None):
        self.state = state
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(
            state.cfg.seed)
        n = state.buffer.n
        perm_gen = torch.Generator().manual_seed(state.cfg.seed + 7919)
        self.perm = torch.randperm(n, generator=perm_gen)   # control "shuffled"

    def __len__(self):
        return self.num_samples

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.num_samples // self.batch_size)

    @property
    def warmup_steps(self) -> int:
        return int(round(self.state.cfg.warmup_epochs * self.steps_per_epoch))

    def _clip_value(self, p: torch.Tensor, mask: torch.Tensor) -> float:
        vals = p[mask]
        if vals.numel() == 0:
            return 1.0
        if vals.numel() > 1_000_000:
            sel = torch.randint(vals.numel(), (1_000_000,), generator=self.generator)
            vals = vals[sel]
        return float(torch.quantile(vals.double(), self.state.cfg.clip_quantile))

    def probabilities(self) -> torch.Tensor:
        """P (N,) float64 con el estado actual del buffer."""
        st, cfg, buf = self.state, self.state.cfg, self.state.buffer
        valid = buf.valid
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise RuntimeError("replay: no quedan ventanas válidas")
        uniform = valid.double() / n_valid
        prioritized = cfg.sampler != "uniform" and st.step >= self.warmup_steps
        if not prioritized:
            P = uniform
            q = float("nan")
        else:
            p = buf.priority.double()
            visited = buf.visited()
            q = self._clip_value(p, visited & valid)
            q = max(q, 1e-8)
            # No visitadas reciben la prioridad de recorte (≈ máx de Schaul et al.).
            p = torch.where(visited, p.clamp(min=1e-8, max=q), torch.full_like(p, q))
            if cfg.sampler == "shuffled":
                p = p[self.perm]          # misma distribución, desacoplada del contenido
            w = p.pow(cfg.alpha) * valid
            P = (1.0 - cfg.eps_uniform) * w / w.sum() + cfg.eps_uniform * uniform
        st.sampler_stats = {
            "ess": float(1.0 / (n_valid * P.square().sum())),
            "q_clip": q,
            "prioritized": float(prioritized),
            "n_valid": float(n_valid),
        }
        return P

    def __iter__(self):
        remaining = self.num_samples
        chunk = max(1, self.state.cfg.chunk_batches) * self.batch_size
        while remaining > 0:
            m = min(chunk, remaining)
            P = self.probabilities()
            cdf = torch.cumsum(P, 0)
            u = torch.rand(m, generator=self.generator, dtype=torch.float64) * cdf[-1]
            idx = torch.searchsorted(cdf, u, right=True).clamp_(max=P.numel() - 1)
            probs = P[idx]
            for i, p in zip(idx.tolist(), probs.tolist()):
                yield (i, p)
            remaining -= m


# ---------------------------------------------------------------------------- #
# Piezas del forward (pesos IS, ventanas, etiquetas)
# ---------------------------------------------------------------------------- #
def importance_weights(sample_prob: torch.Tensor, beta: float) -> Optional[torch.Tensor]:
    """``w_i ∝ P_i^{-β}`` autonormalizados a media 1 en el batch (None si β=0)."""
    if beta == 0:
        return None
    p = sample_prob.double().clamp_min(1e-300)
    logw = -float(beta) * torch.log(p)
    w = torch.exp(logw - logw.max())
    return (w / w.mean()).float()


def replay_weights(batch: dict, state: ReplayState, stage: str):
    """``(w_pred, w_sigreg)`` para el batch; ``(None, None)`` = media simple."""
    if stage != "fit" or state.cfg.sampler == "uniform":
        return None, None
    prob = batch.get("sample_prob", None)
    if prob is None or torch.isnan(prob).any():
        return None, None
    return importance_weights(prob, state.beta), importance_weights(prob, 1.0)


def weighted_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    if w is None:
        return x.mean()
    w = w.to(x.device, x.dtype)
    return (w * x).sum() / w.sum()


def valid_windows(action: torch.Tensor, n_used: int) -> torch.Tensor:
    """True si las acciones usadas (posiciones < n_used) no tienen NaN (borde de secuencia)."""
    a = action[:, :n_used]
    return ~torch.isnan(a.float()).flatten(1).any(1)


def split_time(batch: dict, t_main: int) -> dict:
    """Copia superficial del batch con los tensores temporales recortados a t_main."""
    T = batch["action"].size(1)
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.size(1) == T:
            out[k] = v[:, :t_main]
        else:
            out[k] = v
    return out


@contextmanager
def frozen_bn_stats(module: nn.Module):
    """BatchNorm sigue usando estadísticas del batch en train, pero sin actualizar
    las running stats (momentum=0). Así las pasadas extra de etiquetado no cambian
    lo que el modelo usa en eval/planificación."""
    saved = []
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            nbt = int(m.num_batches_tracked) if m.num_batches_tracked is not None else None
            saved.append((m, m.momentum, nbt))
            m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom, nbt in saved:
            m.momentum = mom
            # En eval (val / inference_mode) BN no cuenta batches: no hay nada que restaurar.
            if nbt is not None and int(m.num_batches_tracked) != nbt:
                with torch.no_grad():
                    m.num_batches_tracked.fill_(nbt)


@torch.no_grad()
def rollout_predictions(model, z_ctx: torch.Tensor, act_emb: torch.Tensor,
                        horizon: int, history_size: int) -> torch.Tensor:
    """Rollout autoregresivo de ``horizon`` pasos, idéntico a ``JEPA.rollout``.

    z_ctx: (B, HS, D) latentes de los frames 0..HS-1.
    act_emb: (B, >= HS+K-1, A) embedding de la acción aplicada en cada frame.
    Devuelve (B, K, D): predicciones de los frames HS..HS+K-1.
    """
    HS = int(history_size)
    need = HS + horizon - 1
    assert act_emb.size(1) >= need, f"act_emb tiene {act_emb.size(1)} pasos, se necesitan {need}"
    emb = z_ctx
    preds = []
    for k in range(horizon):
        a = act_emb[:, : HS + k]
        p = model.predict(emb[:, -HS:], a[:, -HS:])[:, -1:]
        preds.append(p)
        emb = torch.cat([emb, p], dim=1)
    return torch.cat(preds, dim=1)


def replay_collect(model, full_batch: dict, emb: torch.Tensor, act_emb: torch.Tensor,
                   valid: torch.Tensor, state: ReplayState, history_size: int,
                   t_main: int, stage: str) -> dict:
    """Etiquetas de K horizontes (sin gradiente) -> ``state.last``. Devuelve logs.

    Reutiliza los latentes del camino principal (frames 0..t_main-1) y codifica
    sólo los frames restantes, en el mismo modo (train/eval) que el forward y con
    las running stats de BatchNorm congeladas.
    """
    K = state.cfg.horizon
    HS = int(history_size)
    need = HS + K
    T = full_batch["action"].size(1)
    if T < need:
        raise ValueError(f"replay.horizon={K} necesita ventanas de {need} frames y el batch "
                         f"trae {T}: revisa data.dataset.num_steps (= history_size + horizon)")
    with torch.no_grad(), frozen_bn_stats(model):
        z_main, a_main = emb.detach(), act_emb.detach()
        if need > t_main:
            extra = {"pixels": full_batch["pixels"][:, t_main:need],
                     "action": full_batch["action"][:, t_main:need]}
            extra = model.encode(extra)
            z_all = torch.cat([z_main, extra["emb"].to(z_main.dtype)], dim=1)
            a_all = torch.cat([a_main, extra["act_emb"].to(a_main.dtype)], dim=1)
        else:
            z_all, a_all = z_main[:, :need], a_main[:, :need]
        preds = rollout_predictions(model, z_all[:, :HS], a_all, K, HS)
        err = (preds.float() - z_all[:, HS:HS + K].float()).pow(2).mean(-1)   # (B, K)

    state.last = {
        "stage": stage,
        "idx": full_batch.get("idx", None),
        "valid": valid.detach(),
        "err": err,
        "z_ctx": z_all[:, :HS].float(),
        "act_k": a_all[:, HS - 1:HS - 1 + K].float(),
        "pred": preds.float(),
    }
    m = valid.to(err.device)
    logs = {}
    if m.any():
        for k in range(K):
            logs[f"replay_err_k{k + 1}"] = err[m, k].mean().detach()
    return logs


# ---------------------------------------------------------------------------- #
# Red de alucinación H
# ---------------------------------------------------------------------------- #
class HallucinationNet(nn.Module):
    """MLP: contexto (+ acciones, + predicciones) -> log-error por horizonte (B, K)."""

    def __init__(self, z_dim: int, a_dim: int, history_size: int, horizon: int,
                 mode: str = "state_action", hidden: int = 512, depth: int = 2):
        super().__init__()
        if mode not in _HNET_INPUTS:
            raise ValueError(f"mode='{mode}' no válido; usa {_HNET_INPUTS}")
        self.z_dim, self.a_dim = int(z_dim), int(a_dim)
        self.history_size, self.horizon, self.mode = int(history_size), int(horizon), mode
        in_dim = self.history_size * self.z_dim
        if mode in ("state_action", "state_action_pred"):
            in_dim += self.horizon * self.a_dim
        if mode == "state_action_pred":
            in_dim += self.horizon * self.z_dim
        self.in_dim = in_dim
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, self.horizon))
        self.net = nn.Sequential(*layers)

    def config(self) -> dict:
        return {"z_dim": self.z_dim, "a_dim": self.a_dim, "history_size": self.history_size,
                "horizon": self.horizon, "mode": self.mode}

    def forward(self, z_ctx, act_k=None, pred=None) -> torch.Tensor:
        parts = [z_ctx.flatten(1)]
        if self.mode in ("state_action", "state_action_pred"):
            parts.append(act_k.flatten(1))
        if self.mode == "state_action_pred":
            parts.append(pred.flatten(1))
        return self.net(torch.cat(parts, dim=1).float())


def hnet_loss(log_pred: torch.Tensor, err: torch.Tensor, weights: Optional[torch.Tensor],
              rank_weight: float = 1.0, margin: float = 0.05, delta: float = 1.0,
              eps: float = 1e-8):
    """Huber en log-error + ranking logístico por pares (por horizonte)."""
    target = torch.log(err.float() + eps)
    w = torch.ones(log_pred.size(0), device=log_pred.device) if weights is None else weights
    w = w.to(log_pred.device).float()
    huber = F.huber_loss(log_pred, target, delta=delta, reduction="none").mean(1)
    l_reg = (w * huber).sum() / w.sum()
    l_rank = log_pred.new_zeros(())
    if rank_weight > 0 and log_pred.size(0) > 1:
        d_pred = log_pred[:, None, :] - log_pred[None, :, :]      # (B, B, K)
        d_tgt = target[:, None, :] - target[None, :, :]
        pair_w = (w[:, None] * w[None, :])[..., None] * (d_tgt > margin)
        denom = pair_w.sum()
        if denom > 0:
            l_rank = (F.softplus(-d_pred) * pair_w).sum() / denom
    return l_reg + rank_weight * l_rank, l_reg.detach(), l_rank.detach()


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Correlación de Spearman (rangos sin corrección por empates)."""
    x, y = x.flatten().double(), y.flatten().double()
    if x.numel() < 3:
        return float("nan")
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = (rx.norm() * ry.norm()).clamp_min(1e-12)
    return float((rx * ry).sum() / den)


# ---------------------------------------------------------------------------- #
# Callback: prioridades, entrenamiento de H, diagnósticos
# ---------------------------------------------------------------------------- #
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _beta_at(cfg: ReplayConfig, step: int, total_steps: Optional[float]) -> float:
    if not total_steps or not math.isfinite(total_steps) or cfg.beta_anneal_frac <= 0:
        return float(cfg.beta_end)
    frac = min(1.0, step / max(1.0, cfg.beta_anneal_frac * total_steps))
    return float(cfg.beta_start + (cfg.beta_end - cfg.beta_start) * frac)


class PrioritizedReplayCallback(Callback):
    """Actualiza el buffer, entrena H (pasiva) y registra diagnósticos.

    Lee la salida del forward desde ``state.last`` (no depende de lo que
    ``training_step`` devuelva a los hooks).
    """

    def __init__(self, state: ReplayState, train_dataset=None, run_dir=None,
                 z_dim: int = 192, a_dim: int = 192, history_size: int = 3):
        super().__init__()
        self.state = state
        cfg = state.cfg
        self.train_dataset = train_dataset
        self.out_dir = Path(run_dir) / "replay" if run_dir is not None else None
        self.hnet = None
        self.opt = None
        if cfg.hnet_enabled:
            self.hnet = HallucinationNet(z_dim, a_dim, history_size, cfg.horizon,
                                         mode=cfg.hnet_input, hidden=cfg.hnet_hidden,
                                         depth=cfg.hnet_depth)
            self.opt = torch.optim.AdamW(self.hnet.parameters(), lr=cfg.hnet_lr,
                                         weight_decay=cfg.hnet_weight_decay)
        self._pending_opt = None
        self._fit_started = False
        self._val = []
        self._total_steps = None

    @property
    def state_key(self) -> str:
        return "PrioritizedReplay"

    # ------------------------------------------------------------------ #
    # Estado (resume)
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        sd = {"replay": self.state.state_dict()}
        if self.hnet is not None:
            sd["hnet"] = self.hnet.state_dict()
            sd["hnet_opt"] = self.opt.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.state.load_state_dict(state_dict["replay"])
        if self.hnet is not None and "hnet" in state_dict:
            self.hnet.load_state_dict(state_dict["hnet"])
            if self._fit_started:
                self.opt.load_state_dict(state_dict["hnet_opt"])
            else:
                self._pending_opt = state_dict["hnet_opt"]

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def on_fit_start(self, trainer, pl_module):
        if self.hnet is not None:
            self.hnet.to(pl_module.device)
            if self._pending_opt is not None:
                self.opt.load_state_dict(self._pending_opt)   # castea al device de los params
                self._pending_opt = None
        self._fit_started = True
        try:
            self._total_steps = float(trainer.estimated_stepping_batches)
        except Exception:
            self._total_steps = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.state.step = int(trainer.global_step)
        self.state.beta = _beta_at(self.state.cfg, self.state.step, self._total_steps)

    def _scores(self, err: torch.Tensor) -> torch.Tensor:
        """Error normalizado por horizonte y agregado: s_i = mean_k e_ik / ē_k."""
        ema = self.state.err_ema.to(err.device)
        return (err / ema.clamp_min(1e-12)).mean(1)

    def _update_ema(self, err: torch.Tensor, valid: torch.Tensor) -> None:
        if not valid.any():
            return
        m = err[valid].mean(0).detach().float().cpu()
        if self.state.err_ema is None:
            self.state.err_ema = m
        else:
            d = self.state.cfg.ema_decay
            self.state.err_ema = d * self.state.err_ema + (1 - d) * m

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        out, self.state.last = self.state.last, None
        if out is None or out["idx"] is None:
            return
        st, cfg = self.state, self.state.cfg
        err = out["err"].float()
        valid = out["valid"].to(err.device)
        idx = out["idx"]
        self._update_ema(err, valid)
        if st.err_ema is None:          # batch sin ventanas válidas antes de tener ema
            st.buffer.update(idx, torch.zeros(len(idx)), err, valid, st.step)
            return
        s = self._scores(err)

        log_pred = None
        logs = {}
        if self.hnet is not None:
            feats = (out["z_ctx"], out["act_k"], out["pred"])
            with torch.no_grad():
                log_pred = self.hnet(*feats)               # predicción ANTES de actualizar H
            if valid.sum() >= 3:
                s_hat = self._scores(log_pred.exp())
                logs["hnet/batch_spearman"] = spearman(s_hat[valid], s[valid])
            # Paso de H: pesos IS con beta=1 -> calibrada bajo la distribución uniforme.
            prob = batch.get("sample_prob", None) if isinstance(batch, dict) else None
            w = None
            if prob is not None and not torch.isnan(prob).any() and cfg.sampler != "uniform":
                w = importance_weights(prob, 1.0)
            if valid.any():
                with torch.enable_grad():
                    self.hnet.train()
                    lp = self.hnet(*[f[valid] for f in feats])
                    loss, l_reg, l_rank = hnet_loss(
                        lp, err[valid], None if w is None else w.to(err.device)[valid],
                        rank_weight=cfg.hnet_rank_weight, margin=cfg.hnet_rank_margin,
                        delta=cfg.hnet_huber_delta)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 1.0)
                    self.opt.step()
                logs.update({"hnet/loss": float(loss.detach()), "hnet/loss_reg": float(l_reg),
                             "hnet/loss_rank": float(l_rank)})

        # Prioridad según el criterio configurado.
        if cfg.priority == "error":
            prio = s
        elif cfg.priority == "surprise":
            prio = (s - self._scores(log_pred.exp())).clamp_min(0.0) + 1e-3
        else:  # progress: |s_prev - s| (primera visita: s)
            idx_cpu = idx.long().cpu()
            prev_err = st.buffer.err[idx_cpu].float().to(err.device)
            seen = st.buffer.visits[idx_cpu].to(err.device) > 0
            prio = torch.where(seen, (self._scores(prev_err) - s).abs(), s)
        prio = torch.where(valid, prio, torch.zeros_like(prio))
        st.buffer.update(idx, prio, err, valid, st.step)

        if cfg.log_every > 0 and st.step % cfg.log_every == 0:
            buf = st.buffer
            visited = buf.visited()
            logs.update({
                "replay/beta": st.beta,
                "replay/frac_visited": float(visited.float().mean()),
                "replay/frac_invalid": float((~buf.valid & visited).sum() / visited.sum().clamp_min(1)),
                "replay/priority_mean": float(prio[valid].mean()) if valid.any() else 0.0,
            })
            for k, v in st.sampler_stats.items():
                if isinstance(v, float) and math.isfinite(v):
                    logs[f"replay/{k}"] = v
            for k in range(cfg.horizon):
                logs[f"replay/err_ema_k{k + 1}"] = float(st.err_ema[k])
            prob = batch.get("sample_prob", None) if isinstance(batch, dict) else None
            if prob is not None and not torch.isnan(prob).any():
                wis = importance_weights(prob, 1.0)
                if wis is not None:
                    logs["replay/is_w_max"] = float(wis.max())
            self._log(pl_module, logs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        out, self.state.last = self.state.last, None
        if out is None or self.hnet is None or getattr(trainer, "sanity_checking", False):
            return
        with torch.inference_mode():
            self.hnet.eval()
            log_pred = self.hnet(out["z_ctx"], out["act_k"], out["pred"])
        valid = out["valid"].to(out["err"].device)
        self._val.append((log_pred[valid].float().cpu(), out["err"][valid].float().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        val, self._val = self._val, []
        if not val or getattr(trainer, "sanity_checking", False) or self.state.err_ema is None:
            return
        log_pred = torch.cat([v[0] for v in val])
        err = torch.cat([v[1] for v in val])
        ema = self.state.err_ema
        s = (err / ema.clamp_min(1e-12)).mean(1)
        s_hat = (log_pred.exp() / ema.clamp_min(1e-12)).mean(1)
        logs = {"hnet/val_spearman": spearman(s_hat, s)}
        for k in range(err.size(1)):
            logs[f"hnet/val_spearman_k{k + 1}"] = spearman(log_pred[:, k], err[:, k])
        self._log(pl_module, logs, on_epoch=True)
        if self.out_dir is not None:
            self._write_calibration(s_hat, s, trainer.current_epoch + 1)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.out_dir is None:
            return
        epoch = trainer.current_epoch + 1
        if self.hnet is not None:
            torch.save({"state_dict": self.hnet.state_dict(), "config": self.hnet.config(),
                        "err_ema": self.state.err_ema, "replay_cfg": vars(self.state.cfg)},
                       self.out_dir / f"hnet_epoch_{epoch}.pt")
        try:
            self._dump_topk(epoch)
        except Exception as e:  # el diagnóstico nunca debe tumbar el entrenamiento
            print(f"[replay] dump top-k falló: {e!r}")

    def on_train_end(self, trainer, pl_module):
        if self.out_dir is not None:
            torch.save(self.state.state_dict(), self.out_dir / "buffer_final.pt")

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log(pl_module, logs: dict, on_epoch: bool = False) -> None:
        clean = {k: float(v) for k, v in logs.items()
                 if v is not None and math.isfinite(float(v))}
        if clean:
            pl_module.log_dict(clean, on_step=not on_epoch, on_epoch=on_epoch)

    def _write_calibration(self, s_hat: torch.Tensor, s: torch.Tensor, epoch: int) -> None:
        order = s_hat.argsort()
        bins = torch.tensor_split(order, 10)
        path = self.out_dir / f"calibration_epoch_{epoch}.csv"
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["decile", "n", "pred_mean", "actual_mean"])
            for d, b in enumerate(bins):
                if b.numel():
                    wr.writerow([d, b.numel(), float(s_hat[b].mean()), float(s[b].mean())])

    def _dump_topk(self, epoch: int) -> None:
        k = int(self.state.cfg.dump_topk)
        if k <= 0:
            return
        buf = self.state.buffer
        mask = buf.visited() & buf.valid
        if not mask.any():
            return
        p = buf.priority.clone()
        p[~mask] = -float("inf")
        top = p.topk(min(k, int(mask.sum()))).indices
        rows = []
        for i in top.tolist():
            base_i = resolve_index(self.train_dataset, i) if self.train_dataset is not None else i
            rows.append([i, base_i, float(buf.priority[i]), int(buf.visits[i]),
                         int(buf.last_step[i])] + [float(x) for x in buf.err[i]])
        with open(self.out_dir / f"topk_epoch_{epoch}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["idx", "dataset_idx", "priority", "visits", "last_step"]
                        + [f"err_k{j + 1}" for j in range(buf.horizon)])
            wr.writerows(rows)
        self._save_topk_png(top.tolist(), self.out_dir / f"topk_epoch_{epoch}.png")

    def _save_topk_png(self, indices, path: Path, size: int = 96) -> None:
        if self.train_dataset is None:
            return
        base = self.train_dataset.base if isinstance(self.train_dataset, IndexedDataset) \
            else self.train_dataset
        rows = []
        for i in indices:
            item = base[int(i)]
            if "pixels" not in item:
                return  # render online: el dataset no trae píxeles
            x = item["pixels"].float()                       # (T, C, H, W) normalizado
            if x.ndim != 4 or x.size(1) != 3:
                return
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            x = (x * std + mean).clamp(0, 1)
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            rows.append(torch.cat(list(x), dim=2))           # (3, size, T*size)
        if not rows:
            return
        grid = torch.cat(rows, dim=1).permute(1, 2, 0).numpy()
        import matplotlib.image as mpimg
        mpimg.imsave(path, grid)
