"""JEPA piramidal multiescala (LiftWM).

Extiende ``jepa.JEPA`` para latentes por escala. Cada frame se representa como un
dict ``{level: (B, T, N_l, D_l)}`` (sin pooling global). El encoder es un
``wavelet.pyramid_encoder.PyramidEncoder`` y el predictor un
``wavelet.hier_predictor.HierarchicalPredictor`` (NCA o ViT por escala).

* ``encode``: píxeles -> pirámide -> projector token-a-token por escala.
* ``predict`` / ``predict_rollout``: predictor por escala + pred_proj por escala.
* ``rollout``: autoregresivo por escala (patrón de ``SpatialJEPA.rollout``).
* ``criterion``: costo CEM = suma ponderada por escala de la MSE espacial del
  último paso pred-vs-goal (``scale_weights``).
* ``get_cost``: se hereda de ``JEPA`` (genérico: encode del goal + rollout + criterion).

Pérdidas: ``pred_loss`` = media ponderada por escala; ``SIGReg`` = media entre
escalas (mantiene la escala de lambda tuneada para un solo término). Las auxiliares
(gate L1, dinámica inversa en la escala fina, rollout) viven en ``pyramid_losses``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jepa import JEPA, detach_clone


class PyramidJEPA(JEPA):

    def __init__(self, encoder, predictor, action_encoder, projector=None,
                 pred_proj=None, scale_weights=None, inverse_dynamics=None):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.inverse_dynamics = inverse_dynamics
        self.grid_hw: dict[int, tuple] = {}
        n_scales = encoder.levels + 1
        if scale_weights is None:
            w = torch.ones(n_scales)
        else:
            w = torch.as_tensor(scale_weights, dtype=torch.float32)
            assert w.numel() == n_scales, f"scale_weights debe tener {n_scales} entradas"
        self.register_buffer("scale_weights", w)

    # ------------------------------------------------------------------ #
    def _tokenwise(self, mlp, x):
        """MLP token a token sobre ``(B, T, N, D)`` (patrón de SpatialJEPA)."""
        b, t, n, _ = x.shape
        y = mlp(rearrange(x, "b t n d -> (b t n) d"))
        return rearrange(y, "(b t n) d -> b t n d", b=b, t=t, n=n)

    def _proj(self, module_dict, level, x):
        if module_dict is None or isinstance(module_dict, nn.Identity):
            return x
        mod = module_dict[str(level)]
        if isinstance(mod, nn.Identity):
            return x
        return self._tokenwise(mod, x)

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        out = self.encoder(pixels)
        emb = {}
        self.grid_hw = {}
        for s in out.scales:
            self.grid_hw[s.level] = s.grid_hw
            tok = rearrange(s.tokens, "(b t) n d -> b t n d", b=b)
            emb[s.level] = self._proj(self.projector, s.level, tok)
        info["emb"] = emb
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb, only_last: bool = False):
        preds = self.predictor(emb, act_emb, grid_hw=self.grid_hw, only_last=only_last)
        return {l: self._proj(self.pred_proj, l, v) for l, v in preds.items()}

    def predict_rollout(self, emb_ctx, act_emb_seq, persist_hidden: bool = True,
                        detach_between: bool = False, history: Optional[int] = None):
        if self.pred_proj is None or isinstance(self.pred_proj, nn.Identity):
            post = None
        else:
            post = {
                l: (lambda y, l=l: self._proj(self.pred_proj, l, y.unsqueeze(1)).squeeze(1))
                for l in emb_ctx
            }
        return self.predictor.rollout(
            emb_ctx, act_emb_seq, grid_hw=self.grid_hw, post=post,
            persist_hidden=persist_hidden, detach_between=detach_between, history=history,
        )

    ####################
    ## Inference only ##
    ####################
    def rollout(self, info, action_sequence, history_size: int = 3):
        """Igual que ``SpatialJEPA.rollout`` pero con dict de latentes por escala.

        pixels: (B, S, T, C, H, W); action_sequence: (B, S, T, action_dim).
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        init_emb = _init["emb"]                       # dict[l] -> (B, H, N_l, D_l)
        emb = info["emb"] = {
            l: v.unsqueeze(1).expand(B, S, *v.shape[1:]) for l, v in init_emb.items()
        }
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = {l: rearrange(v, "b s ... -> (b s) ...").clone() for l, v in emb.items()}
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            ctx = {l: v[:, -HS:] for l, v in emb.items()}
            pred_emb = self.predict(ctx, act_emb[:, -HS:], only_last=True)
            emb = {l: torch.cat([emb[l], pred_emb[l]], dim=1) for l in emb}
            if t < n_steps:
                act = torch.cat([act, act_future[:, t:t + 1]], dim=1)

        info["predicted_emb"] = {
            l: rearrange(v, "(b s) ... -> b s ...", b=B, s=S) for l, v in emb.items()
        }
        return info

    def criterion(self, info_dict: dict):
        """Costo CEM multiescala: Σ_l w_l · MSE espacial del último paso. -> (B, S)."""
        pred = info_dict["predicted_emb"]             # dict[l] -> (B, S, T, N_l, D_l)
        goal = info_dict["goal_emb"]                  # dict[l] -> (B, T, N_l, D_l)
        w = self.scale_weights
        cost = None
        for l in pred:
            p = pred[l][:, :, -1]                     # (B, S, N_l, D_l)
            g = goal[l][:, -1].unsqueeze(1).expand_as(p).detach()
            c = F.mse_loss(p, g, reduction="none").sum(dim=(2, 3))  # (B, S)
            cost = w[l] * c if cost is None else cost + w[l] * c
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Igual que ``JEPA.get_cost`` pero robusto a ``goal_emb`` dict.

        El bucle que renombra las claves ``goal_*`` sólo opera sobre claves que
        son tensores (presentes en ``goal``); ``goal_emb`` aquí es un dict por
        escala y no debe intentar popearse. Sin esto, la 2ª iteración del CEM
        (que reutiliza ``info_dict`` con ``goal_emb`` ya puesto) da KeyError.
        """
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(info_dict.keys()):
            if k.startswith("goal_") and k in goal:
                goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


def _cfg_get(loss_cfg, name, key, default):
    return (loss_cfg.get(name, None) or {}).get(key, default)


def pyramid_losses(model, batch, emb, pred_emb, tgt_emb, cfg, act_emb=None):
    """Auxiliares por escala (gate L1, IDM en la escala fina, rollout). -> (extra, logs).

    ``pred_loss`` y ``sigreg_loss`` se calculan en ``train.py`` (media ponderada y
    media entre escalas respectivamente). Aquí van las pérdidas con gradiente
    opcionales y el diagnóstico, análogo a ``jepa_spatial.spatial_losses``.
    """
    loss_cfg = cfg.loss
    levels = sorted(emb.keys())
    fine = levels[-1]
    extra = pred_emb[fine].new_zeros((), dtype=torch.float32)
    logs = {}

    aux = getattr(model.predictor, "aux", None) or {}

    # Diagnóstico copy_loss por escala (sin gradiente).
    with torch.no_grad():
        for l in levels:
            ctx = emb[l][:, :pred_emb[l].size(1)]
            logs[f"copy_loss_l{l}"] = (ctx - tgt_emb[l]).float().pow(2).mean()

    # Gate L1 (si el predictor expone gate_mean agregado).
    w_gate = float(_cfg_get(loss_cfg, "gate_l1", "weight", 0.0))
    if w_gate > 0 and "gate_mean" in aux:
        gl1 = aux["gate_mean"].float()
        extra = extra + w_gate * gl1
        logs["gate_l1_loss"] = gl1.detach()

    # Dinámica inversa sobre la escala más fina.
    idm = getattr(model, "inverse_dynamics", None)
    if idm is not None:
        HS = pred_emb[fine].size(1)
        x0, x1 = emb[fine][:, :HS], emb[fine][:, 1:HS + 1]
        target = batch["action"][:, :HS].float()
        gate_map = aux.get(f"{fine}.gate_map", aux.get("gate_map"))
        pred_a = idm(x0, x1, gate_map)
        idm_loss = F.mse_loss(pred_a.float(), target)
        w_idm = float(_cfg_get(loss_cfg, "idm", "weight", 1.0))
        extra = extra + w_idm * idm_loss
        logs["idm_loss"] = idm_loss.detach()

    # Rollout multi-paso por escala (media entre escalas).
    w_roll = float(_cfg_get(loss_cfg, "rollout", "weight", 0.0))
    if w_roll > 0 and act_emb is not None:
        M = int(_cfg_get(loss_cfg, "rollout", "steps", 2))
        T = emb[fine].size(1)
        assert 1 <= M <= T - 1, f"loss.rollout.steps={M} necesita T>={M+1}, hay T={T}"
        t0 = T - 1 - M
        n_ctx = 2 if cfg.history_size > 1 else 1
        ctx = {l: emb[l][:, max(0, t0 - n_ctx + 1):t0 + 1] for l in levels}
        preds = model.predict_rollout(
            ctx, act_emb[:, t0:t0 + M],
            persist_hidden=bool(_cfg_get(loss_cfg, "rollout", "persist_hidden", True)),
            detach_between=bool(_cfg_get(loss_cfg, "rollout", "detach", True)),
            history=int(cfg.history_size),
        )
        per_scale = []
        for l in levels:
            target = emb[l][:, t0 + 1:t0 + 1 + M]
            per_step = (preds[l] - target).float().pow(2).mean(dim=(0, 2, 3))
            per_scale.append(per_step[-1] if _cfg_get(loss_cfg, "rollout", "last_only", False)
                             else per_step.mean())
        r_loss = torch.stack(per_scale).mean()
        extra = extra + w_roll * r_loss
        logs["rollout_loss"] = r_loss.detach()

    return extra, logs
