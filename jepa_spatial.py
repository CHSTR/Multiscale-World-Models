"""JEPA espacial: predice la grilla de patch tokens en vez de un vector global.

El ``JEPA`` base reduce cada frame a un vector (CLS / ``pooled``), ``(B, T, D)``.
Un predictor local como el NCA necesita vecindario, así que aquí cada frame es
la grilla de patch tokens del encoder, ``(B, T, N, D)`` con ``N = gh * gw``.

* ``encode``: toma los últimos ``N`` tokens de ``last_hidden_state`` (en todos
  los encoders del repo los patches van al final: DINOv2 ``[CLS, patches]``,
  DINOv3 ``[CLS, dyn, storage, patches]``, ViT HF ``[CLS, patches]``) y les
  aplica el ``projector`` token a token.
* ``predict`` / ``rollout`` / ``criterion``: iguales al base pero con la
  dimensión espacial ``N`` y pasando ``grid_hw`` al predictor.
* El costo de planning es la MSE sumada sobre toda la grilla del último paso.
* ``inverse_dynamics`` (opcional): cabeza que predice ``a_t`` desde
  ``(x_t, x_{t+1})`` del encoder. Ancla la representación a lo que la acción
  controla (ver ``spatial_losses``).
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jepa import JEPA, detach_clone


class InverseDynamicsHead(nn.Module):
    """Predice la acción (cruda, normalizada) desde dos grillas consecutivas.

    Cada token ve ``[x_t, x_{t+1} - x_t]`` -> MLP; luego se agrega sobre la
    grilla con pesos ``w`` y una cabeza MLP predice la acción.

    ``pool``:
      * ``gate``: ``w`` = gate del NCA normalizado. El gradiente de esta loss
        llega al gate y lo empuja a abrirse donde la acción tiene efecto.
      * ``attn``: ``w`` = softmax de un score aprendido por token.
      * ``mean``: promedio uniforme.
      * ``auto``: ``gate`` si el predictor expone ``gate_map``, si no ``attn``.
    """

    POOLS = ("auto", "gate", "attn", "mean")

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256,
                 pool: str = "auto"):
        super().__init__()
        assert pool in self.POOLS, f"pool {pool!r} no válido (opciones: {self.POOLS})"
        self.pool = pool
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x0, x1, gate_map=None):
        """x0, x1: ``(B, T, N, D)``; gate_map: ``(B, T, N)`` o None -> ``(B, T, A)``."""
        h = self.token_mlp(torch.cat([x0, x1 - x0], dim=-1))  # (B, T, N, H)
        mode = self.pool
        if mode == "auto":
            mode = "gate" if gate_map is not None else "attn"
        if mode == "gate":
            assert gate_map is not None, "pool='gate' requiere predictor.gate=true"
            w = gate_map / (gate_map.sum(-1, keepdim=True) + 1e-6)
        elif mode == "attn":
            w = torch.softmax(self.score(h).squeeze(-1).float(), dim=-1).to(h.dtype)
        else:
            w = torch.full(h.shape[:-1], 1.0 / h.size(-2), device=h.device, dtype=h.dtype)
        pooled = (w.unsqueeze(-1).to(h.dtype) * h).sum(dim=-2)
        return self.head(pooled)


def spatial_losses(model, batch, emb, pred_emb, tgt_emb, cfg):
    """Pérdidas auxiliares y diagnóstico para JEPA espacial (emb ``(B, T, N, D)``).

    Devuelve ``(extra_loss, logs)``: ``extra_loss`` se suma a la loss total y
    ``logs`` es un dict de escalares para loguear (todos ya separados del grafo).

    Diagnóstico (sin gradiente):
      * ``copy_loss``: MSE de predecir ``x_{t+1} = x_t``. Es la línea base que
        el NCA tiene en la inicialización.
      * ``pred_loss_moving`` / ``copy_loss_moving``: las mismas MSE pero sólo
        sobre el ``diag.moving_frac`` de tokens que más cambian.
      * ``moving_ratio`` = pred/copy en tokens móviles. < 1 => el modelo
        predice dinámica; ~1 => sólo copia el frame.
      * ``gate_mean`` y ``gate_moving_ratio`` (gate medio en tokens móviles /
        gate medio en estáticos; > 1 => el gate se abre donde hay movimiento).

    Pérdidas (con gradiente, sólo si su peso > 0 / el módulo existe):
      * ``gate_l1_loss`` (``loss.gate_l1.weight``): media del gate (g >= 0).
      * ``idm_loss`` (``loss.idm.weight``): MSE de la dinámica inversa.
    """
    loss_cfg = cfg.loss
    diag_cfg = cfg.get("diag", None) or {}
    HS = pred_emb.size(1)
    ctx = emb[:, :HS]
    extra = pred_emb.new_zeros((), dtype=torch.float32)
    logs = {}

    aux = getattr(model.predictor, "aux", None) or {}
    gate_map = aux.get("gate_map")

    with torch.no_grad():
        err_pred = (pred_emb - tgt_emb).float().pow(2).mean(-1)  # (B, T, N)
        err_copy = (ctx - tgt_emb).float().pow(2).mean(-1)
        logs["copy_loss"] = err_copy.mean()
        frac = float(diag_cfg.get("moving_frac", 0.1))
        thr = torch.quantile(err_copy.flatten(), 1.0 - frac)
        moving = err_copy >= thr
        if moving.any():
            pm, cm = err_pred[moving].mean(), err_copy[moving].mean()
            logs["pred_loss_moving"] = pm
            logs["copy_loss_moving"] = cm
            logs["moving_ratio"] = pm / (cm + 1e-8)
        if gate_map is not None:
            g = gate_map.float()
            logs["gate_mean"] = g.mean()
            if moving.any() and (~moving).any():
                logs["gate_moving_ratio"] = g[moving].mean() / (g[~moving].mean() + 1e-8)

    w_gate = float((loss_cfg.get("gate_l1", None) or {}).get("weight", 0.0))
    if w_gate > 0 and "gate_mean" in aux:
        gl1 = aux["gate_mean"].float()
        extra = extra + w_gate * gl1
        logs["gate_l1_loss"] = gl1.detach()

    idm = getattr(model, "inverse_dynamics", None)
    if idm is not None:
        assert emb.size(1) >= HS + 1, "inverse dynamics necesita T >= history_size + 1"
        x0, x1 = emb[:, :HS], emb[:, 1:HS + 1]
        target = batch["action"][:, :HS].float()
        pred_a = idm(x0, x1, gate_map)
        idm_loss = F.mse_loss(pred_a.float(), target)
        w_idm = float((loss_cfg.get("idm", None) or {}).get("weight", 1.0))
        extra = extra + w_idm * idm_loss
        logs["idm_loss"] = idm_loss.detach()

    return extra, logs


class SpatialJEPA(JEPA):

    def __init__(self, encoder, predictor, action_encoder, projector=None,
                 pred_proj=None, patch_size: int = 16, inverse_dynamics=None):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.patch_size = patch_size
        self.inverse_dynamics = inverse_dynamics
        self.grid_hw = None  # se fija en el primer ``encode``

    def _tokenwise(self, mlp, x):
        """Aplica un MLP (con BatchNorm1d) token a token sobre ``(B, T, N, D)``."""
        b, t, n, _ = x.shape
        y = mlp(rearrange(x, "b t n d -> (b t n) d"))
        return rearrange(y, "(b t n) d -> b t n d", b=b, t=t, n=n)

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        out = self.encoder(pixels, interpolate_pos_encoding=True)

        gh = pixels.shape[-2] // self.patch_size
        gw = pixels.shape[-1] // self.patch_size
        n = gh * gw
        tokens = out.last_hidden_state
        assert tokens.size(1) >= n, (
            f"el encoder devolvió {tokens.size(1)} tokens, se esperaban >= {n} "
            f"patches (patch_size={self.patch_size})"
        )
        self.grid_hw = (gh, gw)

        patches = rearrange(tokens[:, -n:], "(b t) n d -> b t n d", b=b)
        info["emb"] = self._tokenwise(self.projector, patches)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb, only_last: bool = False):
        """emb: ``(B, T, N, D)``, act_emb: ``(B, T, A)`` -> ``(B, T', N, D)``."""
        preds = self.predictor(emb, act_emb, grid_hw=self.grid_hw, only_last=only_last)
        return self._tokenwise(self.pred_proj, preds)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Igual que ``JEPA.rollout`` pero con latentes ``(…, N, D)``.

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
        init_emb = _init["emb"]  # (B, H, N, D)
        emb = info["emb"] = init_emb.unsqueeze(1).expand(B, S, *init_emb.shape[1:])
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:], only_last=True)
            emb = torch.cat([emb, pred_emb], dim=1)
            if t < n_steps:
                act = torch.cat([act, act_future[:, t : t + 1]], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """MSE entre la grilla predicha del último paso y la del goal, sumada. -> (B, S)."""
        pred = info_dict["predicted_emb"][:, :, -1]  # (B, S, N, D)
        goal = info_dict["goal_emb"][:, -1]           # (B, N, D)
        goal = goal.unsqueeze(1).expand_as(pred).detach()
        cost = F.mse_loss(pred, goal, reduction="none")
        return cost.sum(dim=tuple(range(2, cost.ndim)))
