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
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from jepa import JEPA, detach_clone


class SpatialJEPA(JEPA):

    def __init__(self, encoder, predictor, action_encoder, projector=None,
                 pred_proj=None, patch_size: int = 16):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.patch_size = patch_size
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
