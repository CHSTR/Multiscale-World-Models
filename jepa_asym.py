"""Asymmetric JEPA: separate input/target projectors + next-``n_future`` prediction.

This is the Fase B model. It extends :class:`jepa.JEPA` with three changes:

* ``encode`` uses two projectors -- ``projector_in`` over the input token and
  ``projector_tgt`` over the target tokens -- and exposes ``emb_in`` and
  ``emb_tgt`` (in addition to the legacy ``emb`` alias = ``emb_tgt[...,0,:]``).
* ``predict`` accepts ``n_future`` and forwards it to the (masked) predictor,
  so a single call predicts ``n_future`` future frames at once.
* ``rollout`` is vectorised (no autoregressive loop): it encodes the real
  context, runs every planning action through the action encoder once and makes
  a single ``predict(x_ctx, c, n_future=T-H_ctx)`` call.

Training loss lives in ``train.asym_forward``.
"""

import torch
from einops import rearrange
from torch import nn

from jepa import JEPA, detach_clone


class AsymJEPA(JEPA):
    """JEPA with separate in/target projectors and n-step prediction."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector_in,
        projector_tgt,
        pred_proj=None,
        num_preds=None,
    ):
        # Initialise the JEPA parent so the standard attributes
        # (self.encoder / predictor / action_encoder / projector / pred_proj)
        # all exist. ``super().__init__()`` (no args) used to be called here,
        # but that resolves to JEPA.__init__(encoder, predictor, action_encoder,
        # ...) which requires those positional args -> every AsymJEPA
        # instantiation crashed. See PROGRESS_DIRECT_PREDICTOR.md.
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector_in,
            pred_proj=pred_proj,
        )

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector_in = projector_in or nn.Identity()
        self.projector_tgt = projector_tgt or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Planning horizon the model was trained to predict (used by rollout).
        self.num_preds = num_preds

    def encode(self, info):
        """Encode observations -> emb_in, emb_tgt, emb (alias), act_emb.

        ``emb_in``  = ``projector_in(tok_in)``            (B, T, D)
        ``emb_tgt`` = ``projector_tgt(tok_tgt)``          (B, T, K, D)
        ``emb``     = ``emb_tgt[..., 0, :]`` (compat alias) (B, T, D)
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)

        emb_in = self.projector_in(output.tok_in)
        emb_tgt = self.projector_tgt(output.tok_tgt)
        info["emb_in"] = rearrange(emb_in, "(b t) d -> b t d", b=b)
        info["emb_tgt"] = rearrange(emb_tgt, "(b t) k d -> b t k d", b=b)
        info["emb"] = info["emb_tgt"][:, :, 0, :]  # alias: first target

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, x_ctx, act_emb, n_future=0):
        """Predict ``n_future`` future embeddings in one shot.

        ``x_ctx``: (B, ctx, D)
        ``act_emb``: (B, ctx+n_future, A_emb)
        returns (B, ctx+n_future, D)
        """
        preds = self.predictor(x_ctx, act_emb, n_future=n_future)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=x_ctx.size(0))
        return preds

    ####################
    ##  Inference    ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Vectorised rollout (no autoregressive loop).

        ``info["pixels"]``: (B, S, H_ctx, C, H, W) -- the visible context frames.
        ``action_sequence``: (B, S, T, action_dim) -- full planning horizon.

        Encodes the real context (``emb_in``), runs every action through the
        action encoder once, and predicts the ``T - H_ctx`` future frames with a
        single ``predict(x_ctx, c, n_future=T-H_ctx)`` call, returning
        ``info["predicted_emb"]`` of shape (B, S, T, D).
        """
        assert "pixels" in info, "pixels not in info_dict"
        H_ctx = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]

        # The planning horizon must fit the horizon the model can predict.
        plan_cfg = info.get("plan_config", None)
        horizon = (
            getattr(plan_cfg, "horizon", None)
            if plan_cfg is not None
            else info.get("horizon")
        )
        if horizon is not None and self.num_preds is not None:
            assert (
                horizon <= self.num_preds
            ), f"plan_config.horizon ({horizon}) > model num_preds ({self.num_preds})"

        # Encode the real context for every (batch, sample).
        px = rearrange(info["pixels"], "b s t ... -> (b s) t ...")
        ctx_info = {"pixels": px}
        if "action_history" in info:
            ctx_info["action"] = rearrange(
                info["action_history"], "b s ... -> (b s) ..."
            )
        ctx_info = self.encode(ctx_info)
        emb_in = ctx_info["emb_in"].detach().clone()  # (BS, H_ctx, D)

        # All planning actions -> (BS, T, A_emb)
        acts = rearrange(action_sequence, "b s t a -> (b s) t a")
        act_emb = self.action_encoder(acts)

        # One-shot prediction of the future frames.
        pred = self.predict(emb_in, act_emb, n_future=T - H_ctx)  # (BS, T, D)
        pred_rollout = rearrange(pred, "(b s) t d -> b s t d", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Inherited JEPA cost, goal embedding picked from the target projector."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        # goal = furthest look-ahead target per position.
        k_H = goal["emb_tgt"].size(2) - 1
        info_dict["goal_emb"] = goal["emb_tgt"][:, :, k_H]

        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost
