import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # registra el formato HDF5
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa_spatial import spatial_losses
from jepa_pyramid import pyramid_losses
from module import SIGReg, sigreg_input
from prioritized_replay import (replay_collect, replay_weights, split_time, valid_windows,
                                weighted_mean)
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback, StarletVisCallback


def lejepa_forward(self, batch, stage, cfg, pixels=None, state_col="state", replay=None):
    """encode observations, predict next states, compute losses.

    ``replay`` (``prioritized_replay.ReplayState``, opcional): replay priorizado +
    etiquetas de K horizontes para la red de alucinación H. La ventana trae
    ``history_size + K`` frames, pero pred_loss y SIGReg usan sólo los primeros
    ``history_size + num_preds`` (mismo cálculo que el baseline); con
    ``replay=None`` el forward es exactamente el original.
    """

    if pixels is not None:
        # data.render.online=true: el batch trae estados crudos, no píxeles.
        batch["pixels"] = pixels(batch[state_col])

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    full_batch = None
    if replay is not None:
        # Antes de nan_to_num: ventanas que cruzan un borde de secuencia en alguna
        # acción usada por el rollout de etiquetado (posiciones < HS + K - 1).
        valid = valid_windows(batch["action"], ctx_len + replay.cfg.horizon - 1)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    if replay is not None:
        full_batch = batch
        batch = split_time(full_batch, ctx_len + n_preds)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_act = act_emb[:, : ctx_len]
    extra_logs = {}

    if replay is not None and (isinstance(emb, dict) or emb.ndim != 3):
        raise NotImplementedError(
            "replay priorizado (F1) sólo está implementado para el JEPA global "
            "(emb (B, T, D), p. ej. model=dinov3_lora); usa replay=disabled con este modelo."
        )

    if isinstance(emb, dict):
        # LiftWM: latentes por escala. pred_loss = media ponderada por scale_weights;
        # SIGReg = media entre escalas (mantiene la escala de lambda de un solo término).
        ctx_emb = {l: v[:, :ctx_len] for l, v in emb.items()}
        tgt_emb = {l: v[:, n_preds: n_preds + ctx_len] for l, v in emb.items()}
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        w = self.model.scale_weights
        w = w / w.sum()
        output["pred_loss"] = sum(
            w[l] * (pred_emb[l] - tgt_emb[l]).pow(2).mean() for l in emb
        )
        if lambd > 0:
            output["sigreg_loss"] = torch.stack([
                self.sigreg(sigreg_input(emb[l], cfg.loss.sigreg.get("max_tokens", None)))
                for l in emb
            ]).mean()
        else:
            output["sigreg_loss"] = next(iter(pred_emb.values())).new_zeros((), dtype=torch.float32)
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        extra, extra_logs = pyramid_losses(
            self.model, batch, emb, pred_emb, tgt_emb, cfg, act_emb=act_emb
        )
        output["loss"] = output["loss"] + extra
        for l in emb:
            extra_logs[f"pred_loss_l{l}"] = (pred_emb[l] - tgt_emb[l]).pow(2).mean().detach()
    else:
        ctx_emb = emb[:, :ctx_len]
        # label. Con el batch por defecto (T = history_size + num_preds) es igual a
        # emb[:, n_preds:]; el recorte deja usar batches más largos (p. ej. para
        # loss.rollout con más pasos) sin cambiar el teacher forcing.
        tgt_emb = emb[:, n_preds: n_preds + ctx_len]
        pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

        # LeWM loss
        w_pred = w_sig = None
        if replay is not None:
            # Pesos IS: w_pred con beta (0 = focal, sin corrección); SIGReg siempre
            # con beta=1 (ECF ponderada) -> isotropía bajo la distribución real.
            w_pred, w_sig = replay_weights(full_batch, replay, stage)
        if w_pred is None:
            output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        else:
            per_sample = (pred_emb - tgt_emb).float().pow(2).mean(dim=(1, 2))   # (B,)
            output["pred_loss"] = weighted_mean(per_sample, w_pred)
        if lambd > 0:
            sig_in = sigreg_input(emb, cfg.loss.sigreg.get("max_tokens", None))
            output["sigreg_loss"] = (self.sigreg(sig_in) if w_sig is None
                                     else self.sigreg(sig_in, weights=w_sig))
        else:
            # Sin parámetros entrenables antes de emb (encoder congelado y sin
            # projector) SIGReg tiene gradiente cero: no se calcula.
            output["sigreg_loss"] = pred_emb.new_zeros((), dtype=torch.float32)
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

        if replay is not None:
            # Etiquetas de K horizontes (sin gradiente) -> replay.last (lo lee el callback).
            extra_logs.update(replay_collect(
                self.model, full_batch, emb, act_emb, valid, replay,
                history_size=ctx_len, t_main=ctx_len + n_preds, stage=stage,
            ))

        # JEPA espacial (emb (B,T,N,D)): gate L1, dinámica inversa, rollout y diagnóstico.
        if emb.ndim == 4:
            extra, extra_logs = spatial_losses(
                self.model, batch, emb, pred_emb, tgt_emb, cfg, act_emb=act_emb
            )
            output["loss"] = output["loss"] + extra

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict.update({f"{stage}/{k}": v for k, v in extra_logs.items()})
    if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'level_weights'):
        for i, w in enumerate(self.model.encoder.level_weights):
            losses_dict[f"{stage}/level_weight_{i}"] = w.detach()
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    fraction = dataset_cfg.pop("fraction", 1.0)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    # data.render.online=true: el dataset no trae `pixels`; los frames se
    # dibujan en el device con stable-envs a partir de la columna de estado.
    render_cfg = cfg.data.get("render", None)
    online = bool(render_cfg and render_cfg.get("online", False))
    state_col = str(render_cfg.get("state_column", "state")) if online else "state"

    transforms = [] if online else [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
    ]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if online and col == state_col:
                continue  # el renderer necesita el estado crudo, sin normalizar
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    if transforms:
        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

    # Grab one sample frame for starlet visualization
    sample_frame = None
    if not online and cfg.model.encoder._target_ == "wavelet.starlet_encoder.StarletEncoder":
        for i in range(min(100, len(dataset))):
            try:
                row = dataset.get_row_data(i)
                sample_frame = row["pixels"].unsqueeze(0)  # (1, 3, H, W)
                break
            except Exception:
                continue

    if fraction < 1.0:
        n = int(len(dataset) * fraction)
        dataset = torch.utils.data.Subset(dataset, range(n))

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # Replay priorizado (replay=per_is|per_focal|per_shuffled|uniform_labels): el
    # sampler vive en el proceso principal y entrega claves (idx, P_idx); los
    # datasets se envuelven para que cada ventana traiga su idx y su probabilidad.
    replay_state = None
    replay_cfg = cfg.get("replay", None)
    if replay_cfg is not None and replay_cfg.get("enabled", False):
        from prioritized_replay import (IndexedDataset, PrioritizedSampler, ReplayConfig,
                                        ReplayState)
        rcfg = ReplayConfig.from_cfg(replay_cfg)
        train_set, val_set = IndexedDataset(train_set), IndexedDataset(val_set)
        replay_state = ReplayState(rcfg, n=len(train_set))
        sampler = PrioritizedSampler(
            replay_state, num_samples=len(train_set), batch_size=cfg.loader.batch_size,
            generator=torch.Generator().manual_seed(cfg.seed + 1),
        )
        train = torch.utils.data.DataLoader(train_set, **cfg.loader, sampler=sampler, drop_last=True)
        print(f"[replay] sampler={rcfg.sampler} priority={rcfg.priority} K={rcfg.horizon} "
              f"alpha={rcfg.alpha} beta={rcfg.beta_start}->{rcfg.beta_end} eps={rcfg.eps_uniform} "
              f"warmup_steps={sampler.warmup_steps} hnet={rcfg.hnet_enabled}({rcfg.hnet_input}) "
              f"N_train={len(train_set)}")
    else:
        train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    pixels = None
    if online:
        from online_render import Pixels
        stats = spt.data.dataset_stats.ImageNet
        pixels = Pixels(
            OmegaConf.to_container(render_cfg.env, resolve=True),
            cfg.img_size, stats["mean"], stats["std"],
        )

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg, pixels=pixels, state_col=state_col,
                        replay=replay_state),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    callbacks = [object_dump_callback]
    if sample_frame is not None:
        callbacks.append(StarletVisCallback(
            run_dir=run_dir,
            sample_frame=sample_frame,
            levels=cfg.model.encoder.get("levels", 4),
        ))
    if replay_state is not None:
        # Buffer + H (pasiva) + diagnósticos en <run_dir>/replay/ (H con su propio
        # optimizador: no comparte gradiente, clipping ni scheduler con el WM).
        from prioritized_replay import PrioritizedReplayCallback
        callbacks.append(PrioritizedReplayCallback(
            replay_state, train_dataset=train_set, run_dir=run_dir,
            z_dim=cfg.embed_dim, a_dim=cfg.model.action_encoder.emb_dim,
            history_size=cfg.history_size,
        ))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # Resume: `resume_from=<path a un last.ckpt>` restaura estado completo
    # (optimizer/scheduler/epoch/RNG) y continúa hasta trainer.max_epochs.
    # Sin `resume_from` el comportamiento es el original (init desde
    # weights_epoch_*.ckpt si existe, sólo pesos).
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        ckpt_path = Path(resume_from).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_from apunta a un archivo inexistente: {ckpt_path}")
        weights_only = False
    else:
        ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
        ckpt_path = ckpt_path if ckpt_path.exists() else None
        weights_only = True

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        weights_only=weights_only,
    )

    manager()
    if pixels is not None:
        pixels.close()
    return


if __name__ == "__main__":
    run()
