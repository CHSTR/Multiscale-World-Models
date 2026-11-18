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
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback, StarletVisCallback


def lejepa_forward(self, batch, stage, cfg, pixels=None, state_col="state"):
    """encode observations, predict next states, compute losses."""

    if pixels is not None:
        # data.render.online=true: el batch trae estados crudos, no píxeles.
        batch["pixels"] = pixels(batch[state_col])

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_act = act_emb[:, : ctx_len]
    extra_logs = {}

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
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        if lambd > 0:
            output["sigreg_loss"] = self.sigreg(
                sigreg_input(emb, cfg.loss.sigreg.get("max_tokens", None))
            )
        else:
            # Sin parámetros entrenables antes de emb (encoder congelado y sin
            # projector) SIGReg tiene gradiente cero: no se calcula.
            output["sigreg_loss"] = pred_emb.new_zeros((), dtype=torch.float32)
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

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
        forward=partial(lejepa_forward, cfg=cfg, pixels=pixels, state_col=state_col),
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
