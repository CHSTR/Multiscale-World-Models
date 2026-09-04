"""Entrenamiento del decoder de diagnóstico (CLSDecoder) sobre un encoder CONGELADO.

Replica la herramienta de visualizacion de LeWM (App. D del paper, "Decoder
(Visualization Only)"). El decoder NUNCA propaga gradiente al world model: sirve
solo para inspeccionar que informacion retiene el embedding, no para entrenarlo.
Anadir reconstruccion como termino de perdida degrada el control (Tab. 7 del
paper: 96.0 -> 86.0 en Push-T), asi que aqui el encoder va en eval() y con
requires_grad_(False) sin excepcion.

Funciona con cualquier encoder del repo (ViT plano, Starlet, SWT, WaveViT):
la dimension latente se infiere con un forward de prueba en vez de hardcodearse.

Uso tipico
----------
    python decoder_train.py --ckpt dataset/checkpoints/starlet_l4/weights_epoch_10.pt

    # decodificar el espacio post-projector (el que ve SIGReg y el predictor)
    python decoder_train.py --ckpt ... --source emb

    # barrido de checkpoints para reproducir la Fig. 10 del paper
    for e in 1 3 5 10; do
        python decoder_train.py --ckpt dataset/checkpoints/lewm/weights_epoch_${e}.pt \
            --steps 20000 --out-dir outputs/decoder/epoch_${e}
    done
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import stable_pretraining as spt
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registra el formato HDF5

from utils import get_img_preprocessor


# --------------------------------------------------------------------------- #
#  Decoder
# --------------------------------------------------------------------------- #


class CLSDecoder(nn.Module):
    """Decodifica un unico embedding global a una imagen RGB.

    Nota sobre la arquitectura: el key/value de la cross-attention es un solo
    token (el CLS proyectado), asi que el softmax es identicamente 1.0 y la
    atencion no selecciona nada. Cada capa se reduce a un broadcast aditivo del
    mismo vector sobre las P queries, seguido de un MLP por posicion. Tampoco
    hay self-attention entre queries: cada parche se decodifica de forma
    independiente. Es deliberadamente debil, y eso es bueno para un diagnostico:
    si la imagen sale bien, la informacion estaba en el latente y no la invento
    el decoder.
    """

    def __init__(
        self,
        cls_dim=384,
        img_size=224,
        patch_size=16,
        dim=256,
        heads=8,
        depth=3,
    ):
        super().__init__()

        assert img_size % patch_size == 0, "img_size debe ser divisible por patch_size"

        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.queries, std=0.02)

        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "norm1": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )

        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

        self.patch_size = patch_size

    def forward(self, x):
        """x: (B, D) -> (B, 3, img_size, img_size), en espacio normalizado."""
        B = x.size(0)
        P = self.num_patches

        kv = self.cls_proj(x).unsqueeze(1)  # (B, 1, dim)
        q = self.queries.expand(B, -1, -1)  # (B, P, dim)

        for layer in self.layers:
            attn_out = layer["cross_attn"](q, kv, kv)[0]
            q = layer["norm1"](q + attn_out)
            mlp_out = layer["mlp"](q)
            q = layer["norm2"](q + mlp_out)

        patches = self.to_pixels(q)  # (B, P, patch_dim)
        patches = patches.reshape(B, P, self.patch_size, self.patch_size, 3)

        H = W = int(self.num_patches**0.5)
        patches = patches.reshape(B, H, W, self.patch_size, self.patch_size, 3)
        img = patches.permute(0, 5, 1, 3, 2, 4)
        img = img.reshape(B, 3, H * self.patch_size, W * self.patch_size)

        return img


# --------------------------------------------------------------------------- #
#  Encoder congelado
# --------------------------------------------------------------------------- #


def encode_frames(model, pixels, source="cls"):
    """pixels: (N, C, H, W) ya normalizadas -> (N, D).

    Tolera encoders que devuelven un objeto HF (.last_hidden_state) o un tensor
    de tokens directo, y encoders que no aceptan interpolate_pos_encoding.
    """
    try:
        out = model.encoder(pixels, interpolate_pos_encoding=True)
    except TypeError:
        out = model.encoder(pixels)

    tokens = getattr(out, "last_hidden_state", out)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]

    cls = tokens[:, 0]
    if source == "cls":
        return cls
    return model.projector(cls)


@torch.no_grad()
def infer_latent_dim(model, img_size, device, source):
    dummy = torch.zeros(2, 3, img_size, img_size, device=device)
    return encode_frames(model, dummy, source=source).size(-1)


def load_world_model(ckpt, device):
    model = swm.wm.utils.load_pretrained(ckpt)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


# --------------------------------------------------------------------------- #
#  Datos
# --------------------------------------------------------------------------- #


def build_dataset(cfg):
    """Reutiliza exactamente la config del run de entrenamiento del encoder.

    Critico: el decoder tiene que ver la misma normalizacion de imagen
    (ImageNet stats + resize) que vio el encoder, o las reconstrucciones salen
    mal por una razon que no tiene que ver con el latente.
    """
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    dataset_cfg.pop("fraction", None)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    dataset = swm.data.load_dataset(name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    dataset.transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=cfg.img_size
    )
    return dataset


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def denormalize(x):
    """Deshace la normalizacion ImageNet para poder mirar las imagenes."""
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def save_preview(real, recon, path, n=6):
    real = denormalize(real[:n]).cpu().numpy().transpose(0, 2, 3, 1)
    recon = denormalize(recon[:n]).cpu().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, len(real), figsize=(2.2 * len(real), 4.8))
    axes = axes.reshape(2, -1)
    for i in range(len(real)):
        axes[0, i].imshow(real[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("real")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(path.stem, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig  # el caller decide si cerrarla (p.ej. tras loguearla en wandb)


# --------------------------------------------------------------------------- #
#  Entrenamiento
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="ruta relativa al cache dir, p.ej. dataset/checkpoints/lewm/weights_epoch_10.pt")
    p.add_argument("--config", default=None, help="config.yaml del run (por defecto: junto al ckpt)")
    p.add_argument("--source", choices=["cls", "emb"], default="cls",
                   help="cls = token crudo del encoder (paper); emb = post-projector (espacio de planificacion)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preview-every", type=int, default=1000)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="usa bf16 en el forward del decoder")
    p.add_argument("--wandb", action="store_true",
                   help="loguea en wandb el preview (real vs recon) cada --preview-every steps")
    p.add_argument("--wandb-entity", default="chstr")
    p.add_argument("--wandb-project", default="lewm")
    p.add_argument("--wandb-name", default=None,
                   help="nombre del run (por defecto: decoder_<ckpt_stem>_<source>)")
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(swm.data.utils.get_cache_dir())

    # -- config del run de entrenamiento del encoder
    cfg_path = Path(args.config) if args.config else (cache_dir / args.ckpt).parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No encuentro {cfg_path}. Pasa --config con el config.yaml del run."
        )
    cfg = OmegaConf.load(cfg_path)

    out_dir = Path(args.out_dir) if args.out_dir else (cache_dir / args.ckpt).parent / f"decoder_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- world model congelado
    model = load_world_model(args.ckpt, device)
    latent_dim = infer_latent_dim(model, cfg.img_size, device, args.source)
    print(f"[decoder] encoder congelado | source={args.source} | latent_dim={latent_dim}")

    # -- wandb (opcional)
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"decoder_{Path(args.ckpt).stem}_{args.source}"
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name)
        wandb.config.update(
            {"ckpt": args.ckpt, "source": args.source, "latent_dim": latent_dim,
             "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
             "patch_size": args.patch_size, "dim": args.dim, "heads": args.heads,
             "depth": args.depth}
        )
        print(f"[decoder] wandb run: {args.wandb_entity}/{args.wandb_project}/{run_name}")

    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    assert n_trainable == 0, f"el world model tiene {n_trainable} params entrenables, deberia ser 0"

    # -- datos
    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    stream = infinite(loader)

    # -- decoder
    decoder = CLSDecoder(
        cls_dim=latent_dim,
        img_size=cfg.img_size,
        patch_size=args.patch_size,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
    ).to(device)
    print(f"[decoder] {sum(p_.numel() for p_ in decoder.parameters()) / 1e6:.2f}M params")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup))
        * 0.5
        * (1 + torch.cos(torch.tensor(min(s, args.steps) / args.steps * 3.141592653589793)).item()),
    )

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.amp
        else torch.autocast(device_type=device.type, enabled=False)
    )

    running = 0.0
    for step in range(args.steps):
        batch = next(stream)
        pixels = batch["pixels"].float().to(device, non_blocking=True)

        # (B, T, C, H, W) -> (B*T, C, H, W); cada frame es una muestra independiente
        if pixels.ndim == 5:
            pixels = pixels.flatten(0, 1)

        with torch.no_grad():
            z = encode_frames(model, pixels, source=args.source).detach()

        with amp_ctx:
            recon = decoder(z)
            loss = F.mse_loss(recon.float(), pixels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()

        if (step + 1) % args.log_every == 0:
            print(f"step {step + 1:>7d}/{args.steps}  mse {running / args.log_every:.5f}  lr {sched.get_last_lr()[0]:.2e}")
            running = 0.0

        if (step + 1) % args.preview_every == 0 or (step + 1) == args.steps:
            decoder.eval()
            with torch.no_grad():
                recon = decoder(z).float()
                fig = save_preview(pixels, recon, out_dir / f"recon_step_{step + 1}.png")
                if args.wandb:
                    import wandb
                    wandb.log({"recon/real_vs_recon": wandb.Image(fig),
                               "recon/mse": loss.item()}, step=step + 1)
                plt.close(fig)
            decoder.train()
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "cls_dim": latent_dim,
                    "img_size": cfg.img_size,
                    "patch_size": args.patch_size,
                    "dim": args.dim,
                    "heads": args.heads,
                    "depth": args.depth,
                    "source": args.source,
                    "encoder_ckpt": args.ckpt,
                    "step": step + 1,
                },
                out_dir / "decoder.pt",
            )

    print(f"[decoder] listo. Salidas en {out_dir}")
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
