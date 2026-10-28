"""Media y desviación estándar por canal de los patch tokens del encoder congelado.

Las usa ``SpatialJEPA(token_stats_path=...)`` para estandarizar los tokens
(``(x - mean) / std``) antes del predictor y del costo de planning. Es una
transformación lineal invertible: no descarta información, sólo evita que el
MSE quede dominado por los pocos canales de DINO con mucha varianza.

Las estadísticas son por dataset (entorno): calcúlalas una vez por ``data=``.
Se acumulan en float64 sobre todos los tokens ``(frames x N)`` de ``n_rows``
filas al azar del split completo.

Uso (mismos overrides de hydra que ``train.py``)::

    python compute_token_stats.py +experiment=nca_frozen data=pusht \\
        +stats.out=token_stats/pusht_dinov3s16.pt +stats.n_rows=4000

Salida: ``{"mean": (D,), "std": (D,), "count": int, "patch_size": int,
"img_size": int, "dataset": str}``.
"""

import os
from pathlib import Path

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registra el formato HDF5
import torch
from einops import rearrange
from omegaconf import OmegaConf

from utils import get_img_preprocessor


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    st = OmegaConf.to_container(cfg.get("stats", None) or {}, resolve=True)
    n_rows = int(st.get("n_rows", 4000))
    batch_size = int(st.get("batch_size", 32))
    seed = int(st.get("seed", 0))
    device = torch.device(st.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    patch_size = int(cfg.model.get("patch_size", 16))

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset_cfg.pop("fraction", None)
    out = Path(st.get("out", f"token_stats/{Path(dataset_name).stem}_p{patch_size}.pt"))

    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=os.environ.get("LOCAL_DATASET_DIR", None),
        **dataset_cfg,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    )

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[: min(n_rows, len(dataset))].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx), batch_size=batch_size, shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    encoder = hydra.utils.instantiate(cfg.model.encoder).to(device).eval().requires_grad_(False)

    s1 = s2 = None
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixels = rearrange(batch["pixels"].float().to(device), "b t ... -> (b t) ...")
            n = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
            tokens = encoder(pixels, interpolate_pos_encoding=True).last_hidden_state
            # Mismo recorte que SpatialJEPA.encode: los N últimos tokens son los patches.
            x = tokens[:, -n:].reshape(-1, tokens.size(-1)).double()
            s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
            s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
            count += x.size(0)
            if i % 20 == 0:
                print(f"[stats] {i * batch_size}/{len(idx)} filas, {count} tokens")

    mean = s1 / count
    std = (s2 / count - mean * mean).clamp_min(0).sqrt()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": mean.float().cpu(), "std": std.float().cpu(), "count": count,
        "patch_size": patch_size, "img_size": int(cfg.img_size), "dataset": dataset_name,
    }, out)

    ratio = (std.max() / std.clamp_min(1e-12).min()).item()
    print(f"[stats] guardado {out}: D={mean.numel()} tokens={count} "
          f"std min/med/max = {std.min():.4f}/{std.median():.4f}/{std.max():.4f} "
          f"(max/min = {ratio:.1f})")


if __name__ == "__main__":
    run()
