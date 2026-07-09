import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

from wavelet.starlet_torch import starlet_conv4d

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class StarletVisCallback(Callback):
    """Save starlet decomposition weighted by learned level_weights each epoch."""

    def __init__(self, run_dir: str, sample_frame: torch.Tensor, levels: int = 4):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sample_frame = sample_frame  # (1, 3, H, W) normalized
        self.levels = levels

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        encoder = pl_module.model.encoder
        if not hasattr(encoder, 'level_weights'):
            return

        w = encoder.level_weights.detach()
        x = self.sample_frame.to(device=w.device)

        with torch.no_grad():
            bands = starlet_conv4d(x, self.levels, scale=w, filter=encoder.filter)
            bands = bands.view(1, self.levels + 1, -1, *x.shape[2:])[0]  # (L+1, 3, H, W)

        n = self.levels + 1
        fig, axes = plt.subplots(2, (n + 2) // 2, figsize=(5 * ((n + 2) // 2), 8))
        axes = axes.ravel()
        axes[0].imshow(self._to_np(x))
        axes[0].set_title("original", fontsize=9)
        axes[0].axis("off")
        for i in range(n):
            ax = axes[i + 1]
            band = self._to_np(bands[i])
            ax.imshow(band)
            label = f"L{i+1}" if i < self.levels else "residual"
            ax.set_title(f"{label} (w={w[i]:.3f})", fontsize=9)
            ax.axis("off")
        for j in range(n + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Starlet (filter={encoder.filter}) — epoch {trainer.current_epoch + 1}", fontsize=11)
        plt.tight_layout()
        path = self.run_dir / f"starlet_epoch_{trainer.current_epoch + 1}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.cpu().numpy().squeeze()  # remove batch dim if present
        if arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)
        lo, hi = arr.min(), arr.max()
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)