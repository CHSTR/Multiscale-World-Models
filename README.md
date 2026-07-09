# LeWM — Wavelet Multiescala

Repo original: **LeWM** — world model basado en JEPA con encoder ViT.

Yo simplemente implementé transformaciones multiescala basadas en wavelets
insertadas antes del patch embedding del encoder. Las variantes:

- **Starlet** (à trous): descomposición isotrópica, `C*(L+1)` canales.
- **SWT** (stationary wavelet transform): 3 orientaciones por nivel + LL, comprimidas vía `channel_proj` 1×1.
- **WaveViT**: DWT de Haar dentro de los bloques de attention (compresión de K/V 4×).

Todo el resto (training loop, JEPA, solver CEM, evaluación) es del repo original.

## Estructura

```
wavelet/          # encoders + transformaciones wavelet
config/train/model/  # configs de hydra para cada variante
viz/              # scripts de evaluación y visualización
train.py          # entry point de entrenamiento
```