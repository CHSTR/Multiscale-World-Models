# LeWM — Wavelet Multiescala

Repo original: **LeWM** — world model basado en JEPA con encoder ViT.

Yo simplemente implementé transformaciones multiescala basadas en wavelets
insertadas antes del patch embedding del encoder. Las variantes:

- **Starlet** (à trous): descomposición isotrópica, `C*(L+1)` canales.
- **SWT** (stationary wavelet transform): 3 orientaciones por nivel + LL.
- **WaveViT**: DWT de Haar dentro de los bloques de attention (compresión de K/V 4×).

Todo el resto (training loop, JEPA, solver CEM, evaluación) es del repo original.

[![wandb](https://img.shields.io/badge/wandb-logs%20de%20entrenamiento-FFCC00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/chstr/lewm/workspace?nw=nwuserchstr)

Los logs de todos los modelos entrenados están en [wandb](https://wandb.ai/chstr/lewm/workspace?nw=nwuserchstr).

## Instalación

Instrucciones de instalación y configuración del entorno en el [repo original](https://github.com/lucas-maes/le-wm).

## Entrenamiento

```bash
source .venv/bin/activate

# ViT plano (baseline)
python train.py data=tworoom model=lewm

# Starlet (L niveles)
python train.py data=tworoom model=starlet model.encoder.levels=4

# SWT (L niveles)
python train.py data=pusht model=swt model.encoder.levels=4

# WaveViT
python train.py data=tworoom model=wave_vit
```

Los checkpoints se guardan en `dataset/checkpoints/<output_model_name>/`.

## Evaluación

```bash
# Un modelo
python viz/eval_reward.py \
  policy=dataset/checkpoints/<modelo>/weights_epoch_10.pt \
  eval.dataset_name=dataset/datasets/tworoom.h5 \
  eval.num_eval=100 eval.goal_offset_steps=50 \
  eval.eval_budget=100 \
  --config-name=tworoom
```
Se puede utilizar también el eval.py original.

## Mapas de atención

```bash
python viz/generate_attention_gif.py \
  --ckpt dataset/checkpoints/<modelo>/weights_epoch_10.pt \
  --start 685385 --nframes 30 --dataset tworoom \
  --mode heads --aggregate-layers \
  --output-dir outputs/attn
```

## Estructura

```
wavelet/             # encoders + transformaciones wavelet
config/train/model/  # configs de hydra para cada variante
config/eval/         # configs de evaluación
viz/                 # scripts de evaluación y visualización
train.py             # entry point de entrenamiento
```
