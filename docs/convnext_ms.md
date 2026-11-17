# ConvNeXt-MS — etapas de ConvNeXt como escalas del world model

> Rama `convnext-ms` (sale de `liftwm`). Reutiliza `PyramidJEPA`, el predictor por
> escala, las pérdidas por escala y el costo CEM por escala de LiftWM; sólo cambia
> el encoder.

## Idea

LiftWM construye la pirámide con un lifting scheme en píxeles y un encoder pequeño
por sub-banda. ConvNeXt ya es jerárquico: sus 4 etapas producen mapas a
`img/4, img/8, img/16, img/32`. `ConvNeXtPyramidEncoder` toma un subconjunto de
etapas y entrega cada una como una escala (`PyramidOutput`), con una sola pasada
del backbone y pesos compartidos entre escalas.

```
píxeles (B·T, 3, 224, 224)
   └─ ConvNeXt: stem → etapa0 (56²) → etapa1 (28²) → etapa2 (14²) → etapa3 (7²)
                                     │               │               │
                          neck LN+Linear   neck LN+Linear   neck (norm DINOv3)+LN+Linear
                                     │               │               │
                              level 2 (fina)   level 1          level 0 (gruesa)
```

Convención igual que LiftWM: `level 0` = más gruesa. Las etapas intermedias de
DINOv3 ConvNeXt no tienen norma propia (`norms[0..2] = Identity`), por eso cada
escala pasa por un neck `LayerNorm → Linear(dim_etapa → embed_dim)`.

## Escalas por defecto (`stages=[1,2,3]`, img 224)

| level | etapa | dim ConvNeXt-T | grilla | tokens |
|---|---|---|---|---|
| 0 | 3 | 768 | 7×7 | 49 |
| 1 | 2 | 384 | 14×14 | 196 |
| 2 | 1 | 192 | 28×28 | 784 |

Las etapas posteriores a la última usada se podan (no se ejecutan).

## Inicialización y ajuste

| `init` | `finetune` | Qué entrena | Params entrenables (tiny, 3 escalas) |
|---|---|---|---|
| `dinov3` | `lora` (default) | LoRA `pwconv1/2` + normas + stem + necks | ~0,81 M / 28,6 M |
| `dinov3` | `full` | todo | 28,1 M |
| `dinov3` | `frozen` | sólo necks | ~0,26 M |
| `scratch` | `full` | todo, init aleatoria | 28,1 M |

- **DINOv3:** ConvNeXt T/S/B/L destilados del ViT-7B (LVD-1689M). Pesos:
  `model.encoder.ckpt_path` > `$DINOV3_CONVNEXT_CKPT` > hub oficial.
- **DINOv2:** no publicó modelos ConvNeXt (sólo ViT), así que no hay opción `dinov2`.
- **Desde cero:** `model=convnext_ms_scratch` (end-to-end, `drop_path_rate=0.1`).

## Comandos

```bash
P=/home/chr/le-wm/.venv/bin/python
# DINOv3 ConvNeXt-T + LoRA (cube):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P train.py data=ogb model=convnext_ms \
    loader.batch_size=32 output_model_name=cnms_dinov3_lora_cube \
    model.encoder.ckpt_path=/ruta/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
# Desde cero:
$P train.py data=ogb model=convnext_ms_scratch loader.batch_size=32 output_model_name=cnms_scratch_cube
# Variantes:
#   model.encoder.finetune=full|frozen
#   model.encoder.size=small|base|large
#   model.encoder.out_grids=[7,14,14]          # escala fina 784 -> 196 tokens
#   model.scale_weights=[1,1,2]                # más peso a la fina en pred_loss y CEM
#   model.encoder.starlet_levels=2             # ablación: starlet antes del stem
```

Si cambias el número de escalas (`stages`), ajusta `n_scales` en `predictor`,
`projector` y `pred_proj` (deben coincidir con `len(stages)`).

## Costo

Con grillas nativas el predictor ve 49 + 196 + 784 = 1029 tokens por frame (3 ViT
espaciales), parecido a `liftwm_vit` (1078 tokens, 4 escalas), que necesitó
`batch_size=32` en la 3090. La escala fina concentra casi todo el costo de
atención. `out_grids=[7,14,14]` baja a 441 tokens (~0,12× el costo total de
atención, Σ N²) a cambio de perder resolución en la escala fina.

El backbone corre una sola vez por frame para todas las escalas.

## Tests

`tests/test_convnext_pyramid.py` (13 tests, sin descargas: la ruta DINOv3 usa un
checkpoint falso cargado por el mismo hub). Suite completa: 57 passed.

## Próximos pasos posibles

- Comparar con `liftwm_vit` y el baseline DINOv3+LoRA (74% cube) en las mismas
  condiciones de eval.
- Matriz estática α (capas × escalas) en un predictor compartido con cross-attention
  a todas las escalas, para ver qué capa usa qué escala.
- Atención block-causal gruesa→fina entre escalas (en vez de predictores independientes).
