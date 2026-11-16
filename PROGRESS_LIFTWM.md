# PROGRESS_LIFTWM

World model JEPA multiescala con pirámide wavelet (lifting scheme) y predictor por
escala **intercambiable ViT/NCA**. Rama `liftwm`. Diseño y plan originales en
`/home/chr/.opencode/plan/2026-11-16-liftwm-{design,plan}.md`.

## Estado: pipeline completo y verificado (T1–T5). Falta T6 (entrenamiento real + eval).

### Componentes
| Archivo | Qué hace |
|---|---|
| `wavelet/lifting.py` | `LiftingScheme` 2D separable invertible (init Haar/CDF 5/3, filtros aprendibles). `Pyramid(approx, details)`. Reconstrucción perfecta fp32 (~7e-7) y bf16 (<0.016). |
| `wavelet/pyramid_encoder.py` | `PyramidEncoder`: píxeles → pirámide → tokens espaciales por escala `(B,N_l,D_l)`, sin pooling global. `ScaleTokens`, `PyramidOutput`. |
| `wavelet/vit_predictor.py` | `ViTSpatialPredictor`: transformer espacial **no causal** + AdaLN-zero con la acción (coherente con `ARPredictor`/`MaskedPredictor` de la rama `direct-predictor`). Misma interfaz que NCA. |
| `wavelet/hier_predictor.py` | `HierarchicalPredictor`: router por escala sobre cualquier predictor con interfaz espacial (NCA o ViT). |
| `wavelet/factories.py` | `scale_mlp_dict`, `hier_nca`, `hier_vit` (ModuleDict por escala para Hydra). |
| `jepa_pyramid.py` | `PyramidJEPA`: encode/predict/rollout/criterion con dict-emb por escala. `get_cost` heredado de `JEPA`. `pyramid_losses` (gate L1, IDM escala fina, rollout). |
| `train.py` | Rama `isinstance(emb, dict)`: pred_loss = media ponderada por `scale_weights`; SIGReg = **media** entre escalas. Baselines intactos. |
| `config/train/model/liftwm.yaml` | Config NCA (Stage 1 Haar fijo). |
| `config/train/model/liftwm_vit.yaml` | Config ViT. |

### Verificación
- `.venv_hf/bin/python -m pytest tests/ -q` → **43 passed** (lifting 9, encoder 4, predictores 8, PyramidJEPA 6, smoke 4 + repo 29 sin regresiones).
- Smoke-train real (tworoom, 3 batches) en el entorno canónico: **NCA y ViT** entrenan, loguean losses por escala y guardan checkpoint.
- `get_cost` (camino CEM completo: encode goal + rollout + criterion) testeado.

## Decisiones (builder-check resuelto)
- **Q1 grids:** `stem_patch=4`, `levels=3` → grids por escala **7/7/14/28** (49/49/196/784 tokens). El token fino cubre 8×8 px originales (más fino que DINO-16 → preserva orientación). Configurable. Para la 3090 (24 GB): `loader.batch_size=64` + `grad_checkpoint=true` (ya activo en NCA). Si OOM/tiempo: subir `stem_patch` o bajar batch.
- **Q2 entorno:** **tests** en `.venv_hf` (tiene pytest). **Entrenamiento y eval** en `/home/chr/le-wm/.venv` (swm 0.1.0 con `solver.CEMSolver(model=...)`, el que produjo los baselines; `.venv_hf` tiene el swm nuevo SIN `solver`, no sirve para el eval CEM).
- **Q3 SIGReg:** media entre escalas (mantiene la escala de λ=0.09 tuneada para un término). `scale_weights` pondera `pred_loss` y `criterion` (no SIGReg).
- **Predictor intercambiable:** `model=liftwm` (NCA) vs `model=liftwm_vit` (ViT). Misma interfaz, mismo JEPA/planning.

## Cómo correr (entorno canónico `/home/chr/le-wm/.venv/bin/python`)

### Stage 1 — Haar fijo (valida pipeline + baseline ablación)
```bash
P=/home/chr/le-wm/.venv/bin/python
# cube (juez principal) y pusht (sanity), NCA:
$P train.py data=ogb   model=liftwm loader.batch_size=64 output_model_name=liftwm_haar_cube
$P train.py data=pusht model=liftwm loader.batch_size=64 output_model_name=liftwm_haar_pusht
# variante ViT:
$P train.py data=ogb   model=liftwm_vit loader.batch_size=64 output_model_name=liftwm_vit_cube
```

### Stage 2 — lifting aprendible (ablación vs Stage 1)
```bash
$P train.py data=ogb model=liftwm loader.batch_size=64 \
    model.encoder.learnable_lifting=true output_model_name=liftwm_learn_cube
```

### Eval (CEM, 50 eps / budget 50 / horizon 5 — paridad con baseline 74%)
```bash
$P viz/eval_reward.py \
    policy=dataset/checkpoints/liftwm_haar_cube/weights_epoch_10.pt \
    --config-name=cube
# (los checkpoints se guardan en ~/.stable_worldmodel/checkpoints/<output_model_name>/;
#  ajustar la ruta policy al checkpoint real)
```

## Objetivos y próximos pasos (T6)
- **Juez:** cube > **74%** (mejor baseline DINOv3+LoRA; DINO-WM reporta 86%). Push-T ≥ ~94% (no caer).
- Entrenar Stage 1 (NCA y ViT) en cube/pusht, evaluar, comparar.
- Stage 2 (lifting aprendible) → ablación fijo vs aprendible.
- Si no mejora: ajustar `scale_weights` (peso de escalas finas en criterion), `stem_patch`,
  pasos NCA / depth ViT, dims por escala.
- Ablación clave que prueba la hipótesis: ¿preservar los detalles finos en latentes
  propios + costo por escala recupera la orientación que el pooling global descartaba?

## Riesgos abiertos
- Cómputo: escala fina 784 tokens × 4 predictores; vigilar memoria/tiempo en la 3090.
  - **ViT OOM con batch 64** (23 GB). Fix: `loader.batch_size=32` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (~12 GB, estable a 2.7 it/s).
  - Tiempo ViT cube: ~51k steps/epoch × 10 ≈ **52 h** a batch 32. Para iterar rápido: menos epochs, `stem_blocks=1`, `depth=2`, o `stem_patch` mayor (menos tokens, pero re-descarta detalle).
- `stem_patch` grande podría re-descartar detalle (tradeoff con cómputo) — es la perilla arquitectónica principal.
- Colapso de representación con lifting aprendible (anti-goal): SIGReg por escala + ablación Stage 1 vs 2 lo vigilan.
