# Probe lineal y no-lineal sobre latentes de DINOv3 en Cube

Fecha: 2026-10-29

## Intención y por qué

El paper reporta probing para analizar qué información contienen los latentes
aprendidos. Queremos responder: ¿la representación de DINOv3 entrenada con JEPA
en Cube captura el estado físico de la escena? Un probe lineal mide si la
información es linealmente accesible; un probe no-lineal (MLP) mide si está
presente pero entrelazada.

## Objetivo

Entrenar probes lineal y MLP sobre features congeladas de checkpoints de Cube y
reportar, por cantidad física: **MSE** y **coeficiente de correlación de
Pearson** entre predicción y ground truth.

## No-objetivos

- No se entrena ni ajusta el world model ni el encoder.
- No se exploran arquitecturas de probe más allá de ridge lineal y MLP de 2 capas.
- No se reportan métricas de rollout/planificación (eso ya existe).

## Datos

`ogbench/cube_single_expert.h5` (2.01M frames, 10k episodios).

Variables objetivo (8 grupos, 27 dimensiones):

| Grupo | Columnas | Dim |
|---|---|---|
| joint_pos | `proprio_joint_pos` | 6 |
| joint_vel | `proprio_joint_vel` | 6 |
| effector_pos | `proprio_effector_pos` | 3 |
| effector_yaw | `proprio_effector_yaw` | 1 |
| gripper | `proprio_gripper_opening`, `_contact`, `_vel` | 3 |
| block_pos | `privileged_block_0_pos` | 3 |
| block_quat | `privileged_block_0_quat` | 4 |
| block_yaw | `privileged_block_0_yaw` | 1 |

Muestreo: **200,000 frames** uniformes sobre las filas (seed 0).
Split: **por episodio**, 80/20 train/val (`ep_idx`), sin fuga entre splits.
Normalización: targets z-score con estadísticas de train (solo train).

## Modelos y features

Checkpoints (encoder congelado, `swm.wm.utils.load_pretrained`):

| Checkpoint | Rol |
|---|---|
| `dinov3_s_lora_cube/weights_epoch_5.pt` | mejor epoch de DINOv3 (74.0 SR) |
| `dinov3_s_lora_cube/weights_epoch_10.pt` | epoch final |
| `starlet_L3_cube/weights_epoch_10.pt` | contraste wavelet-scratch (74.0 SR) |

Por cada checkpoint, dos espacios de features (384d ambos):

1. **CLS**: token CLS del encoder.
2. **patch-mean**: media de los 256 tokens de parche (se excluyen CLS y los
   `n_storage_tokens` del backbone).

Imágenes normalizadas exactamente como en entrenamiento/eval
(`utils.get_img_preprocessor`: ImageNet stats + resize 224).

## Probes

- **Lineal**: ridge regression cerrada sobre features z-scoreadas; alpha
  elegido en una grilla sobre validación (1e-3 .. 1e4).
- **No-lineal**: MLP `D → 256 → D_out`, GELU, dropout 0.1, AdamW, early
  stopping por val, máximo 100 epochs.

## Métricas

Por grupo (promedio sobre dimensiones) y por probe:

- **MSE** sobre targets normalizados (z-units, comparable entre grupos) y MSE
  en unidades originales en el CSV.
- **Pearson r** (media de las correlaciones por dimensión).

## Salidas

- `assets/probe_results.csv` (todas las combinaciones modelo × feature × probe × grupo).
- Tabla markdown para el paper.
- Figura: heatmap `cantidad física × (modelo, feature, probe)` con Pearson r.

## Implementación

- Script nuevo `probe_train.py` en la raíz, reutilizando:
  - `decoder_train.load_world_model`, `decoder_train.build_dataset`,
  - `utils.get_img_preprocessor`.
- Fase 1: extracción de features por checkpoint, cacheada en
  `outputs/probe/features_<modelo>_<fuente>.npz` (features + targets + ep_idx).
- Fase 2: entrenamiento de probes sobre las features cacheadas.
- Tests de sanidad:
  - features aleatorias → r ≈ 0;
  - target como feature → r ≈ 1;
  - split sin episodios compartidos.

## Restricciones / riesgos

- Velocidades y yaw no son directamente observables en un frame: un r bajo en
  `joint_vel`/`gripper_vel` es un resultado esperable del protocolo, no una
  falla del modelo. Se documenta en el análisis.
- `block_quat` incluye dimensiones redundantes con `block_yaw`; se reportan igual.
- El checkpoint de DINOv2-Cube está incompleto (1/10 epochs): no se incluye.
