# Experimentos de diagnóstico multiescala

Objetivo: saber **si** el front-end multiescala (Starlet / SWT) afecta, **por qué** y **en qué**. Todo
corre sobre el pipeline existente (`train.py`, `viz/eval_reward.py`); los cambios son retrocompatibles y
los checkpoints anteriores cargan sin modificaciones.

## Hallazgo crítico: bug en DINOv2 (los resultados con DINOv2 no son válidos)

En `src/dinov2/layers/block.py`, `attn_residual_func` hacía `self.attn(self.norm1(x))[0]`. Con
`MemEffAttention` (la clase que usa `dinov2_vits14`), `attn(...)` devuelve un tensor `(B, N, C)`, así que
`[0]` tomaba la atención de la **primera imagen del batch** y, por broadcasting, se la sumaba a todas.
Consecuencia: el CLS de **todas** las imágenes del batch era el de la imagen 0.

- En entrenamiento, los B·T frames de un batch compartían el mismo embedding.
- En planificación, todos los entornos y todos los goals se codificaban como el entorno 0.

Eso explica DINOv2 = 38.7 % en TwoRooms mientras DINO-WM (también DINOv2) llega a 100 %, y el 48.7 %
idéntico de DINOv2 y DINOv2+Starlet en Cube. **DINOv3 no tiene el problema** (usa otra implementación de
bloque). Ya está corregido y hay un test de regresión (`test_no_cross_sample_leakage_in_batch`); los
checkpoints de DINOv2 se entrenaron con el bug, así que hay que re-entrenarlos.

Reproducción mínima (antes del fix): `CLS(b)` dentro del batch `[a, b]` era idéntico a `CLS(a)` y
distinto de `CLS(b)` evaluado solo.

## Hallazgo previo: la init "legacy" no reproduce a DINO

`expand_patch_embed` copiaba los pesos RGB preentrenados a los **3 primeros canales**, pero
`starlet_conv4d` apila `[w1, w2, …, cJ]`: el filtro de DINO veía solo **w1** (el detalle más fino) y el
resto (incluida la aproximación cJ, donde vive casi toda la energía de una imagen natural) recibía la
media de los pesos RGB. Además, el bias del patch embedding no se copiaba.

Con imágenes suaves (energía en baja frecuencia), en el paso 0 y con pesos aleatorios:

| Init | Error relativo del patch embedding vs DINO | cos(CLS, CLS de DINO) |
|---|---|---|
| `legacy` (original) | 79 % | 0.75 |
| `equivalent` (nueva) | 0 % | 1.000 |

Como `x = cJ + Σ wj`, replicar `W_rgb` en todas las bandas (`patch_init=equivalent`) hace que
`conv(W, Σ bandas) = conv(W, x)`: el encoder **es** DINO en el paso 0 y la LoRA no tiene que
reconstruir el preentrenamiento. Hipótesis principal para el −11 pp de DINOv3+Starlet en Cube.

## Qué se agregó

| Pieza | Archivo | Para qué |
|---|---|---|
| Front-end configurable | `wavelet/frontend.py`, `src/encoders/multiscale.py` | Transformadas (`starlet`, `swt`, `rgb_repeat`, `random`), fusión (`replace`, `residual`), init (`legacy`, `equivalent`), ablación de bandas |
| Configs SWT para DINO | `config/train/model/dinov{2,3}_swt_lora.yaml` | SWT con fusión residual (la SWT no suma la imagen, no admite init equivalente) |
| Eval con ablación/perturbación | `viz/eval_reward.py`, `config/eval/*.yaml` | `ablation.*`, `perturb.*`, `output.tag`; trazas con claves para emparejar episodios y `results.jsonl` |
| Importancia de bandas | `diagnostics/band_importance.py` | Δ pérdida de predicción, deriva del latente e Integrated Gradients por banda |
| Probes lineales | `diagnostics/linear_probe.py` | R² latente → estado verdadero, split por episodio |
| Error vs horizonte | `diagnostics/rollout_error.py` | MSE / nMSE / 1−cos del rollout open-loop por paso |
| Estadística pareada | `diagnostics/paired_stats.py` | Δ pp con IC, permutación de signo + Holm, McNemar |
| Perturbaciones | `diagnostics/perturb.py` | blur, ruido, downsample, brillo |
| Generador de matriz | `experiments/matrix.py` | Comandos de entrenamiento/evaluación/diagnóstico con nombres consistentes |
| Fix DINOv2 | `src/dinov2/layers/block.py` | La atención de la imagen 0 se copiaba a todo el batch |
| Tests | `tests/test_multiscale.py` | 41 tests en CPU (equivalencias exactas, sin fuga entre muestras, compatibilidad con checkpoints viejos, estadística) |

Durante el entrenamiento se loguean `band_weight/<banda>` y, en fusión residual, `ms_branch_ratio`
(‖rama multiescala‖ / ‖rama RGB‖): cuánto usa el modelo la multiescala.

## Experimentos

Convención de nombres: `<env>__<variante>__s<seed>` (= `output_model_name`). `experiments/matrix.py`
genera todos los comandos; exporta `STABLEWM_HOME` antes de correrlos.

### 1. Matriz completa con evaluación pareada

```bash
python experiments/matrix.py --groups core --encoders dinov3 dinov2 --seeds 0 1 2 \
    --stages train eval stats --emit run_core.sh
```

Variantes `core`: `rgb`, `starlet-legacy` (lo que ya tenías), `starlet-equiv`, `starlet-resid`, `swt-resid`.
Todas se evalúan con `seed=42`: mismos estados iniciales y goals ⇒ se comparan **episodio a episodio**.

```bash
python -m diagnostics.paired_stats \
  --run "dinov3-rgb=$STABLEWM_HOME/cube__dinov3-rgb__s*/traces_weights_epoch_10_clean_seed42.npz" \
  --run "dinov3-starlet-equiv=$STABLEWM_HOME/cube__dinov3-starlet-equiv__s*/traces_weights_epoch_10_clean_seed42.npz" \
  --ref dinov3-rgb --out reports/cube_dinov3.md
```

Lectura: Δ pp con IC bootstrap sobre episodios; `p (signo)` corregido por Holm. Con una sola réplica
por episodio también reporta McNemar (b/c = episodios que solo gana cada uno).

### 2. Diagnóstico de Cube (init y fusión)

Mismo comando del punto 1 con `--envs cube --encoders dinov3`. Lectura:

- `starlet-equiv` ≈ `rgb` y `starlet-legacy` < `rgb` ⇒ la caída era la init, no la multiescala.
- `starlet-equiv` < `rgb` ⇒ la multiescala daña al entrenar (mirar `band_weight/*` y probes).
- `starlet-resid`: si `ms_branch_ratio` se queda cerca de 0, la rama multiescala no aporta en ese entorno.

### 3. Controles de capacidad

```bash
python experiments/matrix.py --groups controls --stages train eval stats --emit run_controls.sh
```

`rgbrep-equiv` (J+1 copias de x/(J+1)) y `random-equiv` (filtros aleatorios de media cero + residuo) tienen
los mismos canales, la misma init exacta y la misma propiedad Σ bandas = x que `starlet-equiv`, pero sin
estructura de escala. Si igualan a Starlet, la ganancia no es multiescala. `swtmerged-resid` suma las
orientaciones: aísla si la orientación de la SWT aporta.

### 4. Ablación de bandas (sin re-entrenar, sobre checkpoints existentes)

Offline, en el split de validación (barato, poca varianza):

```bash
python -m diagnostics.band_importance data=pusht +policy=<run>/weights_epoch_10.pt seed=<seed_de_train> \
    loader.batch_size=16 +diag.max_batches=30 +diag.modes=[zero,noise] +diag.ig_steps=8
```

Solo se ablaciona el **contexto**; el target se codifica limpio (si no, quitar información lo haría más
fácil de predecir). `seed` debe ser el del entrenamiento para reproducir el split de validación.
Columnas: `delta_loss` (± se), `delta_rel_pct`, `drift_1mcos`, `ig_frac_pct`. Σ IG ≈ L(todo) − L(nada).

En planificación (lo que importa al final):

```bash
python viz/eval_reward.py --config-name=pusht policy=<run>/weights_epoch_10.pt seed=42 \
    'ablation.bands=[w1]' ablation.mode=noise
```

Bandas por nombre (`w1`, `c3`, `LH2`, …), grupos (`details`, `approx`) o índices. Modos: `zero` (quita la
banda), `mean` (deja su DC), `noise` (misma media y desviación, sin información: el más limpio porque no
cambia la estadística de entrada). Funciona también con los encoders desde cero (`starlet`, `swt`).

### 5. Barrido de J

```bash
python experiments/matrix.py --groups jsweep --encoders dinov3 --stages train eval stats   # DINO, J = 1..4
python experiments/matrix.py --groups scratch-jsweep --stages train eval stats             # desde cero, J = 1..5
```

Relacionar el J óptimo con el tamaño de los objetos (la escala de wj es 2^(j-1)–2^j px).

### 6. Probes lineales

```bash
python -m diagnostics.linear_probe data=pusht +policy=<run>/weights_epoch_10.pt seed=<seed> \
    +probe.max_frames=20000
```

Defaults: targets = `keys_to_cache` sin `action`; en PushT `state[4]` (ángulo del bloque) se trata como
ángulo (sin/cos + error medio en grados). Overrides: `+probe.targets=[state]`,
`'+probe.angle_dims={state:[4]}'`. Reporta R² en episodios **no vistos** para `cls` (encoder) y `emb`
(latente del predictor). La hipótesis a testear: la multiescala mejora la pose fina (ángulo de la T).

### 7. Salir del techo y robustez

```bash
python experiments/matrix.py --stages hard robust --groups core --emit run_hard.sh
```

- `hard`: goals más lejanos (`eval.goal_offset_steps=50 eval.eval_budget=100`) y menos muestras de CEM
  (`solver.num_samples=100`).
- `robust`: `perturb.kind` ∈ {blur, noise, downsample, brightness} con varias intensidades; se aplica a la
  observación y al goal (`perturb.goal=false` para solo la observación).

Eficiencia de datos: `data.dataset.fraction=0.1|0.25|0.5` en `train.py` (vía `--extra-train` en matrix.py).

### 8. Error de predicción vs horizonte

```bash
python -m diagnostics.rollout_error data=pusht +policy=<run>/weights_epoch_10.pt seed=<seed> \
    data.dataset.num_steps=13 +rollout.horizon=10 +rollout.max_batches=20
```

Rollout open-loop con las acciones reales (mismo esquema que `JEPA.rollout`). Por paso k: MSE, MSE
normalizado por la varianza del latente y 1 − cos. ¿La multiescala estabiliza los horizontes largos?

## Prioridad sugerida

1. **Hoy, sin entrenar:** 4 (band_importance + ablación en eval) y 6 (probes) sobre los checkpoints de
   DINOv3 y de los encoders desde cero (no sobre los de DINOv2: tienen el bug).
2. **En paralelo:** 2 (Cube con `starlet-equiv` y `starlet-resid`, 3 semillas) y re-entrenar DINOv2.
3. Después: 1 completo, 3, 5 y 7.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```
