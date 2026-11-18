# Diseño: banco sintético multiescala "MSWorld" (fase 1, offline)

> **SUPERSEDED (2026-11-18):** reemplazado por el entorno **MultiScaleDial**
> (spec en `/home/chr/Documents/multiescala_env/multiscale-env-spec.md`, que
> se integrará al repo). Su diseño es superior: escala como variable
> independiente (curva dosis-respuesta Δ(k) con k=1..4), dial "balanced" con
> dipolo nulo (sin fuga de información fina a bandas gruesas), metas 2×2
> coarse/fine/both sobre el mismo modelo, y gym env con `set_state` que une
> diagnóstico offline y eval CEM pareada. Se conservan de este doc: el split
> por episodio (train/val separados) y la lección de medir energía por banda
> con sumas (Parseval).

## Intent y por qué

Los benchmarks actuales no pueden separar "lo multiescala aporta" de "lo
multiescala no hace nada": TwoRoom y PushT saturados (>92 % de éxito), cube con
ruido de ±6–7 pp a n=50, y todos los checkpoints DINOv2 invalidados por el bug
de cross-sample leakage. Queremos un banco de pruebas offline, generado
proceduralmente, donde el contenido de información por banda espacial sea
**controlable y verificable**, de modo que la hipótesis multiescala pueda
ganar, empatar o perder — y sepamos por qué.

Fase 1 es **offline**: sin entorno gym, sin planning CEM. El juez son métricas
de diagnóstico sobre el world model entrenado (ablación de bandas, error de
rollout por escala, probes). Si hay señal, la fase 2 (no incluida acá)
implementaría el env swm para éxito pareado por planning.

## Goals

1. Generador determinista por seed de un mundo 2D top-down con knobs por banda,
   salida H5 **idéntica en esquema a `tworoom.h5`** (compatible con
   `swm.data.load_dataset` y `train.py` sin tocar el loader).
2. Variante principal `both` + 3 controles de validez (`coarse-only`,
   `fine-only`, `cluttered`) que difieren en **una sola variable** cada uno.
3. Entrenar desde cero baselines single-scale y variantes multiescala con
   presupuesto idéntico, 3 seeds.
4. Juzgar con métricas offline portadas/adaptadas de
   `exp/multiscale-diagnostics` (band_importance, rollout_error, linear_probe)
   + nueva métrica WAPE.
5. Un **ejemplo visual** (spike) del mundo generado antes de entrenar nada:
   el humano decide si seguir.

## Non-goals

- Env gym/swm con `set_state`/goal para eval CEM (fase 2).
- Modelos DINO-LoRA (fase 1: solo entrenamientos desde cero, baratos).
- Métricas pareadas de éxito (sin éxito no hay pareo).

## Anti-goals

- Saturación del juez: si el banco lo resuelve un modelo trivial, no sirve.
- Arte de render/aliasing de 1 px que haga "ganar" a lo multiescala por bugs
  de dibujo y no por estructura de escala.
- Variantes que difieran en más de una variable (confunde la lectura).

## El mundo (224×224 RGB, top-down)

- **Estructura gruesa (banda baja):** mapa procedural de 2–4 salas conectadas
  por puertas; paredes de 8–16 px. La topología varía por seed.
- **Agente (banda alta, dinámica):** disco/flecha de tamaño ajustable
  `fine_size_px` (default 6, rango 4–16). Dinámica de punto con velocidad
  limitada y colisión contra paredes. Es el elemento fino **móvil**.
- **Bloque grande móvil (banda baja, dinámica):** rectángulo de
  `coarse_size_px` (default 64, rango 48–96) con **trayectoria paramétrica
  determinista** (velocidad constante + rebote contra paredes; seed por
  episodio). Da dinámica real a la escala gruesa: sin esto, la escala gruesa
  sería trivial de predecir (todo estático). Determinista a propósito: un
  bloque browniano tendría ruido irreducible que penalizaría a cualquier
  modelo por innovaciones que no puede predecir; con trayectoria determinista
  el rollout de un solo futuro es una métrica justa.
- **Distractores de banda media:** parches/texturas de 16–64 px, amplitud
  `clutter_amp` (0 = mundo limpio). No informativos (no predicen nada),
  estáticos por episodio pero recolocados entre episodios.
- **Render:** PIL con supermuestreo 2× y downsample (antialiasing): espectro
  suave sin alias. Nada de bordes de 1 px crudos.
- **Acción:** velocidad 2D `float32` (dim 2, como tworoom). Trayectorias =
  random walk suave hacia waypoints aleatorios **navegables** (sin salirse de
  las salas). No se necesita política experta: la fase 1 mide predicción
  action-conditioned, no planificación.
- **Estado para probes:** `pos_agent` (2,), `pos_block` (2,); `observation`
  concatena ambos. `proprio` = pos_agent (paridad con tworoom donde
  proprio = posición del agente).

### Esquema H5 (clon de tworoom.h5, tamaño reducido)

`pixels` uint8 (N,224,224,3) · `action` float32 (N,2) · `proprio` float32
(N,2) · `observation` float64 (N,4) · `pos_agent` float32 (N,2) ·
`pos_block` float32 (N,2) · `ep_idx` int32 · `step_idx` int64 · `ep_len`
int32 (n_eps,) · `ep_offset` int64 (n_eps,) · `id` int64 · `reward` float64
(0 en todo; clave requerida por formato). Default **3,000 episodios × 92
steps ≈ 276k frames (~4 GB)**, emitidos como dos archivos con topologías
disjuntas: train (2,700 eps) + val (300 eps) — paridad de *formato* con
tworoom, no de tamaño (ajustable por CLI; para curvas de data-efficiency se
puede subir).

## Controles de validez

| Variante | Cambio único respecto de `both` | Expectativa si lo multiescala sirve |
|---|---|---|
| `both` (principal) | — | la multiescala **debe** ganar |
| `coarse-only` | `fine_size_px`: 6 → 48 (agente grande; todo lo dinámico queda en banda gruesa) | sin ventaja (nada relevante vive en la banda fina) |
| `fine-only` | sala única vacía: sin paredes **y sin bloque grande** | sin ventaja (nada relevante vive en la banda gruesa) |
| `cluttered` | `clutter_amp`: 0 → 1 | la ventaja, si existe, debería crecer (distractores no informativos de banda media) |

Semántica común: cada control hace que **una sola banda contenga toda la
información relevante** (agente y bloque en el mismo régimen de frecuencia).
La expectativa "sin ventaja" aplica a ambas métricas (state-rollout del
agente y del bloque). Si la multiescala "gana" en coarse-only o fine-only,
la ventaja no es multiescala sino capacidad — y hay que mirar
`band_importance` para distinguir.

**Puerta de validez (antes de entrenar nada; umbrales concretos):**
1. Energía de movimiento por banda medida con el `LiftingScheme` existente
   (Δ entre frames consecutivos, 5,000 frames de val): en `both`, ≥ 60 % de
   la energía de movimiento del agente debe caer en las bandas finas
   (detalles 1–2) y ≥ 60 % de la del bloque en las bandas gruesas
   (aprox + detalles 2–3, según L=3).
2. Probe ridge frame→estado (5,000 frames): limpio, R² ≥ 0.9 tanto para
   `pos_agent` como `pos_block`.
3. Operadores de ablación (corregido: downsample+upsample es low-pass, NO
   quita la banda gruesa): **quitar banda fina** = blur gaussiano (low-pass);
   **quitar banda gruesa** = residuo high-pass `x − blur(x)`. Pasa la puerta
   si: tras quitar la fina, R²(pos_agent) cae a ≤ 0.3; tras quitar la gruesa,
   R²(pos_block) cae a ≤ 0.3. Si no pasa, el mundo no testea multiescala y
   se corrige antes de entrenar.

## Unidad experimental, seeds y splits

- **Jerarquía de seeds:** un seed de generador fijo (1000) produce el
  dataset; la **topología se remuestrea por episodio** (cada episodio es una
  sala nueva) → la unidad de generalización es el episodio/topología, no el
  frame.
- **Dataset fijo compartido:** cada variante del mundo se genera **una vez**
  y la usan todos los modelos y seeds. Los run seeds {0,1,2} varían solo el
  entrenamiento (init/order), nunca regeneran datos.
- **Split por episodio, no por ventana:** el generador emite **dos archivos**
  `msworld_<variante>_train.h5` (2,700 eps) y `msworld_<variante>_val.h5`
  (300 eps) con topologías disjuntas (rangos de seed de episodio disjuntos).
  Entrenamiento usa el train; todos los diagnósticos (probes, rollouts,
  band_importance, puerta de validez) usan el val. Esto evita que el split
  interno por ventanas de `train.py:141-143` deje frames vecinos del mismo
  episodio en ambos lados (mediría otra cosa).

## Modelos y presupuesto

Todos **desde cero**, mismo budget de steps/epochs, seeds {0,1,2}, nombres
`<world>__<variante>__s<seed>` (convención EXPERIMENTOS.md).

- **Single-scale:** `lewm` (ViT plano, el baseline histórico) + control
  `lewm` con input downsampled (anti-aliased) = modelo al que le falta la
  banda fina "por construcción".
- **Multiescala:** `starlet`, `swt`, `wave_vit`, `liftwm` (NCA),
  `convnext_ms`.

**Matriz obligatoria:** `both` × {lewm, lewm-ds, starlet, swt, wave_vit,
liftwm, convnext_ms} × 3 seeds = **21 runs**. `liftwm_vit` NO está en la
matriz obligatoria (el eje predictor NCA-vs-ViT es ortogonal a la hipótesis
multiescala; se entrena solo para la variante multiescala que mejor dé en
`both`, como chequeo extra). Controles (`coarse-only`, `fine-only`,
`cluttered`) × {lewm, la mejor multiescala} × 3 seeds = 18 runs opcionales,
se corren después de `both`, solo si `both` muestra separación.

**Regla de recorte si el cómputo aprieta** (una sola GPU, budget aprobado:
**2 epochs por run**): primero se baja el budget por run, después se
recortan variantes en este orden: `wave_vit` → `swt` (quedan `starlet` +
`liftwm` + `convnext_ms` como representantes multiescala, y `lewm`/`lewm-ds`
como baselines). Las comparaciones que deben sobrevivir cualquier recorte:
`both` × {lewm, lewm-ds, starlet, liftwm} × 3 seeds.

## Juez (métricas offline, portadas de `exp/multiscale-diagnostics`)

**Métrica primaria cross-modelo (para rankear):** *state-space rollout error*.
Los latentes JEPA tienen escala y base arbitrarias → el WAPE latente **no**
es comparable entre checkpoints. Solución estándar: probe lineal congelado
por checkpoint (latente→`pos_agent`/`pos_block`, entrenado en val con split
por episodio) aplicado autoregresivamente sobre el rollout, y medir el error
de posición en el espacio del estado a horizontes 1/5/10. Los targets están
en unidades físicas (px), comparables entre modelos.

**Regla de decisión (fase 1, aprobada):** la multiescala *gana* si en `both` su
state-rollout error (media de 3 seeds) supera al de `lewm` con separación
mayor que el spread entre seeds; *pierde* si queda claramente abajo en
`both`; si emparda en `both` pero también en los controles, el veredicto es
"no aporta". `cluttered` se reporta como evidencia secundaria (si la brecha
crece con distractores) pero **no** es requisito para declarar victoria.
`band_importance` es el chequeo de mecanismo (dice *por qué*), no el veredicto.

1. **band_importance (modo `noise`):** Δ pérdida de predicción por banda con
   la estadística preservada — diagnóstico **dentro** de cada checkpoint
   (uso de bandas), no comparable entre modelos. Adaptación necesaria:
   ablación de canales del frontend starlet/swt (encoders desde cero) y de
   escalas de la pirámide en liftwm / convnext_ms.
2. **state-rollout + WAPE latente por escala vs horizonte (1/5/10):** el
   state-rollout es la métrica primaria; el WAPE latente
   (Σ|pred−real|/Σ|real|, nuevo, ~20 líneas; MAPE literal descartado:
   inestable en señales que cruzan el cero) se reporta por escala solo como
   diagnóstico interno del checkpoint.
3. **linear_probe por escala:** R² de `pos_agent` (info fina) y `pos_block`
   (gruesa). Hipótesis: con `clutter_amp` alto, la multiescala decodifica
   mejor la posición fina.
4. Opcional: curvas de data-efficiency (`data.dataset.fraction`
   0.1/0.25/0.5) para `both`.

## Piezas técnicas

1. `data_gen/gen_msworld.py` — generador + CLI (`--variant --seed
   --n-episodes --out`). Solo numpy/PIL (ya instalados), CPU.
2. `config/train/data/msworld.yaml` — keys_to_load `pixels,action,proprio`;
   keys_to_cache `action,proprio` (+ pos_* para probes).
3. Port de `diagnostics/` a esta rama: `stats_core` (+WAPE), `rollout_error`
   (per-escala), `linear_probe`, `band_importance` (adaptado por familia de
   encoder). Fuente: rama `exp/multiscale-diagnostics`.
4. `experiments/matrix_ms.py` — genera los comandos train/diag con nombres
   consistentes.
5. Tests: determinismo por seed, compatibilidad loader (`load_dataset` +
   `get_row_data` + split val), energía por banda en rango esperado,
   reconstrucción de estados desde frames (puerta de validez 2).

## Ejemplo (spike) — lo primero que se muestra

Un PNG grid: (a) 3 frames de un episodio `both` (inicio/medio/fin),
(b) la pirámide lifting de un frame (con `viz/liftwm_visualize.py`),
(c) los 4 mundos lado a lado, (d) barras de energía por banda del nivel
wavelet. Con eso el humano decide si el mundo "se ve" multiescala antes de
gastar GPU.

## Approaches consideradas

1. **Sintético offline con perillas (elegida):** control total, barato,
   verifica su propia validez antes de entrenar; sin env.
2. **Sintético + env swm para CEM:** la métrica final de verdad, pero el
   triple de trabajo; solo si la fase 1 da señal.
3. **Contaminar cube/pusht con distractores por banda:** lo más barato,
   pero mide robustez a distractores, no beneficio multiescala per se.
   Descartada como vía principal.

Trade-off asumido: un mundo sintético tiene menos validez externa que un
benchmark "real"; lo compensamos con controles de validez internos y con la
fase 2 (env) si la señal lo justifica.

## Queda a criterio del builder

- Estructura de salas: BSP/grid particionado vs plantillas paramétricas.
- Forma del agente (disco vs flecha con orientación — la flecha añade un
  ángulo fino que los probes pueden aprovechar; propuesta: flecha).
- Cómo minimizar el diff al portar `band_importance` por familia de encoder
  y `linear_probe` (split por episodio ya resuelto con train/val separados).

## Decisiones ya tomadas (no reabrir sin motivo)

- Bloque grande: trayectoria determinista (velocidad constante + rebotes),
  no browniano — ver Unidad experimental / El mundo.
- Controles redefinidos: fine-only sin paredes **y sin bloque**;
  coarse-only con agente agrandado. Operador de ablación gruesa =
  residuo high-pass, no downsample+upsample.
- Métrica primaria cross-modelo: state-space rollout error vía probes
  congelados; WAPE latente solo diagnóstico interno.
- Dataset fijo compartido, split por episodio (train/val en archivos
  separados), run seeds solo para entrenamiento.
