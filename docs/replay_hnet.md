# Replay priorizado por sorpresa + red de alucinación H (F1)

> Rama `per-hnet` (sale de `convnext-stable-env`). Sólo JEPA global
> (`emb (B, T, D)`), p. ej. `model=dinov3_lora`. Con `replay=disabled` (default)
> el entrenamiento es exactamente el de antes.

## Qué hace

1. **Etiquetas de K horizontes.** Cada ventana trae `history_size + K` frames
   (`K = replay.horizon = 5`, el horizonte del planificador). El camino principal
   (pred_loss + SIGReg) usa sólo los primeros `history_size + num_preds`, igual que
   el baseline. Desde el contexto se corre, sin gradiente, el mismo rollout
   autoregresivo que `JEPA.rollout` (el que usa CEM):

   ```
   frames:   z0  z1  z2 │ z3   z4   z5   z6   z7        (HS = 3, K = 5 → num_steps = 8)
   rollout:              ẑ3 → ẑ4 → ẑ5 → ẑ6 → ẑ7
   error:                e1   e2   e3   e4   e5        e_k = ‖ẑ − z‖² / D
   ```

   `e1` coincide con el error teacher-forced de la última posición de contexto
   (test `test_forward_main_loss_equals_baseline_and_fills_labels`).

2. **Buffer (memoria episódica).** Tabla por índice de ventana, en CPU:
   `priority`, `err (N, K)`, `last_step`, `visits`, `valid` (~23 B por ventana,
   ~23 MB por millón). Vive en el callback y va en el checkpoint de Lightning, así
   que `resume_from` lo restaura.

3. **Sampler.** `P(i) = (1-ε)·p_i^α / Σp^α + ε/N_valid` sobre ventanas válidas,
   con muestreo con reemplazo. Detalles:
   - `p_i = mean_k e_ik / ē_k`: cada horizonte normalizado por su media móvil, para
     que el paso 5 no domine.
   - Las ventanas no visitadas reciben la prioridad de recorte (cuantil 0,99).
   - Durante `warmup_epochs` el muestreo es uniforme; el buffer se llena igual.
   - El sampler entrega claves `(i, P_i)`, así que la probabilidad exacta con la
     que se muestreó cada ventana viaja con el batch.

4. **Pesos IS.** `w_i ∝ P_i^{-β}`, autonormalizados en el batch.
   - **pred_loss:** usa β de `beta_start` a `beta_end`. β = 0 es reponderación
     focal, sin corrección.
   - **SIGReg:** usa siempre β = 1, con la ECF ponderada (`SIGReg(x, weights=w)`).
     Así la isotropía vale bajo la distribución real de los datos y no bajo la del
     batch sesgado.

5. **Ventanas de borde.** `nan_to_num` tapaba acciones NaN. Ahora se detectan antes
   (en posiciones `< HS + K − 1`) y, una vez vistas, esas ventanas salen del
   muestreo, también en el control uniforme. `replay/frac_invalid` reporta qué
   fracción son.

6. **H (memoria semántica), pasiva en F1.** Es un MLP que va de
   `(z_ctx, acciones de los K pasos)` a `log ê_{1..K}`.
   - Se entrena en el callback con su propio AdamW y entradas detached: no comparte
     gradiente, clipping ni scheduler con el world model.
   - **Pérdida:** Huber en log-error más ranking logístico por pares dentro del
     batch, con pesos IS β = 1 para que quede calibrada bajo la distribución
     uniforme.
   - **Entrada:** `replay.hnet.input` = `state | state_action | state_action_pred`.

## Presets (`replay=...`)

| Preset | Sampler | β pred_loss | Pregunta |
|---|---|---|---|
| `uniform_labels` | uniforme | — | **Control**: mismas ventanas, etiquetas y H; sin priorizar |
| `per_is` | priorizado | 0,4 → 1 | ¿Priorizar mejora el mismo objetivo? |
| `per_focal` | priorizado | 0 | ¿Desplazar el objetivo hacia lo difícil mejora la planificación? |
| `per_shuffled` | prioridades permutadas | 0,4 → 1 | ¿El efecto viene de la sorpresa o sólo de muestrear no uniforme? |

**Compara contra `uniform_labels`, no contra el baseline sin replay.** El baseline
usa ventanas de 4 frames (el set de ventanas válidas cambia) e incluye las de borde.

```bash
P=/home/chr/le-wm/.venv/bin/python
for R in uniform_labels per_is per_focal per_shuffled; do
  $P train.py data=ogb model=dinov3_lora replay=$R output_model_name=dinov3_lora_cube_$R
done
# Perillas: replay.alpha=1.0  replay.eps_uniform=0.2  replay.warmup_epochs=0.5
#           replay.horizon=5  replay.hnet.input=state_action_pred  replay.priority=progress
```

## Qué mirar

| Métrica | Dónde | Lectura |
|---|---|---|
| `hnet/val_spearman`, `_k{1..5}` | por época (val, modo eval) | ¿La alucinación es predecible? Si queda en ~0, F2/F3 sobran |
| `hnet/batch_spearman` | train, antes del paso de H | Validación progresiva (fuera de muestra) |
| `replay/ess` | train | Tamaño efectivo de P / N. Con ~0,9 o más PER casi no cambia la distribución (sube `alpha` o revisa la dispersión del error) |
| `replay/frac_invalid` | train | Fracción de ventanas de borde entre las visitadas |
| `replay/err_ema_k*`, `fit/replay_err_k*` | train | Curva de error del rollout |
| `replay/topk_epoch_E.{csv,png}` | `<run_dir>/replay/` | Las 16 ventanas con mayor prioridad: ¿contactos o artefactos? |
| `replay/calibration_epoch_E.csv` | `<run_dir>/replay/` | Error real medio por decil de Ĥ |
| `hnet_epoch_E.pt`, `buffer_final.pt` | `<run_dir>/replay/` | H y el buffer para análisis offline y F2/F3 |

Además, la pérdida sobre el decil más difícil y el éxito en CEM en cube (offsets
largos, varias semillas) frente al control.

## Costo

- **Por paso:** encoder sin gradiente sobre `K − num_preds = 4` frames extra, más
  K forwards del predictor.
- **Carga de datos:** se leen 8 frames por ventana en vez de 4. Si la GPU queda
  esperando, sube `num_workers`.
- **Sampler:** unos 1–4 ms por batch con 1–4 M de ventanas.
- **H:** despreciable.

## Matices

- **Etiquetas en modo train.** En entrenamiento se calculan con el modelo en modo
  train (dropout de LoRA y del predictor activos, BatchNorm con estadísticas del
  batch pero con las running stats congeladas, así que la pasada extra no cambia
  lo que usa el planificador). En validación todo va en modo eval, como CEM. Por
  eso `hnet/val_spearman` mide lo que importa para F3.
- **Sesgo de escala de H.** H predice log-error, así que `exp(ê)` sesga hacia abajo
  la media (Jensen). Para un umbral en F3, fija τ como cuantil de Ĥ en validación,
  no en unidades absolutas de error.
- **Prioridades obsoletas.** Las prioridades se vuelven obsoletas porque el
  encoder se mueve. La mezcla ε y la prioridad de recorte para no visitadas evitan
  la inanición. F2 usará H para re-puntuar.

## Próximo (no implementado)

- **F2:** usar H para puntuar ventanas obsoletas o no visitadas, con latentes
  cacheados por época.
- **F3:** H como disparador de replanificación en MPC.
