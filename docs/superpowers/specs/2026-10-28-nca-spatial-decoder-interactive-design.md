# Decoder espacial + demo interactiva para el NCA latente

## Intención

Poder interactuar en tiempo real con el checkpoint `nca_tworoom_frozen`
(`SpatialJEPA` + `LatentNCAPredictor`). El usuario ancla la demo en un frame real
de TwoRoom, aplica acciones y ve el siguiente estado **imaginado** decodificado a
píxeles, junto con métricas de deriva y el mapa de gate del NCA.

El world model no se reentrena. El decoder es puramente de visualización y se
entrena a posteriori con el encoder congelado, igual que el `CLSDecoder` actual
del repo.

## Por qué hace falta un decoder nuevo

El checkpoint produce latentes espaciales `(B, T, 196, 192)`:
`14×14` patch tokens, no un vector CLS global. El `CLSDecoder` existente espera
`(B, D)` y el decoder de `outputs/decoder_dinov3_tworoom/decoder.pt` pertenece a
otro modelo. Por eso no sirve directamente.

## Componentes

1. **`nca_decoder.py` — `SpatialDecoder`**
   - Entrada: `(B, 196, 192)` → reshape `(B, 192, 14, 14)`.
   - Salida: `(B, 3, 224, 224)` en espacio normalizado ImageNet.
   - Arquitectura convolucional con 4 etapas de upsample (`14→28→56→112→224`),
     `ConvTranspose2d` + normalización + activación.
   - Sin attention y sin gradiente hacia el encoder.

2. **`train_nca_decoder.py`**
   - Carga el checkpoint del world model y congela todo.
   - Dataset TwoRoom con `num_steps=1`, misma normalización que el encoder.
   - Codifica frames con `model.encode(...)["emb"]`, toma la grilla del frame y
     entrena el decoder con MSE.
   - Split por episodios para validación.
   - Guarda `decoder.pt` con metadatos: `latent_dim`, `grid_hw`, `patch_size`,
     `img_size`, `source="emb"`, checkpoint de origen y métricas.
   - Defaults: 10k pasos, batch 32, AdamW, lr 3e-4.

3. **`interactive_nca.py` — app Gradio**
   - Carga checkpoint + decoder una vez.
   - Ancla aleatoria o por índice de frame del dataset.
   - Contexto de 3 frames reales codificados.
   - Controles: pad 2D de acción, magnitud, aplicar, deshacer, reiniciar.
   - Salida: frame real del ancla, frame imaginado, estado de deriva
     (`|z|²/D`, rectitud, drift), mapa de gate.
   - Cada paso: `model.predict` con historial 3 + `decoder` en `no_grad`.
   - Objetivo de latencia: <1 s por paso en la RTX 3090.

## Flujo de inferencia

```
frame real x_t  ──encoder──▶  emb_t (196×192)
                                  │
           historial 3 frames + acción
                                  ▼
                         NCA (eval_steps=8)
                                  ▼
                         emb_{t+1} (196×192)
                                  ▼
                         SpatialDecoder
                                  ▼
                         imagen imaginada
```

## Datos y entrenamiento del decoder

- Dataset: `tworoom.h5`.
- Checkpoint inicial: `~/.stable_worldmodel/checkpoints/nca_tworoom_frozen/weights_epoch_1.pt`.
- Config: `outputs/2026-10-27/14-28-43/.hydra/config.yaml`.
- El encoder se usa en `eval()` y `requires_grad_(False)`.
- Validación: MSE de reconstrucción y PSNR en episodios no vistos.
- No se usa el decoder para calcular ninguna loss del world model.

## Git

- Rama de trabajo: `feature/nca-decoder-interactive`.
- Tag de respaldo: `backup/pre-nca-decoder-20261028`.
- No se tocan los cambios sucios de `wandb/` ni `wandb_resume.json`.
- Commits separados: diseño, decoder + entrenamiento, app, documentación.

## No objetivos

- No reentrenar el world model ni el encoder.
- No añadir un decoder al entrenamiento del NCA.
- No simular física real: la demo muestra el rollout imaginado del modelo.
- No prometer fidelidad más allá del horizonte entrenado (~5 pasos).

## Criterios de aceptación

1. `train_nca_decoder.py` entrena y guarda un `decoder.pt` cargable.
2. El decoder reconstruye frames de TwoRoom de forma reconocible.
3. `interactive_nca.py` arranca con el checkpoint y el decoder.
4. Aplicar una acción actualiza la imagen imaginada en <1 s.
5. Reset/undo/ancla funcionan y las métricas de deriva se actualizan.
6. Los comandos quedan documentados en el README o en el docstring del script.
