---
title: LeWM interactivo (pushT)
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
tags:
  - world-model
  - jepa
  - gradio
  - pushT
---

# LeWM interactivo · pushT

Demo interactiva de un **world model basado en JEPA** con encoder DinoV3+LoRA,
entrenado sobre el entorno **pushT** (un brazo robotico que empuja un objeto).

Elige una **ancla** (3 frames reales de pushT, ya codificados), y a partir de ahí
**tus acciones** (pad 2D) mueven el mundo imaginado: cada accion genera un paso
latente con el predictor y se decodifica a pixeles. El encoder no vuelve a
intervenir.

- **Horizonte fiable**: ~5 pasos. Mas alla, lo que ves sigue siendo plausible
  pero deja de ser fiel.
- **Indicadores de deriva**: `|z|²/D` (≈1 bajo SIGReg), rectitud de la velocidad
  latente y deriva desde el ancla.
- **Test de ciclo**: ejecuta una secuencia cerrada (ida y vuelta) y mide si el
  modelo vuelve al mismo latente.

## Detalles tecnicos

- Encoder: `DinoV3Encoder` (vits16, 384 dim) con LoRA (`r=8, alpha=16`).
- Predictor: `ARPredictor` (depth 6, heads 16), condicionado a acciones.
- Decoder: CLSDecoder (cross-attention sobre el latente post-projector).
- Las anclas son clips de pusht pre-codificados: no hace falta subir dataset.

Los pesos se cargan desde `weights/` y el backbone DinoV3 base se descarga del
hub la primera vez.