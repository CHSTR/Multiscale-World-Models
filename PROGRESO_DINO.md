# PROGRESO_DINO

## Objetivo
Comparar un ViT entrenado desde cero vs. encoders DINO preentrenados con LoRA
para el WM de `two-room`:
- **ViT-S/14 desde 0** (baseline, sin pretrained).
- **DINOv2-S** (`src/encoders/dinov2_encoder.py`) con LoRA.
- **DINOv3-S** (`src/encoders/dinov3_encoder.py`) con LoRA.

## P1 — Encoders y LoRA
- `src/encoders/dinov2_encoder.py` — envuelve `src/dinov2` ViT.
- `src/encoders/dinov3_encoder.py` — envuelve `src/dinov3` ViT.
- `src/encoders/vit_lora.py` — LoRA manual (sin `peft`) sobre `Linear`: sustituye
  `qkv`/`proj` por `LoRALinear`, copia pesos, `A` normal, `B`=0 (delta inicial 0).
- Fix de `set_param` en `src/dinov2/layers/lora.py` (compatible con el LoRA
  manual / freeze por capas).
- `src/utils.py` registran `ViT-S/14` (índices de capas para positions).

## P2 — Configs
- `dinov2_lora`, `dinov3_lora` (base 384-d).
- `dinov2_starlet_lora` — `in_chans=15` (3 canal × 5).
- `dinov3_starlet_lora`.
- Projector 384 → 192.

## Comandos
```
# baseline ViT desde 0
python train.py data=tworoom model=lewm

# DINO con LoRA (comparación directa, mismo dataset que el baseline)
python train.py data=tworoom model=dinov2_lora
python train.py data=tworoom model=dinov3_lora

# variantes starlet / multicanal
python train.py data=tworoom model=dinov2_starlet_lora
python train.py data=tworoom model=dinov3_starlet_lora

# overrides útiles
python train.py data=tworoom model=dinov2_lora model.encoder.ckpt_path=... lora_r=8
```

## Notas / decisiones
- **PEFT no se usa**: LoRA manual propio en `src/encoders/vit_lora.py` (r=8,
  alpha=16 => escala 2.0). Targets = `qkv`, `proj`.
- **Multicanal**: 3 canal → 15 (= 3 × (4+1)). Se amplía el conv del
  patch-embedding copiando los canales RGB y relleno `mean`.
  Smoke: `smoke_dino_encoders.py` (4 configs × 3ch/15ch OK).
- **SIGReg**: ortogonaliza la capa de registro; es ortogonal al LoRA (que parte
  de weight 0.0). No interfieren.
- **DINOv2**: vía hub interno (`src/dinov2`).
- **DINOv3**: path local de checkpoints,
  `/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`.

## Limpieza
- Eliminados (muertos, sin referencias): `src/model_LN_prompt.py`
  (legacy SBIR, importaba `experiments.options` y `src.clip` inexistentes),
  `src/encoders/clip_encoder.py` (`src.clip` inexistente),
  `src/utils.py` (helpers LoRA nunca importados; los encoders usan
  `src/encoders/vit_lora.py`).
- Renombres con sentido: `src/encoders/lora.py` → `src/encoders/vit_lora.py`
  (LoRA ViT + patch-expand; evita confusión con `utils.py` raíz y con
  `src/dinov2/layers/lora.py` vendor), `smoke_dino.py` →
  `smoke_dino_encoders.py`.
