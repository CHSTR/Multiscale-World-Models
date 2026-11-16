# LiftWM — Arquitectura del World Model Multiescala

> JEPA piramidal con **lifting scheme wavelet aprendible** y predictor por escala
> **intercambiable (ViT / NCA)**. Rama `liftwm`.
>
> Documentos relacionados: [`PROGRESS_LIFTWM.md`](../PROGRESS_LIFTWM.md) (estado y comandos),
> diseño/plan originales en `/home/chr/.opencode/plan/2026-11-16-liftwm-{design,plan}.md`.

---

## 1. Motivación

El diagnóstico que motiva la arquitectura (evidencia del propio repo):

- **Probes** (`assets/probe_results.md`): la *posición* del bloque/efector está casi
  perfecta en todos los encoders (r≈0.999), pero la **orientación** (yaw/quat) está
  ausente (r≈0.03–0.17), incluso en DINOv3.
- **Evals** (`models_server/checkpoints/*/*_results.txt`): Push-T casi saturado
  (98% starlet-desde-cero); **Cube es el margen** (mejor 74% DINOv3+LoRA). Starlet
  como canales de entrada **empeora** cube (52–62%).
- DINO-WM logra 86% en cube planificando sobre **patch features**, sin pooling global.

**Conclusión:** el detalle fino (orientación/contacto) se pierde aguas arriba y el
pooling global lo descarta. LiftWM conserva latentes **espaciales por escala** desde
los píxeles hasta el planning, en vez de colapsar a un vector.

---

## 2. Vista general

```
                         píxeles  (B·T, 3, 224, 224)
                              │
                    ┌─────────▼──────────┐
                    │   LiftingScheme    │  invertible, Haar/CDF5/3 init,
                    │  (wavelet/lifting) │  filtros aprendibles opcionales
                    └─────────┬──────────┘
                              │  Pyramid { approx LL_L, details[LH,HL,HH]_l }
                    ┌─────────▼──────────┐
                    │   PyramidEncoder   │  stem conv + pos_embed + bloques
                    │ (por escala, L+1)  │  atención espacial NO causal
                    └─────────┬──────────┘
                              │  {level: tokens (B·T, N_l, D_l)}
                    ┌─────────▼──────────┐
                    │  projector por esc │  MLP token-a-token
                    └─────────┬──────────┘
                              │  emb {level: (B, T, N_l, D_l)}
              ┌───────────────┼────────────────┐
              │               │                │
     ┌────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────────┐
     │ Hierarchical  │ │  SIGReg por  │ │  CEM planning   │
     │  Predictor    │ │  escala +    │ │  criterion =    │
     │ NCA o ViT     │ │  pred_loss   │ │  Σ w_l·MSE_l    │
     │ (por escala)  │ │  ponderada   │ │  (espacial)     │
     └───────────────┘ └──────────────┘ └─────────────────┘
```

Principio: **nada se colapsa a un vector global**. Cada escala mantiene su grilla
espacial; lo grueso captura estructura/posición (dinámica lenta) y lo fino
orientación/bordes/contacto (dinámica rápida).

---

## 3. Flujo de tensores (config por defecto: L=3, img=224, D=192, stem_patch=4)

| Etapa | Forma | Nota |
|---|---|---|
| píxeles | `(B·T, 3, 224, 224)` | `T = history_size + num_preds` |
| `Pyramid.approx` | `(B·T, 3, 28, 28)` | LL más gruesa (224/2³) |
| `Pyramid.details[0]` | `(B·T, 9, 112, 112)` | detalle fino, 3C canales `[LH,HL,HH]` |
| `Pyramid.details[1]` | `(B·T, 9, 56, 56)` | |
| `Pyramid.details[2]` | `(B·T, 9, 28, 28)` | detalle grueso |
| tokens escala 0 | `(B·T, 49, 192)` | grid 7×7 (approx) |
| tokens escala 1 | `(B·T, 49, 192)` | grid 7×7 (detail₂) |
| tokens escala 2 | `(B·T, 196, 192)` | grid 14×14 (detail₁) |
| tokens escala 3 | `(B·T, 784, 192)` | grid 28×28 (detail₀, fino) |
| `emb` (tras projector) | `{l: (B, T, N_l, 192)}` | dict por escala |
| predicción | `{l: (B, T, N_l, 192)}` | teacher forcing |
| `criterion` | `(B, S)` | S = candidatos CEM |

---

## 4. Componentes

### 4.1 Lifting scheme — `wavelet/lifting.py`

Descomposición wavelet 2D **separable** (filas → columnas) por *lifting scheme*:

```
split polyphase:  even = x[0::2],  odd = x[1::2]
predict:          d = odd − P(even)          # detalle
update:           s = even + U(d)            # aproximación
normalización:    approx = s·√2,  detail = −d/√2
```

- **Invertibilidad estructural:** `inverse` deshace `undo update → undo predict → merge`
  con los mismos filtros. Vale para **cualquier** valor de P/U → los filtros pueden
  aprenderse sin romper la reconstrucción perfecta.
- **Inicializaciones** (kernel 3, conv depthwise):
  - `haar`: `P=[0,1,0]` (identidad), `U=[0,0.5,0]`. Da `LL=(a+b+c+d)/2`, coincide
    exactamente con el DWT Haar de `module.WaveletAttention._dwt`.
  - `cdf53`: `P=[0,0.5,0.5]`, `U=[0.25,0.25,0]` (CDF 5/3 de JPEG2000).
- **Filtros aprendibles** (`learnable=True`): `nn.Parameter` por nivel/orientación
  (`predict_w/h`, `update_w/h`), init a los coeficientes exactos. Stage 1 fijo,
  Stage 2 aprendible (ablación).
- **2D por nivel:** ancho produce `low_w/high_w`; alto sobre `low_w`→`LL,LH`, sobre
  `high_w`→`HL,HH`. Detalles apilados `[LH,HL,HH]` (3C). Recursión sobre `LL`.
- **Bordes:** padding `replicate` (funciona con tamaño espacial 1), idéntico en el
  inverso → no afecta reconstrucción.
- **Precisión:** en bf16/fp16 computa internamente en fp32 y redondea solo salidas.
  Reconstrucción: fp32 ~7e-7, bf16 <0.016, fp64 con filtros aleatorios ~1e-9.
- **Validación:** `H,W % 2^levels == 0` (224=2⁵·7 → hasta levels=5).

```python
@dataclass
class Pyramid:
    approx: Tensor            # (B,C,H/2^L,W/2^L)
    details: list[Tensor]     # len L; details[i]=(B,3C,H/2^(i+1),W/2^(i+1)); [0]=fino

LiftingScheme(levels, in_channels=3, init="haar", learnable=False)
  .forward(x) -> Pyramid
  .inverse(p) -> Tensor
```

### 4.2 PyramidEncoder — `wavelet/pyramid_encoder.py`

Convierte cada sub-banda en una grilla de tokens. **L+1 escalas**:

| level | fuente | resolución sub-banda | grid (stem_patch=4) | N_l |
|---|---|---|---|---|
| 0 | `approx` (LL_L) | 28×28 | 7×7 | 49 |
| 1 | `details[L-1]` | 28×28 | 7×7 | 49 |
| 2 | `details[L-2]` | 56×56 | 14×14 | 196 |
| 3 | `details[0]` (fino) | 112×112 | 28×28 | 784 |

Por escala: `Conv2d(in_ch, D_l, stem_patch, stem_patch)` (patchify) → flatten a tokens
→ `pos_embed` aprendible → `stem_blocks` `EncoderBlock` (atención espacial **no causal**,
`module.Attention(causal=False)` + `FeedForward`) → LayerNorm.

- `in_ch`: escala 0 = C; escalas ≥1 = 3C.
- `dims` default `[192]*(L+1)`. `stem_patch` controla el tradeoff detalle/cómputo.
- El token fino (stem_patch=4) cubre 4×4 px de sub-banda = **8×8 px originales**
  (más fino que DINO-16 → preserva orientación).

```python
@dataclass
class ScaleTokens: tokens; grid_hw; level
@dataclass
class PyramidOutput: scales: list[ScaleTokens]   # len L+1

PyramidEncoder(levels=3, dims=None, stem_blocks=2, stem_patch=4,
               img_size=224, learnable_lifting=False, ...)
  .forward(pixels) -> PyramidOutput
```

### 4.3 Predictores por escala (intercambiables)

**Interfaz espacial común** (la implementa `nca.LatentNCAPredictor`):
```python
forward(x:(B,T,N,D), c:(B,T,A), grid_hw=None, only_last=False) -> (B,T',N,D)
rollout(x_ctx:(B,Tc,N,D), c_seq:(B,M,A), grid_hw=None, post=None,
        persist_hidden=True, detach_between=False, history=None) -> (B,M,N,D)
```

**`HierarchicalPredictor`** (`wavelet/hier_predictor.py`): router puro sobre un
`nn.ModuleDict` keyed por `str(level)`. Acción compartida, `grid_hw` por escala,
`post` (pred_proj) por escala. No asume NCA. Agrega `aux`/`aux_rollout` por nivel.

**Dos implementaciones intercambiables:**

- **NCA** (`nca.LatentNCAPredictor`, existente): regla local compartida, K pasos
  internos, canales ocultos (scratchpad), gate opcional, daño latente. Receptive
  field local que crece con K. Determinista en eval (`eval_mode=fixed_mask`).
- **ViT** (`wavelet/vit_predictor.py`, nuevo): `ViTSpatialPredictor`. Un transformer
  sobre los N tokens de cada frame, atención **espacial no causal** (receptive field
  global en una pasada) + **AdaLN-zero con la acción** (`module.modulate`, bloques
  identidad al init estilo DiT). Velocidad `x_t−x_{t-1}` como canal extra. Sin canales
  ocultos (`persist_hidden` se ignora). Coherente con el predictor transformer
  `ARPredictor`/`MaskedPredictor` de la rama `direct-predictor`. Determinista en eval.

Ambos comparten interfaz → `PyramidJEPA` y el planning no cambian al alternar.
Fábricas Hydra: `wavelet/factories.py` (`hier_nca`, `hier_vit`, `scale_mlp_dict`).

### 4.4 PyramidJEPA — `jepa_pyramid.py`

Extiende `jepa.JEPA`. Latentes = `dict[level → (B,T,N_l,D_l)]`.

- **`encode`**: píxeles → `PyramidEncoder` → projector token-a-token por escala
  (`module.MLP`, patrón `SpatialJEPA._tokenwise`). Guarda `grid_hw` por escala.
- **`predict`**: `HierarchicalPredictor` + `pred_proj` por escala.
- **`predict_rollout`**: rollout por escala con `post` = pred_proj por nivel.
- **`rollout`**: autoregresivo por escala (patrón `SpatialJEPA.rollout`);
  `predicted_emb` dict `[l] → (B,S,T,N_l,D_l)`.
- **`criterion`** (costo CEM):
  ```
  cost = Σ_l  w_l · MSE( pred[l][:,:,-1],  goal[l][:,-1] )   sumado sobre (N_l, D_l)
  ```
  → `(B,S)`. `w_l = scale_weights` (buffer, default unos).
- **`get_cost`**: **heredado de `JEPA`** (genérico: encode del goal + rollout +
  criterion). Compatible con `stable_worldmodel.solver.CEMSolver(model=Costable)`.

### 4.5 Pérdidas

En `train.py` (rama `isinstance(emb, dict)`):
- **`pred_loss`** = media **ponderada** por `scale_weights` (normalizados) de la MSE
  por escala.
- **`SIGReg`** = **media** entre escalas de `sigreg(sigreg_input(emb_l, max_tokens))`.
  Media (no suma) para mantener la escala de λ=0.09 tuneada para un solo término.
  `max_tokens` submuestrea tokens por escala (costo SIGReg ∝ n_samples).
- **`loss`** = `pred_loss + λ·sigreg_loss + extra`.

`pyramid_losses` (auxiliares, análogo a `spatial_losses`):
- `gate_l1` (si el predictor expone `gate_mean` agregado).
- `idm` (dinámica inversa sobre la **escala fina**, opcional).
- `rollout` multi-paso por escala (media), opcional.
- Diagnóstico `copy_loss_l{l}` por escala.

Los baselines (global `(B,T,D)` y espacial `(B,T,N,D)`) siguen su rama original intacta.

---

## 5. Planning CEM

`viz/eval_reward.py` → `swm.policy.WorldModelPolicy` + `swm.solver.CEMSolver(model=...)`.
El solver llama `model.get_cost(info, action_candidates)`:
1. encode del goal → `goal_emb` dict por escala.
2. `rollout` de los candidatos → `predicted_emb` dict.
3. `criterion` → costo `(B,S)` ponderado por escala.

Paridad con baseline: 50 eps, budget 50, horizon 5, `num_samples=300`, `n_steps=30`,
`topk=30`. Determinismo: predictores deterministas en eval → costo sin ruido entre
candidatos.

---

## 6. Configuración y ejecución

Configs Hydra: `config/train/model/liftwm.yaml` (NCA), `liftwm_vit.yaml` (ViT).
`n_scales = encoder.levels + 1` (levels=3 → 4).

**Entorno canónico** (swm 0.1.0 con `solver.CEMSolver`, el de los baselines):
`/home/chr/le-wm/.venv/bin/python`. Tests en `.venv_hf` (tiene pytest).

```bash
P=/home/chr/le-wm/.venv/bin/python
# ViT cube (batch 32 por memoria; expandable_segments):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P train.py \
    data=ogb model=liftwm_vit loader.batch_size=32 \
    output_model_name=liftwm_vit_cube \
    wandb.enabled=true wandb.config.entity=chstr wandb.config.project=lewm
# NCA: model=liftwm.  Stage 2 aprendible: model.encoder.learnable_lifting=true
# Eval:
$P viz/eval_reward.py policy=<ckpt>/weights_epoch_10.pt --config-name=cube
```

---

## 7. Decisiones de diseño y tradeoffs

| Decisión | Elección | Porqué / tradeoff |
|---|---|---|
| Pirámide en píxeles vs features DINO | **píxeles** | DINO ya perdió la orientación (probes); la pirámide debe ver el detalle fino. |
| Grid fino | `stem_patch=4` → 28×28=784 tokens | 8×8 px/token, más fino que DINO-16. Mayor `stem_patch` = más rápido pero re-descarta detalle. |
| Predictor | **ViT y NCA intercambiables** | El usuario aún no decide; misma interfaz, ablación directa. |
| SIGReg por escala | **media** | Preserva escala de λ=0.09 (anti-colapso). |
| `scale_weights` | pondera pred_loss + criterion | Permite enfatizar escalas finas en planning. |
| Lifting | Haar fijo (Stage 1) → aprendible (Stage 2) | Valida pipeline primero; ablación fija-vs-aprendible gratis. |
| Memoria ViT | batch 32 + expandable_segments | batch 64 → OOM (24 GB) por 784 tokens × 4 ViT. |
| Entorno | le-wm/.venv para train/eval | `.venv_hf` tiene swm nuevo SIN `solver` → no eval CEM. |

**Anti-goals vigilados:** colapso de representación (SIGReg por escala + ablación),
costo CEM ruidoso (predictores deterministas en eval), pérdida de invertibilidad
(garantizada por estructura + tests).

---

## 8. Mapa de archivos

```
wavelet/lifting.py          LiftingScheme + Pyramid
wavelet/pyramid_encoder.py  PyramidEncoder + ScaleTokens/PyramidOutput + EncoderBlock
wavelet/vit_predictor.py    ViTSpatialPredictor + SpatialBlock
wavelet/hier_predictor.py   HierarchicalPredictor (router por escala)
wavelet/factories.py        scale_mlp_dict / hier_nca / hier_vit
jepa_pyramid.py             PyramidJEPA + pyramid_losses
train.py                    rama dict-emb (baselines intactos)
config/train/model/liftwm.yaml      NCA
config/train/model/liftwm_vit.yaml  ViT
tests/test_lifting.py               9 tests
tests/test_pyramid_encoder.py       4 tests
tests/test_hier_predictor.py        3 tests
tests/test_vit_predictor.py         5 tests
tests/test_pyramid_jepa.py          6 tests
tests/test_liftwm_smoke.py          4 tests (configs, forward, ckpt, get_cost)
```

---

## 9. Estado de verificación

- **43 tests verdes** (`.venv_hf/bin/python -m pytest tests/ -q`), sin regresiones.
- Smoke-train real (tworoom) para **NCA y ViT**: entrenan, loguean por escala, guardan ckpt.
- `get_cost` (camino CEM completo) testeado.
- **Pendiente (T6):** entrenamiento real cube/pusht + eval + ablación Stage 1 vs 2.
  Juez: cube > 74%, Push-T ≥ ~94%.
