# Diseño: prototipos frame-level para JEPA (Etapa 1)

Fecha: 2026-10-01. Base acordada: `exp/multiscale-diagnostics` + `dinov3_starlet_lora`.

## 1. Intento compartido

Validar con cambios mínimos si un bottleneck discreto por frame ayuda al
world model, sin romper rollout, planning ni checkpoints actuales.

Lo dicho por el usuario vs supuestos:

- Dicho: latente actual es CLS por frame (`jepa.py:38` → `(B,T,D)`),
  sin stop-grad en target (`train.py:36`), rollout realimenta `pred_emb`
  en el mismo espacio post-projector 192-d, planning es MSE continuo
  (`jepa.py:112`). Verificado en código el 2026-10-01.
- Supuesto: `CEMSolver` usa `JEPA.get_cost` tal cual (no verificado en
  `stable_worldmodel`; se asume y se mantiene la interfaz).
- Supuesto: 1 GPU (`devices: auto` sin `all_reduce`; SIGReg ya documenta
  "single-GPU!").

Éxito Etapa 1: entrena sin colapso (perplejidad/entropía de clusters
sanas), `predict`/`rollout`/`criterion` intactos, ablaciones medibles en
two-room, Push-T y Cube + error de rollout por horizonte.

## 2. Alcance acordado

- Solo Etapa 1 frame-level. Etapa 2 patch-level queda fuera de esta rama.
- Sin híbrido residual `ẑ = Σπc + r(h)` en esta rama (futura ablación).
- Variante elegida: A SwAV compartido
  `logits = norm(h)·norm(C)/τ`, `ẑ = Σ π_k c_k`.
  Descartadas: B Linear separada, C solo CE auxiliar.

## 3. Arquitectura y componentes

### `module.py` (nuevo)

- `PrototypeHead(K, D)`: `nn.Parameter (K, D)`, normalizado en cada
  forward; `forward(h) → (logits, pi, z)` con `pi = softmax(logits)`.
- `sinkhorn(scores, eps, iters)`: sin gradiente, ~3 iters.
  Defaults: `K=64, tau=0.1, eps=0.05, iters=3`.

### `jepa.py` (integración compatible)

- `JEPA.__init__(..., proto_head=None)` igual que `projector`/`pred_proj`,
  para que `save_pretrained`/`load_pretrained` lo incluyan vía `cfg.model`.
- `predict(emb, act_emb)` intacto (devuelve `ẑ` continuo) → no se rompen
  `rollout`, `viz/rollout_decoded.py`, `decoder_train.py`, `eval.py`.
- Nuevo `predict_proto(...)` solo para entrenamiento → `(ẑ, logits)`.

### `train.py` (`lejepa_forward`)

- Target con `torch.no_grad()`:
  `scores = norm(tgt_emb)·norm(C)ᵀ / τ`, `q = sinkhorn(scores)`.
- Pérdida:
  `λ_mse·MSE(ẑ, tgt_emb) + λ_ce·CE(logits, q) + 0.09·SIGReg`
  con `weight_mse=1.0, weight_ce=1.0` por defecto.
- `self.log` existente filtra `"loss" in k`, así que `proto_ce_loss`
  aparece solo; perplejidad y entropía de `q` se loguean aparte.
- Congelar prototipos durante el primer epoch (estilo SwAV) queda como
  opción del plan, no decidido.

## 4. Configs

- Nuevo `config/train/model/dinov3_starlet_lora_proto.yaml` (copia de
  `dinov3_starlet_lora.yaml` + bloque `proto_head`). Modelos actuales
  intactos.
- `config/train/lewm.yaml`: agregar `loss.proto` con
  `K, tau, eps, iters, weight_ce, weight_mse`.
- Todo instanciado con `hydra.utils.instantiate`.

## 5. Flujo de datos

`encode (CLS)` → `(B,T,192)` → `predict_proto` → `(ẑ, logits)` vs
target `tgt_emb` (no-grad) → `q` → losses. `predict` clásico sigue
alimentando `rollout`/`criterion` con vectores continuos 192-d.

## 6. Restricciones

- No tocar `src/dinov2` ni `src/dinov3` (código vendor).
- Checkpoints previos deben cargar igual (defaults = comportamiento previo).
- Sin `all_reduce` (1 GPU).

## 7. Testing y evaluación

- Sintaxis + smoke de `PrototypeHead`/`sinkhorn` en CPU.
- Entrenamiento corto que verifique: loss baja, perplejidad > 1,
  entropía no colapsada, `rollout` produce `(B,S,T,D)` continuo.
- Ablaciones Etapa 1: baseline vs solo-CE-auxiliar vs mezcla `ẑ=Σπc`.
  Métricas: éxito two-room/Push-T/Cube, error rollout vs horizonte,
  uso de clusters.
- `eval.py` sin cambios (interfaz `predict` intacta). Costo de planning
  por distribución (CE π vs q del goal) queda como ablación opcional.

## 8. Riesgos

- SIGReg vs clusters: gaussiana isotrópica sin estructura de clusters
  (hipótesis más incierta).
- Colapso inicial por co-adaptación sin EMA → monitorear perplejidad
  desde epoch 0.
- Pérdida de detalle: `ẑ` limitado a la envolvente convexa de prototipos
  puede degradar planning fino.

## 9. Fuera de alcance (Etapa 2)

Patches `last_hidden_state[:, 1+n_storage:]`, helper por encoder,
`ARPredictor` a `T·N` tokens con máscara block-causal, `projector`/
`pred_proj` con `BatchNorm1d` sobre tokens aplanados, SIGReg sobre tokens,
`criterion` sumado sobre N, `decoder_train.py`/viz. Todo para un spec
separado.
