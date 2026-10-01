# Prototype JEPA frame-level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SwAV-shared frame-level prototypes to JEPA training without breaking predict/rollout/planning.

**Architecture:** `PrototypeHead(K,D)` + no-grad `sinkhorn` in `module.py`; `JEPA(proto_head=None)` + `predict_proto` in `jepa.py`; proto loss in `train.py:lejepa_forward`; new model yaml + `loss.proto` in configs. `predict` stays intact so rollout/criterion/CEM keep continuous 192-d vectors.

**Tech Stack:** PyTorch, Hydra instantiate, Lightning (`spt.Module`), BatchNorm1d MLPs, single GPU (no all_reduce).

**Spec:** `docs/superpowers/specs/2026-10-01-proto-jepa-frame-design.md`

## Global Constraints

- Do NOT touch `src/dinov2` nor `src/dinov3` (vendor code).
- `predict(emb, act_emb) -> Tensor (B,T,D)` keeps its signature and behavior; only `predict_proto` returns logits.
- Checkpoints actuales deben cargar igual: `proto_head=None` por defecto; defaults = comportamiento previo.
- Single GPU: no `all_reduce` en `sinkhorn`/logging.
- Valores del spec: `K=64, tau=0.1, eps=0.05, iters=3, weight_ce=1.0, weight_mse=1.0, sigreg weight=0.09`.
- Rama: `exp/proto-jepa-frame` (base `exp/multiscale-diagnostics`); sin push.

## Review Focus

- Colapso a pocos clusters con `K=64` y encoder congelado parcial: se espera perplejidad >> 1 y entropía de `q` alta; si `pi` concentra > 0.99 en < 4 clusters el entrenamiento no valida la idea.
- `pred_proj` usa `BatchNorm1d` sobre `(B*T, D)`: en `eval()` vs `train()` las estadísticas cambian `h` y por tanto `logits`; se espera mismo shape y valores finitos en ambos modos.
- Checkpoints viejos sin bloque `proto_head` deben instanciar con `None` y entrenar igual que antes; se espera `predict` idéntico con y sin head.
- `get_cost`/CEM debe seguir usando `predict` (continuo), nunca `predict_proto`; se espera costo MSE finito sobre `(B,S)`.
- `sinkhorn` en `bf16` (`precision: bf16` en `lewm.yaml`) puede dar NaN: se espera computar scores en `float32` y devolver `q` finita que suma 1 por fila/columna según marginales.

---

### Task 1: PrototypeHead + sinkhorn en module.py

**Files:**
- Modify: `module.py` (append, no tocar clases existentes)
- Test: `tests/test_proto.py` (nuevo)

**Interfaces:**
- Consumes: `torch`, `torch.nn.functional as F` (ya importados en `module.py`).
- Produces:
  - `sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor` (decorado `@torch.no_grad()`, acepta `(..., K)`, devuelve `q` misma forma, filas suman ~1).
  - `PrototypeHead(torch.nn.Module).__init__(self, num_prototypes: int = 64, dim: int = 192, tau: float = 0.1)`; `self.prototypes = nn.Parameter(torch.randn(K, D) * 0.02)`.
  - `PrototypeHead.forward(self, h: torch.Tensor) -> tuple[logits: Tensor, pi: Tensor, z: Tensor]` con formas `(..., K), (..., K), (..., D)`; `logits = h_norm @ C_norm.T / tau`.

- [ ] **Step 1: Write the failing test**

```python
def test_prototype_head_shapes():
    m = PrototypeHead(num_prototypes=64, dim=192, tau=0.1)
    h = torch.randn(4, 3, 192)
    logits, pi, z = m(h)
    assert logits.shape == (4, 3, 64)
    assert pi.shape == (4, 3, 64)
    assert z.shape == (4, 3, 192)
    assert torch.allclose(pi.sum(-1), torch.ones(4, 3), atol=1e-5)

def test_sinkhorn_rows_sum_to_one():
    q = sinkhorn(torch.randn(8, 64) / 0.1, eps=0.05, iters=3)
    assert q.shape == (8, 64)
    assert torch.isfinite(q).all()
    assert torch.allclose(q.sum(-1), torch.ones(8), atol=1e-3)

(Convención fijada: `sinkhorn` devuelve matriz row-stochastic, cada fila suma 1.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_proto.py -v`
Expected: FAIL with "PrototypeHead not defined" / file not found.

- [ ] **Step 3: Implement `sinkhorn` + `PrototypeHead` in `module.py`**

`@torch.no_grad()` en `sinkhorn`; normalizar con `F.normalize(..., dim=-1)`; `pi = softmax(logits, -1)`; `z = pi @ C_norm`. Sin tocar `SIGReg`/otras clases.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_proto.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add module.py tests/test_proto.py
git commit -m "feat(proto): PrototypeHead SwAV K64 + sinkhorn sin gradiente"
```

### Task 2: JEPA proto_head + predict_proto

**Files:**
- Modify: `jepa.py:13-55`
- Test: `tests/test_proto.py` (append)

**Interfaces:**
- Consumes: `PrototypeHead` de Task 1 (import perezoso dentro de `jepa.py` para no romper `stable_worldmodel`).
- Produces:
  - `JEPA.__init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None, proto_head=None)`.
  - `JEPA.predict_proto(self, emb: Tensor (B,T,D), act_emb: Tensor) -> tuple[z: Tensor (B,T,D), logits: Tensor (B,T,K)]` = `self.predict(...)` + `self.proto_head(h)`; si `proto_head is None` lanza `RuntimeError`.

- [ ] **Step 1: Write the failing test**

```python
def test_predict_proto_shapes_and_predict_intact():
    jepa = make_mini_jepa(proto_head=PrototypeHead(64, 16, tau=0.1), dim=16)
    emb = torch.randn(2, 3, 16)
    act = torch.randn(2, 3, 4)
    z, logits = jepa.predict_proto(emb, act)
    assert z.shape == (2, 3, 16) and logits.shape == (2, 3, 64)
    p1 = jepa.predict(emb, act)
    assert p1.shape == (2, 3, 16) and torch.isfinite(p1).all()
```

(`make_mini_jepa` = helper del test con `ARPredictor`/`Embedder`/`MLP` mínimos sobre CPU.)

```python
def test_predict_proto_none_head_raises_but_predict_works():
    jepa = make_mini_jepa(proto_head=None, dim=16)
    emb = torch.randn(2, 3, 16)
    act = torch.randn(2, 3, 4)
    assert jepa.predict(emb, act).shape == (2, 3, 16)
    with __import__("pytest").raises(RuntimeError):
        jepa.predict_proto(emb, act)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_proto.py::test_predict_proto_shapes_and_predict_intact -v`
Expected: FAIL with "predict_proto not defined".

- [ ] **Step 3: Implement `proto_head` + `predict_proto` in `jepa.py`**

Guardar `self.proto_head`; `predict` sin cambios; `predict_proto` llama a `self.predict` y luego al head. `None` → `RuntimeError("proto_head no configurado")`.

- [ ] **Step 4: Run tests (nuevo + anteriores)**

Run: `python -m pytest tests/test_proto.py -v`
Expected: PASS. Además `python -m pytest tests/test_multiscale.py -q` sigue PASS (no se tocó multiscale).

- [ ] **Step 5: Commit**

```bash
git add jepa.py tests/test_proto.py
git commit -m "feat(proto): JEPA predict_proto con head opcional, predict intacto"
```

### Task 3: lejepa_forward con pérdida proto

**Files:**
- Modify: `train.py:18-54`
- Test: `tests/test_proto.py` (append, test de loss pura sin hydra/trainer)

**Interfaces:**
- Consumes: `JEPA.predict_proto`, `sinkhorn`, `self.sigreg`.
- Produces: en `output`: `pred_loss` (MSE existente), `proto_ce_loss`, `loss = weight_mse*mse(z) + weight_ce*CE + 0.09*sigreg` donde el término MSE usa `ẑ` de `predict_proto` cuando hay head; si `proto_head is None` el forward es idéntico al actual. Helper `_proto_losses(...) -> dict(mse, ce, loss, q, perplexity, entropy)` con scores en `float32` (aunque el modelo corra en `bf16`). Lee `cfg.loss.proto = {eps, iters, weight_ce, weight_mse, tau}` y `cfg.loss.sigreg.weight`.

- [ ] **Step 1: Write the failing test**

```python
def test_proto_loss_terms_finite():
    out = proto_losses(z=torch.randn(4, 2, 16), logits=torch.randn(4, 2, 64), tgt=torch.randn(4, 2, 16), eps=0.05, iters=3)
    assert torch.isfinite(out["mse"]) and torch.isfinite(out["ce"])
    assert out["loss"] == out["mse"] + out["ce"]  # pesos 1.0 en el helper
    assert torch.isfinite(out["perplexity"]) and out["perplexity"] > 1.0
    assert torch.isfinite(out["entropy"]) and out["entropy"] > 0.0
```

(Si se implementa inline en `lejepa_forward` sin helper, el test construye el mismo cálculo con `sinkhorn` + `F.cross_entropy` sobre `(B*T, K)` y `F.mse_loss`; el plan fija ese helper como función privada `_proto_losses` en `train.py` para testearla.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_proto.py::test_proto_loss_terms_finite -v`
Expected: FAIL with "_proto_losses not defined".

- [ ] **Step 3: Implement `_proto_losses` + rama proto en `lejepa_forward` en `train.py`**

Target con `torch.no_grad()`: `scores = norm(tgt) @ norm(C).T / tau`, `q = sinkhorn(scores)`, `CE(logits.reshape(-1,K), q.reshape(-1,K))`; `MSE(z, tgt)`; `loss = w_mse*mse + w_ce*ce + 0.09*sigreg`. `C` = prototipos normalizados del head. Log: `proto_ce_loss` entra en `losses_dict` por el filtro `"loss" in k`; perplejidad/entropía de `q` con `self.log` aparte (`train/proto_perplexity`, `train/proto_entropy`). Sin head → camino actual byte-idéntico.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_proto.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_proto.py
git commit -m "feat(proto): loss MSE+CE+SIGReg con target Sinkhorn no-grad"
```

### Task 4: Configs + smoke instantiate

**Files:**
- Create: `config/train/model/dinov3_starlet_lora_proto.yaml` (copia de `dinov3_starlet_lora.yaml` + bloque `proto_head`)
- Modify: `config/train/lewm.yaml` (agregar `loss.proto`; NO cambiar `sigreg`)
- Test: append en `tests/test_proto.py` o script inline (sin GPU)

**Interfaces:**
- Consumes: bloques hydra existentes (`${history_size}`, `${embed_dim}`).
- Produces:
  - `proto_head: {_target_: module.PrototypeHead, num_prototypes: 64, dim: ${embed_dim}, tau: 0.1}`.
  - `loss.proto: {tau: 0.1, eps: 0.05, iters: 3, weight_ce: 1.0, weight_mse: 1.0}`.

- [ ] **Step 1: Write the failing test**

```python
def test_proto_yaml_instantiates():
    head = hydra.utils.instantiate({"_target_": "module.PrototypeHead", "num_prototypes": 64, "dim": 192, "tau": 0.1})
    h = torch.randn(2, 3, 192)
    logits, pi, z = head(h)
    assert z.shape == (2, 3, 192)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_proto.py::test_proto_yaml_instantiates -v`
Expected: FAIL antes de Task 1; tras Tasks 1-3 debe PASS (este task lo fija como regresión de config).

- [ ] **Step 3: Crear yaml proto + `loss.proto`**

Copiar `dinov3_starlet_lora.yaml` byte a byte y solo añadir `proto_head`; en `lewm.yaml` solo añadir bloque `loss.proto` (mantener `sigreg.weight: 0.09`).

- [ ] **Step 4: Verificar**

Run: `python -m pytest tests/test_proto.py tests/test_multiscale.py -q`
Expected: PASS. Y `python -c "from omegaconf import OmegaConf; c=OmegaConf.load('config/train/model/dinov3_starlet_lora_proto.yaml'); print(c.proto_head._target_)"` → `module.PrototypeHead`.

- [ ] **Step 5: Commit**

```bash
git add config/train/model/dinov3_starlet_lora_proto.yaml config/train/lewm.yaml tests/test_proto.py
git commit -m "feat(proto): config starlet proto K64 + loss.proto defaults"
```
