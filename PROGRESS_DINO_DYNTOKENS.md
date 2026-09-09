# Phase 1 report — dynamic tokens in the DINOv3 ViT wrapper

State: **done / green**. No training performed. `k=0` behavior unchanged.

## What changed

### Back end (official ViT backbone) — `src/dinov3/models/vision_transformer.py`
- `prepare_tokens_with_masks` (v1): concat `[cls, prompt.expand(B), storage, x]`
  in one step (no in-place `+=` / separate storage concat).
- `forward_features_list`: `prompt = prompt_list[0].shape[1]` (per-crop list API;
  the shape of the *second* dim of the single prompt crop equals `k`).
- CLS region slicing: added the missing `+1` for the CLS token
  (`[: n_storage_tokens + prompt + 1]`). This fixes a latent off-by-one where
  the 4th storage token silently leaked into the patch bucket. **Safe**: `k=0`
  `last_hidden_state` is byte-identical (wrapper concatenates in the same order;
  only internal bucketing changed).
- Output dict: `"x_prompt_tokens" = x_norm_cls_reg[:, 1:1+n_prompt]`,
  `"x_storage_tokens" = x_norm_cls_reg[:, 1+n_prompt:]`.
- `_get_intermediate_layers_not_chunked`: accepts/propagates `prompt=`.

### LoRA helpers — `src/encoders/vit_lora.py`
- `freeze_all(model)`: freezes 100% (incl norms + LORA + storage + dyn tokens).
- `apply_lora_to_vit`: early-returns `[]` when `r <= 0` (no-LoRA backbone).

### Wrapper — `src/encoders/dinov3_encoder.py`
- New args: `n_dyn_tokens=0`, `readout="cls"`, `dyn_init="cls"`,
  `freeze_mode="lora_norm_patch"`. Assertions + options (`READOUT_OPTIONS`,
  `DYN_INIT_OPTIONS`, `FREEZE_MODE_OPTIONS`).
- `self.readout_dim` = 384, or `k*384` for `dyn_concat`.
- `self.dyn_tokens`: `nn.Parameter` of shape `(1, k, 384)` (broadcast over
  batch). Created **after** freeze so it is trainable by default; in `all`
  mode explicitly frozen.
- `_create_dyn_tokens`: `cls` -> copy of `self.model.cls_token` +
  `N(0, 0.02*sigma)`, `storage` -> copy of `storage_tokens` (or normal),
  `normal` -> `N(0, 0.02)`. `sigma = base.abs().mean()`.
- `forward`: builds `[CLS, dyn, storage, patches]`; computes `pooled` per
  `readout`; returns `types.SimpleNamespace(last_hidden_state, pooled,
  dyn_tokens, cls)`. `prompt` kwarg is accepted but unused internally (the
  tokens are always fed via `self.dyn_tokens`).
- `prepare_tokens_with_masks` no longer used by the wrapper (kept in the
  backbone; unused).

### JEPA — `jepa.py`
- `encode`: `emb = out.pooled if out.pooled is not None else out.last_hidden_state[:, 0]`
  (2-line change; falls back to CLS for encoders that don't expose `.pooled`).

## New init modes + `init_std` (this change)

### Wrapper — `src/encoders/dinov3_encoder.py`
- New arg `init_std: float | None = None`. `None` preserves the legacy noise
  scale `0.02 * base.abs().mean()` exactly; a float is used directly as the
  Gaussian `std`.
- `DYN_INIT_OPTIONS` extended to `("cls", "cls_noise", "zeros", "storage", "normal")`.
- `_create_dyn_tokens`:
  - `cls`      -> **exact** copy of `cls_token`, noise forced to `0` even if
    `init_std > 0` (baseline-identical point).
  - `cls_noise`-> copy of `cls_token` + `N(0, init_std)` per token (helper
    `_dyn_noise`).
  - `zeros`    -> `nn.Parameter` of exact zeros; degenerate by design with any
    readout that reads the dynamic tokens (`dyn_first` / `dyn_mean` /
    `dyn_concat`) — the "no prior" checkpoint.
  - `storage`  / `normal` unchanged (legacy).
- Module + class docstrings document the init table.

### Smoke test — `smoke_dino_encoders.py`
- Removed the old `cls != raw CLS` noise assertion (no longer valid).
- New `dyn_init` section: `zeros` must be exactly zero before the forward;
  `cls` must equal the source `cls_token` (atol `1e-4`) and keep
  `pooled == cls` for the zero-noise point; `cls_noise` with `init_std=0.5`
  must differ from the raw CLS.

### PROGRESS
- `init_std` is now a real configurable arg (open item 3 resolved).

## Verification (smoke test, CPU, green)
`smoke_dino_encoders.py` composes all 4 configs via hydra, instantiates the
encoder, runs a forward, and (for dinov3) checks the dynamic-token feature:
token-sequence shape, `readout_dim`, off-by-one placement, slice correctness
for all 4 readouts, and that cls-init noise differs from the raw CLS register.

Key results:
- k=0 base (dinov3): `last_hidden_state=(2, 201, 384)`, `readout=cls` =>
  `pooled == last_hidden_state[:, 0]` (the path `jepa.encode` now uses).
- k=4: `last_hidden_state=(2, 205, 384)` = `[CLS(1), dyn(4), storage(4), patches(196)]`.
  `dyn_concat` => `pooled=(2, 1536)`, `readout_dim=1536`.
- noise added at init: `max|dyn_tokens - raw CLS| ≈ 5.2e-3` (nonzero, expected).
- **`ALL SMOKE CHECKS PASSED`.**

Run:
```
timeout 900 python smoke_dino_encoders.py
```

## Fase 2 — `dinov3_frozen` (DINO-WM-style, backbone 100% congelado)

State: **done / green**. No training performed. Config verification only.

### What changed
- New file `config/train/model/dinov3_frozen.yaml`, copied from
  `dinov3_lora.yaml` and modified **only** in the encoder block:
  - `lora_r: 0`        -> `apply_lora_to_vit()` early-returns `[]` (no LoRA added).
  - `freeze_mode: "all"` -> `freeze_all()` freezes 100% of the backbone (norms,
    patch-embed and storage tokens included).
  - `n_dyn_tokens: 0`   -> no dynamic tokens (`[CLS, storage, patches]`).
- Everything else kept **byte-identical** to `dinov3_lora.yaml`:
  predictor / action_encoder / projector / pred_proj.

This is the **0-parameters-of-the-sweep** ("suelo") point: the encoder
contributes **zero** trainable params; only the projector, predictor and action
encoder learn (the classic DINO-WM recipe: frozen visual backbone + MLP head).
The yaml documents the sweep launch command (dataset `tworoom`, 10 epochs,
3 seeds).

### Verification (instantiating the full JEPA model via hydra, CPU, no training)
`model=dinov3_frozen` (local `vit_small/16` checkpoint; `action_encoder.input_dim`
overridden to `4` just to instantiate — train.py fills it from the dataset).
Counts of `requires_grad` params per submodule:

| submodule      | total      | trainable |
|----------------|-----------:|----------:|
| encoder        | 21,601,152 | **0**     |
| projector      |  1,185,984 | 1,185,984 |
| predictor      | 10,791,360 | 10,791,360|
| action_encoder |    156,146 |   156,146 |

- `model.encoder` has **0 trainable params** (backbone fully frozen).
- projector + predictor + action_encoder = **12,133,490** trainable params —
  these are the only ones that learn.
- Forward pass sanity check (`k=0`, `readout=cls`):
  `pooled=(2,384)` and `pooled == last_hidden_state[:,0]` (the `jepa.encode`
  path) — finite, correct.
- The encoder constructor log prints
  `[trainable=0/21,601,152 params]`, `n_dyn=0`, `freeze_mode=all`, `lora_layers=0`.

Run (temp script, then removed — the canonical check is via the overrides above;
keep a permanent copy where you want):
```
PYTHONPATH=$PWD python verify_frozen_encoder_tmp.py   # PASSED
```

### Training command (dataset tworoom, 10 epochs, 3 seeds)
```
export DINOV3_CKPT=/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
for seed in 3072 1234 5678; do \
  python train.py model=dinov3_frozen data=tworoom \
    trainer.max_epochs=10 seed=$seed & \
done
```
(The optimizer is a single `model_opt` over the trainable params only — the
frozen backbone never appears in the param graph, so it stays frozen.)

## Decisions / open items (for review)
1. **CLS off-by-one fix is a safe change**: `k=0` outputs are unchanged; only
   internal region bucketing was corrected. Flagging in case downstream code
   depends on the (buggy) previous bucketing.
2. **Optimizer / `spt.Module`**: optimizer is a *single* `model_opt` over the
   `'model'` parameter group (`config/train/optimizers.yaml`), not per-module
   groups. `spt.Module.__getattr__` (in `src/spt/module.py`) forwards attribute
   access to its `self.module` submodules, so freeze state propagates through
   the wrapper. **Not implemented here** (per task), but noted.
3. **`readout` / `dyn_init` / `init_std` config plumbing**: the base `lewm`
   hydra config does not declare these keys, so they must be added with the
   `+` prefix when overridden (e.g. `+model.encoder.n_dyn_tokens=4`). The
   constructor defaults apply otherwise. `init_std` (the noise scale `sigma`)
   is currently computed as `base.abs().mean()` — not yet a standalone
   configurable arg.
4. **Determinism**: the forward has LoRA dropout active in the backbone's
   `is_training=True` path, so outputs are not bit-for-bit deterministic across
   passes. The slice-based checks compare tensors from the *same* forward pass,
   so they remain valid.

## Next steps
- Phase 2: integrate dynamic-token generation into JEPA's cross-attention
  (batched) and confirm the `k=0` fallback is still exact.
- Decide whether to make `init_std` a configurable arg (item 3).
- (Optional) formalize the optimizer group handling for `spt.Module` if a
  per-module optimizer is required downstream.

## Nota de entorno (.venv)

- El `.venv` del workspace (`/home/chr/Multiscale-World-Models/.venv`) está
  **vacío** (sin `hydra`/`torch`) y **no sirve** para ejecutar el smoke test
  (`smoke_dino_encoders.py`).
- El entorno canonico para correr el smoke / tests es
  `/home/chr/le-wm/.venv`.
- **Pendiente (futuro)**: decidir si se rellena el `.venv` del workspace o se
  documenta formalmente que el entorno canónico es `/home/chr/le-wm/.venv`.
