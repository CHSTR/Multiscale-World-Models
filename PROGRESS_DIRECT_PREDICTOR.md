# Direct-Predictor Phase A — Progress

## Scope (Phase A only)
1. Refactor `src/encoders/dinov3_encoder.py` to the new contract.
2. Verify the DINOv3 backbone supports the direct prompt path (`vision_transformer.py`).
3. Extend `smoke_dino_encoders.py` with Phase 0 (P0) asserts.
4. Record this progress.

No training, no push, no commit.

## Contract (new)
- Args: `n_in_tokens=1`, `n_tgt_tokens=1`, `tok_init {cls,cls_noise,normal}`,
  `tie_tgt_to_in=False`, `freeze_mode="lora_norm_patch"`.
- Single `nn.Parameter prompt_tokens` of shape `(1, n_in+n_tgt, 384)`.
- Forward:
  ```
  features = self.model(x, prompt=self.prompt_tokens, is_training=True)
  tok = features["x_prompt_tokens"]
  tok_in = tok[:,0]          # (BT, 384)
  tok_tgt = tok[:,1:]        # (BT, K, 384)
  if tie_tgt_to_in: tok_tgt = tok_in.unsqueeze(1)
  ```
- Returns `SimpleNamespace(last_hidden_state, cls, tok_in, tok_tgt, pooled)`
  with `pooled = tok_in`, `readout_dim = 384`.
- `starlet`/`LoRA`/`freeze`/`ckpt` unchanged.

## Legacy compat (decision — documented)
- `n_dyn_tokens` kept as a compat alias only:
  - `n_dyn_tokens = k` => `n_in_tokens = 1`, `n_tgt_tokens = k - 1`.
  - `k <= 0` => `n_tgt_tokens = 0` (single prompt token only).
- This keeps `config/train/model/dinov3_frozen.yaml` (`n_dyn_tokens: 0`)
  loadable WITHOUT editing the yaml.
- `readout`/`dyn_init`/`dyn_tokens` removed entirely (single API path).
- `pooled` now = `tok_in` (previously CLS). Frozen backbone is still
  deterministic; JEPA.encode falls back to `last_hidden_state[:,0]` if needed.

## Backbone change (minimal)
- `vision_transformer.py::forward_features_list` now accepts a single prompt
  tensor `(1,N,D)` OR a per-crop list/tuple; `n_prompt = prompt.shape[1]`.
- `forward_features` passes `prompt` positionally.
- CLS-region slicing uses int `n_prompt`:
  - `x_prompt_tokens = x_norm_cls_reg[:, 1:1+n_prompt]`
  - `x_storage_tokens = x_norm_cls_reg[:, 1+n_prompt:]`
- `_get_intermediate_layers_not_chunked` forwards a single-tensor prompt to
  `prepare_tokens_with_masks`.

## Verification (smoke GREEN)
Run: `/home/chr/le-wm/.venv/bin/python smoke_dino_encoders.py`
- P0: tok_in=(2,384) tok_tgt=(2,1,384) pooled==tok_in readout_dim=384
- default: tok_in=(2,384) tok_tgt=(2,1,384) last_hidden_state=(2,203,384)
- n_tgt=3: tok_tgt=(2,3,384) prompt_tokens=(1,4,384)
- tie: torch.equal(tok_tgt, tok_in.unsqueeze(1)) == True
- cls: max|prompt_tokens - raw CLS| = 0.00e+00
- cls_noise: diff ~1.6e+00 (init_std=0.5)
- normal: prompt tokens differ from raw CLS
- n_dyn=0: prompt_tokens=(1,1,384) tok_in=(2,384)  [frozen, 0 trainable]
- n_dyn=4: prompt_tokens=(1,4,384)  (n_in=1, n_tgt=3)
Captured to /tmp/smoke_out.txt.

## Environment
- Canonical env: `/home/chr/le-wm/.venv` (workspace `.venv` is empty).
- torch 2.12.1+cu130, CUDA available.
- Checkpoint: /home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
- dinov3_vits16: img_size=224, patch16, embed_dim=384, n_storage_tokens=4,
  sequence=[CLS(1)+prompt(n_in+n_tgt)+storage(4)+patches(196)].
