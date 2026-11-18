# MultiScaleDial: integración y matriz E1 — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar el entorno MultiScaleDial al pipeline swm (entrenar desde cero + eval CEM pareada) y correr el experimento E1 (barrido de un dial, k=1..4) con reporte Δ(k) por modo de meta.

**Architecture:** El env vive como paquete propio `swm_msdial/` en este repo y se registra en gymnasium al importarlo (el swm instalado en el venv de entrenamiento es una copia no editable en site-packages y NO se toca). Los datasets H5 clonan el esquema de `tworoom.h5`; los sets de eval son H5 fijos (fila 0 = inicio, fila 25 = meta) compartidos por todos los modelos. El success viaja por `terminated` del env; el éxito por componente viaja por claves `ever_pos`/`ever_dial` del info que el propio env acumula.

**Tech Stack:** Python 3.10, gymnasium, h5py, numpy/scipy, hydra, stable_worldmodel 0.1.0, PyTorch.

**Spec:** `docs/superpowers/specs/2026-11-18-msdial-env-spec.md` (copia de `/home/chr/Documents/multiescala_env/multiscale-env-spec.md`; se copia al repo en la Task 1 y viaja con este plan). Implementación de referencia: `/home/chr/Documents/multiescala_env/msdial_env.py`.

## Global Constraints

- **Dos venvs, roles fijos:** tests con `./.venv_hf/bin/python -m pytest` (tiene pytest, gymnasium, scipy, h5py, torch); entrenamiento/eval/swm con `/home/chr/le-wm/.venv/bin/python` (swm 0.1.0 con `CEMSolver`). Ninguna tarea mezcla los roles.
- **No modificar site-packages** ni el clone `/home/chr/stable-worldmodel`: el registro del env se hace al importar `swm_msdial` desde nuestros scripts.
- **Tolerancias de éxito:** `eps_p = 0.05`, `eps_th = 0.3 rad` (spec §3).
- **Dial:** `r_k = 0.9·R·2^{-(k-1)}` con `R=0.12`, `pos_step=0.03`, `rot_step=0.25`, `sensor_noise=2/255`, supersampling 4×, estilo principal `balanced` (spec §1).
- **Protocolo eval:** frameskip 5, horizon 5, `eval.num_eval=200`, `goal_offset_steps=25`, `eval_budget=50`, CEM por defecto del repo (`num_samples=300, n_steps=30, topk=30`), seed de eval fija (spec §3).
- **Presupuesto entrenamiento:** `trainer.max_epochs=2` (aprobado); extenderlo es decisión explícita del humano en la Task 7, no del implementador.
- **Datasets en disco:** `~/.stable_worldmodel/datasets/msdial/*.h5` (no se commitean; los tests usan `/tmp/opencode`).
- **Nombres de runs:** `msdial_k<K>_<style>__<arm>__s<seed>` (= `output_model_name`), convención del spec §3 / EXPERIMENTOS.md.
- **La última fila de cada episodio del colector queda sin sucesor** (par (pixels_t, action_t) sin s_{t+1}): es intencional y válido para ventanas JEPA; no "corregir".

## Review Focus

1. **Goal no fijado** (`_set_goal_state` nunca llamado): el env debe seguir corriendo sin terminar jamás — test `test_step_terminates_only_with_goal` (Task 1).
2. **Wrap de θ en ±π**: meta θ=π−0.05 y estado θ=−π+0.05 están a 0.1 rad, no a π−0.1; la tolerancia debe usar `wrap()`, no resta directa — test `test_goal_boundary_wrap` (Task 1).
3. **Bordes de episodio en el loader**: `load_chunk(ep, 0, 26)` debe devolver exactamente 26 filas del episodio pedido, sin mezclar el siguiente — smoke `swm_msdial/smoke_loader.py` (Task 2, corre con venv le-wm).
4. **Dimensión de acción 2+K vs el encoder**: `train.py:120` resuelve `input_dim = frameskip·dim(action)` solo si el H5 tiene columna `action` con la dim correcta del nivel — smoke train de 3 batches (Task 5) es el test real.
5. **Sets de eval compartidos**: fila 0 y fila 25 de cada episodio de eval deben contener EXACTAMENTE el estado inicial y el goal (pixels y columna `state`); si el filler estático contamina esos extremos, todos los modelos ven otra meta — tests `test_eval_rows_*` (Task 4).

---

### Task 1: Paquete `swm_msdial` — env gym + registro + oráculo

**Files:**
- Copy: `/home/chr/Documents/multiescala_env/multiscale-env-spec.md` → `docs/superpowers/specs/2026-11-18-msdial-env-spec.md`
- Copy (base, se adapta): `/home/chr/Documents/multiescala_env/msdial_env.py` → `swm_msdial/env.py`
- Create: `swm_msdial/__init__.py`
- Test: `tests/test_msdial_env.py`

**Interfaces (produce, usados por Tasks 2–5):**
- `swm_msdial.register()` — registra `swm/MultiScaleDial-v0` (idempotente); se ejecuta al `import swm_msdial`.
- `MultiScaleDialEnv(levels=(k,), img_size=224, supersample=4, body_radius=0.12, pos_step=0.03, rot_step=0.25, sensor_noise=2/255, distractor_band=None, distractor_amp=0.08, distractor_dynamic=False, max_steps=200, dial_style="balanced", render_mode="rgb_array")` — constructor idéntico a la referencia.
- Nuevos vs. la referencia: `self.goal: np.ndarray | None`; `_set_state(state: np.ndarray) -> None` (clip pos, wrap θ); `_set_goal_state(goal_state: np.ndarray) -> None`; `step()` devuelve `terminated = goal is not None and ‖p−g_p‖<eps_p and max|wrap(θ−g_θ)|<eps_th`; `reset()` limpia `self._ever_pos/_ever_dial = False`; `info` de `step()` lleva: `state` ([p, sinθ, cosθ] float32), `ever_pos`, `ever_dial` (bool, OR acumulado del episodio), `success_pos`, `success_dial` (bool del paso), `dpos` (float), `dth_max` (float).
- `sample_goal(start, mode)` con `mode ∈ {"coarse","fine","both"}` y `scripted_policy(state, target, rng, noise=0.3, gain_p=8.0, gain_th=1.5)` — sin cambios vs. referencia.

- [ ] **Step 1: Copiar spec + referencia al repo (procedencia)**

```bash
cp /home/chr/Documents/multiescala_env/multiscale-env-spec.md docs/superpowers/specs/2026-11-18-msdial-env-spec.md
cp /home/chr/Documents/multiescala_env/msdial_env.py swm_msdial/env.py   # crear dir
```

Header de `swm_msdial/env.py`: nota de procedencia ("adaptado de la implementación de referencia, /home/chr/Documents/multiescala_env").

- [ ] **Step 2: Escribir los tests (fallan)**

`tests/test_msdial_env.py` — assertion clave de cada test:

```python
def test_register_and_make():          # import swm_msdial; gym.make("swm/MultiScaleDial-v0", levels=(2,), max_episode_steps=100) funciona
def test_set_state_roundtrip():        # get_state(set_state(s)) == s (tras clip/wrap) para s con θ fuera de [-π,π]
def test_step_terminates_only_with_goal():  # sin goal: 30 steps, terminated siempre False; con goal en tolerancia: terminated True en ≤50 steps con scripted_policy
def test_goal_boundary_wrap():         # goal θ=π−0.05, state θ=−π+0.05 → success()["dial"] True (dth≈0.1 < 0.3)
def test_render_shape_dtype():         # render() → (224,224,3) uint8
def test_sample_goal_modes():          # coarse: ‖Δp‖>0.3 y θ igual; fine: p igual y |Δθ|∈[π/2,π]; both: ambos cambian
def test_ever_flags_reset():           # tras alcanzar goal (ever_*=True), reset() → ever_pos/ever_dial False
def test_oracle_reachability():        # 20 eps × {coarse,fine,both}, 50 steps con scripted_policy sin ruido → success all == 1.0
```

- [ ] **Step 3: Correr y verificar que fallan**

Run: `./.venv_hf/bin/python -m pytest tests/test_msdial_env.py -q`
Expected: FAIL (ModuleNotFoundError / AttributeError `_set_state`).

- [ ] **Step 4: Implementar en `swm_msdial/env.py` + `swm_msdial/__init__.py`**

`__init__.py`: `from .env import MultiScaleDialEnv, scripted_policy, register` y `register()` al importar. Adaptaciones sobre la referencia según el bloque Interfaces (goal, `_set_*`, `terminated`, flags `ever_*` en info). El cálculo de `terminated` usa `wrap()` del módulo; `sample_goal` sin cambios.

- [ ] **Step 5: Correr tests → PASS; commit**

Run: `./.venv_hf/bin/python -m pytest tests/test_msdial_env.py -q` → PASS (8 tests).

```bash
git add swm_msdial/ tests/test_msdial_env.py docs/superpowers/specs/2026-11-18-msdial-env-spec.md
git commit -m "feat(msdial): env gym MultiScaleDial-v0 con goal, terminated y ever_* + tests"
```

---

### Task 2: Colector H5 (train/val) + compatibilidad loader

**Files:**
- Create: `swm_msdial/collect.py`
- Create: `swm_msdial/smoke_loader.py` (verificación con venv le-wm, no pytest)
- Test: `tests/test_msdial_collect.py`

**Interfaces:**
- Consumes: `MultiScaleDialEnv`, `scripted_policy` (Task 1).
- Produce: `collect(levels, style, episodes, val_episodes, steps, seed, out_dir) -> (train_path, val_path)`; H5 con columnas EXACTAS (clon de `tworoom.h5`):
  `pixels` (N,224,224,3) uint8 · `action` (N,2+K) float32 en [-1,1] · `state` (N,2+K) float64 [p,θ] · `observation` (N,2+2K) float64 [p,sinθ,cosθ] · `proprio` (N,2+2K) float64 (=observation) · `ep_idx` int32 · `step_idx` int64 · `ep_len` (n_eps,) int32 · `ep_offset` (n_eps,) int64 · `id` int64 · `reward` (N,) float64 (ceros).
  Train y val en archivos separados con estados disjuntos (episodios de val usan seeds con offset `10_000_000 + seed`).
- CLI: `--levels 2 --style balanced --episodes 1800 --val-episodes 200 --steps 100 --seed 0 --out-dir ~/.stable_worldmodel/datasets/msdial`; nombres `msdial_train_k{K}_{style}.h5` / `msdial_val_k{K}_{style}.h5`.
- Política de recolección: `scripted_policy` con waypoints cada 25 pasos + 30% acciones uniformes (referencia `collect.py`).

- [ ] **Step 1: Tests (fallan)** — `tests/test_msdial_collect.py` sobre un H5 chico (2+2 eps, 10 steps, out en `/tmp/opencode`):

```python
def test_schema_exacta():      # columnas, dtypes y shapes del bloque Interfaces, fila a fila
def test_episodios_contiguos():  # ep_idx/step_idx consecutivos por episodio; ep_len/ep_offset consistentes
def test_acciones_rango():     # action ∈ [-1,1], dim == 2+K
def test_train_val_disjuntos():  # ningún vector state de val aparece en train (comparación por episodio: centroides distintos)
```

- [ ] **Step 2: Run → FAIL** (`./.venv_hf/bin/python -m pytest tests/test_msdial_collect.py -q`).

- [ ] **Step 3: Implementar `swm_msdial/collect.py`** — port de la referencia con writer h5py (esquema del bloque Interfaces). Los ids globales (`id`) son secuenciales; `ep_offset` = índice de la primera fila del episodio.

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Smoke de compatibilidad con el loader real (venv le-wm)**

Run: `/home/chr/le-wm/.venv/bin/python swm_msdial/smoke_loader.py --file /tmp/opencode/msdial_train_k2_balanced.h5`
Expected: `load_dataset` OK, `load_chunk(ep=0, 0, 26)` → 26 filas, `get_dim("action") == 3`, `get_row_data(0)["pixels"].shape == (224,224,3)`.

- [ ] **Step 6: Commit**

```bash
git add swm_msdial/collect.py swm_msdial/smoke_loader.py tests/test_msdial_collect.py
git commit -m "feat(msdial): colector H5 train/val (esquema tworoom) + smoke loader"
```

---

### Task 3: Auditoría espectral en el repo

**Files:**
- Create: `swm_msdial/audit_band_energy.py` (port de la referencia, importando `from swm_msdial.env import MultiScaleDialEnv`)
- Test: `tests/test_msdial_audit.py`

**Interfaces:**
- Produce: `audit(style: str, k: int, img_size: int = 224) -> np.ndarray` (6 fracciones por banda, orden B0..B5); CLI `--out reports/msdial_band_audit.md` escribe la tabla completa (k=1..4 × {wedge, balanced}).

- [ ] **Step 1: Test (falla)** — `test_audit_balanced_coarse_fraction()`: con `img_size=96`, k=3, balanced → `f[4]+f[5] < 0.02` (la fuga gruesa del dial fino es despreciable).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Port (adaptación mínima: import del env desde el paquete; bands() y SIG sin cambios).**

- [ ] **Step 4: Run test → PASS.**

- [ ] **Step 5: Correr la auditoría completa y verificar contra la spec §1.1**

Run: `./.venv_hf/bin/python swm_msdial/audit_band_energy.py --out reports/msdial_band_audit.md`
Expected: k=1 wedge `coarse(B4+B5) ≈ 0.339`, k=1 balanced `≈ 0.032` (±0.01) — idéntico a la tabla ya reproducida de la spec.

- [ ] **Step 6: Commit** (incluye `reports/msdial_band_audit.md`).

---

### Task 4: Sets de eval fijos + config de eval + `eval_reward` por componente + oráculo en el wrapper real

**Files:**
- Create: `swm_msdial/gen_eval_set.py`
- Create: `config/eval/msdial.yaml`
- Modify: `viz/eval_reward.py` (import de registro + reporte por componente + `results.jsonl`)
- Create: `swm_msdial/smoke_eval_integration.py` (venv le-wm)
- Test: `tests/test_msdial_eval_set.py`

**Interfaces:**
- Consumes: env Task 1; `swm.World`, `_apply_callables` de `stable_worldmodel.world.world` (solo en el smoke).
- Produce: H5 de eval `msdial_eval_k{K}_{style}_{mode}.h5` por (k, estilo, modo): 200 episodios de **exactamente 26 filas**; fila 0 = (pixels render(start), state=start), filas 1–24 = filler estático en start, fila 25 = (pixels render(goal), state=goal); columnas = esquema del colector (Task 2) + `mode` no (el modo va en el nombre de archivo). Meta en `~/.stable_worldmodel/datasets/msdial/eval_meta.json` (seeds, tolerancias, pares start/goal).
- `gen_eval_set(levels, style, episodes, seed, out_dir)`; CLI `--levels 2 --style balanced --episodes 200 --modes coarse fine both`.
- `viz/eval_reward.py` después de la Task: (a) `import swm_msdial` arriba del main; (b) guarda `results.jsonl` junto al txt de resultados con `{"successes": [...], "success_pos": [...], "success_dial": [...], "dpos": [...], "dth_max": [...], "config": {...}}` por episodio (leído de `world.infos` al final del evaluate: `ever_pos`, `ever_dial`, `dpos`, `dth_max`); (c) imprime `success_pos/success_dial` además de `success_rate`.

- [ ] **Step 1: Tests (fallan)** — `tests/test_msdial_eval_set.py` sobre un set chico (4 eps, `/tmp/opencode`):

```python
def test_eval_rows_count():        # cada episodio tiene exactamente 26 filas
def test_eval_rows_start_goal():   # fila 0: state == start; fila 25: state == goal (meta del json)
def test_eval_mode_semantics():   # fine: state[:2] fila25 == fila0 y |wrap(θ25−θ0)| ∈ [π/2, π]; coarse: Δp > 0.3 y θ igual
def test_eval_goal_pixels_render():  # pixels fila 25 se renderizó EN el goal: difiere de fila 0 en fine y en coarse
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implementar `gen_eval_set.py`** — usa `env.set_state` + `env.render` para start y goal; `sample_goal` con rng de seed fija; filler estático.

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: `config/eval/msdial.yaml`** (clon estructural de `tworoom.yaml`):

```yaml
defaults: [launcher: local, solver: cem, _self_]
world:
  env_name: swm/MultiScaleDial-v0
  num_envs: ${eval.num_eval}
  max_episode_steps: 100
  env_type: single
  ob_type: states
  multiview: False
  width: 224
  height: 224
  levels: [2]            # override por nivel: world.levels=[k]
  dial_style: balanced   # override wedge: world.dial_style=wedge
dataset:
  stats: ${eval.dataset_name}
  keys_to_cache: [action, state]
plan_config: {horizon: 5, receding_horizon: 5, action_block: 5}
policy: random
eval:
  num_eval: 200
  goal_offset_steps: 25
  eval_budget: 50
  img_size: 224
  dataset_name: msdial/msdial_eval_k2_balanced_fine.h5   # override por (k, style, mode)
  callables:
    - {method: _set_state, args: {state: {value: state}}}
    - {method: _set_goal_state, args: {goal_state: {value: goal_state}}}
output: {filename: msdial_results.txt}
```

(Verificar contra `swm.World.__init__`: `levels` y `dial_style` van por `**kwargs` → `gym.make`; si `env_type/ob_type/multiview/width/height` no son kwargs aceptados por `World` o `MultiScaleDialEnv`, eliminarlos — el smoke del Step 6 lo detecta.)

- [ ] **Step 6: Oráculo dentro del wrapper real (venv le-wm, spec §7.2)**

`swm_msdial/smoke_eval_integration.py`: genera un set chico (8 eps), construye `swm.World(env_name="swm/MultiScaleDial-v0", num_envs=8, image_shape=(224,224), max_episode_steps=100, levels=(2,))`, aplica los callables del config con `_apply_callables` (importándolo de `swm.world.world`), corre 50 steps con `scripted_policy(info["state"], goal)` y reporta `terminated/ever_pos/ever_dial`.
Run: `/home/chr/le-wm/.venv/bin/python swm_msdial/smoke_eval_integration.py`
Expected: ≥ 7/8 episodios con `terminated=True` y `ever_pos=ever_dial=True`.

- [ ] **Step 7: Modificar `viz/eval_reward.py`** (per Interfaces) + smoke con `policy=random` sobre el set chico (venv le-wm): corre sin error, imprime `success_pos/success_dial`, escribe `results.jsonl`.

- [ ] **Step 8: Commit**

```bash
git add swm_msdial/gen_eval_set.py swm_msdial/smoke_eval_integration.py config/eval/msdial.yaml viz/eval_reward.py tests/test_msdial_eval_set.py
git commit -m "feat(msdial): sets de eval fijos, config hydra y eval por componente (pos/dial/all)"
```

---

### Task 5: Configs de entrenamiento + smoke train + generador de matriz

**Files:**
- Create: `config/train/data/msdial.yaml`
- Create: `config/train/model/lewm_s8.yaml` (control de resolución: copia de `lewm.yaml` con `patch_size: 8`)
- Create: `config/train/model/lewm_rgbrep.yaml` (control de canales; ver Step 3)
- Create: `experiments/matrix_msdial.py`
- Test: `tests/test_matrix_msdial.py`

**Interfaces:**
- Produce: `matrix_msdial.emit(arms, datasets, seeds, stages) -> list[str]` (comandos completos train/eval); CLI `--emit run_E1.sh`.
- Brazos del track desde cero (spec §4): `lewm`, `lewm_s8` (resolución), `lewm_rgbrep` (canales), `starlet`, `swt`, `wave_vit`, `convnext_ms`.
- Datasets E1 para la matriz de brazos (spec §7.6: brazos en k=1..3): k∈{1,2,3} balanced + k=2 wedge (4 archivos train). k=4 es punto extremo opcional (spec §1.1: 96% de energía <1px), fuera de la matriz.
- `action_encoder.input_dim` NO se configura: `train.py:120` lo resuelve (`frameskip × dim(action)`).

- [ ] **Step 1: `config/train/data/msdial.yaml`** — clon de `tworoom.yaml` con `name: msdial/msdial_train_k2_balanced.h5` (default), `keys_to_load: [pixels, action, proprio]`, `keys_to_cache: [action, observation]`, `frameskip: 5`, `num_steps: ${eval:'${num_preds} + ${history_size}'}`.

- [ ] **Step 2: Control de resolución `lewm_s8.yaml`** — copia de `lewm.yaml` cambiando solo `encoder.patch_size: 8` (y mantener `${img_size}`).

- [ ] **Step 3: Control de canales `lewm_rgbrep.yaml`**

Portar desde la rama `exp/multiscale-diagnostics` (con `git show exp/multiscale-diagnostics:wavelet/frontend.py` y `git show exp/multiscale-diagnostics:src/encoders/multiscale.py` como fuente): el front-end `rgb_repeat` (J+1 copias de x/(J+1)) con fusión replace e init equivalent, montado sobre el mismo ViT del `lewm` base. Misma cantidad de canales que `starlet` con J=4 → el control aísla "más canales" de "estructura de escala". Config yaml apunta `_target_` al encoder portado (ruta elegida: `src/encoders/multiscale.py` + `wavelet/frontend.py`, mismas rutas que en la rama origen).

- [ ] **Step 4: Test (falla) → implementar `experiments/matrix_msdial.py` → PASS**

```python
def test_matrix_counts_and_names():
    cmds = matrix_msdial.emit(arms=ARMS, datasets=DATASETS_E1, seeds=[0,1,2], stages=["train","eval"])
    assert len([c for c in cmds if "train.py" in c]) == 7 * 4 * 3      # 84 train (k1-3 balanced + k2 wedge)
    assert all("output_model_name=msdial_k" in c and "__s" in c for c in cmds if "train.py" in c)
    assert all("viz/eval_reward.py" in c and "--config-name=msdial" in c for c in cmds if "eval_reward" in c)
```

Comando de train emitido (patrón): `/home/chr/le-wm/.venv/bin/python train.py data=msdial model=<arm> data.dataset.name=msdial/msdial_train_k<K>_<style>.h5 trainer.max_epochs=2 output_model_name=msdial_k<K>_<style>__<arm>__s<seed> seed=<seed>`.
Comando de eval emitido (patrón): `viz/eval_reward.py policy=<ckpt> --config-name=msdial eval.dataset_name=msdial/msdial_eval_k<K>_<style>_<mode>.h5 world.levels=[<K>] world.dial_style=<style>` (3 modos × cada run).

- [ ] **Step 5: Smoke train (venv le-wm)** — con el H5 chico de la Task 2 en `/tmp/opencode`:

Run: `/home/chr/le-wm/.venv/bin/python train.py data=msdial model=lewm data.dataset.name=/tmp/opencode/msdial_train_k2_balanced.h5 trainer.max_epochs=1 loader.batch_size=8 trainer.limit_batches=3 output_model_name=msdial_smoke_lewm` y luego lo mismo con `model=starlet`.
Expected: ambos corren 3 batches, loguean losses, guardan checkpoint; `input_dim` resuelto = 15.

- [ ] **Step 6: Commit**

```bash
git add config/train/data/msdial.yaml config/train/model/lewm_s8.yaml config/train/model/lewm_rgbrep.yaml \
        src/encoders/multiscale.py wavelet/frontend.py experiments/matrix_msdial.py tests/test_matrix_msdial.py
git commit -m "feat(msdial): configs train (data, s8, rgbrep) + generador de matriz E1"
```

---

### Task 6: Sondas lineales + estadística pareada

**Files:**
- Create: `swm_msdial/probe.py`
- Create: `swm_msdial/paired_stats.py`
- Test: `tests/test_msdial_probe.py`, `tests/test_msdial_paired_stats.py`

**Interfaces:**
- Produce: `probe(policy_path: str, data_name: str, targets: list[str], out: str) -> dict` — carga el modelo con `swm.wm.utils.load_pretrained`, codifica N episodios de val (split por episodio), ajusta ridge latente→targets (default targets = `observation` = [p, sinθ, cosθ]; ángulo vía sin/cos, error adicional en grados), reporta R² por target en episodios no vistos; CLI `--policy --data --out reports/...json`.
- Produce: `paired_stats(run_a_jsonl: str, run_b_jsonl: str, key: str, out: str) -> dict` — Δ pp con IC bootstrap por episodio, test de permutación de signo con corrección Holm, McNemar (b/c); opera sobre los `results.jsonl` de la Task 4.
- Fuente de adaptación (no copia ciega): `git show exp/multiscale-diagnostics:diagnostics/linear_probe.py` y `paired_stats.py` — cambia el acceso al loader (API `swm.data.load_dataset` de este venv) y los targets.

- [ ] **Step 1: Tests (fallan)**

```python
def test_probe_recovers_linear_map():   # latentes sintéticos z con y = W z + ruido leve → R² > 0.95
def test_paired_stats_known_counts():   # éxitos sintéticos con b=8, c=2 → Δpp=12, p(McNemar) < 0.05, dirección correcta
```

- [ ] **Step 2: Run → FAIL. Implementar (Step 3) → PASS (Step 4).**

- [ ] **Step 5: Commit**

```bash
git add swm_msdial/probe.py swm_msdial/paired_stats.py tests/test_msdial_probe.py tests/test_msdial_paired_stats.py
git commit -m "feat(msdial): sondas ridge por episodio + estadística pareada (McNemar, signo+Holm)"
```

---

### Task 7: Piloto k=1 balanced — gate de sanidad antes de la matriz

**Files:**
- Create: `reports/msdial_pilot_k1.md` (salida del gate; sin código nuevo)

**Interfaces:**
- Consumes: Tasks 2 (dataset real), 4 (eval), 6 (sondas). Corre todo con venv le-wm.

- [ ] **Step 1: Generar el dataset real k=1 balanced** — `collect.py --levels 1 --style balanced --episodes 1800 --val-episodes 200` (~15 min CPU, 2000 eps × 100 steps).

- [ ] **Step 2: Generar los sets de eval k=1 balanced (3 modos × 200)** + `eval_meta.json`.

- [ ] **Step 3: Entrenar el baseline** — `train.py data=msdial model=lewm trainer.max_epochs=2 output_model_name=msdial_k1_balanced__lewm__s0 seed=0` (GPU, horas; background). Este run entra a la matriz E1 (mismo nombre, no se repite).

- [ ] **Step 4: Evaluar + sondas del piloto** — eval CEM en 3 modos (`policy=<ckpt>`; para el gate alcanza `eval.num_eval=50`) + `probe.py` sobre val + piso por azar: eval con `policy=random` (ya soportado) fija el piso del dial (~9.5%).

- [ ] **Step 5: Gate (spec §7.5) — escribir `reports/msdial_pilot_k1.md` y PREGUNTAR al humano:**
  - `coarse` cerca del oráculo (success_pos ≥ 90%): si no, el problema es el pipeline, no la escala → parar y diagnosticar.
  - `fine` por encima del piso por azar del dial: señal de que el modelo lee θ.
  - R²(θ) del probe > 0.5.
  - Si el piloto sale "subentrenado" (coarse lejos del oráculo): decisión del humano subir `max_epochs` antes de la matriz; el default aprobado sigue siendo 2.

- [ ] **Step 6: Commit del reporte**

```bash
git add reports/msdial_pilot_k1.md
git commit -m "report(msdial): piloto k=1 baseline lewm — gate de sanidad pre-matriz"
```

---

### Task 8: Matriz E1 completa + reporte Δ(k)

**Files:**
- Create: `reports/msdial_E1.md` (con la tabla Δ(k) y veredictos H1–H5)

**Interfaces:**
- Consumes: todo lo anterior. Corre con venv le-wm, GPU.

- [ ] **Step 1: Datos y sets de eval restantes + matriz**

Generar datasets train/val y sets de eval para k∈{2,3} balanced y k=2 wedge (k=1 ya existe del piloto; ~45 min CPU total). Luego `matrix_msdial.py --emit run_E1.sh` y lanzar: **84 train** (7 brazos × {k1,k2,k3 balanced, k2 wedge} × 3 seeds, `max_epochs=2`; el run del piloto `msdial_k1_balanced__lewm__s0` ya existe — el script lo salta si el checkpoint está). Orden: por dataset, seeds anidad; cada train seguido de sus 3 evals (200 eps, modos coarse/fine/both).
- [ ] **Step 2: Sondas por run** — `probe.py` sobre cada checkpoint (targets [p, sinθ, cosθ]).
- [ ] **Step 3: Estadística pareada + piso por azar** — `paired_stats.py` lewm vs cada brazo multiescala por (k, modo) sobre los `results.jsonl`; más una eval `policy=random` por (k, modo) como control negativo/piso (spec §3).
- [ ] **Step 4: Reporte `reports/msdial_E1.md`** — tabla Δ(k) por modo con IC, veredicto H1–H5 del spec §5, sondas R²(θ_k), y la lectura de resultados del spec (negativo publicable / "resolución barata" / solo-wedge). Incluir gráfico `reports/msdial_E1_delta.png` (Δ(k) por modo con IC, matplotlib).
- [ ] **Step 5: Commit del reporte**

```bash
git add reports/msdial_E1.md
git commit -m "report(msdial): matriz E1 completa — Δ(k) por modo, veredicto H1-H5"
```

- [ ] **Step 6 (gate, NO automático): E2/E3** — solo si E1 es informativo (spec §7.8); nueva conversación de diseño, no parte de este plan.
