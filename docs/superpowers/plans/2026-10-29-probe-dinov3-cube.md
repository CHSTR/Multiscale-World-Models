# Probe lineal y MLP sobre latentes de Cube — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar probes lineales y MLP sobre features congeladas de checkpoints de Cube y reportar MSE y Pearson r por cantidad física.

**Architecture:** Un script CLI (`probe_train.py`) en dos fases: extracción de features cacheada a `.npz` y ajuste de probes sobre las features cacheadas. Un segundo script (`probe_report.py`) agrega los CSV por run y genera la tabla markdown y el heatmap. Los tests usan un encoder stub y arrays sintéticos: nunca cargan el dataset real ni un checkpoint.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pandas, matplotlib, `stable_worldmodel` (loader), pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-probe-dinov3-cube-design.md`

## Global Constraints

- Split **por episodio**, 80/20, sin fuga entre train y val.
- Targets z-scoreados con estadísticas **solo de train**.
- Métricas por grupo: **MSE** (z-units y unidades originales) y **Pearson r** (media de correlaciones por dimensión).
- Features: **CLS** (384d) y **patch-mean** (384d, media de los 256 últimos tokens).
- Normalización de imagen idéntica a entrenamiento: `utils.get_img_preprocessor(source="pixels", target="pixels", img_size=224)`.
- Semilla por defecto: `0`. Sin wandb.

## Review Focus

- Targets constantes o casi constantes (p. ej. `gripper_contact`): Pearson indefinido → debe devolver `0.0`, no `nan` ni excepción.
- Dimensiones de feature con varianza ~0: la estandarización no debe producir `inf`.
- Filas con NaN en algún target: se descartan antes del split.
- Frontera de patch tokens: DINOv3 tiene CLS + storage tokens; el patch-mean debe usar los **últimos** `n_patches` tokens, no `tokens[:, 1:]`.
- Split con episodios de 1 frame: ningún split puede quedar vacío.

---

### Task 1: Dataset, muestreo y split por episodio

**Files:**
- Create: `probe_train.py`
- Create: `tests/test_probe.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `GROUP_SPECS: dict[str, list[str]]` — 8 grupos → columnas del HDF5.
  - `load_cube_dataset(cache_dir: str | None = None) -> dataset`
  - `sample_and_split(dataset, n_frames: int, val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe.py
import numpy as np
from probe_train import GROUP_SPECS, sample_and_split

class FakeDataset:
    def __init__(self, ep_idx):
        self.ep_idx = np.asarray(ep_idx)
    def __len__(self):
        return len(self.ep_idx)
    def get_col_data(self, name):
        assert name == "ep_idx"
        return self.ep_idx

def test_group_specs_cover_eight_groups():
    assert len(GROUP_SPECS) == 8
    assert GROUP_SPECS["joint_pos"] == ["proprio_joint_pos"]
    assert set(GROUP_SPECS["gripper"]) == {
        "proprio_gripper_opening", "proprio_gripper_contact", "proprio_gripper_vel"}

def test_split_has_no_episode_leakage_and_is_deterministic():
    ep = np.repeat(np.arange(20), 10)          # 20 episodios x 10 frames
    ds = FakeDataset(ep)
    tr1, va1 = sample_and_split(ds, n_frames=100, val_frac=0.2, seed=0)
    tr2, va2 = sample_and_split(ds, n_frames=100, val_frac=0.2, seed=0)
    assert tr1 == tr2 and va1 == va2
    assert set(ep[tr1]).isdisjoint(set(ep[va1]))
    assert len(tr1) + len(va1) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'probe_train'`

- [ ] **Step 3: Implement `probe_train.py` (constants + loader + split)**

Implementar `GROUP_SPECS` exactamente con estas columnas:

```python
GROUP_SPECS = {
    "joint_pos": ["proprio_joint_pos"],
    "joint_vel": ["proprio_joint_vel"],
    "effector_pos": ["proprio_effector_pos"],
    "effector_yaw": ["proprio_effector_yaw"],
    "gripper": ["proprio_gripper_opening", "proprio_gripper_contact", "proprio_gripper_vel"],
    "block_pos": ["privileged_block_0_pos"],
    "block_quat": ["privileged_block_0_quat"],
    "block_yaw": ["privileged_block_0_yaw"],
}
```

`load_cube_dataset` usa `swm.data.load_dataset("ogbench/cube_single_expert.h5", transform=None, cache_dir=..., keys_to_load=["pixels", *all_target_columns], num_steps=1, frameskip=1)` y luego `dataset.transform = get_img_preprocessor(...)`.

`sample_and_split`: `rng = np.random.default_rng(seed)`; elegir `n_frames` índices uniformes con `rng.choice(len(dataset), size=min(n_frames, len(dataset)), replace=False)`; episodios únicos → `rng.permutation`; primeros `int(round(val_frac * n_episodes))` son val; partir los índices muestreados según el episodio de cada fila.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add probe_train.py tests/test_probe.py
git commit -m "feat(probe): cube dataset sampling and episode-level split"
```

---

### Task 2: Extracción de features CLS / patch-mean con caché

**Files:**
- Modify: `probe_train.py`
- Modify: `tests/test_probe.py`

**Interfaces:**
- Consumes: `load_cube_dataset`, `sample_and_split`.
- Produces:
  - `extract_features(model, dataset, indices, source: str, batch_size: int = 256, target_columns: list[str] | None = None, transform=None) -> dict`
    con `features: np.ndarray (N, 384)`, `targets: dict[str, np.ndarray]`.
  - `parse_args(argv=None) -> argparse.Namespace`

- [ ] **Step 1: Write the failing test**

```python
import torch
from torch import nn
import types
from probe_train import extract_features

class StubEncoder(nn.Module):
    """Devuelve [CLS(3), storage(2), patches(4)] con valores conocidos."""
    def __init__(self, n_patches=4):
        super().__init__()
        self.n_patches = n_patches
    def forward(self, x, **kw):
        b = x.size(0)
        cls = torch.full((b, 1, 3), 10.0)
        storage = torch.zeros(b, 2, 3)
        patches = torch.arange(self.n_patches, dtype=torch.float32).view(1, -1, 1).expand(b, -1, 3)
        return types.SimpleNamespace(last_hidden_state=torch.cat([cls, storage, patches], dim=1))

class StubModel(nn.Module):
    def __init__(self, n_patches=4):
        super().__init__()
        self.encoder = StubEncoder(n_patches)

class TinyDataset:
    def __init__(self, n=6):
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return {"pixels": torch.zeros(1, 3, 224, 224),
                "proprio_joint_pos": torch.full((1, 6), float(i))}

def test_patch_mean_uses_last_tokens_not_tokens_after_cls():
    ds = TinyDataset()
    out = extract_features(
        StubModel(n_patches=4), ds, list(range(6)), source="patch_mean", batch_size=3,
        target_columns=["proprio_joint_pos"])
    # patches valen 0,1,2,3 -> media 1.5; si tomara storage+patches daria 0.6
    assert out["features"].shape == (6, 3)
    assert np.allclose(out["features"], 1.5)

def test_targets_are_concatenated_and_cls_is_used():
    ds = TinyDataset()
    out = extract_features(
        StubModel(), ds, [0, 1], source="cls", batch_size=2,
        target_columns=["proprio_joint_pos"])
    assert np.allclose(out["features"], 10.0)
    assert out["targets"]["proprio_joint_pos"].shape == (2, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_features'`

- [ ] **Step 3: Implement `extract_features`**

Diseño:
- Acepta opcionalmente `dataset` y `transform` para los tests (si no, usa los argumentos del CLI ya construidos).
- Itera con `torch.utils.data.Subset(dataset, indices)` y `DataLoader(batch_size, shuffle=False)`.
- Mueve `pixels` a GPU si hay CUDA; `model.encoder` en `eval()` y `torch.no_grad()`.
- `source == "cls"` → `tokens[:, 0]`.
- `source == "patch_mean"` → `n_patches = (img_size // patch_size) ** 2`; `tokens[:, -n_patches:].mean(1)`. **No** usar `tokens[:, 1:]`.
- Targets: para cada columna pedida, apilar `batch[col].reshape(b, -1).float()`; concatenar por grupo al guardar.
- Devuelve arrays float32 en CPU.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add probe_train.py tests/test_probe.py
git commit -m "feat(probe): CLS and patch-mean feature extraction"
```

---

### Task 3: Métricas y probe lineal (ridge)

**Files:**
- Modify: `probe_train.py`
- Modify: `tests/test_probe.py`

**Interfaces:**
- Consumes: features/targets de Task 2.
- Produces:
  - `mse_and_pearson(y_true, y_pred) -> tuple[float, float]`
  - `fit_ridge(X_tr, Y_tr, X_va, Y_va, alphas) -> tuple[np.ndarray, np.ndarray, float]`
  - `standardize(train, other) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `drop_nan_rows(targets: dict[str, np.ndarray]) -> np.ndarray` (máscara booleana)

- [ ] **Step 1: Write the failing test**

```python
from probe_train import mse_and_pearson, fit_ridge

def test_mse_and_pearson_perfect_and_random():
    y = np.arange(100.0).reshape(-1, 1)
    mse, r = mse_and_pearson(y, y)
    assert mse == pytest.approx(0.0)
    assert r == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    y_rand = rng.normal(size=(200, 1))
    _, r_rand = mse_and_pearson(y, y_rand)
    assert abs(r_rand) < 0.2

def test_mse_and_pearson_constant_target_returns_zero():
    y = np.ones((50, 1))
    mse, r = mse_and_pearson(y, np.zeros((50, 1)))
    assert np.isfinite(mse)
    assert r == 0.0

def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 16))
    W = rng.normal(size=(16, 3))
    Y = X @ W
    W_hat, b_hat, alpha = fit_ridge(X[:1500], Y[:1500], X[1500:], Y[1500:], [1e-4, 1e-2, 1.0])
    pred = X[1500:] @ W_hat + b_hat
    _, r = mse_and_pearson(Y[1500:], pred)
    assert r > 0.99

def test_standardize_zero_variance_dim_is_safe():
    X = np.ones((20, 4)); X[:, 1] = np.arange(20)
    Z, _, _ = standardize(X, X)
    assert np.isfinite(Z).all()

def test_drop_nan_rows_removes_rows_with_any_nan_target():
    Y = {"a": np.array([[1.0], [np.nan]]), "b": np.array([[0.0], [1.0]])}
    mask = drop_nan_rows(Y)
    assert mask.tolist() == [True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

- `mse_and_pearson`: MSE sobre todas las dims; para Pearson, `np.corrcoef` por columna; columnas con std < 1e-8 → 0.0; devolver media finita. Si no hay dims válidas → 0.0.
- `standardize(train, other)`: media/std de train; `std = np.where(std < 1e-8, 1.0, std)`.
- `fit_ridge`: estandarizar X e Y con train; para cada alpha resolver `(XᵀX + αI)⁻¹ XᵀY` en float64; predecir val; elegir alpha con mayor Pearson medio en val; devolver `W, b` en **espacio estandarizado**.
- `drop_nan_rows`: máscara `True` para filas sin NaN en ninguna columna de ningún grupo.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add probe_train.py tests/test_probe.py
git commit -m "feat(probe): MSE/Pearson metrics and ridge probe"
```

---

### Task 4: Probe MLP con early stopping

**Files:**
- Modify: `probe_train.py`
- Modify: `tests/test_probe.py`

**Interfaces:**
- Produces:
  - `fit_mlp(X_tr, Y_tr, X_va, Y_va, hidden: int = 256, epochs: int = 100, patience: int = 10, seed: int = 0) -> nn.Module`

- [ ] **Step 1: Write the failing test**

```python
import torch
from probe_train import fit_mlp

def test_mlp_beats_linear_on_nonlinear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8)).astype("float32")
    Y = np.stack([X[:, 0] * X[:, 1], np.tanh(X[:, 2])], axis=1)
    model = fit_mlp(X[:3000], Y[:3000], X[3000:], Y[3000:], epochs=30, patience=5)
    with torch.no_grad():
        pred = model(torch.from_numpy(X[3000:])).numpy()
    _, r = mse_and_pearson(Y[3000:], pred)
    assert r > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe.py::test_mlp_beats_linear_on_nonlinear_relation -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `fit_mlp`**

`nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, D_out))`; AdamW lr 1e-3, weight_decay 1e-4; batch 4096; hasta `epochs`; early stopping con paciencia sobre MSE de validación; restaurar el mejor `state_dict`; features y targets **ya estandarizados fuera** (la función recibe arrays y estandariza internamente con estadísticas de train, devolviendo el modelo sobre features estandarizadas).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add probe_train.py tests/test_probe.py
git commit -m "feat(probe): MLP probe with early stopping"
```

---

### Task 5: CLI, caché y CSV por run

**Files:**
- Modify: `probe_train.py`
- Modify: `tests/test_probe.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces:
  - `run_probe(ckpt, source, n_frames, out_dir, cache_dir, seed) -> list[dict]`
  - Filas con columnas: `checkpoint, source, probe, group, dim, mse_z, mse_raw, pearson`.

- [ ] **Step 1: Write the failing test**

```python
from probe_train import make_rows

def test_rows_have_expected_columns_and_group_dim():
    y = np.random.default_rng(0).normal(size=(50, 3))
    rows = make_rows("ckpt.pt", "cls", "ridge", y, y, {"block_pos": ["a", "b", "c"]})
    assert {r["group"] for r in rows} == {"block_pos"}
    assert rows[0]["dim"] == 3
    assert rows[0]["pearson"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe.py::test_rows_have_expected_columns_and_group_dim -v`
Expected: FAIL

- [ ] **Step 3: Implement `make_rows` + `run_probe` + `parse_args`**

- `make_rows(checkpoint, source, probe, y_true, y_pred, group_specs)` → una fila por grupo; `mse_z = MSE(y_true, y_pred)`, `mse_raw = mse_z * var_train_por_grupo` (la varianza de train se pasa como argumento opcional; si no, `mse_raw = mse_z`), `pearson` con `mse_and_pearson`.
- `run_probe`: cargar modelo (`swm.wm.utils.load_pretrained`, eval, requires_grad_(False)), dataset, sample/split, extraer features de train y val, cachear `outputs/probe/features_<ckpt_stem>_<source>.npz` (si existe, reutilizar), entrenar ridge y MLP por grupo, escribir `outputs/probe/<ckpt_stem>_<source>.csv`.
- CLI: `--ckpt` (repetible), `--source {cls,patch_mean}`, `--n-frames`, `--out-dir`, `--seed`, `--skip-extract`.

- [ ] **Step 4: Run tests + smoke real**

Run: `pytest tests/test_probe.py -v`
Expected: PASS (13 tests)

Run smoke (500 frames, 1 checkpoint):
```bash
/home/chr/Multiscale-World-Models/.venv_hf/bin/python probe_train.py \
  --ckpt /home/chr/Multiscale-World-Models/models_server/checkpoints/dinov3_s_lora_cube/weights_epoch_5.pt \
  --source cls --n-frames 500 --out-dir outputs/probe_smoke
```
Expected: crea `outputs/probe_smoke/weights_epoch_5_cls.csv` con 8 grupos × 2 probes = 16 filas.

- [ ] **Step 5: Commit**

```bash
git add probe_train.py tests/test_probe.py
git commit -m "feat(probe): CLI, feature cache and per-run CSV"
```

---

### Task 6: Reporte agregado (tabla + heatmap)

**Files:**
- Create: `probe_report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: CSVs de Task 5.
- Produces:
  - `aggregate(csv_paths: list[str]) -> pandas.DataFrame`
  - `write_markdown(df, path) -> None`
  - `write_heatmap(df, path) -> None`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from probe_report import aggregate

def test_aggregate_keeps_latest_run_per_key(tmp_path):
    df = pd.DataFrame([
        {"checkpoint": "a", "source": "cls", "probe": "ridge", "group": "g",
         "dim": 3, "mse_z": 0.1, "mse_raw": 0.1, "pearson": 0.9, "run": 0},
        {"checkpoint": "a", "source": "cls", "probe": "ridge", "group": "g",
         "dim": 3, "mse_z": 0.05, "mse_raw": 0.05, "pearson": 0.95, "run": 1},
    ])
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    out = aggregate([str(p)])
    assert len(out) == 1
    assert out.iloc[0]["pearson"] == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

- `aggregate`: leer CSVs, ordenar por `run`, `drop_duplicates(keep="last", subset=[checkpoint, source, probe, group])`.
- `write_markdown`: tabla `group × (checkpoint, source, probe)` con `MSE (z)` y `r`.
- `write_heatmap`: `pearson` con filas = grupos y columnas = `checkpoint/source/probe`; `matplotlib` con LaTeX (`text.usetex=True`) y `RdYlGn`, vmin=0, vmax=1; guardar en `assets/probe_heatmap.png`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add probe_report.py tests/test_report.py
git commit -m "feat(probe): aggregate report and heatmap"
```

---

## Verificación final (manual)

1. Extraer features de los 3 checkpoints (200k frames) — ~15–25 min:
```bash
python probe_train.py \
  --ckpt <dinov3 ep5> --ckpt <dinov3 ep10> --ckpt <starlet_L3 ep10> \
  --source cls --source patch_mean --n-frames 200000
```
(la CLI acepta `--source` repetible o se corre dos veces)
2. Generar CSV + tabla + heatmap:
```bash
python probe_report.py outputs/probe/*.csv --out assets/probe_results.csv
```
3. Sanity checks: con `--shuffle-targets`, Pearson ≈ 0; con `--identity-features`, Pearson ≈ 1.
