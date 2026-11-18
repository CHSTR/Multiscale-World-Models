# TODO MultiScaleDial — estado en vivo

Actualizado: 2026-11-18 ~16:12 (hora local)

## Hecho

- [x] Spec MultiScaleDial adoptada + plan de integración (`docs/superpowers/specs/2026-11-18-msdial-env-spec.md`, `docs/superpowers/plans/2026-11-18-msdial-integration.md`)
- [x] **T1** Env gym `swm/MultiScaleDial-v0` (goal/terminated/ever_*) — 8 tests
- [x] **T2** Colector H5 train/val (esquema tworoom, blosc) + smoke con loader real
- [x] **T3** Auditoría espectral en repo — reproduce spec §1.1 (0.339/0.032)
- [x] **T4** Sets de eval fijos + `config/eval/msdial.yaml` + eval por componente (pos/dial/all) + `results.jsonl`; oráculo en wrapper real 8/8; `policy=random` → dial 14.3% ≈ piso
- [x] **T5** Configs train (data, `lewm_s8`, `lewm_rgbrep` init equivalente) + generador de matriz con grupos (14 brazos: 7 desde cero + 7 preentrenados)
- [x] **T6** Sondas ridge por episodio + estadística pareada (McNemar exacto, Holm)
- [x] Sets de eval k=1 (3 modos × 200 eps)
- [x] Disco resuelto: `datasets/msdial` movido a `/run/media/chr/phd/Multiescale_bkup/msdial` (1.3 TB libres) vía symlink
- [x] Suite: **85 tests verdes**; rama `msdial` con commits T1–T6 + extensión de matriz

## En curso (background)

- [ ] **T7.1** Recolección dataset k=1: `msdial_train_k1_balanced.h5` (4.7 / ~13.3 GB, ~36%) → ETA ~17:20
- [ ] **T7.1b** Recolección datasets **k=2 y k=3** + sus sets de eval (paralelo, CPU) → ETA ~17:50
- [ ] **T7.2** Cadena piloto (`scripts/run_pilot_k1.sh`, PID 82153): espera k=1 → mueve al externo + symlink → entrena `lewm` k=1 (2 ep, ~15 min) → eval CEM 3 modos (50 eps) → sonda R²
- [ ] **T7.3** Gate de sanidad: `coarse` cerca del oráculo (≥90% pos), `fine` > piso por azar (~9.5%), R²(θ) > 0.5 → decide si se lanza el barrido de 12 h

## Pendiente (barrido 12 h)

- [ ] **Fase A** k=2 × 12 brazos × seed 0 (~5.5 h): lewm, lewm_s8, lewm_rgbrep, starlet, swt, wave_vit, convnext_ms, dinov2_lora, dinov2_starlet_lora, dinov3_lora, dinov3_starlet_lora, dinov3_convnext_tiny
- [ ] **Fase B** k=1 y k=3 × {lewm, starlet, convnext_ms} × seed 0 (~2 h) → curva Δ(k) de 3 puntos
- [ ] **Fase C** Evals 2 modos (coarse/fine) × 30 eps por run (~3 h; se ajusta con el tiempo real del piloto)
- [ ] Reporte agregado: tabla por familia, Δ(k), sondas R²(θ)
- [ ] Review final del branch + cierre de rama

## Notas / riesgos

- Presupuesto: 12 h desde ~17:30 GPU → cierre ~05:30. El barrido arranca sólo si el gate del piloto es sano.
- 1 seed ⇒ conclusiones **direccionales**, sin IC entre seeds (los sets de eval sí son pareados).
- Fuera del barrido: wedge/robustez (E3), diales simultáneos (E2), k=4 extremo, ConvNeXt small/base, seeds múltiples.
- Cada run escribe checkpoint + `results.jsonl`: nada se pierde; se puede retomar.
