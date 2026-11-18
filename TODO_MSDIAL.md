# TODO MultiScaleDial — estado en vivo

Actualizado: 2026-11-18 ~16:40 (hora local)

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

## PAUSADO (19:00, a pedido del usuario)

Todo detenido: 0 procesos, GPU libre. El barrido se cortó tras el primer run
completo (`lewm` k=2). El `starlet` k=2 terminó y guardó checkpoint (sin eval).

Resultados disponibles:
- **Piloto k=1** (`lewm`, 2 ep): coarse pos **2%**, fine dial **30%**, both 0% · sonda R² **−0.79**.
- **k=2** (`lewm`, 2 ep, fine 30 eps): dial **100%**, both **53.3%** ← el dial se resuelve a 2 epochs en k=2.
- Evals k=1 ×3 modos (50 eps) + k=2 lewm fine (30 eps); `reports/probe_k1_lewm_s0.json`.
- Checkpoints: `msdial_k1_balanced__lewm__s0`, `msdial_k2_balanced__lewm__s0`, `msdial_k2_balanced__starlet__s0`.
- Datasets k1/k2/k3 + eval sets íntegros (locales, 50 GB) — listos para retomar.

Pregunta abierta al retomar: k=1 vs k=2 (¿dificultad real del anillo grande o run atípico?)
y presupuesto de epochs para el control de posición.

## En curso (background)

- [x] ~~k=1 primera recolección~~ → **matada por timeout de 1 h** (16:31). La cadena del piloto lanzó un `mv` en plena escritura de k2/k3: detenido a tiempo, copia parcial borrada, fuentes intactas.
- [ ] **T7.1** Recolección dataset k=1 (RELANZADA, PID 101481, timeout 3 h): ETA ~18:10–18:45
- [ ] **T7.1b** Recolección datasets **k=2 y k=3** + sets de eval (siguen vivas desde 16:05): ETA ~17:05
- [ ] **T7.2** Cadena piloto blindada (`scripts/run_pilot_k1.sh`, PID 101611): espera los **tres** collectors → **verifica integridad de los H5** → mueve al externo + symlink → entrena `lewm` k=1 (2 ep) → eval 3 modos (50 eps) → sonda R²
- [ ] **T7.3** Gate de sanidad: `coarse` cerca del oráculo (≥90% pos), `fine` > piso por azar (~9.5%), R²(θ) > 0.5 → decide si se lanza el barrido de 12 h

## Pendiente (barrido 12 h)

- [ ] **Fase A** k=2 × 12 brazos × seed 0 (~5.5 h): lewm, lewm_s8, lewm_rgbrep, starlet, swt, wave_vit, convnext_ms, dinov2_lora, dinov2_starlet_lora, dinov3_lora, dinov3_starlet_lora, dinov3_convnext_tiny
- [ ] **Fase B** k=1 y k=3 × {lewm, starlet, convnext_ms} × seed 0 (~2 h) → curva Δ(k) de 3 puntos
- [ ] **Fase C** Evals **fine × 30 eps** por run (~1.5 h; recortada para entrar en las 12 h por el retraso del incidente; se agrega coarse si el piloto mide eval rápido)
- [ ] Reporte agregado: tabla por familia, Δ(k), sondas R²(θ)
- [ ] Review final del branch + cierre de rama

## Notas / riesgos

- Presupuesto: 12 h desde ~17:30 GPU → cierre ~05:30. El barrido arranca sólo si el gate del piloto es sano.
- 1 seed ⇒ conclusiones **direccionales**, sin IC entre seeds (los sets de eval sí son pareados).
- Fuera del barrido: wedge/robustez (E3), diales simultáneos (E2), k=4 extremo, ConvNeXt small/base, seeds múltiples.
- Cada run escribe checkpoint + `results.jsonl`: nada se pierde; se puede retomar.
