# MultiScaleDial — entorno controlado para probar información multiescala en LeWM

*Spec v0 — nov. 2026. Implementación de referencia: `msdial_env.py`, `collect.py`, `oracle_check.py`, `audit_band_energy.py`.*

## 0. Por qué hace falta un entorno nuevo

Los benchmarks actuales no pueden responder "¿lo multiescala supera la línea base?":

- **TwoRoom está saturado** (97–100 %): no hay margen para que se vea el efecto.
- **PushT y Cube: las diferencias están dentro del ruido.** Con n≈50 episodios, el IC95 de una tasa ~93 % es ≈ ±7 pp; con n=200, ≈ ±3.5 pp. `original_starlet_pusht` 94.7 vs `dinov3_s_lora_pusht` 92.7, o `starlet_L3_cube` 74.0 vs `dinov3_s_lora_cube` 74.0, son empates estadísticos.
- **La información relevante para la tarea es gruesa en los tres**: posición del agente o bloque, de decenas de px. Un front-end wavelet no tiene nada exclusivo que aportar ahí.
- **Starlet sobre DINOv3 empeora** (Cube 62.7 vs 74.0). Lo más probable es que sea un desajuste de distribución en el patch embedding preentrenado más LoRA, no evidencia en contra de lo multiescala. Hay que separar el track preentrenado del track desde cero.

**Principio de diseño:** la escala de la información relevante tiene que ser la **variable independiente**. Así se obtiene una curva dosis-respuesta Δ(k) = éxito_multiescala(k) − éxito_base(k) en vez de un punto aislado. Entre variantes cambia una sola cosa: la escala visual del estado fino. La dinámica, la dimensión de acción, el cuerpo y el presupuesto de planificación se mantienen idénticos.

## 1. El entorno

**Estado:** `p ∈ [R, 1−R]²` (posición del cuerpo, escala gruesa) y `θ_k ∈ S¹` para cada nivel activo k.
**Acción:** `a ∈ [−1,1]^(2+K)` = `(dx, dy, dθ_1..dθ_K)`, con `p += 0.03·a_xy` y `θ += 0.25·a_θ` por paso de entorno.
**Observación:** 224×224 en gris replicado a RGB, para evitar atajos de color. Supersampling 4× con box-filter, para que el aliasing no cree moiré de baja frecuencia (eso sería una fuga de lo fino a lo grueso). Ruido de sensor σ = 2/255.

**Dial de nivel k:** anillo `[r_k/2, r_k]` con `r_k = 0.9·R·2^−(k−1)` y una cuña de 60° que marca θ_k. Los niveles están anidados y no se solapan, con la misma estructura de octavas que J en Starlet/SWT.

| k | r_ext (px) | relación con un patch de 16 px |
|---|---|---|
| 1 | 24.2 | ocupa ~3×3 patches |
| 2 | 12.1 | ~1.5×1.5 patches |
| 3 | 6.0 | dentro de 1 patch |
| 4 | 3.0 | sub-patch (caso extremo) |

### 1.1 Auditoría espectral (ya ejecutada)

Fracción de la energía dependiente de θ_k que cae en bandas más gruesas que medio patch (σ > 8 px), medida como Σ_px Var_θ[B_j(I_θ)]:

| k | `wedge` (cuña oscura) | `balanced` (dipolo nulo) |
|---|---|---|
| 1 | 0.339 | 0.032 |
| 2 | 0.071 | 0.001 |
| 3 | 0.007 | 0.000 |
| 4 | 0.0005 | 0.000 |

- La cuña oscura simple tiene **momento dipolar**: desplaza el centroide de brillo en la dirección θ, lo que le da una huella de baja frecuencia proporcional al área. El estilo `balanced` parte la cuña radialmente (mitad interior +δ, exterior −δ, con corte en r_m = r_ext·(9/16)^{1/3}) para que ∫r²f(r)dr = 0. Así la fuga a escala gruesa baja en un factor O((r/σ)²).
- **Usar `balanced` como condición principal** y `wedge` como variante con fuga "natural". El contraste entre ambas es en sí un experimento: si el baseline mejora mucho con `wedge`, está leyendo θ desde la huella gruesa.
- `balanced` desplaza toda la energía ~1 octava hacia lo fino (k=4: 96 % en <1 px). En la práctica k=4 puede quedar fuera del alcance de cualquier encoder.

**Lección:** en imágenes sintéticas sin ruido, una prueba lineal de "presencia" no sirve. Un ridge sobre el crop recupera θ con R² ≈ 1 incluso tras un blur de σ = 8 px, porque el filtro gaussiano es invertible sin ruido. La métrica correcta es la **fracción de energía por banda** (o la SNR), no la decodificabilidad.

## 2. Variantes (en orden)

1. **E1 — barrido de un dial** `levels=(k,)`, k ∈ {1,2,3,4}. La acción es 3D en todas las variantes. Es el experimento principal: la curva Δ(k).
2. **E2 — diales simultáneos** `levels=(1,2,3)`, acción 5D. Prueba la afirmación fuerte, "hace falta información de varias escalas a la vez", donde una representación de una sola escala tiene que priorizar. Como la acción tiene más dimensiones, el CEM es más difícil; por eso es obligatoria la cota de oráculo (§3).
3. **E3 — distractores pasabanda** `distractor_band=b` (DoG sobre ruido blanco), estáticos o dinámicos (`distractor_dynamic=True`, fondo con random walk que es impredecible). Barrer b respecto de la banda del dial: ¿la separación por bandas ayuda a ignorar distractores en una banda ajena y perjudica cuando el distractor cae en la misma banda del dial?
4. **(Opcional) E4 — escala temporal** (Punto 5 de la agenda): diales con distinta constante de tiempo o inercia. Mantenerlo separado de E1–E3; no mezclar ejes.

**Sin paredes ni obstáculos** en E1–E3. El costo terminal ‖ẑ_H − z_g‖² ya tiene una falla conocida con desvíos no monótonos, y mezclarla confundiría representación con planificación.

## 3. Protocolo de evaluación

- **Diseño 2×2 de metas** (`sample_goal(start, mode)`):
  - `coarse`: cambia p (‖Δp‖ > 0.3) y θ se mantiene.
  - `fine`: p se mantiene y |Δθ| ∈ [π/2, π].
  - `both`: cambian ambos.
  - **Predicción de especificidad:** la ganancia multiescala debe aparecer en `fine`/`both` y no en `coarse`. Si aparece también en `coarse`, es capacidad o regularización, no escala.
- **Éxito por componente:** `success()` devuelve `pos`, `dial`, `all` y distancias continuas (‖Δp‖, |Δθ|), con ε_p = 0.05 y ε_θ = 0.3 rad. La probabilidad de acertar el dial al azar es ≈ 9.5 %. Reportar también las distancias continuas, que tienen más potencia estadística que la tasa binaria.
- **Pareado:** el mismo conjunto fijo de (inicio, meta) para todos los modelos, generado con semillas y guardado en disco. Fijar también la semilla del CEM. Prueba de McNemar sobre éxito por episodio y bootstrap pareado para la diferencia.
- **Tamaño:** ≥200 episodios por (modelo, k, modo) y ≥3 semillas de entrenamiento. Reportar media ± IC entre semillas. Se acabaron los "n=1".
- **Presupuesto:** el mismo que TwoRoom (frameskip 5, mismo horizonte y mismas iteraciones del CEM). `oracle_check.py` confirma que con estado verdadero se llega al 100 % en los tres modos con 50 pasos de entorno, así que las metas son alcanzables.
- **Cotas:**
  - **Oráculo de estado:** el mismo predictor y el mismo CEM, pero con z = estado verdadero [p, sin θ, cos θ]. Separa el límite del planificador del límite de la representación. Imprescindible en E2.
  - **Control negativo:** encoder sin entrenar (como `dinov3_untrained`).
- **Sondas lineales** (como en LeWM, Tab. 3) desde z hacia p y hacia (sin θ_k, cos θ_k), con R² por nivel. Es la evidencia mecanística independiente del CEM: si el multiescala gana en éxito, debería ganar primero en R²(θ_k).

## 4. Modelos y controles

**Track principal, desde cero** (el entorno es OOD para DINOv3, y eso es deseable):

| brazo | qué aísla |
|---|---|
| ViT-S/16 desde cero (línea base) | referencia |
| ViT-S/16 + Starlet J=3 | efecto del front-end multiescala |
| WaveViT / SWT | familia de transformada |
| ConvNeXt-T desde cero | multiescala jerárquica nativa |
| **Control de canales:** ViT-S/16 con un front-end de la misma cantidad de canales pero sin multiescala (p. ej. conv 1×1 fija aleatoria, o RGB replicado) | descarta "más canales o parámetros" |
| **Control de resolución:** ViT-S/8 (o entrada de 448 con /16) | la alternativa barata a lo multiescala |

**Track preentrenado** (separado): DINOv3-S LoRA vs DINOv3-S LoRA + Starlet. Se interpreta aparte.

**Reglas:** los mismos hiperparámetros en todos los brazos (lr, batch, `history_size`, épocas, latente de 192, mismo predictor y projector), con params y FLOPs reportados. Antes de empezar, corregir o al menos fijar las desviaciones conocidas respecto del paper (lr uniforme también para el predictor, batch chico que afecta la ECF de SIGReg, `history_size: 3`).

## 5. Hipótesis (pre-registrar antes de correr)

- **H1:** en `coarse`, todos los brazos empatan, para todo k (sanity check).
- **H2:** en `fine`/`both`, el éxito de la línea base cae al crecer k; el del multiescala cae más lento. Δ(k) es creciente para k=1..3.
- **H3:** R²(θ_k) de la sonda sigue el mismo patrón que el éxito.
- **H4:** el control de canales no reproduce la ganancia.
- **H5 (E2):** con varios diales, la ventaja multiescala es mayor que la suma de las ventajas por nivel de E1, lo que indicaría competencia por capacidad en el latente.

**Lectura de resultados:**
- Δ(k) ≈ 0 con IC estrechos: el front-end no aporta información que la línea base no pueda extraer. Es un resultado negativo publicable.
- Δ(k) > 0 pero ViT-S/8 lo iguala: lo multiescala es "resolución barata". Sigue siendo una afirmación válida, pero distinta.
- La ganancia solo aparece con `wedge`: el baseline usa la fuga gruesa y el multiescala lee la señal fina directamente. Es interesante, pero debe reportarse así.

## 6. Integración con stable_worldmodel / Multiscale-World-Models

1. Copiar `msdial_env.py` junto al entorno TwoRoom y registrarlo (`gym.register("swm/MultiScaleDial-v0", ...)`) siguiendo el mismo wrapper.
2. **`set_state(state)`** existe para renderizar la imagen meta, que es como la eval de LeWM obtiene z_g. Verificar que el wrapper de evaluación lo invoque igual que en TwoRoom.
3. **Datos:** `collect.py` mezcla un controlador proporcional ruidoso hacia waypoints aleatorios (en p y θ) con un 30 % de acciones uniformes, para cubrir el espacio de acciones. Corre a ~340 fps en un proceso de CPU: 2000 ep × 100 pasos ≈ 10 min por nivel (paralelizar por nivel). Adaptar el writer al formato del loader de TwoRoom (`pixels`, `action`, `state`, `episode_idx`).
4. **Normalización de acciones:** usar las estadísticas del dataset nuevo. Ojo con el bug de offset ya visto en el diagnóstico de rollout.
5. Mantener frameskip 5. Con `rot_step=0.25`, un giro de π son ~13 pasos de entorno (~3 pasos de planificación); con `pos_step=0.03`, cruzar la arena son ~36 pasos.

## 7. Checklist de ejecución

1. [ ] Correr `audit_band_energy.py` y guardar la tabla (§1.1) en el repo como parte del entorno.
2. [ ] Integrar el entorno y comprobar con `oracle_check.py` dentro del wrapper real.
3. [ ] Generar los datasets E1 (k=1..4, `balanced`), más k=2 con `wedge`.
4. [ ] Generar y guardar en disco los conjuntos de eval fijos (3 modos × ≥200 episodios × k).
5. [ ] Entrenar la línea base ViT-S/16 desde cero en k=1 y verificar que `coarse` y `fine` estén cerca del oráculo (si no, el problema es el pipeline y no la escala).
6. [ ] Entrenar todos los brazos del track principal en k=1..3, con 3 semillas.
7. [ ] Correr sondas lineales y eval CEM pareada, y graficar Δ(k) por modo con IC.
8. [ ] Solo si E1 es informativo: E2, luego E3, luego el track preentrenado.
