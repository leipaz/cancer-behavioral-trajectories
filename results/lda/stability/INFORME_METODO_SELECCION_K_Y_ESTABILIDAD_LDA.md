# Informe metodológico — Selección de K y estabilidad LDA

Documento para double-check: **qué se ha hecho, en qué orden, con qué datos y qué significa cada métrica**.

**Entrega canónica:** `results/lda/stability/`  
**Informe numérico + figuras:** `lda_stability_report.docx` (también `.md` / `.pdf`)  
**Embeddings (réplicas / barrido K):** `data/revision/user_embeddings_revision_a0.csv`  
**Referencia LDA:** `models/lda/LDA_model_vdec25.gensim` (**sin reentrenar**; corpus histórico **PMP lock 2025-09-11**)

---

## 0. Procedencia del corpus

| | Referencia `LDA_model_vdec25` | Este análisis (K + estabilidad + alarma) |
|--|--|--|
| Daily / fechas | **PMP lock 2025-09-11** | **DB1 auditado / fechas reconciliadas** |
| Documento | Secuencia de day-types bajo el lock | Secuencia de day-types del seguimiento DB1 reconciliado |
| ¿Se reentrena? | No (congelada) | Réplicas a K=6; ancla = referencia |

**Comparación de fechas (DB1 − PMP, n=175):** mediana Δ inicio / fin / n_días = **0** (`csv/comparacion_fechas_PMP_lock_vs_DB1_por_paciente.csv`). La reconciliación **no modifica** los pesos del modelo publicado.

---

## 1. Objetivo

1. **¿Cuántos temas (K)?** → coherencia no supervisada.  
2. **¿El LDA se reproduce a K=6?** → estabilidad a semilla y leave-20%-out.  
3. **¿La señal mensual / alarma se mantiene?** → réplicas alineadas sobre los **1019** landmarks collapsed de `base_analysis` (θ fijo).

---

## 2. Cadena de trabajo

```
Referencia congelada: PMP lock → VQ-VAE → LDA_model_vdec25
                                              ▲
                                              │ Hungarian / cosine (sin outcomes)
Daily DB1 reconciliado → embeddings ──────────┘
        │
        ├─ Barrido K (coherencia) → reconfirma K=6
        ├─ Estabilidad Level 1 (50 semillas) y Level 2 (50× leave-20%-out)
        ├─ Chequeo de passes (100 vs 1000 vs 5000, from-scratch)
        └─ Estabilidad mensual + alarma (1019 landmarks, θ fijo; reutiliza réplicas)
```

Scripts:

- `scripts/03_analysis/lda/run_lda_coherence_sweep.py`
- `scripts/03_analysis/lda/run_lda_stability.py`
- `scripts/03_analysis/lda/run_lda_monthly_alarm_stability.py` (`--reuse-models`)

---

## 3. Dataset y unidad de análisis

| Concepto | Valor |
|----------|--------|
| Pacientes (N) | **175** |
| Documento LDA | 1 paciente = secuencia completa de day-types (VQ-VAE `a0`) |
| Estabilidad de temas / pUF global | Sobre el documento completo |
| Estabilidad de alarma | Landmarks **mensuales collapsed** del informe de granularidad (`base_analysis`) |

---

## 4. Paso A — Selección de K (coherencia)

- Corpus BOW por paciente; split **70/30** por paciente (seed 42).  
- Candidatos **K ∈ {2, 6, 10, 14, 18}**; **5** inicializaciones; **50** passes.  
- Regla: maximizar coherencia media **c_v** de validación → **K = 6** (igual que el modelo publicado).  
- Salidas: `csv/coherence/coherence_coarse_grid.csv`, `coherence_selection_meta.json`, figura `figures/lda_coherence_k_selection.*`.

---

## 5. Paso B — Tema = P(day-type | tema)

Un tema es una distribución sobre day-types (suma 1). LDA no fija etiquetas entre runs: lo comparable es la **firma** de day-types, no el índice numérico del tema.

---

## 6. Paso C — Alineación Hungarian (sin outcomes)

Cada tema de la referencia se empareja con un tema de la réplica maximizando **cosine** de `P(day-type | tema)`. **No** se usan desenlaces clínicos.

---

## 7. Paso D — Hipertópicos F/U y pUF

| Hipertópico | Temas (tras alinear) |
|-------------|----------------------|
| Favourable (F) | T0–T2 |
| Unfavourable (U) | T3–T5 |

**pUF (documento):** \(P(T_3)+P(T_4)+P(T_5)\) sobre la secuencia completa del paciente (señal global).  
**pUF mensual / carga:** en el análisis de alarma, pUF por bloque de 30 días y \(S_m\) = suma de los últimos 3 bloques (= `AUC_W`).

---

## 8. Paso E — Estabilidad de temas (Level 1 / Level 2)

| Parámetro | Referencia publicada | Réplicas |
|-----------|----------------------|----------|
| **passes** | **100 000** | **100** |
| **iterations** | **50** | **50** |
| K | 6 | 6 |
| n réplicas | — | 50 + 50 |

### 8.1 Level 1 — Semilla

50 reentrenos con toda la cohorte (n=175); cosine por tema + F/U + Pearson de pUF global.

### 8.2 Level 2 — Leave-20%-out

50 máscaras (~140 in / ~35 out). Cosine de temas y pUF global sobre pacientes del ajuste (diseño del informe de estabilidad de temas).

### 8.3 Chequeo de passes (¿100 es suficiente?)

Diseño fijo a priori (5 semillas + 5 leave-20%-out): entrenar **from-scratch** a **100 / 1000 / 5000** passes.

- ELBO (bound) from-scratch: cambio relativo ~**0.05%**.  
- Temas / pUF entre budgets: mediana cosine / Pearson **> 0.99**.  
- Se **mantienen 100 passes** por coste de la batería 50+50.  
- La curva continua pass-a-pass **no** se usa para elegir el corte (ver `csv/convergence_passes/`).

---

## 9. Paso F — Estabilidad mensual y de alarma

Anclado a `results/alarm/base_analysis/granularity_report.docx` y  
`decision_points_collapsed.csv`.

| Ítem | Valor |
|------|------:|
| Landmarks evaluables | **1019** (159 pos / 860 neg) |
| Reloj | Tiempo_obs digital / t_evento_eb2 |
| W / H | 3 meses / 120 días |
| Corte (suma de 3 pUF) | **θ = 1.0596384140662845** (no se reoptimiza por réplica) |
| AUROC / sens / spec de referencia | **0.701 / 0.881 / 0.450** |

Método: mismas réplicas alineadas (VQ-VAE y diccionario congelados; Hungarian; F/U como arriba). Por réplica vs referencia:

1. Correlación de pUF mensual y de carga de 3 bloques.  
2. Fracción de decisiones de alarma que cambian (0/1) a θ fijo.  
3. AUROC, sensibilidad y especificidad en los mismos landmarks.

**Leave-20%-out:** sí se hace **inferencia** sobre los ~35 excluidos; los resultados se presentan **por separado** para landmarks de pacientes **in-fit (140)** y **held-out (35)**. Es estabilidad **dentro de DB1**, no validación DB2/DB3.

Salidas: `csv/monthly_alarm/`; figura `figures/lda_stability_monthly_alarm.*`; detalle numérico en **§5** de `lda_stability_report.docx`.

---

## 10. Métricas (lectura rápida)

| Métrica | Pregunta |
|---------|----------|
| Cosine por tema / F/U | ¿Misma firma de day-types? |
| Pearson pUF global | ¿Misma ordenación de pacientes en U? |
| Pearson pUF mensual / load | ¿Misma señal mes a mes / carga de 3 bloques? |
| Decision-flip fraction | ¿Cuántas alarmas 0/1 cambian? |
| AUROC / Sens / Spec (θ fijo) | ¿Se mantiene el punto de operación? |

---

## 11. Headline numérico (entrega actual)

### Temas — Level 1 / Level 2

| | Level 1 (seed) | Level 2 (leave-20%) |
|--|--:|--:|
| Median cosine (pares tema–réplica) | **0.915** | **0.867** |
| Unfavourable hypertopic cosine (median) | **0.916** | **0.885** |
| pUF global Pearson (median [IQR]) | **0.849 [0.795, 0.890]** | **0.772 [0.675, 0.858]** |

### Alarma — mediana [IQR] (50 réplicas; sin fila “all 1019” leave-out)

| Panel | Load r | Flip fraction | AUROC |
|-------|-------:|--------------:|------:|
| Seed (all 1019) | **0.832 [0.777, 0.870]** | **0.155 [0.130, 0.174]** | **0.679 [0.660, 0.689]** |
| Leave-20% in-fit | **0.755 [0.650, 0.810]** | **0.210 [0.165, 0.254]** | **0.658 [0.635, 0.686]** |
| Leave-20% held-out | **0.755 [0.633, 0.851]** | **0.194 [0.151, 0.255]** | **0.665 [0.605, 0.739]** |

---

## 12. Archivos de esta carpeta

| Ruta | Contenido |
|------|-----------|
| `lda_stability_report.docx` (+ md/pdf) | Informe numérico + figuras |
| `csv/coherence/` | Barrido K |
| `csv/level1_seed/`, `csv/level2_resample/` | Estabilidad de temas |
| `csv/convergence_passes/` | Chequeo 100/1000/5000 passes |
| `csv/monthly_alarm/` | Estabilidad mensual + alarma (1019) |
| `figures/` | Figuras embebidas en el informe |
| Este archivo | Explicación metodológica |

Réplicas LDA guardadas (reutilizables): `results/lda/_archive_pre_clean_1019/replica_models/` (o ruta de modelos del run).

---

## 13. Cómo regenerar

```bash
cd /export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories

python scripts/03_analysis/lda/run_lda_coherence_sweep.py \
  --k-list 2,6,10,14,18 --n-inits 5 --passes 50

python scripts/03_analysis/lda/run_lda_stability.py \
  --mode both --n-reps 50 --passes 100 --leave-out 0.2

python scripts/03_analysis/lda/run_lda_monthly_alarm_stability.py \
  --n-seed 50 --n-resample 50 --reuse-models \
  --out-dir results/lda/stability/csv/monthly_alarm \
  --models-dir results/lda/_archive_pre_clean_1019/replica_models
```

---

## 14. Mensaje corto

La referencia `LDA_model_vdec25` (PMP lock) **no** se reentrena. Estabilidad y K usan DB1 reconciliado; se reconfirma **K=6**; alineación Hungarian por cosine **sin outcomes**. **100 passes** en réplicas bastan frente a 1000/5000 (temas/pUF). La extensión a **bloques mensuales y alarma** usa los **1019** landmarks de `base_analysis` con θ fijo y métricas separadas in-fit / held-out.
