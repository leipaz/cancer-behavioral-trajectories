# Informe metodológico — Selección de K y estabilidad LDA

Documento para double-check / respuesta a colaboradores: **qué se ha hecho, en qué orden, con qué datos y qué significa cada métrica**.

**Entrega canónica (corpus reconciliado CNIO/DB1):** `results/lda/stability/`  
**Embeddings:** `data/processed/lda/user_embeddings_from_pkl.csv`  
**Daily upstream:** `data/revision/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv`  
**Referencia LDA:** publicada (`LDA_model_vdec25`), sin reentrenar

---

## 0. Decisión operativa

Análisis **único** sobre embeddings reconciliados. El LDA de referencia permanece el **publicado** (entrenado históricamente); no se reentrena. La prueba de estabilidad sobre las nuevas fechas es consistente con ese modelo, así que se mantiene como ancla de temas / F–U / pUF del paper y alarmas.

## 1. Objetivo

Responder dos preguntas distintas:

1. **¿Cuántos temas (K) usar?** → selección no supervisada por coherencia.  
2. **Una vez fijado K = 6, ¿el LDA se reproduce?** → estabilidad frente a semilla y frente a submuestreo de pacientes.

La referencia con la que se compara siempre es el **modelo publicado** usado en alarmas:  
`models/lda/LDA_model_vdec25.gensim` (+ diccionario `dictionary_LDA_vdec25.dict`).

---

## 2. Cadena de trabajo (orden)

```
① Daily reconciliado (rangos CNIO/DB1)
        │
        ▼
② VQ-VAE → secuencias de day-types por paciente
   data/processed/lda/user_embeddings_from_pkl.csv
        │
        ▼
③ Barrido de K (coherencia c_v, train/val 70/30 por paciente)
        │
        ▼
④ Fijar K = 6 (modelo publicado; validado en la selección original)
        │
        ▼
⑤ Estabilidad Level 1 (50 semillas) y Level 2 (50× leave-20%-out)
   · alinear temas a la referencia (Hungarian + cosine)
   · medir cosine de P(day-type | tema) y concordancia de pUF
```

Scripts:

- `scripts/03_analysis/lda/run_lda_coherence_sweep.py`  
- `scripts/03_analysis/lda/run_lda_stability.py`

---

## 3. Dataset y unidad de análisis

| Concepto | Valor |
|----------|--------|
| Corpus | **Reconciliado** (fechas CNIO/DB1), no el lock-date histórico |
| Input | `data/processed/lda/user_embeddings_from_pkl.csv` |
| Pacientes (N) | **175** |
| Documentos | **175** (1 documento = 1 paciente) |
| Contenido de un documento | Secuencia de **day-types** (códigos VQ-VAE modelo `a0`) de todos los días observados de ese paciente |
| Ventanas mensuales | **No** se usan en este análisis (sí en alarmas / DCABP longitudinal) |

Cada day-type es un entero del codebook VQ-VAE (p. ej. 0…255). El documento del paciente es la lista de esos códigos a lo largo del seguimiento.

---

## 4. Paso A — Selección de K (coherencia)

### 4.1 Qué se hace

1. Se construye un corpus LDA: un BOW por paciente a partir de sus day-types.  
2. Se parte **70 % train / 30 % validación por paciente** (`train_test_split` sobre índices de documentos; seed fija).  
3. Para cada candidato **K ∈ {2, 6, 10, 14, 18}** se entrenan **5 inicializaciones** (semillas distintas).  
4. Se calcula coherencia media **c_v** en train y en validación.  
5. **Regla:** se elige el K que **maximiza la coherencia media de validación**.

### 4.2 Resultado (selección original del modelo publicado)

Maximiza en **K = 6**. Ese es el K del LDA publicado (`LDA_model_vdec25`).

El barrido documenta la **selección original** del modelo (no un rediseño ad hoc del pipeline de alarmas). Sobre embeddings reconciliados se vuelve a obtener el mismo óptimo (K = 6); los números de estabilidad de esta carpeta usan ya ese K.

### 4.3 Configuración del barrido

| Parámetro | Valor |
|-----------|--------|
| Passes por inicialización | 50 |
| Inicializaciones por K | 5 |
| Split | 70/30 **por paciente** |

Salidas: `coherence_coarse_grid.csv`, `coherence_selection_meta.json`,  
figura `figures/lda_coherence_k_selection.*`

---

## 5. Paso B — Qué es un “tema” y por qué P(day-type \| tema)

Un tema LDA **no** es un escalar: es una distribución sobre el vocabulario de day-types.

\[
P(\text{day-type}=d \mid \text{tema}=t)
\]

Vector de longitud ≈ número de day-types del diccionario; suma 1.  
Intuición: “el tema *t* se caracteriza por usar mucho el day-type 12, algo el 40, poco el 200…”.

**Por qué comparar eso entre entrenamientos:**  
LDA no fija etiquetas. El “tema 3” de una run no tiene por qué ser el mismo patrón que el “tema 3” de otra. Lo comparable es la **firma de day-types** de cada tema.

---

## 6. Paso C — Alineación a la referencia (sin outcomes)

Antes de medir estabilidad:

1. Se toma la matriz de temas de la **referencia** publicada.  
2. Se toma la de la **réplica** (nuevo entrenamiento).  
3. Se empareja cada tema ref con un tema réplica maximizando **cosine** entre sus vectores `P(day-type | tema)` (**algoritmo húngaro**).

Así T0_ref se compara con “su” pareja, no con un índice arbitrario.

**Importante:** el mapping **no usa outcomes clínicos** (progresión, etc.). Solo similitud de distribuciones de day-types.

---

## 7. Paso D — Hipertópicos F / U y patient pUF

### 7.1 Hipertópicos (perfil F/U)

Tras alinear, se usan las etiquetas del modelo publicado:

| Hipertópico | Temas |
|-------------|--------|
| **Favourable (F)** | 0, 1, 2 |
| **Unfavourable (U)** | 3, 4, 5 |

El **perfil** F o U es la **media** de las distribuciones `P(day-type | tema)` de esos temas (firma media del bloque F o U).

### 7.2 patient pUF

Para cada paciente, el LDA asigna probabilidades a los 6 temas sobre **todo su documento** (secuencia completa).

\[
\text{pUF} = P(T_3) + P(T_4) + P(T_5)
\]

Es la masa de probabilidad en el hipertópico **unfavourable**.

| ¿Qué es? | Respuesta |
|----------|-----------|
| ¿Global / ventana / longitudinal? | **Un escalar global por paciente** (documento completo) |
| ¿Es lo mismo que “unfavourable”? | **Sí**, a nivel paciente: pUF = probabilidad de U |
| Favourable análogo | \(P(T_0)+P(T_1)+P(T_2)\) (≈ 1 − pUF) |

---

## 8. Paso E — Estabilidad (dos niveles)

Configuración común:

| Parámetro | Valor |
|-----------|--------|
| K | 6 |
| Réplicas | 50 |
| Passes por réplica | **100** |
| Referencia | `LDA_model_vdec25.gensim` (**~100 000 passes** en su entrenamiento histórico) |

Los ~100 000 passes son del **modelo de referencia publicado**, no de las réplicas de estabilidad. Las réplicas usan 100 passes a propósito (chequeo reproducible y asequible).

### 8.1 Level 1 — Estabilidad a la semilla

- Mismos 175 pacientes.  
- 50 reentrenos con distinta `random_state`.  
- Tras alinear: cosine tema a tema + métricas F/U + Pearson de pUF.

### 8.2 Level 2 — Estabilidad a los datos (leave-20%-out)

- En cada réplica se deja fuera ~20 % de pacientes al azar.  
- Se entrena con el ~80 % restante.  
- Cosine de temas igual que en Level 1.  
- **Pearson de pUF solo sobre los pacientes usados al entrenar** (no sobre los dejados fuera).

---

## 9. Qué significa cada métrica del informe

| Métrica | ¿Qué compara? | Pregunta que responde |
|---------|----------------|------------------------|
| **Cosine por tema** (T0…T5) | Vectores `P(day-type \| tema)` ref vs réplica (tras alinear) | ¿Cada tema tiene la misma firma de day-types? |
| **Favourable / Unfavourable hypertopic cosine** | Firma media F o U (perfiles de temas) | ¿El *concepto* F/U se redefine igual? |
| **Patient pUF Pearson r** | Escalar pUF_ref vs pUF_réplica **entre pacientes** | ¿Los pacientes quedan ordenados igual en “cuánto U tienen”? |

Resumen:  
- Cosine hipertópico U → estabilidad del **patrón** unfavourable.  
- Pearson pUF → estabilidad de la **puntuación** unfavourable por paciente (señal que alimenta la alarma).

---

## 10. Headline (entrega reconciliada, esta carpeta)

### Level 1 (semillas)

| Resumen | Valor |
|---------|------:|
| Median cosine (todos los pares tema–réplica) | **0.915** |
| Unfavourable hypertopic cosine (median) | **0.916** |
| Patient pUF Pearson r (median [IQR]) | **0.849 [0.795, 0.890]** |

### Level 2 (leave-20%-out)

| Resumen | Valor |
|---------|------:|
| Median cosine (todos los pares tema–réplica) | **0.867** |
| Unfavourable hypertopic cosine (median) | **0.885** |
| Patient pUF Pearson r (median [IQR]) | **0.772 [0.675, 0.858]** |

Interpretación breve: temas (sobre todo F) y señal pUF se reproducen bien; Level 2 baja un poco al retirar pacientes, como es esperable.

---

## 11. Archivos de esta entrega

| Ruta | Contenido |
|------|-----------|
| `coherence_*.csv` / `coherence_selection_meta.json` | Barrido K |
| `level1_seed_*.csv` / `level1_seed_summary.json` | Estabilidad semillas |
| `level2_resample_*.csv` / `level2_resample_summary.json` | Leave-20%-out |
| `stability_cosine_summary.json` | Resumen conjunto |
| `run_meta.json` | Parámetros del run |
| `../figures/lda_coherence_k_selection.*` | Figura selección K |
| `../figures/stability/*.svg` (también pdf/png) | Paneles vectoriales estabilidad |
| `lda_stability_report.md` / `.html` / `.pdf` | Informe numérico + figuras |
| Este archivo | Explicación metodológica paso a paso |

---

## 12. Cómo regenerar

```bash
cd /export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories

./env/bin/python scripts/03_analysis/lda/run_lda_coherence_sweep.py \
  --k-list 2,6,10,14,18 --n-inits 5 --passes 50

./env/bin/python scripts/03_analysis/lda/run_lda_stability.py \
  --mode both --n-reps 50 --passes 100 --leave-out 0.2
```

---

## 13. Mensaje corto (para acompañar el envío)

Se documenta la selección original de **K = 6** por coherencia de validación (70/30 por paciente) y la estabilidad del LDA a K fijo frente a semilla y leave-20%-out, sobre embeddings **reconciliados** CNIO (175 docs = 175 pacientes). Los temas se alinean a la referencia publicada por cosine de `P(day-type | tema)` sin outcomes. **pUF** es la probabilidad global del hipertópico unfavourable por paciente. Artefactos y SVG en `results/lda/stability/` y `results/lda/figures/stability/`.
