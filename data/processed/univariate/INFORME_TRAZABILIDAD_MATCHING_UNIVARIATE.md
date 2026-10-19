# Memo de trazabilidad — Matching univariante (Figura 2 / Tabla S1 / you vs you)

Documento para double-check: qué se ha hecho, con qué inputs, y dónde están las salidas.  
Uso previsto: **Figura 2**, **Tabla S1** y análisis **you vs you** (basal vs pre-PD / matched), con el matching revisado por cambios de fecha.

**Última regeneración:** 2026-10-19 (tarde) — daily CNIO dedup + **validez origen** + **clip VQ-VAE**; V1/V2 con repetición por `match_set_id` (**stats sobre 1847 day-rows en basal**, no 1539); 27 variables; cohorts + stats regenerados.

---

## 1. Objetivo y alcance

Rehacer el ensamblaje univariante que en su momento alimentó Figura 2 y Tabla S1, **respetando el nuevo matching 1:5** y el **daily filtrado a fechas DB1/CNIO**.

| Qué | Descripción |
|-----|-------------|
| Comparación entre grupos | Early PD (caso) vs no-Early PD (controles matched) |
| Comparación within (you vs you) | Ventana basal vs ventana pre-PD / matched **dentro** de cada grupo |
| Estadísticos | Media, mediana [Q1–Q3], min/max, varianza, Mann–Whitney |

**No** se reentrena VQ-VAE/LDA en este bloque: aquí solo daily + ventanas del Excel de matching.

---

## 2. Cadena de trabajo (orden)

```
① Matching revisado (Excel 1:5, ventanas por paciente)
        │
        ▼
② Daily EB2 → rangos DB1/CNIO → dedup (user, date)
        │
        ▼
③ Preproceso de valores: validez origen + clip VQ-VAE (→ NaN)
        │
        ▼
④ Extraer días del daily en Ventana 1 y Ventana 2 (fechas absolutas del Excel)
        │
        ▼
⑤ Tabla auxiliar de fechas + tablas resumen / you vs you
```

Script: `scripts/03_analysis/revision/run_univariate_matching_revision.py`

---

## 3. Inputs (versión revisión)

| Input | Ruta | Rol |
|-------|------|-----|
| Matching | `data/processed/univariate/Base_Matching_matched_1to5_followup.xlsx` | Define casos/controles, sets y **fechas** de cada ventana |
| Daily (usado en stats) | `data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv` | CNIO + dedup + validez origen + **CLIP_INFO VQ-VAE** |
| Daily previo (sin clip VQ) | `…_dedup.csv` / `…_dedup_validity.csv` | Intermedios; **no** son la entrada actual de las tablas |

### 3.0 Preproceso del daily (antes del univariante)

- **Daily (CNIO):** solo días dentro de Inicio/Fin DB1.  
- **Dedup:** una fila por `(user, date)` (keep first).  
- **Limpieza de valores (→ NaN):**  
  - `heart_rate_hr_mean` / `hr_min` / `hr_max`: **−1 → NaN** (validez origen)  
  - `sleep_start` **> 86 400 s (24 h) → NaN** (validez origen)  
  - luego clip VQ-VAE (fuera de rango → NaN):  
    - `sleep_start` ∈ [−22 500, 25 000]  
    - `location_distance` ∈ [20, 95 000]  
    - `location_time_home` ≥ 120  
    - `sleep_duration` ∈ [3 600, 54 000]  
    - `activity_walking` ∈ [120, 15 000]  
    - `app_usage_total` ∈ [180, 35 000]  
    - `location_clusters_count` ∈ [1, 15]  
    - `steps_steps_total` ∈ [150, 25 000]

### 3.1 Reglas del matching (según diseño clínico)

- **Caso:** Early PD (PD antes de 180 días).  
- **Controles:** 5 por caso (1:5), misma patología; tiempo at-risk del control **mayor** que el tiempo observado del caso.  
- En pulmón: resampling si no alcanzan 5 controles únicos (sigue siendo válido; ~3 controles únicos/caso en promedio).  
- Columnas de estructura: `match_set_id`, `match_role` (case/control), `index_case_id`, `Early PD`.  
- **Ventanas relevantes (fechas absolutas, no solo la etiqueta “1 a 14”):**
  - **Ventana 1 (basal):** `Fechas inicio ventana 1` → `Fecha fin ventana 1`
  - **Ventana 2 (pre-PD / matched):** `Fecha inicio ventana 2` → `Fecha fin ventana 2`

En esta entrega del Excel, **todas** las ventanas tienen longitud **14 días** de calendario; si en revisiones futuras alguna basal es más corta (PD muy temprana), mandan las columnas de fecha.

### 3.2 Conteos de la base de matching

| Concepto | N |
|----------|---|
| Filas (caso + controles) | 144 |
| Sets (`match_set_id`) | 24 |
| Casos (Early PD = 1) | 24 |
| Controles (filas) | 120 |
| IDs únicos | 116 (controles reutilizados entre sets) |
| IDs del matching presentes en daily | **116 / 116** (ninguno faltante) |

Ejemplo set 1: caso **91004** + 5 controles; las fechas absolutas de cada participante son distintas (cada uno con su propio HDM), pero la etiqueta de ventana relativa al caso es la misma (`1 a 14` / `58 a 71`).

---

## 4. Cómo se ensamblan los días (cuidado con fechas)

1. Por cada fila del matching se toman las fechas de V1 y V2.  
2. Del daily se conservan filas con  
   `Inicio_ventana ≤ date_time ≤ Fin_ventana` (inclusive).  
3. **Basal (V1) y pre-PD (V2):** se conserva `match_set_id`. **No se deduplica por ID+fecha entre sets.** Un control reutilizado en varios sets puede aportar el mismo día basal **una vez por set** (matching válido; fechas de observación distintas por set). Dentro del mismo set no se repite `(ID, fecha)`.  
4. Si un día de la ventana no existe en el daily, **no se imputa** para estas tablas: baja la cobertura.  
5. **Unidad de las stats (basal V1):** medias, N y Mann–Whitney se calculan sobre las **1847 day-rows** de la cohorte (no sobre los 1539 `ID+fecha` únicos). Lo mismo en espíritu para V2 (**1876** day-rows).

### 4.1 Verificación QC (run 2026-10-19)

| Check | Resultado |
|-------|-----------|
| Fechas auxiliar = fechas Excel matching (144 filas) | **0 mismatches** en V1 inicio/fin y V2 inicio/fin |
| Días en cohort V1 fuera de ventana | **0 / 1847** |
| Días en cohort V2 fuera de ventana | **0 / 1876** |
| Duplicados `(ID, fecha, match_set)` en V1 y V2 | **0** (sí puede repetirse ID+fecha entre sets distintos) |
| Longitud calendario V1 / V2 | **14 días en las 144 filas** |
| Cobertura media V1 / V2 | **91.6% / 93.1%** |
| Filas con cobertura &lt; 50% | V1: 11 · V2: 10 |

### 4.2 Cohortes generadas

| Cohort | IDs | Day-rows | Únicos ID+fecha | E.PD day-rows | no-E.PD day-rows |
|--------|-----|----------|-----------------|---------------|------------------|
| Ventana 1 (basal / first 15) | 112 | 1847 | 1539 | 335 | 1512 |
| Ventana 2 (pre-PD / previous 15) | 111 | 1876 | 1846 | 294 | 1582 |

En V1, day-rows (1847) > únicos ID+fecha (1539) porque los controles reutilizados cuentan **una vez por set**. **Ese 1847 es el N de las stats basales** (y el `full_stats` reproduce esas filas). Los 1539 no son una cohorte alternativa: son solo el conteo sin clave de set.

Archivos: `data/processed/univariate/cohorts/cohort_ventana1_basal` y `cohort_ventana2_prePD_matching` (csv/xlsx).

---

## 5. Tabla auxiliar de fechas (doble cara del matching)

Además del Excel de matching “estructural”, se genera una tabla auxiliar con **las fechas concretas** que usa cada ventana, más cobertura frente al daily (`matching_ventanas_fechas_auxiliar.xlsx`). Las filas con cobertura &lt; 100% y su motivo están en `matching_cobertura_menor_100.csv`.

Columnas clave del auxiliar:

- Estructura: `ID`, `match_set_id`, `match_role`, `index_case_id`, `Early_PD`, etiquetas `ventana1_etiqueta` / `ventana2_etiqueta`, `V2_i`, `V2_e`
- Fechas absolutas: `v1_fecha_inicio`, `v1_fecha_fin`, `v2_fecha_inicio`, `v2_fecha_fin`
- QC: `v1_n_dias_calendario`, `v2_n_dias_calendario`, `v1_dias_en_daily`, `v2_dias_en_daily`, `v1_cobertura`, `v2_cobertura`

**Interpretación:** el matching define *quién* va con *quién*; el auxiliar fija *qué calendario* se está usando en V1 y V2 para no confundir la etiqueta relativa (“58 a 71”) con las fechas absolutas de cada paciente.

---

## 6. Variables y estadísticos

### 6.1 Variables usadas (27)

`sleep_start`, `location_distance`, `location_time_home`, `sleep_duration`, `activity_walking`, `app_usage_total`, `location_clusters_count`, `steps_steps_total`, `practiced_sport`, `heart_rate_hr_mean`, `heart_rate_hr_baseline`, `heart_rate_hr_min`, `heart_rate_hr_max`, `heart_rate_sleep_min`, `heart_rate_sleep_max`, `heart_rate_sleep_mean`, `oxygen_saturation_max_spo2`, `oxygen_saturation_min_spo2`, `oxygen_saturation_mean_spo2`, `total_unlocks`, `total_communication`, `total_social_networks`, `total_others`, `deep_time`, `light_time`, `rem_time`, `emotions_valence_pred`

Los NaNs se excluyen variable a variable en Mann–Whitney (igual que el pipeline previo).

### 6.2 Unidad de análisis

Cada valor diario dentro de la ventana cuenta como una observación. Si el mismo control entra en varios sets, su día basal (V1) —y el pre-PD (V2) si aplica— se cuenta **una vez por set**. **No son duplicados erróneos:** es la reutilización del matching 1:5.

| Ventana | Day-rows en cohorte (= base de stats) | Únicos ID+fecha (informativo) |
|---------|----------------------------------------|-------------------------------|
| V1 basal | **1847** | 1539 |
| V2 pre-PD | **1876** | 1846 |

NaNs se excluyen variable a variable (tras el preproceso del daily).

### 6.3 Tests

- **Entre grupos:** Mann–Whitney E.PD vs no-E.PD, por variable y por periodo (First 15 / Previous 15).  
- **You vs you (intra-grupo):** Mann–Whitney basal vs pre-PD **dentro** de E.PD y **dentro** de no-E.PD.

---

## 7. Salidas (archivos)

| Archivo | Contenido | Uso típico |
|---------|-----------|------------|
| `cohort_ventana1_basal.csv` / `.xlsx` | Daily summaries de la ventana 1 (basal / primeros ~15 días) | Cohorte basal; auditoría fila a fila |
| `cohort_ventana2_prePD_matching.csv` / `.xlsx` | Daily summaries de la ventana 2 (pre-PD / previos ~15 días matched) | Cohorte pre-PD; auditoría fila a fila |
| `variable_summary_full_stats_EPD.csv` | N, mean, median IQR, min, max, var, p (E.PD vs no-E.PD) | Tabla tipo S1 / suplemento |
| `variable_summary_mean_medianIQR_var_EPDvsnoEPD_2dates.csv` | Igual sin min/max | Versión compacta |
| `variable_period_comparison_intra_group.csv` | Basal vs pre-PD within group | **You vs you** |
| `matching_ventanas_fechas_auxiliar.xlsx` | Matching + fechas absolutas V1/V2 + días en daily y cobertura | Tracking / QC de fechas de ventana |
| `matching_cobertura_menor_100.csv` | Filas del matching con cobertura &lt; 100% y motivo | Quién no llega a cobertura completa y por qué |

---

## 8. Relación con el resto de la revisión de fechas

1. Primero se fijaron rangos **DB1/CNIO** y se filtró el daily EB2.  
2. El matching nuevo trae **sus propias fechas de ventana** (basal / pre-PD), ya alineadas al diseño clínico revisado.  
3. El univariante **solo usa días que existen** en el daily filtrado dentro de esas ventanas.  
4. Por eso es crítico: matching Excel + daily filtrado deben ser la misma generación de fechas; no mezclar ventanas viejas con daily nuevo (ni al revés).

---

## 9. Checklist de double-check (rápido)

- [x] Excel matching = `Base_Matching_matched_1to5_followup.xlsx` (1:5, 24 sets).  
- [x] Daily = `…_dedup_vqclip.csv` (CNIO + dedup + validez origen + clip VQ-VAE).  
- [x] Ventanas leídas de columnas de **fecha**, no solo de la etiqueta “1 a 14”.  
- [x] Tabla auxiliar de fechas generada y cruzada 0 mismatches vs Excel.  
- [x] Cohortes V1/V2: 0 días fuera de ventana; 0 dups `(ID, fecha)` dentro del mismo set; basales repetidos entre sets (como V2).  
- [x] Stats basales sobre **1847** day-rows (no 1539).  
- [x] 27 variables en tablas de resultados.  
- [x] Salidas en `results/univariate/tables/revision/` (no las dic2025).  
- [ ] Revisar filas con cobertura baja en el auxiliar si algún set preocupa.  
- [x] You vs you = `variable_period_comparison_intra_group.csv` / `.xlsx`.  
- [x] E.PD vs no-E.PD = `variable_summary_full_stats_EPD.csv` / `.xlsx`.

---

## 10. Re-ejecución

```bash
cd /export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories
./env/bin/python scripts/03_analysis/revision/run_univariate_matching_revision.py
```

El script regenera las cohorts, la tabla auxiliar de fechas y las tablas de resumen.

---

## 11. Mensaje corto (para acompañar el envío)

Se ha vuelto a ensamblar el univariante del matching 1:5 con el daily filtrado a fechas DB1/CNIO, **más preproceso de validez de origen y clip VQ-VAE** (mismo criterio que el modelo). Las ventanas basal y pre-PD salen del Excel de matching. **No deduplicamos por ID+fecha entre sets:** las stats basales usan las **1847** day-rows de la cohorte V1 (los 1539 únicos son solo informativos). You vs you e inter-grupo regenerados sobre ese daily. Auxiliar de fechas: `matching_ventanas_fechas_auxiliar.xlsx`.
