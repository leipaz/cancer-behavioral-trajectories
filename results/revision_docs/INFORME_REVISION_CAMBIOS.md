# Informe de revisión — proceso y cambios

**Modelos:** VQ-VAE y LDA **congelados** (no se reentrenan).

Hilo conductor de la revisión:

1. **Cambiar / fijar las fechas** de inicio y fin por paciente (rango anterior → rango nuevo DB1 CNIO).  
2. **Filtrar el daily EB2** con esas fechas nuevas.  
3. **Recalcular topics** (y alarmas) solo sobre los días que quedan tras el filtro.

---

## 1. Resumen ejecutivo

1. Se actualizan los rangos oficiales a **Inicio / Fin datos DB1** (`rangos_fechas.xlsx`).  
2. Ese rango se usa como **máscara** sobre el daily EB2.  
3. Sobre el daily filtrado se repite VQ-VAE → LDA → topics mensuales.  
4. La mayoría queda bien; en pocos casos (sobre todo **41025**) el nº de meses de topics cambia porque, al cortar con las fechas DB1, cambia qué extremos del daily entran al densify del VQ-VAE.

---

## 2. Proceso (en este orden)

```
① Fechas anteriores (rango HDM / seguimiento previo)
        │  actualización clínica CNIO
        ▼
② Fechas nuevas DB1: Inicio_datos_DB1 – Fin_datos_DB1
        │  filtro por paciente
        ▼
③ Daily EB2 restringido a [Inicio_DB1, Fin_DB1]
        │
        ▼
④ Extracción de topics (VQ-VAE + LDA congelados)
        │
        ▼
⑤ Audit 178 / alarmas / análisis asociados
```

### 2.1 Paso ①→② — Cambios de fecha (rango anterior vs DB1 nuevo)

**Qué es un “cambio de fecha” en esta revisión:**  
comparar el **rango de inicio/fin que se usaba antes** (típicamente `Date_start_HDM` + `Obs_time`, o el rango HDM previo documentado) con el **rango nuevo CNIO/DB1** (`Inicio_datos_DB1`, `Fin_datos_DB1`).

| Concepto | Fuente |
|----------|--------|
| Rango anterior | HDM / seguimiento previo (`inicio_rango_anterior_HDM`, `fin_rango_anterior_HDM` en tablas de cambio; o `Date_start_HDM` + `Obs_time` en Subjects) |
| Rango nuevo | `rangos_fechas.xlsx` → `Inicio_datos_DB1`, `Fin_datos_DB1` |

**Números (178 pacientes):**

| Criterio | N |
|----------|---|
| Mismo inicio **y** mismo fin (HDM = DB1) | **80** |
| Cambia inicio y/o fin (HDM ≠ DB1) | **98** |
| Marcados explícitamente `date_change=True` en el Excel de rangos | **18** |

El flag **18** es la lista que CNIO etiquetó como cambio relevante en `rangos_fechas`.  
En la práctica, al comparar rango anterior vs DB1, **cambian más pacientes (~98)**; muchos con desfases pequeños de unos días. Los 18 son el subconjunto marcado, no el total de diferencias inicio/fin.

Tablas: `dias_cambio_date_change_True.csv` (detalle de los 18), `rangos_fechas.xlsx` (rangos nuevos para todos).

### 2.2 Paso ②→③ — Filtrar el daily EB2

Con el rango **nuevo** DB1 por ID:

```text
conservar día  ⇔  Inicio_datos_DB1  ≤  date_time  ≤  Fin_datos_DB1
```

| Entrada | Salida |
|---------|--------|
| Daily EB2 revisión (`daily_summary_eb2prod_revised.csv` / oficial) | `daily_summary_oficial_filtrado_rangos_CNIO.csv` |
| + columnas VQ-VAE | `oncology_daily_summary_model_input_rangos.csv` |

**Sin ningún día EB2 dentro del rango DB1:** `31015`, `61004`, `63006` (no se puede calcular perfil/topics).

Tras el filtro, el daily puede:

- **Cubrir bien** el rango DB1 (inicio/fin EB2 ≈ inicio/fin DB1) → mayoría (~131/175).  
- **Empezar tarde / acabar antes** respecto al papel DB1 → lista en `fechas_CNIO_vs_EB2_no_coinciden.csv` (**44** IDs).  
  Eso no es un “segundo cambio de fechas clínicas”: es **cobertura del wearable** dentro del rango ya fijado en ②.

### 2.3 Paso ③→④ — Cálculo de topics

Sobre el daily **ya filtrado**:

1. VQ-VAE `a0` congelado → embeddings por día  
2. LDA vdec25 congelado → topics globales y por bloques de 30 días (`mes*_topic*`)  
3. Escritura en audit: `178_audit_oct_2_2026_topics_revision.xlsx`

Comandos: `PIPELINE_REVISION.md`.

**Nota densify VQ-VAE:** entre el primer y el último día **que quedan** tras el filtro, los huecos internos se pasan a cero y reciben código. Si el filtro elimina un extremo antiguo, cambia la longitud de esa secuencia y por tanto el nº de “meses” LDA.

---

## 3. Impacto en topics vs BBDD anterior

Comparando nº de meses con `mes*_topic*` no nulos (Base Sept 2025 vs audit revisión):

| ID | Meses antes | Meses ahora | Δ | Lectura |
|----|-------------|-------------|---|---------|
| **41025** | 6 | 0 | −6 | Extremo; ver §4 |
| 31005 | 13 | 12 | −1 | Menor |
| 73006 | 5 | 4 | −1 | Menor |
| 23008 | 15 | 16 | +1 | Gana |
| 72001, 61005 | … | … | +1 | Ganan poco |
| 51016 | 0 | 3 | +3 | Antes sin meses |
| 41026 | 4 | 12 | +8 | Más serie tras nuevo rango/daily |
| 73011 | 0 | 9 | +9 | Antes casi sin serie |

La **mayoría** no cambia el número de meses. Los saltos grandes se explican por el **filtro DB1 + qué días EB2 sobreviven**, no por reentrenar modelos.

---

## 4. Caso 41025 (cambio de primera fecha / extremos)

Encaja exactamente en la cadena ①→②→③→④:

| Etapa | 41025 |
|-------|--------|
| Rango anterior (HDM) | Inicio ~**2024-10-08**, Obs_time 169 → fin ~**2025-03-26** |
| Rango nuevo DB1 | **2024-10-15 → 2025-09-11** (`date_change`) |
| Daily que entraba “antes” al densify | Extremos **08-oct** y **26-mar** (solo 2 filas reales, muy separadas) → densify hinchaba ~170 días → **6 meses** LDA |
| Daily tras filtro DB1 | El **08-oct queda fuera** (inicio DB1 = 15-oct). Quedan **8 días** (**26-mar → 2-abr**) |
| Topics ahora | **0 meses** (serie demasiado corta para bloques de 30 con el criterio actual) |

**Por qué “antes se cubría el rango y ahora no”:**  
antes el densify cubría el tramo entre dos puntas lejanas (incluida la primera fecha antigua). Al pasar a fechas DB1, esa **primera fecha ya no entra** en el filtro; la secuencia deja de estirarse y desaparecen los meses artificiales.

No se ha forzado relleno hasta fin DB1 en el audit final (se probó y se descartó para no reintroducir meses inventados en el resto de la cohorte).

---

## 5. Criterio para rellenar topic predominante (first month / 6 months / 1 year)

Esto es independiente del filtro de fechas: se aplica sobre la **secuencia de embeddings** del paciente tras el daily filtrado.

### 5.1 Cómo se construyen los “meses” (bloques de 30 días)

Función: `get_topic_distributions_and_predominant_topics_relaxedwindow`  
(`window_size = 30`, `min_embeddings = 15`).

| Días de embeddings (n) | Qué se hace |
|------------------------|-------------|
| **n &lt; 15** | No se calcula ningún mes → todo vacío / NaN |
| **15 ≤ n &lt; 30** | **Un solo mes** con los días que hay (mes incompleto permitido solo aquí) |
| **n ≥ 30** | Meses = bloques consecutivos de **30** días: días 1–30, 31–60, … |

**Último tramo incompleto (n ≥ 30):** no se inventa un mes con &lt;30 días sueltos. Se toma una ventana **relaxed** con los **últimos 30** días de la serie y se cierra.  
Ejemplo: 160 días → bloques en 0–30, 30–60, 60–90, 90–120, 120–150 y luego, al no llegar a 180, un último bloque = días 131–160 (últimos 30).

Cada bloque produce `mesN_topic0…5` (probs LDA) y un **topic predominante del mes** = argmax de esas 6 probs.

### 5.2 Columnas resumen: cuándo se rellenan y cuándo no

| Columna | Condición para rellenarla | Valor que se pone | Si el horizonte no está completo |
|---------|---------------------------|-------------------|----------------------------------|
| **`topic_first_month`** | Existe al menos el **mes 1** (hace falta **n ≥ 15**) | Topic predominante del **mes 1** | Si no hay mes 1 → **NaN** (no se inventa) |
| **`topic_6_months`** | Hay **≥ 6 bloques mensuales** (en la práctica **n ≳ 150–180** días) | Topic predominante del horizonte a 6 meses* | Si hay &lt; 6 meses → **NaN** (no se rellena con el mes 3 o 4 “por aproximación”) |
| **`topic_first_year`** | Hay **≥ 12 bloques mensuales** (**n ≳ 330–360** días) | Topic predominante del horizonte a 1 año* | Si hay &lt; 12 meses → **NaN** |

\*En la BBDD histórica (Base Sept), `topic_6_months` / `topic_first_year` coinciden en la mayoría de casos con la **moda** (topic más frecuente) de los predominantes de `mes1…mes6` / `mes1…mes12`. No se rellenan si faltan meses del horizonte.

**Regla práctica:**  
- Mes “suelto” incompleto solo se tolera al inicio (serie corta 15–29 días → un mes).  
- Para **6 months** y **1 year**, el criterio es **tener ese número de bloques**, no “promediar lo que haya” si faltan meses.  
- Un mes concreto (`mesN_*`) existe cuando el algoritmo ha generado el N-ésimo bloque; si no, esas celdas quedan vacías.

### 5.3 Relación con 41025

Con **8 días** tras el filtro DB1: **n &lt; 15** → no hay mes 1 → `topic_first_month`, `topic_6_months` y `topic_first_year` quedan vacíos. Antes, al densificar ~170 días artificiales, sí se generaban ≥6 bloques y por eso aparecía `topic_6_months`.

---

## 6. Mensajes cortos para comunicar

1. **Primero** se fijan las fechas nuevas DB1 respecto a las anteriores.  
2. **Después** se filtra el daily EB2 con esas fechas.  
3. **Después** se recalculan los topics sobre ese daily.  
4. El flag de **18** es la lista marcada; al comparar rangos, **más pacientes** tienen inicio/fin distintos (muchos por pocos días).  
5. **41025** cambia fuerte porque el nuevo inicio DB1 **quita la primera fecha** que antes anclaba un densify largo.

---

## 7. Archivos clave

| Archivo | Rol en la cadena |
|---------|------------------|
| `rangos_fechas.xlsx` | Fechas **nuevas** DB1 (y flag `datee_change`) |
| `dias_cambio_date_change_True.csv` | Detalle de los 18 marcados (anterior HDM vs nuevo CNIO) |
| `daily_summary_eb2prod_revised.csv` | Daily EB2 de partida |
| `daily_summary_oficial_filtrado_rangos_CNIO.csv` | Daily **tras** filtro DB1 |
| `oncology_daily_summary_model_input_rangos.csv` | Input VQ-VAE |
| `profiles_per_sample_oncology_revision.pkl` / `user_embeddings_revision_a0.csv` | Embeddings |
| `178_audit_oct_2_2026_topics_revision.xlsx` | Topics / audit final |
| `fechas_CNIO_vs_EB2_no_coinciden.csv` | Tras el filtro: DB1 vs cobertura real EB2 |
| `PIPELINE_REVISION.md` | Cómo re-ejecutar ③→④ |
