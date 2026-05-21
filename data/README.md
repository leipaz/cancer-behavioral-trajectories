# Datos

Solo se versionan aquí datos **procesados y anonimizados** listos para análisis. Los datos en bruto o identificables deben quedarse en `data/raw/` (local, no en git) o fuera del repositorio.

## Estructura

| Ruta | Contenido esperado |
|------|-------------------|
| `processed/behavioral/` | Series o resúmenes de monitoreo conductual (p. ej. actividad, sueño, pasos) alineados por paciente y ventana temporal |
| `processed/physiologic/` | Señales o derivados fisiológicos (p. ej. frecuencia cardíaca, HRV) con la misma granularidad |
| `processed/clinical/` | Progresión, estadio, tratamientos, biomarcadores de estado del huésped, fechas de eventos |
| `metadata/` | Diccionario de variables (`variables.csv` o similar), definición de cohorte, mapas de códigos |

## Convenciones sugeridas

- Nombres de archivo: `snake_case`, prefijo de cohorte o estudio si aplica (`cohort_behavioral_daily.parquet`).
- Incluir una columna de identificador anonimizado (`patient_id`) consistente entre carpetas.
- Documentar en este README cada archivo añadido: origen, transformación, versión, fecha.

## Archivos previstos (rellenar al subir datos)

<!-- Ejemplo:
- `processed/behavioral/daily_features.parquet` — agregados diarios por paciente
- `processed/physiologic/hrv_windows.parquet` — HRV en ventanas de 5 min
- `processed/clinical/progression_events.csv` — fechas de progresión RECIST u otro criterio
- `metadata/variable_dictionary.csv` — nombre, unidad, descripción
-->

_(Sin archivos de datos aún.)_

## Acceso

Si los datos no pueden publicarse abiertamente, describir aquí el procedimiento de solicitud (comité de ética, acuerdo de transferencia, contacto).
