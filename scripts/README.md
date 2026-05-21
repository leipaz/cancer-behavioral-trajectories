# Scripts

Organización por etapas del flujo de análisis. Prefijo numérico = orden de ejecución.

| Carpeta | Uso |
|---------|-----|
| `01_import/` | Lectura de wearables, EHR, tablas clínicas; unión por `patient_id` |
| `02_preprocess/` | Filtrado, imputación, sincronización temporal, control de calidad |
| `03_features/` | Trayectorias conductuales y fisiológicas; variables para modelos |
| `04_analysis/` | Asociación con progresión, estado biológico del huésped, predicción |
| `05_figures/` | Figuras del manuscrito (salida típica: `results/figures/`) |

Convención: un script principal por análisis (`04_progression_models.R`) y funciones auxiliares en el mismo directorio o en `scripts/utils/` si crece el proyecto.
