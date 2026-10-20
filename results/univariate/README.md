# Univariate matching — canónico (2026-10-20)

**Única versión vigente.** Daily: `data/daily_summaries/daily_datesdb1_enriched_preprocess.csv`  
(fechas DB1/CNIO → enriched → QC: HR −1→NaN; tiempos ±24 h / ≤24 h; **sin** VQ CLIP).

| Salida | Ruta |
|--------|------|
| Tablas stats | `tables/*.csv` |
| Figuras dens/hist | `figures/univariate_matched_*.png` |
| QC rangos (opcional) | `figures/qc_ranges_*.png`, `tables/qc_*.csv`, `tables/pct_dias_presente_por_variable.csv` |
| Cohortes | `data/processed/univariate/cohorts/` |
| Matching | `data/processed/univariate/Base_Matching_matched_1to5_followup.csv` |
| Informe | `data/processed/univariate/02_INFORME_MATCHING_UNIVARIATE.docx` |

Stats sobre **day-rows** de cohorte (V1=1847, V2=1876), **sin** deduplicar ID+fecha entre `match_set`.

Script: `scripts/03_analysis/revision/run_univariate_matching_revision.py`
