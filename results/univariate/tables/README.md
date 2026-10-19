# Univariate tables (CNIO reconciled matching)

Canonical tables from the 1:5 matching + definitive daily
`data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv`.

| File | Content |
|------|---------|
| `variable_summary_full_stats_EPD.*` | E.PD vs no-E.PD (basal / pre-PD) |
| `variable_summary_mean_medianIQR_var_EPDvsnoEPD_2dates.*` | Compact version |
| `variable_period_comparison_intra_group.*` | You vs you (intra-group) |
| `matching_ventanas_fechas_auxiliar.*` | Matching + absolute V1/V2 dates + coverage |
| `matching_window_coverage.csv` | Days in daily per window |

Inputs: `data/processed/univariate/` (matching Excel + cohorts).
