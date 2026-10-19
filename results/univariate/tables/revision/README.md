# Univariate tables — revision (Evento_DB1 / rangos CNIO)

Matching: `data/processed/univariate/Base_Matching_matched_1to5_followup.xlsx` (1:5, same pathology, at-risk ≥ case FU).
Daily: `data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv` (origin validity + VQ-VAE CLIP_INFO → NaN; from dedup CNIO). + enriched left-join (total_unlocks, total_communication, total_social_networks, total_others, deep_time, light_time, rem_time) → `daily_summary_oficial_filtrado_rangos_CNIO_enriched.csv`.

Windows per matching row (absolute calendar dates from the matching Excel):
- **Ventana 1 (basal / first ~15d)** = `Fechas inicio ventana 1` → `Fecha fin ventana 1`
- **Ventana 2 (pre-PD / matched)** = `Fecha inicio ventana 2` → `Fecha fin ventana 2`

Controls reused across sets keep each set's own ventana 2.

| File | Content |
|------|---------|
| `variable_summary_full_stats_EPD.csv` | Mean, median IQR, min/max, var, MW p (E.PD vs no-E.PD) |
| `variable_summary_mean_medianIQR_var_EPDvsnoEPD_2dates.csv` | Same without min/max |
| `variable_period_comparison_intra_group.csv` | First vs Previous within E.PD / no-E.PD |
| `matching_window_coverage.csv` | Days available in daily per window (long) |
| `matching_ventanas_fechas_auxiliar.csv/.xlsx` | Matching structure + concrete V1/V2 dates + coverage |

Also copied to `data/processed/univariate/matching_ventanas_fechas_auxiliar.xlsx`.

Variables used (27): sleep_start, location_distance, location_time_home, sleep_duration, activity_walking, app_usage_total, location_clusters_count, steps_steps_total, practiced_sport, heart_rate_hr_mean, heart_rate_hr_baseline, heart_rate_hr_min, heart_rate_hr_max, heart_rate_sleep_min, heart_rate_sleep_max, heart_rate_sleep_mean, oxygen_saturation_max_spo2, oxygen_saturation_min_spo2, oxygen_saturation_mean_spo2, total_unlocks, total_communication, total_social_networks, total_others, deep_time, light_time, rem_time, emotions_valence_pred
