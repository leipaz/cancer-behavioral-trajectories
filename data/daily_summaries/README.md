# Daily summaries (DB1 date window)

Only these three for analysis (plus legacy `oncology_*` / zip archives):

| File | Role |
|------|------|
| `daily_datesdb1.csv` | Oficial filtrado por fechas DB1/CNIO + dedup. **Sin** limpieza de valores ni VQ CLIP |
| `daily_datesdb1_enriched.csv` | Igual + 7 features enriched (`total_unlocks`, `total_communication`, `total_social_networks`, `total_others`, `deep_time`, `light_time`, `rem_time`) |
| `daily_datesdb1_enriched_preprocess.csv` | Enriched + control de calidad: HR −1→NaN; tiempos ±24 h / ≤24 h → NaN. **Sin** VQ CLIP |

**Preprocess time rules**

- `sleep_start`, `sleep_end_time`: fuera de **[−86400, +86400]** → NaN
- Duraciones en segundos: fuera de **[0, 86400]** → NaN
- `location_time_home`, `location_time_work` (min): fuera de **[0, 1440]** → NaN
