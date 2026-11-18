# Cobertura real (punto 2A) — provenance

- Landmarks: `results/lda/val_2/anidada_estratificada/objetivo.csv` (1013 / 114).
- Scores: baseline pack `Sensitivity_to_artificial_missingness/alarm_scored_objetivo_baseline.csv` (load after VQ-VAE; metrics use all 1013).
- Raw daily (pre-impute): `data/daily_summaries/DB1/daily_summary_checked.csv`.
- 9 input variables = VQ-VAE measure set (excl. `weekend`).
- Windows: 90 d and 30 d ending at `decision_date`; expected cells 810 / 270.
- Bandas SAP: `<50%`, `50–80%`, `≥80%` (fixed before looking at performance).
- IC: patient-clustered bootstrap (1000), seed 20261116.
- Source map in `PROVENANCE.json`: sleep, steps and walking are wearable (primary; mobile fallback not split by day). Location and app use are mobile. `practiced_sport` is derived from activity and is excluded from both source means. The heart-rate / SpO2 day-level rule is not used.
- `practiced_sport` is counted only when `activity_walking` survived the 120–15000 s filter. A sport flag on a day whose walking was removed by that filter is missing.
- Script: `scripts/03_analysis/alarm/run_coverage_real_objetivo.py`.

Still pending (2B): comparator availability alone vs DCABP vs DCABP+availability.
