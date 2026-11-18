# Provenance — sensitivity missingness

- Pipeline cohort (mask → VQ-VAE → LDA): all DB1 daily patients (n=175)
- Sport rule: practiced_sport_0_without_walking_or_running_is_missing. Applied before the baseline missingness count and before the extra mask.
- A stored 0 with raw walking or running present stays observed. Recorded ones stay as stored.
- Mask applied before VQ-VAE imputation; learned parameters frozen
- Alarm / landmark metrics: the 1013 decisions in objetivo.csv (114 / 155+ / 858−)
- Alarm denominator in every scenario/seed: 1013 loads
- Baseline load is unchanged by the sport rule (a0 fills a missing cell with 0)
- Seed CIs: deletion-experiment variability (Student t, 5 seeds), not a patient bootstrap
- Empty-days/variables table: patient-mean rates on each expected calendar
- Threshold: 1.0596384140662845
- Seeds: [101, 102, 103, 104, 105]
- Rates: [0.1, 0.2, 0.3, 0.4, 0.5]
- Script: `scripts/03_analysis/alarm/run_sensitivity_db1_objetivo.py`
- Report: `scripts/03_analysis/alarm/build_punto1_informe.py`
- Run directory: `results/missingness/ramp_db1_175_sport_missing`
