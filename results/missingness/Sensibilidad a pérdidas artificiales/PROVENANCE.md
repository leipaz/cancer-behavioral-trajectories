# Provenance — sensitivity missingness

- Pipeline cohort (mask → VQ-VAE → LDA): all DB1 daily patients (n=175)
- Alarm / landmark metrics: the 1013 decisions in objetivo.csv
- Those 1013 are the monthly collapsed landmarks. decision_points_collapsed has six further rows from the t+1 clock; they are not in this alarm.
- A decision past the profile keeps its day and uses the last 30 profile days.
- All 1013 decisions have a load. Baseline MAE versus the stored collapsed load: 0.0019.
- Threshold: 1.0596384140662845
- Model: VQ-VAE a0 + LDA_model_vdec25 (frozen)
- Seeds: [101, 102, 103, 104, 105]
- Rates: [0.1, 0.2, 0.3, 0.4, 0.5]
- Script: `scripts/03_analysis/alarm/run_sensitivity_db1_objetivo.py`
