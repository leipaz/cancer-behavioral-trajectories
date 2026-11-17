# Provenance — sensitivity missingness

- Pipeline cohort (mask → VQ-VAE → LDA): all DB1 daily patients (n=175)
- Mask applied before VQ-VAE imputation; learned parameters frozen
- Alarm / landmark metrics: the 1013 decisions in objetivo.csv (114 / 155+ / 858−)
- Alarm denominator in every scenario/seed: 1013 loads (no drop for 15-day rule)
- 15-day rule: availability label only (after mask, before imputation)
- Seed CIs: deletion-experiment variability (not patient bootstrap)
- Empty-days/variables table: patient-mean rates on each expected calendar
- Percentage consecutive mask ≠ fixed SAP gaps of 3 / 7 / 14 days
- Threshold: 1.0596384140662845
- Model: VQ-VAE a0 + LDA_model_vdec25 (frozen)
- Seeds: [101, 102, 103, 104, 105]
- Rates: [0.1, 0.2, 0.3, 0.4, 0.5]
- Script: `scripts/03_analysis/alarm/run_sensitivity_db1_objetivo.py`
