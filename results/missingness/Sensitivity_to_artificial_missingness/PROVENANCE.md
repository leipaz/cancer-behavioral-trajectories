# Provenance — sensitivity missingness

- Pipeline cohort (mask → VQ-VAE → LDA): all DB1 daily patients (n=175)
- Sport rule: practiced_sport_walking_gate_but_stored_ones_kept. Applied before the baseline missingness count and before the extra mask.
- A stored 1 is always kept. A stored 0 stays observed only when activity_walking survived 120–15000 seconds.
- Mask applied before VQ-VAE imputation; learned parameters frozen
- Alarm / landmark metrics: the 1013 decisions in objetivo.csv (114 / 155+ / 858−)
- Alarm denominator in every scenario/seed: 1013 loads
- Baseline = frozen alarm (stored AUC_W in results/alarm/base_analysis/decision_points_collapsed.csv). Masked loads are anchored: AUC_W + (masked re-encoding − unmasked re-encoding); alarm = anchored load >= threshold. Script: scripts/03_analysis/alarm/anchor_sensitivity_to_frozen.py. The CSVs here are the anchored ones; the raw re-encoding metrics stay in the run directory (metrics_by_seed.csv).
- Seed CIs: deletion-experiment variability (Student t, 5 seeds), not a patient bootstrap
- Empty-days/variables table: patient-mean rates on each expected calendar
- Threshold: 1.0596384140662845
- Seeds: [101, 102, 103, 104, 105]
- Rates: [0.1, 0.2, 0.3, 0.4, 0.5]
- Script: `scripts/03_analysis/alarm/run_sensitivity_db1_objetivo.py`
- Report: `scripts/03_analysis/alarm/build_punto1_informe.py`
- Run directory: `results/missingness/ramp_db1_175_sport_keep1`
