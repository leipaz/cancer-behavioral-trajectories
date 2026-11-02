# LDA stability — delivery pack (1019 collapsed landmarks)

Clean folder for the **current** stability analysis only.

## Contents
- `lda_stability_report.docx` — main report (also `.md` / `.html` / `.pdf`)
- `figures/` — K-selection, Level 1/2 cosine + heatmaps, monthly-alarm panel
- `csv/`
  - `coherence/` — K sweep (`coherence_coarse_grid.csv`, …)
  - `level1_seed/` / `level2_resample/` — topic cosine stability
  - `monthly_alarm/` — **1019** collapsed landmarks, θ = 1.0596384140662845, digital observation time
  - `convergence_passes/` — 100 vs 1000/5000 pass check
  - date comparison + cosine summaries

## Alarm lock (base_analysis)
- Report: `results/alarm/base_analysis/granularity_report.docx`
- Landmarks: `results/alarm/base_analysis/decision_points_collapsed.csv` (**1019**)
- θ (sum scale) = **1.0596384140662845**

## Not in this pack
- Saved LDA replicas (`replica_models/`) and bulky `per_point_scores.csv` live under `_archive_pre_clean_1019/` (and/or original paths) for reuse, not for delivery.
