# Monthly-block + alarm stability (collapsed, digital observation time)

## Starting point (locked)
- Report: `results/alarm/base_analysis/granularity_report.docx` (§2 Collapsed monthly)
- Landmarks: `results/alarm/base_analysis/decision_points_collapsed.csv`
- **1019** evaluable monthly-collapsed landmarks (159 pos / 860 neg; 116 patients)
- Clock: **Tiempo_obs digital** / t_evento_eb2 (labels in `target`)
- W = 3 months, H = 4 months (120 days)
- Fixed cut: **S_m = AUC_W ≥ 1.0596384140662845** (θ/3 = 0.3532128046887615; Youden on prior 1013; not re-tuned per replica)

## Method
- Freeze VQ-VAE day-types + dictionary; re-fit LDA replicates (K=6, 100 passes)
- Hungarian align topics by cosine of P(day-type|topic); **no outcomes**
- After alignment: F=T0–2, U=T3–5, pUF = P(U)
- 30-day block pUF at month ends → 3-block load (`AUC_W` / `S_m`) at each of the 1019 points
- Leave-20%-out: inference on held-out patients allowed; metrics split **in_fit** vs **held_out**

## Outputs
- `design_meta.json`, `base_landmarks_1019_collapsed.csv`
- `per_replica_summary.csv`, `per_point_scores.csv`, `headline_summary.json`
- Replica LDA models (reusable): `results/lda/stability/replica_models/seed/rep_XXX.gensim` (+ `.json` with alignment perm / train_idx)
  and `results/lda/stability/replica_models/resample/rep_XXX.gensim`. Re-run with `--reuse-models` to skip training.
