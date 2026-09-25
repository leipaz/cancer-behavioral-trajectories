# Alarm analysis outputs

General results for the rolling-burden alarm and temporal-granularity analysis  
(branch `revision-experiments`). Not a review-specific pack.

| File | Content |
|------|---------|
| `decision_points_W3_H4.csv` | Monthly landmarks W=3, H=4 (Obs_time clock; n=1015) |
| `decision_points_W3_H4_t_evento_eb2.csv` | Same pipeline with eB2 event clock (n=1012) |
| `alarm_metrics_W3_H4.csv` | Monthly metrics |
| `granularity_sweep_summary.csv` | Full granularity × W × H summary |
| `wh_grid_auc.csv` | W×H AUC grid |
| `thresholds_and_scales.csv` | **All** configs: Youden θ, score scale (mean vs sum), mean-equivalent θ |
| `thresholds_and_scales_W90_H120.csv` | Same table filtered to W=90 d, H=120 d (main comparison slice) |
| `participants_events_by_patient_W3_H4.csv` | Per-patient landmark / event counts |
| `cohort_summary_W3_H4.csv` | Aggregate N patients / landmarks / events |
| `reconcile_1012_vs_1015_extra_landmarks.csv` | The 3 landmarks that differ between clocks |
| `missingness_funnel.csv` | Eligibility funnel |

**Report:** [`../alarm_granularity_report.html`](../alarm_granularity_report.html)  
**Scripts:** `scripts/03_analysis/alarm/`

### Score scale (short)

- Most granularities: burden = **mean** of pUF over W → θ ~ 0.4–0.6  
- Monthly collapsed: burden = **sum** of pUF over W months → θ ~ 1.61  
- Mean equivalent for monthly: `θ_mean = θ_sum / W_months` (e.g. 1.61/3 ≈ 0.537). AUC unchanged.
