# Delivery for Rev 2 — granularity / monthly baseline (confirm only)

Folder: `results/alarm/delivery_rev2_granularity/`

## 1. CSVs included

| File | What it is |
|------|------------|
| `decision_points_W3_H4.csv` | **Current MAIN** monthly landmarks (Obs_time clock), n=**1015** |
| `decision_points_W3_H4_Obs_time.csv` | Same as MAIN (explicit name) |
| `decision_points_W3_H4_t_evento_eb2.csv` | Alternate eB2 clock, n=**1012** (first audited baseline) |
| `alarm_metrics_W3_H4.csv` | Metrics for MAIN |
| `granularity_sweep_summary.csv` | Full granularity × W × H sweep |
| `thresholds_and_scales_W90_H120.csv` | Youden θ + **score scale** for each config at W=90, H=120 |
| `wh_grid_auc.csv` | W×H AUC grid |
| `missingness_funnel.csv` | Eligibility funnel |
| `cohort_summary_W3_H4.csv` | N landmarks / patients / events (aggregate) |
| `participants_events_by_patient_W3_H4.csv` | Per-patient: n landmarks, n positive landmarks, PD flag, Obs_time, t_evento_eb2 |
| `reconcile_1012_vs_1015_extra_landmarks.csv` | The **+3** landmarks that explain 1015 vs 1012 |

Scripts: `scripts/03_analysis/alarm/` (already on branch `revision-experiments`).  
Report: `results/alarm/alarm_granularity_report.html`.

## 2. Thresholds and scale (confirm)

At the main granularity comparison (**W = 90 d, H = 120 d**), operating θ = **Youden** per configuration (see `thresholds_and_scales_W90_H120.csv`).

| Family | Score definition | Typical θ scale |
|--------|------------------|-----------------|
| Sampled (daily / weekly / biweekly / monthly_sample) | Rolling **mean** of daily pUF over W days | ~0.39–0.57 |
| Collapsed weekly / biweekly | Rolling **mean** of block pUF | ~0.49–0.51 |
| **Monthly collapsed** | **Sum** of monthly pUF over W months | **~1.61** |

So monthly collapsed θ≈1.61 is **not** comparable numerically to sampled θ≈0.5: different aggregation (sum vs mean). Equivalent mean-scale for monthly is θ/W ≈ 1.61/3 ≈ 0.54.

Fixed reference used alongside Youden for monthly: θ = 1.61 (same operating point as Youden here).

**Mean equivalent for monthly collapsed (W=3):** θ_mean = 1.61 / 3 ≈ **0.537** (Youden exact: 1.610762 → 0.536921). AUC unchanged; only the threshold number changes.

## 3. Participants / events (MAIN = Obs_time, W=3, H=4)

| Quantity | Value |
|----------|------:|
| Decision points (landmarks) | 1015 |
| Patients with ≥1 landmark | 115 |
| Positive landmarks (PD within H) | 159 |
| Patients with PD flag among those | 45 |

Per-patient detail: `participants_events_by_patient_W3_H4.csv`.

**Patient-level confidence intervals (IC):** not produced by this pipeline (aggregate ROC/Sens/Spec only). No new CI analyses were run, per “no repetir / no buscar configs mejores”.

## 4. Reconcile 1012 vs 1015

| Clock | File | N |
|-------|------|--:|
| `t_evento_eb2` (earlier audited monthly file) | `decision_points_W3_H4_t_evento_eb2.csv` | **1012** |
| `Obs_time` (Subjects / current MAIN) | `decision_points_W3_H4.csv` | **1015** |

Same pipeline (W=3, H=4, monthly collapsed, landmark if `month×30 < event_time`).  
Only the **event/censor clock** differs.

The +3 landmarks (all PD cases; start of eB2 later than HDM):

| id | month | day (=m×30) | t_evento_eb2 | Obs_time | Why |
|----|------:|------------:|-------------:|---------:|-----|
| 53001 | 8 | 240 | 236 | 246 | 240 < 246 but not < 236 |
| 72001 | 7 | 210 | 203 | 217 | 210 < 217 but not < 203 |
| 73001 | 8 | 240 | 240 | 241 | 240 < 241 but not < 240 |

Detail CSV: `reconcile_1012_vs_1015_extra_landmarks.csv`.  
Start-date table: `data/processed/clinical/obs_vs_teb2_start_dates.csv`.

Metrics stay essentially the same (AUC ≈ 0.704–0.705; 159 positive landmarks).

## 5. What we are NOT doing

No search for better W/H/granularity; no re-tuning; no new IC analyses — only confirmation and files above.
