# Rolling-burden alarm (`03_analysis/alarm`)

Prospective **rolling-burden** alarm built on LDA / DCABP topic probabilities
(unfavourable topics **3 + 4 + 5** → `pUF`). Downstream of the VQ-VAE → LDA
pipeline; not a VQ-VAE training step.

## Inputs

| Path | Role |
|------|------|
| `data/processed/DCABPs/topics_probs.xlsx` | Monthly topic probabilities (tabla maestra / master) |
| `data/processed/clinical/Subjects_data.xlsx` | Clinical FU (`S_RECORD_id`, `PD_event`, `Obs_time`, …) |
| `results/alarm/tables/daily_sliding_puf.csv` | Daily sliding pUF (built once; reusable) |
| `results/alarm/tables/collapsed_puf_*.csv` | Collapsed block pUF (weekly / biweekly) |

## Definitions (short)

| Term | Meaning |
|------|---------|
| **pUF** | P(topic 3)+P(topic 4)+P(topic 5) |
| **Sampled / sliding** | Daily LDA on last 30 days; burden = rolling mean over **W** days |
| **Collapsed** | One LDA per non-overlapping block (7 / 14 / 30 d); own burden |
| **W** | Lookback (burden window) |
| **H** | Horizon for PD label after the decision |

Paper operating point: **monthly collapsed**, W ≈ 3 months, H ≈ 4 months, θ ≈ 1.61 (sum scale).

**Event/censor clock (paper / MAIN):** `Subjects_data.Obs_time` → **1015** decision points  
(`results/alarm/tables/decision_points_W3_H4.csv`).  
Alternate eB2 clock (`t_evento_eb2`) → 1012; see [`results/alarm/README.md`](../../../results/alarm/README.md).

## Quick start

```bash
# Prefer: orchestrated pipeline (reuse existing pUF tables)
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda

# Manuscript monthly replication only
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --monthly-only

# Rebuild collapsed pUF then sweep
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --rebuild-collapsed --skip-lda

# Also write example patient trajectories
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda --with-trajectories
```

## Scripts

| Script | Role |
|--------|------|
| `run_alarm_pipeline.py` | Orchestrator (monthly → sweep → missingness → optional plots) |
| `rolling_burden_alarm.py` | Core monthly metrics / clinical paths |
| `sliding_burden_alarm.py` | Sampled + collapsed decision points |
| `build_daily_sliding_puf.py` | Build daily sliding pUF (LDA windows) |
| `build_collapsed_puf.py` | Build collapsed weekly / biweekly pUF |
| `run_granularity_sweep.py` | Granularity × (W, H) sweep + monthly replication |
| `analyze_missingness_and_wh_grid.py` | Funnel + AUC heatmaps vs W/H |
| `plot_patient_trajectories.py` | Paper-style trajectories |
| `plot_granularity_trajectories.py` | Sampled vs collapsed trajectories |

## Outputs

All under `results/alarm/`:

| Path | Content |
|------|---------|
| `tables/decision_points_W3_H4.csv` | **MAIN** landmarks (Obs_time, n=1015) |
| `tables/alarm_metrics_W3_H4.csv` | **MAIN** monthly paper metrics |
| `tables/decision_points_W3_H4_t_evento_eb2.csv` | Alternate eB2 clock (n=1012) |
| `tables/granularity_sweep_summary.csv` | AUC / Sens / Spec by granularity |
| `tables/wh_grid_auc.csv` | W×H grid |
| `tables/missingness_funnel.csv` | Eligible patients / decision points |
| `figures/roc_*.png` | ROC curves |
| `figures/summary_granularities_*.png` | Granularity comparison |
| `granularidad_alarma.html` | Human-readable report |

See also [`results/alarm/README.md`](../../../results/alarm/README.md) and notebooks under
[`notebooks/revision_experiments/`](../../../notebooks/revision_experiments/).
