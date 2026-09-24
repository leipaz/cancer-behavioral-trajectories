# Alarm results

Outputs of the **rolling-burden alarm** pipeline
([`scripts/03_analysis/alarm/`](../../scripts/03_analysis/alarm/)).

```
results/alarm/
├── tables/                  # metrics, decision points, pUF series
├── figures/                 # ROC, granularity summary, trajectories, W×H
├── granularidad_alarma.html # revision report (open in browser)
└── granularidad_alarma.md
```

## Key tables

| File | Content |
|------|---------|
| `tables/alarm_metrics_W3_H4.csv` | Monthly paper replication |
| `tables/granularity_sweep_summary.csv` | Sampled vs collapsed by granularity |
| `tables/wh_grid_auc.csv` | Lookback × horizon AUC grid |
| `tables/missingness_funnel.csv` | Eligibility / decision-point counts |
| `tables/daily_sliding_puf.csv` | Daily sliding pUF (large; regenerable) |
| `tables/collapsed_puf_weekly.csv` / `*_biweekly.csv` | Collapsed pUF |

## Regenerate

```bash
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda
```

See [`scripts/03_analysis/alarm/README.md`](../../scripts/03_analysis/alarm/README.md).
