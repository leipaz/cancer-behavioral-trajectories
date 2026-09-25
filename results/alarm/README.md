# Alarm results

Outputs of the **rolling-burden alarm** pipeline
([`scripts/03_analysis/alarm/`](../../scripts/03_analysis/alarm/)).

```
results/alarm/
├── tables/                  # metrics, decision points, pUF series
├── figures/                 # ROC, granularity summary, trajectories, W×H
├── alarm_granularity_report.html  # analysis report (open in browser)
└── alarm_granularity_report.md
```

## Which file is the main result?

**Primary (use this):** event/censor time = `Subjects_data.Obs_time`.

| Role | File | N decision points |
|------|------|-------------------|
| **MAIN** | [`tables/decision_points_W3_H4.csv`](tables/decision_points_W3_H4.csv) | **1015** |
| **MAIN metrics** | [`tables/alarm_metrics_W3_H4.csv`](tables/alarm_metrics_W3_H4.csv) | AUC ≈ 0.704 (θ ≈ 1.61) |

Same content also saved as `*_Obs_time.csv` for clarity when comparing clocks.

### Alternative clock (not main)

| Role | File | N |
|------|------|---|
| Backup / eB2 clock | [`tables/decision_points_W3_H4_t_evento_eb2.csv`](tables/decision_points_W3_H4_t_evento_eb2.csv) | **1012** |

Difference **1015 − 1012 = 3** landmarks that pass `month×30 < Obs_time` but fail `month×30 < t_evento_eb2` (patients **53001** month 8, **72001** month 7, **73001** month 8), because eB2 start is later than HDM start for those IDs.

Related clinical date table: [`data/processed/clinical/obs_vs_teb2_start_dates.csv`](../../data/processed/clinical/obs_vs_teb2_start_dates.csv).

## Key tables

| File | Content |
|------|---------|
| `tables/decision_points_W3_H4.csv` | **MAIN** monthly landmarks (Obs_time) |
| `tables/alarm_metrics_W3_H4.csv` | **MAIN** monthly metrics |
| `tables/decision_points_W3_H4_Obs_time.csv` | Copy of MAIN (explicit name) |
| `tables/alarm_metrics_W3_H4_Obs_time.csv` | Copy of MAIN metrics |
| `tables/decision_points_W3_H4_t_evento_eb2.csv` | Alternate: eB2 event clock (1012) |
| `tables/granularity_sweep_summary.csv` | Sampled vs collapsed by granularity |
| `tables/wh_grid_auc.csv` | Lookback × horizon AUC grid |
| `tables/missingness_funnel.csv` | Eligibility / decision-point counts |
| `tables/daily_sliding_puf.csv` | Daily sliding pUF (large; regenerable) |
| `tables/collapsed_puf_weekly.csv` / `*_biweekly.csv` | Collapsed pUF |

## Report

| File | Content |
|------|---------|
| [`alarm_granularity_report.md`](alarm_granularity_report.md) | Full write-up (MAIN = Obs_time, n=1015) |
| [`alarm_granularity_report.html`](alarm_granularity_report.html) | Same report (open in browser) |

Regenerate HTML from markdown:

```bash
.venv/bin/python -c "
from pathlib import Path
import markdown
from datetime import date
md = Path('results/alarm/alarm_granularity_report.md').read_text()
body = markdown.markdown(md, extensions=['tables','fenced_code'])
Path('results/alarm/alarm_granularity_report.html').write_text(
  '<!DOCTYPE html><html><head><meta charset=utf-8><title>Alarm report</title></head><body>'
  + body + f'<p>Generated {date.today()}</p></body></html>')
"
```

## Regenerate

```bash
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda
```

See [`scripts/03_analysis/alarm/README.md`](../../scripts/03_analysis/alarm/README.md).
