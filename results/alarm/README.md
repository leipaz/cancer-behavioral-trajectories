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

## Key tables

| File | Content |
|------|---------|
| `tables/decision_points_W3_H4.csv` | Monthly landmarks (W=3, H=4) |
| `tables/alarm_metrics_W3_H4.csv` | Monthly metrics |
| `tables/granularity_sweep_summary.csv` | Sampled vs collapsed by granularity |
| `tables/wh_grid_auc.csv` | Lookback × horizon AUC grid |
| `tables/missingness_funnel.csv` | Eligibility / decision-point counts |
| `tables/daily_sliding_puf.csv` | Daily sliding pUF (large; regenerable) |
| `tables/collapsed_puf_weekly.csv` / `*_biweekly.csv` | Collapsed pUF |

## Report

| File | Content |
|------|---------|
| [`alarm_granularity_report.md`](alarm_granularity_report.md) | Full write-up |
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
