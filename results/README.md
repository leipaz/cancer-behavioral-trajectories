# Results

Outputs from scripts. Method-specific folders (`lda`, `vq-vae`) are used for modeling pipelines; **univariate** outputs are shared under `univariate/`.

## Layout

```
results/
├── univariate/
│   ├── figures/
│   └── tables/
├── lda/
│   ├── figures/
│   ├── tables/
│   └── models/
├── vq-vae/
│   ├── figures/
│   ├── tables/
│   └── models/
└── alarm/
    ├── figures/          # ROC, granularity, trajectories, W×H
    ├── tables/           # metrics, pUF, decision points
    ├── alarm_granularity_report.html
    └── README.md
```

| Path | Contents |
|------|----------|
| `univariate/figures/` | KDE and histogram comparisons (E.PD vs no-E.PD) |
| `univariate/tables/` | `variable_summary_full_stats_EPD.csv` |
| `lda/figures/` | LDA top terms, topic vs PD event, decoded profile plots |
| `lda/tables/` | LDA top-10 terms per topic |
| `alarm/` | Rolling-burden alarm (see [`alarm/README.md`](alarm/README.md)) |
| `<method>/figures/` | Paper and supplementary plots per method |
| `<method>/tables/` | Exported tables (CSV, LaTeX, etc.) |
| `<method>/models/` | Saved models (`.rds`, `.pkl`, checkpoints, etc.) |

Scripts in `scripts/04_figures/<method>/` should write to `results/<method>/figures/`.
Alarm scripts live under `scripts/03_analysis/alarm/` and write to `results/alarm/`.

Large or regenerable files may stay out of git; document how to regenerate them in the main `README.md`.
