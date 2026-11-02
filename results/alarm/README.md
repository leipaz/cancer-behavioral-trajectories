# Alarm results

Canonical outputs of the **rolling-burden alarm** pipeline
([`scripts/03_analysis/alarm/`](../../scripts/03_analysis/alarm/)).

## Clinical clock

Only [`data/revision/clock_bbdd.xlsx`](../../data/revision/clock_bbdd.xlsx)
(`fecha_inicio` / `fecha_fin` / `evento` → `t_evento = fin − inicio`).

## Layout (tracked)

```
results/alarm/
├── alarm_granularity_report.{md,html,pdf}   # main report (θ = 1.60)
├── README.md
├── figures/                                 # ROC, trajectories, W×H (report embeds)
├── analysis_figures/                        # delivery panel + simplified traj
├── tables/                                  # small metrics / decision points / traj series
└── runs/Evento_DB1_clock/                   # paper W3/H4 + granularity + PROVENANCE
```

Large regenerable series (`daily_sliding_puf`, `collapsed_puf_*`, `topic_probs_granularities`,
daily landmarks) and PDF fonts live in `results/alarm/_local/` (**gitignored**).

## Operating threshold

Monthly collapsed baseline: **θ = 1.60** (sum of last W=3 monthly pUF).
Final clinical threshold TBD.

## Regenerate

```bash
python scripts/03_analysis/alarm/run_paper_alarm.py \
  --clock data/revision/clock_bbdd.xlsx --run-id Evento_DB1_clock --theta 1.60

python scripts/03_analysis/alarm/write_granularity_report.py --run-id Evento_DB1_clock
```

Full pipeline / plots: [`scripts/03_analysis/alarm/README.md`](../../scripts/03_analysis/alarm/README.md).
