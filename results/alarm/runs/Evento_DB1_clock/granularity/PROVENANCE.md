# Alarm provenance — granularity_experiment

- Generated (UTC): `2026-10-21T09:54:04.537017+00:00`
- Clock source: **user BBDD only** (`/export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories/data/revision/clock_bbdd.xlsx`)
- Audit dates: `fecha_inicio` / `fecha_fin` / `evento`
- Rule: `t_evento = (fecha_fin − fecha_inicio).days` (paper-style; uncapped)
- If `evento=1`, `fecha_fin` is progression date; if `0`, censor at `fecha_fin`

## Clock summary

- n patients: 178
- n events (evento=1): 60
- n censored (evento=0): 118
- t_evento days: min=15, median=276, max=939

## Columns

- `id`, `fecha_inicio`, `fecha_fin`, `evento` → `t_evento`

## Run parameters

- **w_days**: `[60, 90, 120]`
- **h_days**: `[90, 120, 180]`
- **focus_W**: `90`
- **focus_H**: `120`
- **n_summary_rows**: `63`
- **out_dir**: `/export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories/results/alarm/runs/Evento_DB1_clock/granularity`
