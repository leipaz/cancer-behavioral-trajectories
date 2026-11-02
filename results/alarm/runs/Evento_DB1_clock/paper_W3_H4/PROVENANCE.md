# Alarm provenance — paper_alarm_W3_H4

- Generated (UTC): `2026-10-21T13:20:28.126691+00:00`
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

- **master**: `/export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories/data/processed/DCABPs/topics_probs.xlsx`
- **W_months**: `3`
- **H_months**: `4`
- **theta**: `1.60`
- **n_landmarks**: `1016`
- **n_patients**: `116`
- **AUROC**: `0.7001267143709005`
- **out_dir**: `/export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories/results/alarm/runs/Evento_DB1_clock/paper_W3_H4`
