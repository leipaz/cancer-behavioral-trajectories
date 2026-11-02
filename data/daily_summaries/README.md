# Daily summaries

## By clinical DB window (current)

Dailys **filtered to CNIO dates** for each database. One folder per DB:

| Path | Source dates | Role |
|------|----------------|------|
| [`DB1/daily_summary_checked.csv`](DB1/daily_summary_checked.csv) | DB1 Oct 2 2026 | Daily filtered to DB1 window |
| [`DB2/daily_summary_checked.csv`](DB2/daily_summary_checked.csv) | DB2 Oct 2 2026 | Daily filtered to DB2 window |
| [`DB2/daily_summary_from_study_start_to_fin_DB2.csv`](DB2/daily_summary_from_study_start_to_fin_DB2.csv) | study start → Fin_datos_DB2 | **Extended** (DB1∪DB2) for topics anchored on inicio — see `results/by_db/DB2/PASOS_topics_anclados_inicio.md` |
| [`DB3/daily_summary_checked.csv`](DB3/daily_summary_checked.csv) | DB3 Oct 2 2026 | Daily filtered to DB3 window |

Clinical date tables (original CNIO Excel) live separately in
[`../processed/clinical/DB1|DB2|DB3/`](../processed/clinical/).

## Legacy (root of this folder)

Older single-window files (`daily_datesdb1*.csv`, `oncology_*`, zips). Prefer `DB1/` / `DB2/` / `DB3/` for new work.
