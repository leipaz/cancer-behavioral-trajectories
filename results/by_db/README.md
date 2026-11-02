# Results by clinical DB window

One folder per analysis case when outputs grow beyond a single Excel.

```
results/by_db/
├── DB1/          # analyses scoped to DB1 dates / daily
├── DB2/
└── DB3/
```

Alarm runs already live under `results/alarm/runs/<run_id>/` (e.g. `Evento_DB1_clock`).
For new DB2/DB3 alarm or topics work, prefer `results/by_db/DB2/...` or a dedicated
`results/alarm/runs/<run_id>/` so each case stays self-contained (tables + figures + notes).
