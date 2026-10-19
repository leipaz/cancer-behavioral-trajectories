# Revision data (canonical only)

Git tracks **only** the definitive inputs for analyses. Local QC dumps stay untracked (see `.gitignore`).

## Definitive daily

| File | Role |
|------|------|
| `daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv` | **Main daily** for analyses: CNIO date ranges → dedup `(user, date)` → origin validity (HR −1, sleep_start >24h) → VQ-VAE `CLIP_INFO` → NaN |
| `daily_summary_oficial_filtrado_rangos_CNIO_enriched_vqclip.csv` | Same + enriched columns (unlocks, app categories, sleep stages) for univariate |

## Embeddings

| File | Role |
|------|------|
| `user_embeddings_revision_a0.csv` | Per-patient day-type sequences (VQ-VAE `a0`) from the definitive daily; also copied to `data/processed/lda/user_embeddings_from_pkl.csv` |

## Matching / univariate

`univariate/` — matching Excel, window cohorts, QC helpers (see `INFORME_TRAZABILIDAD_MATCHING_UNIVARIATE.md`).

## Do not commit

Backups (`*backup*`, `*copia*`), DB1/DB2/DB3 dumps, PMP_lock copies, `eb2prod_*` intermediates, `dias_*` / `fechas_*` QC tables, `*.pkl`, `missingness_daily/`, etc. They stay local via `.gitignore`.
