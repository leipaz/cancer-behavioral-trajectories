# Data

Only **processed, anonymized** data ready for analysis are versioned here. Raw or identifiable data should remain in `data/raw/` (local, not in git) or outside the repository.

## Structure

| Path | Expected contents |
|------|-------------------|
| `processed/behavioral/` | Behavioral monitoring series or summaries (e.g. activity, sleep, steps) aligned by patient and time window |
| `processed/physiologic/` | Physiologic signals or derivatives (e.g. heart rate, HRV) at the same granularity |
| `processed/clinical/` | Progression, stage, treatments, host-state biomarkers, event dates |
| `metadata/` | Variable dictionary (`variables.csv` or similar), cohort definition, code maps |

Optional method-specific derivatives (when not shared across pipelines):

| Path | Expected contents |
|------|-------------------|
| `processed/lda/` | Inputs or features used only by the LDA workflow |
| `processed/vq-vae/` | Inputs or tensors used only by the VQ-VAE workflow |

Shared cohort tables can stay in `behavioral/`, `physiologic/`, and `clinical/`.

## Suggested conventions

- File names: `snake_case`, with a cohort or study prefix if applicable (`cohort_behavioral_daily.parquet`).
- Include a consistent anonymized identifier column (`patient_id`) across folders.
- Document each added file in this README: source, transformation, version, date.

## Planned files (fill in when uploading data)

<!-- Example:
- `processed/behavioral/daily_features.parquet` — daily aggregates per patient
- `processed/physiologic/hrv_windows.parquet` — HRV in 5-minute windows
- `processed/clinical/progression_events.csv` — RECIST or other progression dates
- `metadata/variable_dictionary.csv` — name, unit, description
-->

_(No data files yet.)_

## Access

If data cannot be shared openly, describe here how to request access (ethics committee, data transfer agreement, contact).
