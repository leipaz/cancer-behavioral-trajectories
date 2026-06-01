# Data

Processed, anonymized datasets used by the VQ-VAE → LDA pipeline ([`notebooks/vq-vae_lda_pipeline/`](../notebooks/vq-vae_lda_pipeline/)). Patient identifiers are numeric codes (`user` / `id`).

## Layout

```
data/
├── daily_summaries/       # Wearable daily summaries (oncology cohort)
├── processed/
│   ├── clinical/          # Progression dates and covariates
│   └── lda/               # LDA inputs and topic–clinical tables
├── output_vq_vae/         # VQ-VAE inference outputs
├── output_lda/            # Reserved for LDA-side exports
├── metadata/              # Reserved (variable dictionary, cohort maps)
└── raw/                   # Source tables with wider feature set (when used locally)
```

---

## `daily_summaries/`

Daily behavioral summaries for the oncology cohort (~175 patients, ~51k patient-days).

| File | Description |
|------|-------------|
| `oncology_patient_ids.csv` | Cohort list: one column `user` (anonymized patient ID). |
| `oncology_patient_ids.txt` | Same IDs, one per line. |
| `oncology_daily_summary_raw.csv` | Full daily table from preprocessing: many wearable / app / sleep / location / audio / circadian columns per `user` and `date_time`. |
| `oncology_daily_summary_model_input.csv` | **VQ-VAE input**: subset of columns used by the model (`sleep_start`, `location_distance`, `location_time_home`, `sleep_duration`, `activity_walking`, `app_usage_total`, `location_clusters_count`, `steps_steps_total`, `weekend`, `practiced_sport`, plus `user`, `date_time`, `service`). |

**Notebook:** step 1 — [`daily_to_profiles.ipynb`](../notebooks/vq-vae_lda_pipeline/daily_to_profiles.ipynb).

---

## `processed/clinical/`

| File | Description |
|------|-------------|
| `Subjects_data.xlsx` | Per-patient clinical table merged in the LDA notebooks. Key fields used in analysis include `id`, `Date_start_HDM`, `PD_event` (progression yes/no), and `Obs_time`. |

**Notebook:** step 3 — [`profiles_to_dcabp.ipynb`](../notebooks/vq-vae_lda_pipeline/profiles_to_dcabp.ipynb).

---

## `processed/lda/`

| File | Description |
|------|-------------|
| `user_embeddings_from_pkl.csv` | One row per patient: `user_id`, `embedding_ids` (list of VQ-VAE day-type IDs, model `a0`, sequence length 30). Built from the profiles PKL. |
| `PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv` | eB2 cohort with day-type sequences, LDA topic probabilities (`topic_0`–`topic_5`), predominant topic, and clinical timing columns in English (`date_start_hdm`, `pd_event`, `pd_event_diff`, `obs_time_final`, `early_pd`, etc.). |

**Notebooks:** step 2 — [`profiles_pkl_to_csv.ipynb`](../notebooks/vq-vae_lda_pipeline/profiles_pkl_to_csv.ipynb); steps 3 and 5 use the eB2 table.


---

## `output_vq_vae/`

Outputs of VQ-VAE inference on daily summaries.

| File | Description |
|------|-------------|
| `profiles_per_sample_oncology.pkl` | Per-patient profile sequences (embedding IDs over time) for all model variants stored in the pickle. |
| `decoded_embedding_vectors_a0.pkl` | Decoded behavioral feature vectors per embedding index (model `a0`), used to interpret LDA topics. |

**Notebooks:** step 1 writes the profiles PKL; step 4 — [`profile_decodification.ipynb`](../notebooks/vq-vae_lda_pipeline/profile_decodification.ipynb) uses the decoded vectors.

---

## `output_lda/` / `metadata/` / `raw/`

| Path | Status |
|------|--------|
| `output_lda/` | Placeholder for future LDA exports not stored under `processed/lda/`. |
| `metadata/` | Placeholder for a variable dictionary or cohort codebook. |
| `raw/` | Intended for identifiable or pre-cleaning source exports (e.g. full daily-summary lock files). Not required to run the published notebooks if `daily_summaries/` and `processed/` are present. |

---

## Conventions

- Use a consistent patient key: `user` in daily summaries, `user_id` or `id` in LDA/clinical tables (notebooks align these on merge).
- Prefer `snake_case` file names.
- When adding a file, extend the tables above with a one-line description of source and role in the pipeline.

## Access

If additional data cannot be shared openly, describe here how to request access (ethics committee, data transfer agreement, contact).
