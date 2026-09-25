# Data

Study datasets for this repository. The **VQ-VAE → LDA** pipeline ([`notebooks/vq-vae_lda_pipeline/`](../notebooks/vq-vae_lda_pipeline/)) uses the folders marked below; other paths are additional processed study data.

## Layout

```
data/
├── daily_summaries/              # pipeline
├── output_vq_vae/                # pipeline
├── processed/
│   ├── clinical/                 # pipeline
│   ├── lda/                      # pipeline
│   ├── DCABPs/
│   ├── DNA/
│   ├── RNA/
│   ├── data_completeness/
│   ├── liquid_biopsy/
│   └── wearables_by_period/
└── references/
    ├── genesets/
    └── xCell/
```

---

## Pipeline data (detail)

### `daily_summaries/`

Daily behavioral summaries for the oncology cohort (~175 patients, ~51k patient-days).

| File | Description |
|------|-------------|
| `oncology_patient_ids.csv` | Cohort list: one column `user` (anonymized patient ID). |
| `oncology_patient_ids.txt` | Same IDs, one per line. |
| `oncology_daily_summary_raw.csv` | Full daily table: wearable / app / sleep / location / audio / circadian columns per `user` and `date_time`. |
| `oncology_daily_summary_model_input.csv` | **VQ-VAE input**: model feature subset plus `user`, `date_time`, `service`. |

**Notebook:** step 1 — [`daily_to_profiles.ipynb`](../notebooks/vq-vae_lda_pipeline/daily_to_profiles.ipynb).

### `processed/clinical/`

| File | Description |
|------|-------------|
| `Subjects_data.xlsx` | Per-patient clinical table (`id`, `Date_start_HDM`, `PD_event`, `Obs_time`, …). |
| `master_table_monthly_topics.xlsx` | Monthly topic probabilities + clinical columns (master table). |
| `obs_vs_teb2_start_dates.csv` | HDM vs eB2 start dates and Obs_time vs `t_evento_eb2`. |
| `date_comparison_subjects_vs_maestra.csv` | Subjects vs master table date check. |

**Notebook:** step 3 — [`profiles_to_dcabp.ipynb`](../notebooks/vq-vae_lda_pipeline/profiles_to_dcabp.ipynb).

### `processed/lda/`

| File | Description |
|------|-------------|
| `user_embeddings_from_pkl.csv` | Per patient: `user_id`, `embedding_ids` (VQ-VAE day-types, model `a0`, window 30). |
| `PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv` | eB2 cohort: day-type sequences, LDA topic probabilities, predominant topic, clinical timing (English column names). |

**Notebooks:** step 2 — [`profiles_pkl_to_csv.ipynb`](../notebooks/vq-vae_lda_pipeline/profiles_pkl_to_csv.ipynb); steps 3 and 5 use the eB2 table.

### `output_vq_vae/`

| File | Description |
|------|-------------|
| `profiles_per_sample_oncology.pkl` | Per-patient profile sequences (embedding IDs over time). |
| `decoded_embedding_vectors_a0.pkl` | Decoded behavioral vectors per embedding index (model `a0`). |

**Notebooks:** step 1 writes the profiles PKL; step 4 — [`profile_decodification.ipynb`](../notebooks/vq-vae_lda_pipeline/profile_decodification.ipynb).

---

## Conventions

- Patient key: `user` in daily summaries, `user_id` or `id` in LDA/clinical tables.
- Prefer `snake_case` file names.

## Access

If additional data cannot be shared openly, describe here how to request access (ethics committee, data transfer agreement, contact).
