# Remote monitoring of behavioral–physiologic trajectories predict progression and reflect biologic host state in metastatic cancer in women

This repository contains the code, analysis-ready data, and workflows used in the study on remote monitoring of behavioral and physiologic trajectories in patients with metastatic cancer.

## Authors

Leire Paz<sup>1,\*</sup>, Leonardo Garma<sup>2,\*</sup>, Sonia Pernas<sup>3,4</sup>, Juan Antonio Guerra<sup>5</sup>, Rosario García-Campelo<sup>6</sup>, Jacobo Rogado<sup>7</sup>, David Vicente-Baz<sup>8</sup>, Josefa Terrasa<sup>9</sup>, Begoña Bermejo<sup>10</sup>, Ruth Vera<sup>11</sup>, Santiago González Santiago<sup>12</sup>, Sandra Gallach<sup>13,14,15</sup>, Berta Nasarre<sup>5</sup>, Bartomeu Fullana<sup>3</sup>, Desirée Jiménez<sup>2</sup>, Cristina Reboredo-Rendo<sup>6</sup>, Oscar Padilla<sup>3</sup>, Rodrigo Oliver<sup>1</sup>, Cristina Simarro<sup>8</sup>, Antonia Perelló<sup>9</sup>, Berta Hernández-Martín<sup>10</sup>, Aída Morillas<sup>2</sup>, Silvana Mourón<sup>2</sup>, María J. Bueno<sup>2</sup>, Antonio Lopez<sup>2</sup>, Pablo Martínez Olmos<sup>1</sup>, Silvia Calabuig<sup>13,14,15</sup>, Ramon Colomer<sup>7,16</sup>, Antonio Artes<sup>1</sup>, Miguel Quintela-Fandino<sup>2,7</sup>

## Affiliations

1. Communications and Signal Theory, Escuela Politécnica Superior, Universidad Carlos III, Leganés (Madrid), Spain
2. Breast Cancer Clinical Research Unit, CNIO – Spanish National Cancer Research Center, Madrid, Spain
3. Breast Cancer Unit, Department of Medical Oncology, Institut Català d’Oncologia, L’Hospitalet de Llobregat (Barcelona), Spain
4. Institut d’Investigació Biomèdica de Bellvitge (IDIBELL), L’Hospitalet de Llobregat (Barcelona), Spain
5. Medical Oncology, Hospital Universitario de Fuenlabrada, Fuenlabrada (Madrid), Spain
6. Medical Oncology, Complejo Hospitalario Universitario de A Coruña – CHUAC, A Coruña, Spain
7. Medical Oncology, Hospital Universitario de La Princesa, Madrid, Spain
8. Medical Oncology, Hospital Universitario Virgen de la Macarena, Sevilla, Spain
9. Medical Oncology, Hospital Universitari Son Espases, Palma, Spain
10. Medical Oncology, Hospital Clínico Universitario, Valencia, Spain
11. Medical Oncology, Hospital Universitario de Navarra, Navarra, Spain
12. Medical Oncology, Hospital San Pedro de Alcántara – Complejo Hospitalario Universitario de Cáceres, Cáceres, Spain
13. Department of Pathology, Universitat de València, Valencia, Spain
14. Centro de Investigación Biomédica en Red Cáncer (CIBERONC), Madrid, Spain
15. Molecular Oncology Laboratory, Fundación Investigación Hospital General Universitario de Valencia, Valencia, Spain
16. Roche Endowed Chair of Precision and Personalized Medicine, Universidad Autónoma de Madrid, Madrid, Spain

## Repository structure

```
.
├── data/
│   ├── daily_summaries/          # Per-patient daily summaries (VQ-VAE model input)
│   ├── processed/
│   │   ├── clinical/             # Subjects_data.xlsx (PD dates, covariates)
│   │   └── lda/                    # LDA inputs / topic tables (e.g. eB2 cohort CSV)
│   ├── output_vq_vae/            # VQ-VAE inference outputs (profiles, decoded embeddings)
│   ├── output_lda/               # LDA-side exports
│   ├── metadata/                 # Variable dictionary, cohort maps
│   └── raw/                      # Source daily summaries and clinical tables
├── models/
│   ├── vq-vae/                   # vqvae_a0.pt, scaler, …
│   └── lda/                      # LDA_model_vdec25.gensim, dictionary_LDA_vdec25.dict
├── notebooks/
│   ├── README.md
│   └── vq-vae_lda_pipeline/      # Main 5-notebook pipeline (see below)
├── scripts/
│   ├── vqvae/                    # VQ-VAE package (model, inference, plots)
│   ├── 01_preprocess/            # Daily-summary cleaning; eB2 column rename
│   ├── 02_univariate_analysis/   # E.PD vs no-E.PD univariate plots
│   ├── 03_analysis/              # lda/, vq-vae/ training and downstream
│   └── 04_figures/
├── results/
│   ├── lda/                      # figures/, tables/
│   ├── univariate/
│   └── vq-vae/
├── requirements.txt
└── LICENSE
```

More detail per folder: [`data/README.md`](data/README.md), [`scripts/README.md`](scripts/README.md), [`results/README.md`](results/README.md).

---

## Analysis guide

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open notebooks under `notebooks/vq-vae_lda_pipeline/` with the same interpreter (e.g. select the `.venv` kernel in Jupyter).

### Main pipeline (notebooks)

End-to-end workflow: **daily wearable summaries → VQ-VAE day-type profiles → LDA topics (DCABPs) → behavioral decoding → temporal evolution**.

All notebooks live in [`notebooks/vq-vae_lda_pipeline/`](notebooks/vq-vae_lda_pipeline/). Run them **in order** (repo root is detected automatically from that folder). See also [`notebooks/vq-vae_lda_pipeline/README.md`](notebooks/vq-vae_lda_pipeline/README.md).

```mermaid
flowchart LR
  A[daily_summaries] --> B[VQ-VAE]
  B --> C[profiles PKL]
  C --> D[user embeddings CSV]
  D --> E[LDA + clinical]
  E --> F[topic decode]
  E --> G[DCABP evolution]
```

| Step | Notebook | What it does | Main inputs | Main outputs |
|------|----------|--------------|-------------|--------------|
| 1 | [`daily_to_profiles.ipynb`](notebooks/vq-vae_lda_pipeline/daily_to_profiles.ipynb) | Runs trained **VQ-VAE** on each patient’s daily summaries; assigns a discrete **day-type profile** (embedding ID) per day | `data/daily_summaries/oncology_daily_summary_model_input.csv`, `models/vq-vae/vqvae_a0.pt` | `data/output_vq_vae/profiles_per_sample_oncology.pkl`, `decoded_embedding_vectors_a0.pkl` |
| 2 | [`profiles_pkl_to_csv.ipynb`](notebooks/vq-vae_lda_pipeline/profiles_pkl_to_csv.ipynb) | Extracts embedding sequences for LDA (`model_type=a0`, window `n=30`) | profiles PKL | `data/processed/lda/user_embeddings_from_pkl.csv` |
| 3 | [`profiles_to_dcabp.ipynb`](notebooks/vq-vae_lda_pipeline/profiles_to_dcabp.ipynb) | Loads **LDA** model and dictionary; topic grid; merges **clinical** data; compares topic distributions by progression status | `models/lda/LDA_model_vdec25.gensim`, `dictionary_LDA_vdec25.dict`, `data/processed/clinical/Subjects_data.xlsx`, eB2 table `data/processed/lda/PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv` | `results/lda/tables/lda_topics_top10.csv`, figures under `results/lda/figures/` |
| 4 | [`profile_decodification.ipynb`](notebooks/vq-vae_lda_pipeline/profile_decodification.ipynb) | Maps each LDA topic’s top-10 profile tokens back to **behavioral feature vectors** (VQ-VAE decoder) | top-10 table, `decoded_embedding_vectors_a0.pkl` | decoded profile tables/plots in `results/lda/` |
| 5 | [`lda_dcabp_evolution.ipynb`](notebooks/vq-vae_lda_pipeline/lda_dcabp_evolution.ipynb) | **Predominant DCABP** in 30-day blocks; cohort heatmaps; per-patient **DCABP evolution** (daily sliding windows) with progression marker | eB2 CSV, LDA model, profiles / embeddings | `results/lda/figures/predominant_topics_per_month.png`, `results/lda/figures/topic_evolution/topic_evolution_<id>.png` |

**Terminology:** *DCABP* = digitally characterized activity-based pattern (LDA topic 0–5).

**Script equivalents** (batch / CI-friendly): univariate work in `scripts/02_univariate_analysis/lda/`; LDA train/assign/merge/plots in `scripts/03_analysis/lda/` (`run_lda_pipeline.py`, `run_decode_profiles.py`, `run_monthly_topics_heatmap.py`). Shared helpers include `scripts/03_analysis/lda/monthly_topics_utils.py` (used by notebook 5).

### Other notebook folders

Add collaborator-specific analyses as new subfolders under `notebooks/` (e.g. `notebooks/<name>/`) without changing the main pipeline folder.

---

## Data and privacy

Raw patient-identifiable data are kept under `data/raw/` and are not part of the published dataset. Processed, anonymized tables in `data/processed/` and `data/daily_summaries/` comply with applicable regulations and informed consent.

## License

Code is under the [Apache License 2.0](LICENSE). Data may have additional use conditions; see `data/README.md`.

## Citation

If you use this material, please cite the manuscript (bibliographic reference pending publication).
