# VQ-VAE → LDA pipeline (notebooks)

Run these notebooks **in order** from the repository root (or with this folder as the Jupyter working directory).

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `daily_to_profiles.ipynb` | VQ-VAE inference on daily summaries → `data/output_vq_vae/profiles_per_sample_oncology.pkl` |
| 2 | `profiles_pkl_to_csv.ipynb` | PKL → `data/processed/lda/user_embeddings_from_pkl.csv` |
| 3 | `profiles_to_dcabp.ipynb` | LDA model, topic plots, clinical merge, PD vs no-PD figures |
| 4 | `profile_decodification.ipynb` | Decode top-10 day-types per topic to behavioral features |
| 5 | `lda_dcabp_evolution.ipynb` | Monthly / daily DCABP evolution heatmaps and per-patient figures |

**Environment:** create a virtualenv, activate it, and run `pip install -r requirements.txt` from the repository root (see main [`README.md`](../../README.md)).

Scripts equivalent to several steps live under `scripts/` — see [`scripts/README.md`](../../scripts/README.md).
