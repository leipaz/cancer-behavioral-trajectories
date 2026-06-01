# Scripts

Pipeline organized by **stage** (numeric prefix) and **method** (`lda`, `vq-vae`).

## Layout

```
scripts/
├── utils/              # Shared helpers (used by multiple methods)
├── 01_preprocess/
│   ├── lda/            # e.g. preprocess_daily_summaries.py
│   └── vq-vae/
├── 02_univariate_analysis/
│   ├── lda/
│   └── vq-vae/
├── 03_analysis/
│   ├── lda/
│   └── vq-vae/
└── 04_figures/
    ├── lda/
    └── vq-vae/
```

| Stage | Purpose |
|-------|---------|
| `01_preprocess/` | Unify and clean the daily summary dataset |
| `02_univariate_analysis/` | Univariate analysis (distributions, associations, QC plots) |
| `03_analysis/` | LDA training, topic assignment, downstream inference |
| `04_figures/` | Manuscript figures → `results/<method>/figures/` |

## `01_preprocess/lda/preprocess_daily_summaries.py`

**Scope:** This script only **unifies and cleans the daily summary dataset** (wearable / Garmin daily summaries). It is not the full LDA or VQ-VAE pipeline.

- **Input:** `data/raw/daily_summary_PMP_lock_2025_09_11_with_extra_garmin.csv`
- **Output:** `data/raw/daily_summary_eb2prod_dic25_enriched_filtered.csv`
- **Cleaning:** heart-rate sentinel values (`-1` → missing), invalid `sleep_start` values (> 24 h → missing)

Clinical merges, drug covariates, and user embeddings belong in **`02_univariate_analysis/`** or later, not in this script.

## `02_univariate_analysis/lda/`

**Univariate analysis** of daily summaries vs Early PD (E.PD vs no-E.PD), first 15 days vs previous 15 days. Requires output of `01_preprocess/lda/preprocess_daily_summaries.py` and the matching table `PD_noPD_Matching_fechas_finales_dic25.csv`.

| Script | Role |
|--------|------|
| `prepare_cohorts.py` | ID overlap QC, build window cohorts |
| `plot_density_comparison.py` | KDE density plots |
| `plot_histogram_comparison.py` | Discrete histogram (count) plots |
| `export_summary_table.py` | Full summary statistics CSV |
| `run_univariate_analysis.py` | Run all steps in order |

**Outputs**

| Path | Content |
|------|---------|
| `results/univariate/figures/univariate_matched_dens_2dates.png` | Density comparison |
| `results/univariate/figures/univariate_matched_hist_absolute_2dates.png` | Histogram comparison |
| `results/univariate/tables/variable_summary_full_stats_EPD.csv` | Summary table (mean, IQR, min, max, p-values) |
| `data/processed/lda/df_15_primeros_EB2_prog.csv` | Window 1 cohort (local) |
| `data/processed/lda/df_15_previos_HDM_prog.csv` | Window 2 cohort (local) |

```bash
source env/bin/activate   # see requirements.txt in repo root
python scripts/02_univariate_analysis/lda/run_univariate_analysis.py
```

## `03_analysis/lda/train_lda_from_profiles.py`

**LDA training** from the VQ-VAE profiles PKL through document construction and ``LdaMulticore``.

- **Input:** `data/processed/vq-vae/profiles_per_sample_oncology_28_12_2025.pkl`
- **Outputs (local):** `data/processed/lda/user_embeddings_from_pkl.csv`, `models/lda/dictionary_LDA_vdec25.dict`, `models/lda/LDA_model_vdec25.gensim`

```bash
source env/bin/activate

# Full pipeline (train → topics → clinical merge → plots)
python scripts/03_analysis/lda/run_lda_pipeline.py

# If model already trained:
python scripts/03_analysis/lda/run_lda_pipeline.py --skip-train
```

| Script | Role |
|--------|------|
| `train_lda_from_profiles.py` | PKL → LDA model + dictionary |
| `assign_lda_topics.py` | Per-user topics + top-10 terms table/plot |
| `merge_clinical_topics.py` | Merge topics with clinical Excel |
| `plot_lda_results.py` | PD vs no-PD topic figures |
| `run_lda_pipeline.py` | Runs all steps |

## `03_analysis/lda/decode_profiles.py`

**Decode LDA top-10 profile tokens** to behavioral feature vectors. Requires LDA top-terms table and VQ-VAE decoded embeddings pickle.

- **Inputs:** `results/lda/tables/lda_topics_top10.csv`, `decoded_embedding_vectors_a0.pkl` (cnio or `data/processed/vq-vae/`)
- **Outputs (local):** `data/processed/lda/decoded_profiles_top10.csv`, `decoded_profiles_top10_scaled.csv`
- **Figures:** `results/lda/figures/decoded_topic_plots/final_pattern_analysis_grid.svg`, `pattern_0_individual.svg`, `pattern_3_individual.svg`; optional `behavioral_features_by_pattern_boxplot.svg`

```bash
python scripts/03_analysis/lda/run_decode_profiles.py
# tables only:
python scripts/03_analysis/lda/run_decode_profiles.py --skip-plots
```

| Script | Role |
|--------|------|
| `decode_profiles.py` | Join top terms with decoded vectors → CSV |
| `plot_decoded_profiles.py` | Pattern grid + standardized boxplot |
| `run_decode_profiles.py` | Runs decode + plots |

## `03_analysis/lda/plot_monthly_topics_heatmap.py`

**Monthly predominant topics heatmap.** Patients sorted by PD event and data availability; black box marks progression month in PD cases.

- **Input:** `PD_cutoff_dic_2025s_eB2_MonthPredom.csv` (cnio) or `data/processed/lda/monthly_predominant_topics.csv`
- **Optional rebuild:** `build_monthly_predominant_topics.py` from per-day embeddings + LDA model

```bash
python scripts/03_analysis/lda/run_monthly_topics_heatmap.py
# regenerate table then plot:
python scripts/03_analysis/lda/run_monthly_topics_heatmap.py --rebuild-table
```

| Script | Role |
|--------|------|
| `build_monthly_predominant_topics.py` | 30-day windows → monthly predominant topic CSV |
| `plot_monthly_topics_heatmap.py` | Heatmap figure |
| `run_monthly_topics_heatmap.py` | Orchestrator |

**Outputs (`results/lda/`)**

| Path | Content |
|------|---------|
| `figures/lda_top_terms_grid.png` | Top tokens per topic |
| `figures/topics_by_event.png` | Predominant topic by PD event |
| `figures/avg_topic_distribution_by_event.png` | Mean topic probs PD vs no-PD |
| `tables/lda_topics_top10.csv` | Top 10 profile tokens per topic |
| `figures/decoded_topic_plots/final_pattern_analysis_grid.svg` | Final z-score grid (manuscript) |
| `figures/decoded_topic_plots/pattern_0_individual.svg` | Pattern 0 panel only |
| `figures/decoded_topic_plots/pattern_3_individual.svg` | Pattern 3 panel only |
| `figures/behavioral_features_by_pattern_boxplot.svg` | Z-scored features by pattern (optional) |
| `data/processed/lda/decoded_profiles_top10.csv` | Decoded behavioral features (raw) |
| `data/processed/lda/decoded_profiles_top10_scaled.csv` | Z-scored decoded features |
| `figures/predominant_topics_per_month.png` | Monthly topic heatmap (PD vs no-PD) |
| `figures/predominant_topics_per_month.svg` | Same heatmap (vector) |
| `data/processed/lda/monthly_predominant_topics.csv` | Monthly table (if rebuilt locally) |
| `data/processed/lda/user_topic_cluster_6topics_lda.csv` | User-level topic assignment |
| `data/processed/lda/patient_topics_clinical.csv` | Topics + clinical (inner merge) |
| `data/processed/lda/merged_topics_clinical.csv` | Wide merge for downstream plots |

## Methods

| Folder | Description |
|--------|-------------|
| `lda/` | Latent Dirichlet Allocation (or related topic / latent structure workflow) |
| `vq-vae/` | Vector-quantized variational autoencoder workflow |

## Conventions

- **One entry script per step**, e.g. `01_preprocess/lda/preprocess_daily_summaries.py`.
- **Shared logic** → `scripts/utils/`.
- **Outputs** → `results/lda/` or `results/vq-vae/`.
- **Method-specific processed data** (if any) → `data/processed/<method>/`.

## Execution order (LDA example)

```text
python scripts/01_preprocess/lda/preprocess_daily_summaries.py
scripts/02_univariate_analysis/lda/...
scripts/03_analysis/lda/...
scripts/04_figures/lda/...
```
