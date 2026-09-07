# Revision experiments

Notebooks and CLIs for paper revision analyses (branch `revision-experiments`).

| Artifact | Purpose |
|----------|---------|
| `cohort_vqvae_profile_occupancy.ipynb` | Equal sample per clinical group + CNIO oncology; occupancy & decode comparison |
| `alarm_replication_monthly.ipynb` | Replicate manuscript rolling-burden alarm at **monthly collapsed** granularity (W=3, H=4, θ=1.61) |
| `alarm_granularity_sweep.ipynb` | Compare **simple/sliding** vs **collapsed** pUF at daily / weekly / biweekly / monthly |

## Alarm system (two constructions)

| Mode | How `pUF` is built | Evaluation grid |
|------|--------------------|-----------------|
| **Simple / sliding** | Each day `d≥30`: LDA on the previous 30 days → `pUF(d)`. Burden = rolling mean over lookback **W** days. | Every day, or subsample every 7 / 14 / 30 days |
| **Collapsed** | Non-overlapping blocks of 7 / 14 / 30 days; one LDA mixture per block → `pUF(period)`. Burden = rolling mean/sum over `round(W/block)` periods. | One landmark per block |

- **W (lookback / load window):** how far back burden is accumulated before the alarm decision.
- **H (horizon):** how far ahead progression (`PD`) is labeled as the target.
- Paper operating point: monthly collapsed, **W=3 months (~90 d)**, **H=4 months (~120 d)**, **θ=1.61**.

### CLI

```bash
# Monthly paper replication
.venv/bin/python scripts/03_analysis/alarm/run_granularity_sweep.py --monthly-only

# Full granularity × (W,H) sweep using existing pUF tables (fast)
.venv/bin/python scripts/03_analysis/alarm/run_granularity_sweep.py --skip-lda

# Rebuild collapsed weekly/biweekly pUF with LDA, then sweep
.venv/bin/python scripts/03_analysis/alarm/run_granularity_sweep.py --rebuild-collapsed --skip-lda

# Paper-style individual trajectories (monthly pUF + rolling AUC/W)
.venv/bin/python scripts/03_analysis/alarm/plot_patient_trajectories.py \
  --patients 62004,23003,31002,41006

# Granularity trajectories: sampled vs collapsed (weekly / biweekly / monthly)
.venv/bin/python scripts/03_analysis/alarm/plot_granularity_trajectories.py \
  --patients 62004,23003,31002,41006 --W 90
```
Outputs:

- `results/alarm/tables/` — `daily_sliding_puf.csv`, `collapsed_puf_*.csv`, `granularity_sweep_summary.csv`, decision points, metrics
- `results/alarm/figures/` — ROC / summary panels (`roc_granularities_W90_H120.*`, `summary_granularities_W90_H120.*`, `roc_W3_H4.*`)

Core modules: `scripts/03_analysis/alarm/{rolling_burden_alarm,sliding_burden_alarm,build_daily_sliding_puf,build_collapsed_puf,run_granularity_sweep}.py`.

## Clinical groups (VQ-VAE occupancy)

| Code | Name | Services | Equal sample |
|------|------|----------|--------------|
| CI | Cognitive impairment | DC_* | yes if ≥ N |
| CNIO | CNIO | cnio_caspar | **aparte** (not sampled) |
| das | das | das | yes |
| DM | DM | DM | yes if ≥ N (usually undersized) |
| ED1 | Eating disorder 1 | ita_* | yes |
| ED2 | Eating disorder 2 | adalmed | yes |
| GM | Gregorio Marañón | GM_colon GM_MCEI | yes |
| PMP | PMP | PMP_* | yes |
| SR | Suicide risk | afsp FJD_smartcrisis_2_0 | yes |
| TMC | Common mental disorder | TMC SJD | yes |

```bash
.venv/bin/python -m scripts.vqvae.sample_cohort_profiles --list-groups
.venv/bin/python -m scripts.vqvae.sample_cohort_profiles --n-per-cohort 90 --modes a0
.venv/bin/python -m scripts.vqvae.compare_cohort_profiles --top-k 15
```

**CNIO note:** train-file `cnio_caspar` ≈11 users. Comparison uses 90 patients from `oncology_daily_summary_model_input.csv`.

Outputs (gitignored):

- `data/daily_summaries/revision_cohort_samples/model_input_<CODE>.csv`
- `data/output_vq_vae/revision_cohorts/profiles_per_sample_<CODE>.pkl`
- `data/output_vq_vae/revision_cohorts/comparison_tables/`
- `results/vq-vae/figures/revision_cohorts/`
