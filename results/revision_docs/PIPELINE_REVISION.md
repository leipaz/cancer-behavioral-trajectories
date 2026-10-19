# Re-run revision pipeline (VQ-VAE + LDA frozen)

Models stay frozen; regenerate profiles → topics → hypertopics → alarms.

## Inputs (ready)

| File | Role |
|------|------|
| `oncology_daily_summary_model_input_rangos.csv` | Daily features clipped to official windows |
| `rangos_fechas.xlsx` | Official start/end per ID |
| `models/vq-vae/*` | Frozen VQ-VAE + scaler |
| `models/lda/*` | Frozen LDA + dictionary |

Missing in daily (cannot run): `31015`, `61004`, `63006`.

## Order

```bash
cd /export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories
source env/bin/activate   # or .venv

# 1) VQ-VAE inference → profiles PKL (a0 = paper)
python scripts/03_analysis/vq-vae/generate_profiles.py \
  --data-csv data/revision/oncology_daily_summary_model_input_rangos.csv \
  --models-dir models/vq-vae \
  --scaler models/vq-vae/preprocessing_scaler_vqvae.npz \
  --output data/revision/profiles_per_sample_oncology_revision.pkl \
  --modes a0 \
  --device cpu

# 2) PKL → embedding_ids CSV
#    notebook: notebooks/vq-vae_lda_pipeline/profiles_pkl_to_csv.ipynb
#    (point PROFILES_PKL to the revision PKL; MODEL_TYPE=a0; N=30)

# 3) LDA assign topics (DO NOT retrain)
python scripts/03_analysis/lda/run_lda_pipeline.py --skip-train

# 4) Monthly topic probs → clone of topics_probs.xlsx
#    (blocks of 30 days; NaN if not evaluable; hypertopic 0–2 bueno / 3–5 malo)
#    notebook: notebooks/vq-vae_lda_pipeline/lda_dcabp_evolution.ipynb
#    + rebuild data/processed/DCABPs/topics_probs.xlsx

# 5) Alarm granularity study (MUST rebuild pUF; no --skip-lda)
python scripts/03_analysis/alarm/run_alarm_pipeline.py
# optional:
python scripts/03_analysis/alarm/run_alarm_pipeline.py --with-trajectories
```

## What the alarm study repeats

Same as revision notebooks / `results/alarm/`:

- Monthly collapsed paper point (W=3, H=4, θ=1.61)
- Granularity sweep (sliding vs collapsed × W/H)
- Missingness funnel + AUC heatmaps
- Report: `results/alarm/alarm_granularity_report.html`

**Yes, this analysis can be fully repeated** once steps 1–4 produce a new `topics_probs.xlsx` and refreshed day-level `embedding_ids`.

## Hypertopic bueno / malo

- Topics **0,1,2** = favourable (bueno)
- Topics **3,4,5** = unfavourable (malo) → `pUF = P3+P4+P5`
- Heatmaps / evolution: `lda_dcabp_evolution.ipynb`
- Alarm burden uses the same split automatically

## Do not

- Retrain VQ-VAE or LDA
- Use `run_alarm_pipeline.py --skip-lda` for this revision (would reuse old pUF)
- Compress calendars or fill missing months with 0 (use NaN)
