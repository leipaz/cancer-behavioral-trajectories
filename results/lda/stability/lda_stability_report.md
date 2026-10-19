# LDA topic-number selection and stability report 

Similarity metric throughout: **cosine** between aligned `P(day-type | topic)` distributions (Hungarian matching to the published reference).

**Data:** revised CNIO/DB1 date ranges — `data/revision/user_embeddings_revision_a0.csv`  
(Reference model unchanged: `models/lda/LDA_model_vdec25.gensim`.)

---

## 1. Unsupervised selection of K

We compared several candidate values of **K** using mean topic coherence (**c_v**) over 5 initializations (train/validation 70/30).  
**Rule:** maximise mean validation coherence → **K = 6** (val. peak = 0.588).

![Coherence vs K](../figures/lda_coherence_k_selection.png)

Published LDA: K = 6 re-trained with many passes (~100 000), `models/lda/LDA_model_vdec25.gensim`.

---

## 2. Level 1 — Stability to random seed

Re-fit LDA **50 times** at K = 6 (all patients; different seeds). Align topics to the reference; report **cosine** similarity of topic–profile distributions and patient **pUF** concordance.

### Headline (cosine)

| Summary | Value |
|---------|------:|
| Median cosine (all topic–replicate pairs) | **0.915** |
| IQR | **[0.844, 0.962]** |
| Median of per-replicate mean cosine (6 topics) | **0.884** |
| IQR of per-replicate mean cosine | **[0.840, 0.902]** |
| Favourable hypertopic cosine (median) | **0.960** |
| Unfavourable hypertopic cosine (median) | **0.916** |
| Patient pUF Pearson r (median [IQR]) | **0.849 [0.795, 0.890]** |

| Topic | Hypertopic | Cosine median [IQR] |
|------:|------------|--------------------:|
| T0 | favourable | 0.981 [0.976, 0.984] |
| T1 | favourable | 0.914 [0.880, 0.940] |
| T2 | favourable | 0.957 [0.936, 0.967] |
| T3 | unfavourable | 0.875 [0.846, 0.923] |
| T4 | unfavourable | 0.884 [0.843, 0.935] |
| T5 | unfavourable | 0.691 [0.633, 0.737] |

![Level 1 cosine](../figures/stability/lda_stability_level1_cosine.png)

![Level 1 profile heatmap](../figures/stability/lda_stability_level1_profile_heatmap.png)

---

## 3. Level 2 — Stability to data (leave-20%-out)

Same K = 6 and cosine alignment, but each replicate **drops 20% of patients at random** (n = 50 replicates) and re-fits LDA on the remainder.

### Headline (cosine)

| Summary | Value |
|---------|------:|
| Median cosine (all topic–replicate pairs) | **0.867** |
| IQR | **[0.723, 0.931]** |
| Median of per-replicate mean cosine | **0.794** |
| Favourable / unfavourable hypertopic cosine (median) | **0.937** / **0.885** |
| Patient pUF Pearson r (median [IQR]) | **0.772 [0.675, 0.858]** |

| Topic | Hypertopic | Cosine median [IQR] |
|------:|------------|--------------------:|
| T0 | favourable | 0.965 [0.953, 0.970] |
| T1 | favourable | 0.882 [0.825, 0.919] |
| T2 | favourable | 0.909 [0.860, 0.937] |
| T3 | unfavourable | 0.815 [0.699, 0.871] |
| T4 | unfavourable | 0.797 [0.611, 0.889] |
| T5 | unfavourable | 0.668 [0.579, 0.753] |

![Level 2 cosine](../figures/stability/lda_stability_level2_resample_cosine.png)

![Level 2 profile heatmap](../figures/stability/lda_stability_level2_profile_heatmap.png)

---

## 4. Interpretation

- **K = 6** maximises mean validation coherence among the candidates evaluated on the reconciled embeddings.
- **Level 1 (seed):** topic–profile cosine has **median > 0.9**; hypertopics and **pUF** remain highly concordant → the alarm input is stable to re-initialisation.
- **Level 2 (leave-20%-out):** cosine remains high for most topics/hypertopics under patient subsampling; use the heatmaps (mean ± SD of day-type weights) to show which profiles stay dominant.
- Results are essentially unchanged vs the lock-date stability report (Δ typically ±0.00–0.01).
- Alarm analyses are **not** re-run unless pUF concordance were poor.

---

## 5. Files

| Path | Content |
|------|---------|
| `coherence_coarse_grid.csv` | K-selection (revision) |
| `level1_seed_*.csv` / `level1_seed_summary.json` | Seed stability (cosine) |
| `level2_resample_*.csv` / `level2_resample_summary.json` | Leave-20%-out (cosine) |
| `../figures/lda_coherence_k_selection.*` | K figure |
| `../figures/stability/lda_stability_level1_cosine.*` | Level 1 |
| `../figures/stability/lda_stability_level2_resample_cosine.*` | Level 2 |
| `../figures/stability/*_profile_heatmap.*` | Mean ± SD profile weights |

```bash
./env/bin/python scripts/03_analysis/lda/run_lda_coherence_sweep.py \
  --k-list 2,6,10,14,18 --n-inits 5 --passes 50

./env/bin/python scripts/03_analysis/lda/run_lda_stability.py \
  --mode both --n-reps 50 --passes 100 --leave-out 0.2
```
