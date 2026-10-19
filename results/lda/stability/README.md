# LDA stability (K=6)

Multi-seed re-fits aligned to the published reference model
(`LDA_model_vdec25.gensim`).

| File | Content |
|------|---------|
| `stability_per_seed.csv` | 6-topic + hypertopic JS; patient fav / pUF correlations |
| `stability_js_by_topic.csv` | Per reference topic, per seed |
| `stability_summary.json` | Aggregate means / SDs |
| `../figures/stability/lda_stability_k6_summary.*` | Topics + hypertopics figure |

## Two levels of comparison

1. **Six topics** — JS similarity of `P(profile | topic)` after alignment.
2. **Two hypertopics** (what the alarm uses conceptually):
   - Favourable = topics 0–2; unfavourable = topics 3–5
   - Profile-level: JS on the equal mixture of those topic distributions
   - Patient-level: Pearson of `P0+P1+P2` and of **pUF = P3+P4+P5**

## Settings used here

- Seeds: 50 (values 0…49)
- Passes per seed: **100**
- Documents: 175; vocabulary size: 255

## Headline results

| Level | Metric | Mean ± SD |
|-------|--------|-----------|
| 6 topics | Mean JS similarity | **0.648 ± 0.048** |
| Hypertopic fav | JS (mixed profiles) | **0.781 ± 0.037** |
| Hypertopic unfav | JS (mixed profiles) | **0.760 ± 0.043** |
| Patient fav | Pearson r | **0.823 ± 0.093** |
| Patient pUF (alarm) | Pearson r | **0.823 ± 0.093** |

Regenerate:

```bash
.venv/bin/python scripts/03_analysis/lda/run_lda_stability.py --n-seeds 50 --passes 100
```
