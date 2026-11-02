# Passes convergence check (LDA stability)

## Question
Are the stability replicates at **100 gensim `passes`** under-trained relative to a larger pass budget?

## Confirmed gensim settings
| Setting | Published reference | Stability replicates |
|--|--|--|
| **`passes`** | **100 000** | **100** |
| **`iterations`** | **50** (E-step per document) | **50** |

So the “100 000” in the report is the gensim **`passes`** parameter, not `iterations`.

## Design (fixed a priori — see `design_meta.json`)
- Same audited DB1 day-type documents as the stability analysis
- Same replicate hyper-parameters (`iterations=50`, symmetric α/η, chunksize=100)
- Level-1 style: seeds `{0,1,2,3,4}` on the full cohort
- Level-2 style: 5 leave-20%-out masks (RNG seed 0; indices locked in `design_meta.json`)
- Pass budgets compared from scratch: **100, 1000, 5000**
- Bound learning curve: one full-cohort seed, logged every 25 passes up to 2000

## What this is not
Not a search for better clinical concordance; not a reason to change K or alarm thresholds by itself.

## Results (headline)
Comparing each fixed job at **100 vs 1000** and **100 vs 5000** passes (topics Hungarian-matched between budgets):

| Comparison | Median topic cosine | Min topic cosine | Median pUF Pearson | Min pUF Pearson |
|--|--:|--:|--:|--:|
| 100 vs 1000 | 0.993 | 0.941 | 0.991 | 0.957 |
| 100 vs 5000 | 0.992 | 0.939 | 0.991 | 0.955 |

**Conclusion:** on this locked design, **100 passes already matches 1000/5000** for topic profiles and patient pUF (and from-scratch ELBO changes only ~0.05%). We keep the replicate budget at **100 passes**. The continuous pass-by-pass bound curve is archived but **not** used to choose the cut. No need to refresh the full 50-replicate set for under-training.

## Outputs
- `design_meta.json` — locked seeds / masks / budgets
- `snapshots_by_budget.csv` — metrics at each budget (vs published reference)
- `pairwise_100_vs_higher.csv` — 100 vs each higher budget
- `bound_curve.csv` + `figures/bound_vs_passes.*` — training objective vs passes
- `figures/passes_budget_agreement.*` — agreement scatter
- `convergence_summary.json` — short wrap-up
