# Topic probabilities by granularity

LDA topic probabilities (`topic_0`…`topic_5`) and unfavourable burden `pUF` (= topic_3+4+5), so rolling-burden scores can be recomputed externally.

| File | Granularity | Unit |
|------|-------------|------|
| `topic_probs_daily_sliding.csv` | Daily sliding (30-day LDA window) | one row per study day |
| `topic_probs_weekly_collapsed.csv` | Weekly collapsed | non-overlapping 7-day blocks |
| `topic_probs_biweekly_collapsed.csv` | Biweekly collapsed | non-overlapping 14-day blocks |
| `topic_probs_monthly_collapsed.csv` | Monthly collapsed | non-overlapping 30-day blocks |

Same content also saved as `collapsed_puf_*.csv` (legacy names).

### Columns (collapsed)

`id`, `period`, `block_size_days`, `decision_day`, `pUF`, `topic_0`…`topic_5`, `Evento PD`, `t_evento_eb2`, `follow_up_days`

Short series are included (`min_embeddings=1`): collapsed uses whatever days are available.

### Rebuild

```bash
.venv/bin/python scripts/03_analysis/alarm/build_collapsed_puf.py \
  --block-size 7 14 30 \
  --topics-day-csv data/processed/lda/PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv \
  --out-dir results/alarm/tables/topic_probs_granularities
```
