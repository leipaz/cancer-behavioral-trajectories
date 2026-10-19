# LDA K-selection and stability (K=6)

Single analysis on the **reconciled CNIO/DB1** patient day-type sequences.

| Item | Path |
|------|------|
| Embeddings (documents) | `data/processed/lda/user_embeddings_from_pkl.csv` (= reconciled `a0`) |
| Daily used upstream | `data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv` |
| Reference LDA | `models/lda/LDA_model_vdec25.gensim` (published; not re-trained) |
| Tables / JSON | this folder |
| Figures | `../figures/lda_coherence_k_selection.*`, `../figures/stability/` |

## Levels

1. **Seed stability** — 50 re-fits, all patients, different seeds.  
2. **Leave-20%-out** — 50 re-fits dropping 20% of patients.

Topics are Hungarian-aligned to the published reference by cosine of `P(day-type | topic)`.  
Hypertopics: F = T0–2, U = T3–5; **pUF** = P3+P4+P5 (one scalar per patient document).

## Settings

- Seeds / reps: 50  
- Passes per replicate: **100** (published reference was trained with ~**100 000** passes)  
- Documents: 175 patients  

## Method write-up

[`INFORME_METODO_SELECCION_K_Y_ESTABILIDAD_LDA.md`](INFORME_METODO_SELECCION_K_Y_ESTABILIDAD_LDA.md) / `.docx`

## Regenerate

```bash
./env/bin/python scripts/03_analysis/lda/run_lda_coherence_sweep.py \
  --k-list 2,6,10,14,18 --n-inits 5 --passes 50

./env/bin/python scripts/03_analysis/lda/run_lda_stability.py \
  --mode both --n-reps 50 --passes 100 --leave-out 0.2
```
