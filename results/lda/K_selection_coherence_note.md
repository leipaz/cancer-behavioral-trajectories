# Topic-number selection (coherence) — Methods note

## What this figure shows
Average **c_v** coherence for LDA as a function of the number of topics **K**, averaging several random initializations (train and held-out validation documents).

## How K = 6 was chosen
1. A coherence sweep over K (grid with step 5 in the development notebook; validation coherence peaked at **K = 9** and declined for larger K).
2. **K = 6** was selected as a **parsimonious** value in the **high-coherence region at small K** (near the validation peak), yielding an interpretable partition into three favourable and three unfavourable behavioural patterns.
3. Coherence sweeps used relatively few passes (for computational feasibility across many K × seeds). The **published model** was then **re-trained at fixed K = 6** with a large number of passes (on the order of 10⁴–10⁵) on the full analysis cohort, and stored under `models/lda/`.

## What this does *not* claim
Coherence justifies the **choice of K**. It does **not** by itself establish topic **stability** across random seeds at fixed K; that is addressed in a separate multi-seed stability analysis (topic–profile distribution similarity after alignment).
