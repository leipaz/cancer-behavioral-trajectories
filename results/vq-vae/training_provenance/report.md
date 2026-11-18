# Provenance of the external training and the HDO-only alternative

This block documents the extract that can be counted in this repository, the identifier overlap with HDO, and what the frozen model does and does not show about external labels. It also fixes the single oncology-only alternative named by SAP section 17.4. That alternative is specified here and has not been fitted.

The four-cohort comparison of reconstruction, topics, vocabulary and examples is a separate report. A similar reconstruction, or a different topic mix, does not show that the representation is free of bias or that it is clinically relevant.

## Population and rules

The participant list is `user_splits.csv` (7,231 participants). Days come from `daily_summary_eb2prod_may26.csv`. HDO is `daily_summary_checked.csv` in DB1 (175 participants).

An observed day is a unique participant-date with at least one of the eight continuous variables inside the frozen clip. Raw rows are kept as a second count because some dates are repeated. The column `vqvae` is reported as stored. It is not evidence that a participant was left out of the VQ-VAE fit: Suicide prevention and Depression / Anxiety disorders are entirely `vqvae = 0` and were part of that training.

This extract is not the archived training manifest required by SAP section 17.4. Where a published Table S3 count exists, it is shown beside the extract. A larger extract is not trimmed here. The comparison report applies that cap separately.

## Labels

The frozen VQ-VAE returns one reconstructed value for each input channel, a quantization term and a code index. Its inputs are the ten signal columns in the frozen feature list. The columns `emotion_inter`, `emotion_intra`, `suicide`, `phq9` and `whodas` sit in `user_splits` and are not in that input list. Their presence in the table is not the same as their use as an optimisation target. The per-variable reconstruction on the Table S3 programs present in the May 2026 daily is in `results/vq-vae/reconstruction_s3_may26/`.

The training script, the stopping rule and the record of the choice between checkpoints a0, a1 and a2 are not in this repository. Until that record is archived, this block does not state that external diagnostic or event labels were absent from variable selection, window selection, hyperparameters or the choice of the final checkpoint.

The LDA uses six topics. Coherence on the external day-type sequences peaked at nine topics; six was kept as a parsimonious point in that high-coherence region. The favourable/unfavourable names use HDO progression and belong to the alarm, not to the external codebook.

## Composition of the extract

Identifier overlap with the 175 HDO participants is zero in every row, including the oncology and HIV programs.

| Program in the extract | Participants | vqvae = 1 | vqvae = 0 | Raw rows | Observed days | First day | Last day | Published N | HDO overlap |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Eating disorders | 3,401 | 3,266 | 135 | 1,088,232 | 698,049 | 2020-04-03 | 2026-05-04 | 2,464 | 0 |
| Cognitive impairment | 858 | 819 | 39 | 411,016 | 227,275 | 2023-09-19 | 2026-05-04 | 197 | 0 |
| das | 1,374 | 1,347 | 27 | 450,619 | 294,021 | 2016-01-01 | 2026-05-04 | — | 0 |
| Suicide prevention | 1,091 | 0 | 1,091 | 275,627 | 189,048 | 2016-01-01 | 2026-05-04 | 2,555 | 0 |
| Depression / Anxiety disorders (TMC) | 140 | 0 | 140 | 32,437 | 21,489 | 2022-10-27 | 2026-05-04 | 149 | 0 |
| SJD | 41 | 41 | 0 | 16,120 | 7,653 | 2019-10-21 | 2024-08-13 | — | 0 |
| Gregorio Marañón | 100 | 100 | 0 | 24,514 | 15,035 | 2022-08-30 | 2026-05-04 | 93 | 0 |
| HIV (DM) | 13 | 13 | 0 | 2,971 | 1,504 | 2023-03-16 | 2026-04-26 | 24 | 0 |
| PMP | 206 | 206 | 0 | 94,284 | 83,555 | 2023-02-16 | 2026-05-03 | — | 0 |
| CNIO | 7 | 0 | 7 | 5,214 | 4,899 | 2021-07-27 | 2026-05-03 | — | 0 |
| Extract total | 7,231 |  |  | 2,401,034 | 1,542,528 |  |  |  | 0 |
| HDO | 175 |  |  | 51,972 | 48,048 | 2023-02-16 | 2025-10-22 |  |  |

HDO is the study cohort and was not part of the VQ-VAE fit. CNIO was not part of that fit. Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders were. Gregorio Marañón and HIV are the oncology and HIV slices of this extract; their counts are not the published 93 and 24, and the original identifier list that would confirm membership one by one is not in the repository. das, SJD and PMP are in the extract, are not rows of the four-cohort comparison, and have no HDO identifier in common with the study. Service-level counts are in `composition_by_service.csv`.

## The one HDO-only alternative

SAP section 17.4 asks for a development-only comparison of the fixed pretrained representation with simple feature models and, where the data allow it, an oncology-only representation fitted inside each training fold. The simple models are the steps and heart-rate scores in SAP section 15.1. They ask whether the alarm adds anything to those signals. They do not ask whether the conclusion depends on the external vocabulary.

The oncology-only representation is this one model, fixed before any look at its progression results:

- k-means with K = 6 on the same eight continuous daily variables.
- The same frozen clip, then `log1p` on the seven positive variables. Center and scale are estimated inside the training fold only.
- The partition is the SAP section 12 patient-level nested cross-validation: five folds, ten repeats, seeds drawn from numpy Generator(20261031). One patient stays in one fold.
- Held-out days receive the fold's already fitted clusters. VAL-SET is not used to fit, to label clusters, or to choose this alternative over another.
- Unfavourable clusters and the threshold use the existing training-only 3/3 rule and the Youden criterion, inside the same training fold. The accumulation is three blocks and the horizon is 120 days.
- The comparison with the frozen external-vocabulary alarm is sensitivity, specificity, AUROC and lead time on the held-out folds.

A VQ-VAE retrained on balanced sources is not part of this block.

The specification is in `hdo_only_alternative.json`. No fit has been run, so this block does not yet report how far the clinical summaries move.
