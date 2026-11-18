# Provenance of the external training and the HDO-only alternative

This block records the March 2024 eB2 extract on which the VQ-VAE applied to CNIO was trained, and the identifier overlap with HDO. The oncology-only alternative named by SAP section 17.4 has been fitted as two steps, k-means and then LDA. The four-cohort comparison of reconstruction, topics, vocabulary and examples is a separate report. A similar reconstruction, or a different topic mix, does not show that the representation is free of bias or that it is clinically relevant.

## Training objective

The training procedure is the one in `documents/P·VQVAE - ACM-Health - Online LaTeX Editor Overleaf.pdf`, sections 3 and 4.2 and Appendix A. The pretraining loss is reconstruction of the ten behavioral variables: mean squared error on the continuous and count variables, class-weighted binary cross-entropy on Weekend and Practiced Sport, a commitment term, and an exponential moving average for the codebook. The checkpoint minimizes that validation objective on entries that remain visible to the model. Synthetically hidden entries do not enter the selection. Emotion valence, clinician suicide-risk events, PHQ-9 item 9 and WHODAS are downstream targets. Those tasks consume the frozen daily codes. They are not the pretraining objective, and the codes are not given clinical labels beforehand. The default network is the implicit-mask model (E0). E1 and E2 condition on the mask and are kept as checks. In this repository those three weights are `vqvae_a0.pt`, `vqvae_a1.pt` and `vqvae_a2.pt`.

The manuscript's own daily table, 7,231 participants and 40 programs, is a different extract. It is not the file used for this checkpoint.

## Training corpus

The file is `data/daily_summaries/vq_vae_train/vqvae_leire/daily_summary_eb2.csv`, extracted in March 2024. The partitions are `train_0.csv`, `validation_0.csv` and `test_0.csv`. Cohort names come from the service-code legend in `data/daily_summaries/vq_vae_train/Relación_Pacientes_Cohortes.xlsx`. An observed day is a unique participant-date with at least one of the eight continuous variables inside the frozen clip. Raw rows are kept as a second count because some dates are repeated.

The extract has 5,530 participants and 1,122,233 raw rows, from 2016-01-01 to 2024-03-13. Every participant is in exactly one partition: 3,647 train, 1,563 validation, 320 test. The eight service groups are the Table S3 cohorts. HDO is not one of them, and neither is the CNIO study cohort. PMP is not a service code in this extract. User 5906 has codes 17 and 23, so that person is counted under both eating disorders and suicide prevention. The participant column therefore sums to 5,531.

| Cohort | Participants | Raw rows | Observed days | First day | Last day | Table S3 N | HDO overlap |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Eating disorders | 2,480 | 494,699 | 310,699 | 2020-04-03 | 2024-03-13 | 2,464 | 0 |
| Cancer patients | 83 | 14,513 | 13,811 | 2021-07-27 | 2024-03-13 | 93 | 0 |
| Prevention of cognitive impairment | 181 | 5,408 | 4,804 | 2023-09-19 | 2024-03-13 | 182 | 0 |
| Cognitive impairment follow-up | 15 | 2,548 | 1,302 | 2023-03-16 | 2024-03-13 | 15 | 0 |
| Suicide prevention | 2,547 | 553,974 | 374,849 | 2016-01-01 | 2024-03-13 | 2,555 | 0 |
| HIV | 24 | 8,418 | 5,870 | 2022-08-30 | 2024-03-13 | 24 | 0 |
| Depression / anxiety disorders | 149 | 27,425 | 19,657 | 2022-10-27 | 2024-03-13 | 149 | 0 |
| Bipolar disorder | 52 | 15,248 | 7,699 | 2019-10-04 | 2024-01-05 | 51 | 0 |
| Extract, unique participants | 5,530 | 1,122,233 | 738,691 | 2016-01-01 | 2024-03-13 | 5,533 | 0 |

Follow-up, HIV and depression / anxiety disorders match the published counts. The other five differ by 1 to 16 people. Table S3 does not list identifiers, so those people are not named. Counts by partition are in `march2024_by_cohort_split.csv`.

The fit did not use every row. Sequences shorter than 257 days were removed from training and sequences shorter than 256 days were removed from validation. All 320 test participants remained. After that filter, 1,085 training participants and 490 validation participants remain, 1,895 in all. Prevention of cognitive impairment contributes 181 participants to the extract and none of them remain. The test partition is entirely suicide prevention.

| Cohort | Train | Validation | Test | Observed days |
| --- | ---: | ---: | ---: | ---: |
| Eating disorders | 556 | 265 | 0 | 230,368 |
| Cancer patients | 22 | 6 | 0 | 9,671 |
| Prevention of cognitive impairment | 0 | 0 | 0 | 0 |
| Cognitive impairment follow-up | 3 | 3 | 0 | 800 |
| Suicide prevention | 461 | 192 | 320 | 288,563 |
| HIV | 12 | 5 | 0 | 4,887 |
| Depression / anxiety disorders | 26 | 12 | 0 | 8,956 |
| Bipolar disorder | 6 | 7 | 0 | 5,278 |
| Unique participants | 1,085 | 490 | 320 | 548,523 |

User 5906 remains in training and is the extra count in both eating disorders and suicide prevention. The day counts above are the days of the participants who remained.

## Labels and overlap

The three partition files contain the ten signal columns only: sleep onset, distance, time at home, sleep duration, walking, app use, places, steps, weekend and practiced sport. Emotion, pain, audio, heart rate and the clinical scales are columns of the complete daily and are not in `train_0.csv`, `validation_0.csv` or `test_0.csv`.

No `user` and no `user_uid` in the March 2024 extract equals an identifier of the 175 HDO participants, of the 48 VAL-SET candidates, or of the 42 VAL-SET participants with a daily file. The extract identifiers stop at 6,982. HDO identifiers start at 11,001. The check is in `march2024_overlap.json`.

The LDA uses six topics. Coherence on the external day-type sequences peaked at nine topics; six was kept as a parsimonious point in that high-coherence region. The favourable/unfavourable names use HDO progression and belong to the alarm, not to the external codebook.

## Relation to the May 2026 comparison corpus

`results/vq-vae/adequacy_external_pretraining/report.md` compares HDO with Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders, taken from `daily_summary_eb2prod_may26.csv` and capped at the Table S3 counts. That report calls those rows training patients. The file is a later extract. It is not `daily_summary_eb2.csv` from March 2024, and it adds programs that have no service code in the March legend: das, PMP, SJD and CNIO. Bipolar disorder is in the March extract and has no service under that name in May 2026.

The Table S3 names and the published counts are the same list in both places. The headcounts are not. Suicide prevention has 2,547 participants in March 2024 and 973 of them remain after the length filter. The May comparison uses the 1,091 participants present in that later file. Prevention of cognitive impairment has 181 participants in March and none remain after the filter, while the comparison treats its May cognitive patients as training patients. Eating disorders are 2,480 in March, against a May file of 3,401 that the comparison caps at 2,464. Depression / anxiety disorders are 149 in March, of whom 38 remain, against 140 used in the comparison.

HDO identifier overlap is zero in both extracts. The service-code legend recovers these Table S3 sizes. The patient-count column of the same workbook places the eight published numbers on different rows. This block follows the code column and the Table S3 names already used in the comparison report.

The May 2026 composition remains in `composition_by_program.csv`. It is not this training corpus.

## The one HDO-only alternative

SAP section 17.4 asks for a development-only comparison of the fixed pretrained representation with simple feature models and, where the data allow it, an oncology-only representation fitted inside each training fold. The simple models are the steps and heart-rate scores in SAP section 15.1. They ask whether the alarm adds anything to those signals. They do not ask whether the conclusion depends on the external vocabulary.

The oncology-only representation is two steps, fixed before any look at its progression results. The k-means replaces the VQ-VAE. The LDA stays, and it is fitted on the cluster sequence rather than on the external codes.

- k-means with K = 6 on the same eight continuous daily variables of the training patients.
- The same frozen clip, then `log1p` on the seven positive variables. Center and scale are estimated inside the training fold only. A missing value is the training mean after scaling.
- LDA with K = 6 on those cluster sequences, with the same 500 passes and 50 iterations as the nested SAP LDA. The third seed of each repeat starts both the k-means and the LDA.
- The partition is the SAP section 12 patient-level nested cross-validation: five folds, ten repeats, seeds drawn from numpy Generator(20261031). One patient stays in one fold.
- Held-out days receive the fold's already fitted clusters and the LDA fitted on the training sequences. VAL-SET is not used to fit, to label topics, or to choose the cut.
- Unfavourable topics and the threshold use the existing training-only 3/3 rule and the Youden criterion, inside the same training fold. The accumulation is three blocks and the horizon is 120 days.
- Development is scored on the same 1013 landmarks as the external-vocabulary alarm. VAL-SET is scored once those cuts exist, with the refit of each outer fold.

A VQ-VAE retrained on balanced sources is not part of this block.

On the development landmarks, the mean of the ten repeat-level means is AUROC 0.618, sensitivity 0.46, specificity 0.69 and a median lead of 92 days. The external-vocabulary alarm on the same landmarks is AUROC 0.617, sensitivity 0.64, specificity 0.52 and a median lead of 95 days. The differences, HDO-only minus external vocabulary, are +0.001 AUROC, −0.18 sensitivity, +0.17 specificity and −3.5 days. Eight of the fifty outer refits could not be split by the 3/3 rule, so those folds drop out of the HDO-only mean. The four repeats that score all 1013 landmarks move in the same direction.

VAL-SET contributes 42 of 48 patients and 7 progression events. Six patients have no observed day in the official window, including one progression (62007). With the development cuts left unchanged, the mean across repeats is AUROC 0.596, sensitivity 0.61, specificity 0.65 and a median lead of 90 days. The AUROC is 0.02 below the development figure. That difference is reported and is not used to change the model. The files are under `hdo_only_lda/`.
