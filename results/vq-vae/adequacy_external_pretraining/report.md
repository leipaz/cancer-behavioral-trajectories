# Adequacy of the external pretraining for HDO

The VQ-VAE a0, its scaler and the 6-topic LDA stay frozen. Nothing is refit. The aim is to see how the model represents HDO patients and whether that reading changes when the external corpus is inspected. A good reconstruction, or a different topic mix across cohorts, does not by itself show absence of bias or clinical relevance.

## Cohorts

The comparison is with these psychiatry cohorts from the VQ-VAE training: Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders. HDO is the study cohort. These four cohorts are training patients. The extract does not contain patients from these programs who were held out of the VQ-VAE fit, so the reference is identified as training data.

Age and coverage are reported because they can contribute to differences in topic occupation. Age is in the HDO clinical file. It is not in the mental-health daily or in `user_splits`, so age is not compared across cohorts.

| Cohort | Patients | Observed days | Median days | Median coverage | Median age |
| --- | ---: | ---: | ---: | ---: | ---: |
| HDO | 175 | 48,048 | 207 | 0.61 | 61 |
| Eating disorders | 2,464 | 555,111 | 136 | 0.21 | not available |
| Cognitive impairment | 197 | 70,886 | 362 | 0.27 | not available |
| Suicide prevention | 1,091 | 189,048 | 122 | 0.29 | not available |
| Depression / Anxiety disorders | 140 | 21,489 | 104 | 0.47 | not available |

Coverage is the share of observed continuous cells over each patient's densified calendar (8 variables × days from the first to the last day). Weekend is excluded. 1 HDO patient, 316 Eating disorders patients, 10 Cognitive impairment patients and 19 Suicide prevention patients have no day with a continuous variable inside range. They do not enter the error, the topics or the vocabulary.

## 1. Reconstruction and quantization

The same frozen preprocessing: clipping, `log1p` on the positive variables and the `vqvae_a0` scaler. Error is computed only on continuous components that were actually observed. Weekend is excluded. Quantization distance is the Euclidean distance from the continuous embedding to the nearest code, on days with at least one observed continuous variable. A zero-filled gap stays inside the sequence the encoder sees; it does not enter the error.

| Cohort | Scaled MAE | Median distance (p10–p90) | Median margin per patient | Median probability of the code |
| --- | ---: | ---: | ---: | ---: |
| HDO | 0.36 | 6.5 (3.7–11.5) | 1.03 | 0.48 |
| Eating disorders | 0.31 | 4.5 (2.5–8.6) | 0.95 | 0.43 |
| Cognitive impairment | 0.33 | 5.7 (3.1–10.3) | 1.01 | 0.46 |
| Suicide prevention | 0.35 | 5.6 (2.6–10.9) | 0.98 | 0.45 |
| Depression / Anxiety disorders | 0.35 | 6.6 (3.6–12.1) | 1.15 | 0.52 |

Mean absolute error in the scaled space, by variable. In parentheses, the median absolute error in original units.

| Variable | HDO | Eating disorders | Cognitive impairment | Suicide prevention | Depression / Anxiety disorders |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sleep onset | 0.19 (1448) | 0.15 (1178) | 0.18 (1363) | 0.19 (1463) | 0.19 (1448) |
| Distance | 0.41 (4456) | 0.33 (6440) | 0.38 (4377) | 0.35 (4328) | 0.38 (4643) |
| Time at home | 0.37 (186) | 0.32 (113) | 0.36 (179) | 0.36 (169) | 0.38 (165) |
| Sleep duration | 0.38 (3235) | 0.38 (3567) | 0.39 (3738) | 0.46 (3978) | 0.46 (3918) |
| Walking | 0.40 (1071) | 0.34 (1054) | 0.35 (1019) | 0.36 (883) | 0.37 (1061) |
| App use | 0.35 (2318) | 0.31 (2858) | 0.33 (1894) | 0.37 (3456) | 0.36 (2755) |
| Places | 0.35 (1) | 0.31 (1) | 0.32 (1) | 0.29 (1) | 0.33 (1) |
| Steps | 0.46 (2399) | 0.32 (1436) | 0.36 (1618) | 0.37 (1326) | 0.37 (1721) |

Correlation, in the scaled space, between the observed value and the reconstruction.

| Variable | HDO | Eating disorders | Cognitive impairment | Suicide prevention | Depression / Anxiety disorders |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sleep onset | 0.94 | 0.97 | 0.96 | 0.97 | 0.96 |
| Distance | 0.71 | 0.79 | 0.70 | 0.77 | 0.73 |
| Time at home | 0.76 | 0.87 | 0.78 | 0.82 | 0.79 |
| Sleep duration | 0.75 | 0.83 | 0.84 | 0.84 | 0.76 |
| Walking | 0.79 | 0.80 | 0.81 | 0.82 | 0.77 |
| App use | 0.83 | 0.86 | 0.90 | 0.87 | 0.82 |
| Places | 0.57 | 0.75 | 0.63 | 0.73 | 0.66 |
| Steps | 0.73 | 0.82 | 0.79 | 0.82 | 0.76 |

HDO by tumor. The median is across patients, not across cells.

| Tumor | Patients | Median MAE | Median distance | Median coverage |
| --- | ---: | ---: | ---: | ---: |
| Breast | 109 | 0.34 | 6.3 | 0.57 |
| Colorectal | 24 | 0.34 | 6.5 | 0.67 |
| Lung | 42 | 0.36 | 6.8 | 0.69 |

HDO by center.

| Center | Patients | Median MAE | Median distance | Median coverage |
| --- | ---: | ---: | ---: | ---: |
| ICO Bellvitge | 28 | 0.34 | 6.3 | 0.61 |
| Fuenlabrada | 24 | 0.37 | 7.4 | 0.79 |
| CHUAC | 23 | 0.35 | 6.1 | 0.57 |
| Son Espases | 22 | 0.34 | 5.7 | 0.57 |
| Virgen Macarena | 22 | 0.34 | 6.7 | 0.72 |
| La Princesa | 19 | 0.36 | 6.8 | 0.70 |
| Navarra | 15 | 0.35 | 6.3 | 0.59 |
| Clínico Valencia | 14 | 0.32 | 5.8 | 0.38 |
| San Pedro de Alcántara | 8 | 0.30 | 5.4 | 0.47 |

Patient-level error is in `reconstruction_by_patient.csv`. A similar MAE across cohorts means the observed day is recovered with a similar residual. It does not show that the representation is free of bias.

## 2. Topic occupation

The same frozen LDA. Each document is a window of 30 days with at least one observed continuous variable, without overlap. The windows have the same length. A patient who does not reach 30 observed days is left out of this section. In mental health the topics are read as T0–T5. The oncological prognostic meaning of favorable or unfavorable is not transferred to them.

The comparison is descriptive. The distributions do not need to differ for the representation to be considered valid. Age is not available for the external reference. Coverage and duration are, in the cohort table: a longer follow-up or a different coverage can change the mix even when the model is the same.

Mean occupation of the windows. Each window has equal weight.

| Cohort | Windows | Patients | T0 | T1 | T2 | T3 | T4 | T5 | JS versus HDO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDO | 1,518 | 162 | 0.02 | 0.20 | 0.26 | 0.19 | 0.16 | 0.17 | 0.000 |
| Eating disorders | 17,511 | 1,960 | 0.08 | 0.06 | 0.50 | 0.10 | 0.18 | 0.08 | 0.067 |
| Cognitive impairment | 2,276 | 175 | 0.04 | 0.15 | 0.30 | 0.10 | 0.16 | 0.25 | 0.016 |
| Suicide prevention | 5,807 | 1,008 | 0.09 | 0.08 | 0.27 | 0.21 | 0.21 | 0.14 | 0.028 |
| Depression / Anxiety disorders | 651 | 140 | 0.02 | 0.15 | 0.27 | 0.26 | 0.14 | 0.16 | 0.005 |

HDO by tumor, the same mean of 30-day windows.

| Tumor | Windows | T0 | T1 | T2 | T3 | T4 | T5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Breast | 1,034 | 0.02 | 0.21 | 0.27 | 0.20 | 0.17 | 0.13 |
| Colorectal | 171 | 0.02 | 0.21 | 0.24 | 0.14 | 0.10 | 0.30 |
| Lung | 313 | 0.01 | 0.18 | 0.23 | 0.19 | 0.15 | 0.24 |

HDO by age tertile, only patients with at least one window and age in the clinical file. This remains descriptive.

| Age tertile | Patients | Median age | Median coverage | T0 | T1 | T2 | T3 | T4 | T5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 28–54 | 57 | 48 | 0.61 | 0.01 | 0.17 | 0.40 | 0.15 | 0.19 | 0.07 |
| 55–65 | 51 | 61 | 0.66 | 0.02 | 0.19 | 0.22 | 0.22 | 0.18 | 0.16 |
| 66–88 | 54 | 70 | 0.63 | 0.04 | 0.17 | 0.13 | 0.16 | 0.20 | 0.30 |

## 3. Vocabulary use

Codes from the 256-entry codebook used on observed days. The effective number is the exponential of the entropy. It is used to see whether the assignment collapses or one code takes over a follow-up as an artifact. Uniform occupation of the 256 codes is not required, and neither is a large number of codes per patient.

| Cohort | Codes used | Effective number | Mass of the most frequent | Mass of the top 10 | Most frequent code |
| --- | ---: | ---: | ---: | ---: | ---: |
| HDO | 254 / 256 | 187 | 1.8 % | 14.2 % | 76 |
| Eating disorders | 256 / 256 | 172 | 3.1 % | 20.0 % | 247 |
| Cognitive impairment | 255 / 256 | 185 | 1.8 % | 14.3 % | 247 |
| Suicide prevention | 255 / 256 | 211 | 1.3 % | 10.6 % | 83 |
| Depression / Anxiety disorders | 254 / 256 | 188 | 1.8 % | 13.2 % | 125 |

By patient, within bands of observed duration. The median effective number and the share of the most used code. A short band accumulates fewer codes without that being a collapse.

| Cohort | Duration | Patients | Median days | Median codes | Median effective number | Share of the top code |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HDO | 1–29 days | 12 | 20 | 12 | 9.7 | 16.7 % |
| HDO | 30–89 days | 27 | 64 | 37 | 26.8 | 10.3 % |
| HDO | 90–179 days | 33 | 142 | 61 | 46.9 | 7.1 % |
| HDO | 180 days or more | 102 | 368 | 106 | 68.7 | 6.4 % |
| Eating disorders | 1–29 days | 188 | 14 | 10 | 8.0 | 20.8 % |
| Eating disorders | 30–89 days | 459 | 55 | 30 | 25.2 | 11.1 % |
| Eating disorders | 90–179 days | 457 | 129 | 53 | 38.1 | 9.2 % |
| Eating disorders | 180 days or more | 1,044 | 374 | 94 | 56.8 | 7.4 % |
| Cognitive impairment | 1–29 days | 12 | 4 | 4 | 3.4 | 43.2 % |
| Cognitive impairment | 30–89 days | 12 | 50 | 34 | 27.4 | 10.3 % |
| Cognitive impairment | 90–179 days | 29 | 123 | 58 | 42.3 | 8.1 % |
| Cognitive impairment | 180 days or more | 134 | 469 | 112 | 64.1 | 6.9 % |
| Suicide prevention | 1–29 days | 64 | 21 | 16 | 14.5 | 15.1 % |
| Suicide prevention | 30–89 days | 343 | 54 | 33 | 27.6 | 9.5 % |
| Suicide prevention | 90–179 days | 278 | 129 | 60 | 45.2 | 7.8 % |
| Suicide prevention | 180 days or more | 387 | 297 | 93 | 62.3 | 6.5 % |
| Depression / Anxiety disorders | 30–89 days | 30 | 82 | 40 | 31.9 | 9.2 % |
| Depression / Anxiety disorders | 90–179 days | 78 | 104 | 51 | 37.4 | 9.7 % |
| Depression / Anxiety disorders | 180 days or more | 32 | 250 | 93 | 62.0 | 6.0 % |

## Days covered by the most used day-types

Each observed day counts once. The five most used codes cover 5.9% of Suicide prevention days and 12.7% of Eating disorders days. Each cohort's own top 25 cover 22.7% to 35.1% of its observed days. At 50 codes, the highest share is 51.1%.

## The 25 day-types with the highest mean share

These 25 codes are ranked by the mean share across the five cohorts, so each cohort has the same weight. Together they cover 23.8% of HDO days, 29.0% of Eating disorders days, 26.6% of Cognitive impairment days, 18.3% of Suicide prevention days, 23.8% of Depression / Anxiety disorders days. The largest single share is code 247 in Eating disorders, 3.1% of that cohort's days. Codes in the own top 25 of every cohort: 62.

## 4. Individual examples

The six examples are HDO patients with at least 30 observed days. The criterion is written on each row. If a patient has already been selected, the next one is taken. The day is the VQ-VAE representation: code, distance, margin over the second code and that day's error. The window is the LDA summary: the T0–T5 mix of that patient's 30-day windows. They are not the same object.

| Patient | Tumor | Center | Criterion | Days | Coverage | MAE | Effective number | Share of the top code | Median distance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 91017 | Breast | CHUAC | Closest to the HDO median in error, coverage, effective number of codes and distance to the code | 207 | 0.64 | 0.35 | 52.8 | 6.3 % | 5.8 |
| 51008 | Breast | Son Espases | Lowest coverage among follow-ups of at least 30 observed days | 47 | 0.06 | 0.31 | 20.8 | 12.8 % | 3.1 |
| 31012 | Breast | Clínico Valencia | Highest reconstruction error among high coverage (75th percentile or above) | 198 | 0.93 | 0.52 | 40.3 | 7.1 % | 10.5 |
| 62001 | Colorectal | Navarra | Highest concentration on a single code among follow-ups of at least 90 days | 97 | 0.64 | 0.38 | 14.6 | 27.8 % | 9.1 |
| 41023 | Breast | ICO Bellvitge | Highest effective number of codes among follow-ups of at least 90 days | 360 | 0.65 | 0.41 | 124.0 | 2.8 % | 6.8 |
| 21002 | Breast | La Princesa | Highest quantization distance among coverage at or above the median | 317 | 0.93 | 0.41 | 25.5 | 12.0 % | 12.8 |

Days chosen within each patient: lowest error, median error and highest error. When days with at least four observed continuous variables exist, the choice is made among them. 51008 has no such day: those days are chosen among days with any observed variable, and each contributes a single one. This is the low-coverage case.

| Patient | Day | Date | Code | Distance | Margin | Day error | Observed variables |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 91017 | Lowest-error day | 2025-09-05 | 128 | 7.34 | 0.79 | 0.12 | 5 |
| 91017 | Median-error day | 2025-07-18 | 35 | 4.60 | 0.66 | 0.35 | 5 |
| 91017 | Highest-error day | 2025-03-07 | 66 | 11.03 | 0.85 | 0.80 | 5 |
| 51008 | Lowest-error day | 2024-09-07 | 184 | 2.89 | 2.71 | 0.00 | 1 |
| 51008 | Median-error day | 2024-10-22 | 96 | 2.30 | 0.72 | 0.28 | 1 |
| 51008 | Highest-error day | 2024-10-04 | 21 | 3.15 | 1.30 | 0.66 | 1 |
| 31012 | Lowest-error day | 2025-04-29 | 161 | 6.84 | 1.00 | 0.26 | 6 |
| 31012 | Median-error day | 2025-05-08 | 125 | 8.18 | 4.29 | 0.52 | 8 |
| 31012 | Highest-error day | 2025-08-23 | 20 | 11.37 | 0.17 | 0.96 | 7 |
| 62001 | Lowest-error day | 2023-09-03 | 120 | 8.70 | 0.94 | 0.10 | 4 |
| 62001 | Median-error day | 2023-08-13 | 61 | 10.03 | 1.81 | 0.35 | 7 |
| 62001 | Highest-error day | 2023-08-07 | 97 | 17.05 | 1.79 | 1.10 | 4 |
| 41023 | Lowest-error day | 2025-04-11 | 225 | 8.61 | 3.87 | 0.10 | 7 |
| 41023 | Median-error day | 2024-09-24 | 117 | 5.16 | 1.94 | 0.41 | 8 |
| 41023 | Highest-error day | 2025-05-13 | 92 | 9.10 | 0.29 | 0.89 | 6 |
| 21002 | Lowest-error day | 2023-12-19 | 208 | 7.78 | 1.54 | 0.18 | 8 |
| 21002 | Median-error day | 2023-06-05 | 224 | 13.27 | 5.09 | 0.39 | 8 |
| 21002 | Highest-error day | 2024-03-25 | 125 | 11.03 | 1.09 | 0.85 | 8 |

Mean LDA-window mix for these patients. This summarizes the sequence. It is not the code of the day.

| Patient | Windows | T0 | T1 | T2 | T3 | T4 | T5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 91017 | 6 | 0.01 | 0.19 | 0.73 | 0.04 | 0.02 | 0.02 |
| 51008 | 1 | 0.17 | 0.01 | 0.01 | 0.01 | 0.81 | 0.01 |
| 31012 | 6 | 0.01 | 0.54 | 0.02 | 0.27 | 0.01 | 0.17 |
| 62001 | 3 | 0.01 | 0.01 | 0.01 | 0.01 | 0.08 | 0.90 |
| 41023 | 12 | 0.13 | 0.26 | 0.24 | 0.17 | 0.11 | 0.09 |
| 21002 | 10 | 0.01 | 0.48 | 0.01 | 0.46 | 0.01 | 0.04 |

## Reading

The comparison is with Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders, psychiatry cohorts from the VQ-VAE training. HDO is the study cohort. The observed day is reconstructed on both sides, with the error and quantization distance in the tables. The 30-day windows need not occupy the topics in the same way; any difference is read together with coverage and duration, and without age in the external cohort. The vocabulary is inspected for collapse or a code that takes over a follow-up, not to require that all 256 codes appear. The examples separate the quantized day from the LDA summary.
