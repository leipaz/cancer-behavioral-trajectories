# Reconstruction of each variable on the May 2026 daily

The repository contains the frozen VQ-VAE a0, its scaler and the inference code. It does not contain the training loop. The module returns one reconstructed value for each input channel, a quantization term and a code index. The input channels are the ten signal columns. The columns `emotion_inter`, `emotion_intra`, `suicide`, `phq9` and `whodas` are not among them. This pass applies the frozen model and measures how each continuous variable is reconstructed. It does not refit the model.

The daily is `daily_summary_eb2prod_may26.csv`. The programs below are the Table S3 cohorts that this extract can name. Every patient of those programs is included. The cap used in the HDO comparison is not applied here. Error is the mean absolute error in the scaled space, on continuous components that were actually observed, after the frozen clip, `log1p` and scaler. A day with no continuous variable inside range does not enter the error.

Table S3 also lists Bipolar disorder, 51 participants. This extract has no service under that name, so that row is not in the table. HDO is not in this daily. Its identifier overlap with the extract is zero.

| Cohort | Published N | Patients in the extract | Patients with an observed day | Observed days | Scaled MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eating disorders | 2,464 | 3,401 | 2,970 | 698,049 | 0.31 |
| Cognitive impairment | 197 | 858 | 792 | 227,275 | 0.32 |
| Suicide prevention | 2,555 | 1,091 | 1,072 | 189,048 | 0.35 |
| Depression / Anxiety disorders | 149 | 140 | 140 | 21,489 | 0.35 |
| Cancer Patients | 93 | 100 | 93 | 15,035 | 0.32 |
| HIV Patients | 24 | 13 | 13 | 1,504 | 0.32 |

Cognitive impairment pools prevention and follow-up, because the extract does not separate them. Depression / Anxiety disorders is the TMC service. Eating disorders and Cognitive impairment in this file are larger than the published counts. Suicide prevention, Depression / Anxiety disorders and HIV Patients are smaller. Cancer Patients has 100 identifiers and 93 of them have an observed day.

Scaled mean absolute error by variable.

| Variable | Eating disorders | Cognitive impairment | Suicide prevention | Depression / Anxiety disorders | Cancer Patients | HIV Patients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sleep onset | 0.15 | 0.17 | 0.19 | 0.19 | 0.16 | 0.16 |
| Distance | 0.33 | 0.38 | 0.35 | 0.38 | 0.44 | 0.35 |
| Time at home | 0.32 | 0.35 | 0.36 | 0.38 | 0.34 | 0.36 |
| Sleep duration | 0.38 | 0.39 | 0.46 | 0.46 | 0.40 | 0.42 |
| Walking | 0.34 | 0.35 | 0.36 | 0.37 | 0.34 | 0.31 |
| App use | 0.31 | 0.32 | 0.37 | 0.36 | 0.31 | 0.32 |
| Places | 0.31 | 0.31 | 0.29 | 0.33 | 0.31 | 0.31 |
| Steps | 0.32 | 0.36 | 0.37 | 0.37 | 0.37 | 0.33 |

The same residual across these cohorts means each observed variable is recovered with a similar error. It does not show that the representation is free of bias. The median absolute error in original units, the correlation and the cell counts are in `reconstruction_by_variable.csv`. Patient-level error is in `patients_*.csv`.
