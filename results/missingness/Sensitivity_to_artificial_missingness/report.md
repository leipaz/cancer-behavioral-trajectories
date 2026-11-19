# Sensitivity to artificial missingness

Masks, VQ-VAE a0 and LDA run on all 175 DB1 patients. Topics, pUF, the three-block load, F/U and the alarm use the 1013 decisions in `objetivo.csv`: 114 patients, 155 positive and 858 negative.

practiced_sport: a stored 1 is always kept. A stored 0 counts only when activity_walking survived 120–15000 seconds; otherwise it is missing, so the extra mask cannot select it.

The extra mask is applied before VQ-VAE imputation. The calendar, origin, windows, frozen VQ-VAE a0, LDA, F/U (topics 3–5) and the full threshold **1.0596384140662845** stay fixed. No parameters are re-learned.

The threshold stays at 1.0596384140662845. The baseline is the frozen alarm: the stored load of the original pipeline on the 1,013 decisions, with AUROC 0.702, sensitivity 0.884, specificity 0.451, PPV 0.225, NPV 0.956. A mask changes the input, so every scenario is re-encoded with VQ-VAE a0 and the LDA. Re-encoding without any mask does not give back the stored load exactly (MAE 0.0019; one decision crosses the threshold), because a few days get other codes. So each masked load is anchored to the frozen one: frozen load + (load with the mask − load re-encoded without it). The baseline is then the frozen alarm exactly, every alarm metric below refers to it, and each scenario carries only the change caused by its mask. Topic and load-change metrics do not depend on the anchor. The sport rule does not change the model input: every stored 1 is kept, and a0 reads a missing sport cell as 0, which is what was stored there.

On the 175 patients the starting missingness uses the same patient-mean definition in every row. Fully empty days 9.4% (81 with none, 94 with at least one). Variables inside a day that still has a measurement 36.3% (median 35.4%, tertiles 24.7% and 44.5%). Over the whole calendar, with a fully empty day as nine empty variables, the mean is 40.7%.

Each masked value below is the mean of seeds 101–105. The bracket is the 95% confidence interval of that seed-mean. It is not a patient-resampling bootstrap.

| Added | AUROC | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| baseline | 0.702 | 0.884 | 0.451 | 0.225 | 0.956 |
| +10% days | 0.706 [0.699, 0.713] | 0.862 [0.845, 0.879] | 0.441 [0.432, 0.451] | 0.218 [0.213, 0.223] | 0.947 [0.940, 0.953] |
| +20% days | 0.706 [0.699, 0.713] | 0.859 [0.844, 0.875] | 0.431 [0.416, 0.446] | 0.214 [0.210, 0.219] | 0.944 [0.939, 0.950] |
| +30% days | 0.691 [0.682, 0.700] | 0.865 [0.853, 0.876] | 0.401 [0.394, 0.409] | 0.207 [0.203, 0.211] | 0.943 [0.937, 0.948] |
| +40% days | 0.695 [0.685, 0.705] | 0.875 [0.863, 0.887] | 0.385 [0.376, 0.393] | 0.204 [0.201, 0.208] | 0.944 [0.939, 0.950] |
| +50% days | 0.677 [0.663, 0.691] | 0.834 [0.813, 0.854] | 0.409 [0.403, 0.414] | 0.203 [0.199, 0.207] | 0.931 [0.924, 0.939] |
| +10% variables | 0.708 [0.702, 0.713] | 0.862 [0.846, 0.878] | 0.451 [0.443, 0.460] | 0.221 [0.217, 0.226] | 0.948 [0.942, 0.954] |
| +20% variables | 0.709 [0.702, 0.716] | 0.870 [0.855, 0.884] | 0.432 [0.426, 0.438] | 0.217 [0.213, 0.220] | 0.948 [0.943, 0.954] |
| +30% variables | 0.700 [0.687, 0.713] | 0.875 [0.860, 0.889] | 0.393 [0.378, 0.408] | 0.207 [0.201, 0.213] | 0.946 [0.938, 0.953] |
| +40% variables | 0.696 [0.687, 0.705] | 0.901 [0.871, 0.930] | 0.341 [0.324, 0.358] | 0.198 [0.191, 0.206] | 0.950 [0.935, 0.965] |
| +50% variables | 0.678 [0.669, 0.687] | 0.899 [0.881, 0.918] | 0.286 [0.275, 0.297] | 0.185 [0.181, 0.190] | 0.940 [0.929, 0.952] |
| +10% consecutive days | 0.692 [0.675, 0.709] | 0.818 [0.795, 0.841] | 0.515 [0.502, 0.528] | 0.234 [0.225, 0.242] | 0.940 [0.932, 0.948] |
| +20% consecutive days | 0.669 [0.643, 0.695] | 0.684 [0.654, 0.714] | 0.586 [0.569, 0.603] | 0.230 [0.223, 0.237] | 0.911 [0.905, 0.918] |
| +30% consecutive days | 0.648 [0.616, 0.681] | 0.559 [0.510, 0.608] | 0.655 [0.634, 0.676] | 0.226 [0.206, 0.247] | 0.891 [0.880, 0.903] |
| +40% consecutive days | 0.641 [0.610, 0.673] | 0.435 [0.368, 0.502] | 0.730 [0.709, 0.752] | 0.225 [0.197, 0.254] | 0.877 [0.865, 0.890] |
| +50% consecutive days | 0.636 [0.601, 0.671] | 0.328 [0.264, 0.392] | 0.801 [0.787, 0.815] | 0.229 [0.195, 0.262] | 0.868 [0.858, 0.879] |

For scattered days and for variables, AUROC stays near 0.70 through +30%, and at +50% it is 0.677 and 0.678. Consecutive days lower the load: at +20% the AUROC is 0.669, against 0.706 when those days are scattered, and 147 alarms turn off with none turning on. At +50% a consecutive block turns 387 alarms off.

## Contiguous gaps of 3, 7 and 14 days

One extra block, separate from the 10–50% ramp. On each of the 175 DB1 patients a single run of 3, 7 or 14 days is hidden, placed at random with seeds 101–105. Every one of the 1013 decisions still has a load.

| Gap | Posterior JS | Posterior MAE | Dominant | Threshold | Turns off | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 days | <0.001 | 0.001 [0.001, 0.002] | 0.995 [0.991, 0.998] | 0.995 [0.991, 0.999] | 4 [0, 7] | 0.703 [0.701, 0.704] | 0.872 [0.857, 0.888] | 0.452 [0.449, 0.454] |
| 7 days | <0.001 | 0.004 [0.003, 0.005] | 0.992 [0.987, 0.996] | 0.991 [0.988, 0.994] | 9 [6, 11] | 0.695 [0.692, 0.699] | 0.859 [0.844, 0.875] | 0.456 [0.454, 0.458] |
| 14 days | 0.001 | 0.010 [0.009, 0.011] | 0.960 [0.955, 0.965] | 0.983 [0.978, 0.987] | 17 [13, 22] | 0.687 [0.679, 0.696] | 0.852 [0.835, 0.869] | 0.465 [0.461, 0.470] |

Per-seed file: `results/missingness/Sensitivity_to_artificial_missingness/gaps_3_7_14_metrics_by_seed.csv`.
