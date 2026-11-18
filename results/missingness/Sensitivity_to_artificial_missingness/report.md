# Sensitivity to artificial missingness

Masks, VQ-VAE a0 and LDA run on all 175 DB1 patients. Topics, pUF, the three-block load, F/U and the alarm use the 1013 decisions in `objetivo.csv`: 114 patients, 155 positive and 858 negative.

A stored practiced_sport 0 is missing when raw walking and running are both empty. Those cells are not observed, so the extra mask cannot select them. A 0 with either duration present stays 0. Recorded ones stay as stored.

The extra mask is applied before VQ-VAE imputation. The calendar, origin, windows, frozen VQ-VAE a0, LDA, F/U (topics 3–5) and the full threshold **1.0596384140662845** stay fixed. No parameters are re-learned.

At baseline, AUROC 0.703, sensitivity 0.884, specificity 0.452, PPV 0.226, NPV 0.956. The baseline load is the same as before this rule, because a0 fills a missing sport cell with 0.

On the 175 patients the starting missingness uses the same patient-mean definition in every row. Fully empty days 9.2% (83 with none, 92 with at least one). Variables inside a day that still has a measurement 35.6% (median 34.6%, tertiles 23.9% and 44.5%). Over the whole calendar, with a fully empty day as nine empty variables, the mean is 40.0%.

Each masked value below is the mean of seeds 101–105. The bracket is the 95% confidence interval of that seed-mean. It is not a patient-resampling bootstrap.

| Added | AUROC | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| baseline | 0.703 | 0.884 | 0.452 | 0.226 | 0.956 |
| +10% days | 0.705 [0.696, 0.715] | 0.861 [0.847, 0.874] | 0.448 [0.442, 0.453] | 0.220 [0.217, 0.222] | 0.947 [0.942, 0.951] |
| +20% days | 0.704 [0.698, 0.711] | 0.866 [0.850, 0.881] | 0.425 [0.420, 0.431] | 0.214 [0.210, 0.218] | 0.946 [0.940, 0.952] |
| +30% days | 0.698 [0.692, 0.705] | 0.862 [0.838, 0.886] | 0.406 [0.395, 0.417] | 0.208 [0.201, 0.214] | 0.942 [0.932, 0.952] |
| +40% days | 0.691 [0.683, 0.699] | 0.863 [0.838, 0.888] | 0.388 [0.374, 0.402] | 0.203 [0.201, 0.205] | 0.940 [0.932, 0.949] |
| +50% days | 0.675 [0.665, 0.685] | 0.835 [0.801, 0.869] | 0.410 [0.403, 0.417] | 0.204 [0.198, 0.209] | 0.932 [0.920, 0.945] |
| +10% variables | 0.708 [0.702, 0.714] | 0.858 [0.843, 0.873] | 0.450 [0.440, 0.460] | 0.220 [0.215, 0.225] | 0.946 [0.940, 0.952] |
| +20% variables | 0.709 [0.697, 0.721] | 0.870 [0.849, 0.890] | 0.433 [0.420, 0.446] | 0.217 [0.210, 0.224] | 0.948 [0.940, 0.957] |
| +30% variables | 0.699 [0.692, 0.705] | 0.876 [0.866, 0.887] | 0.398 [0.396, 0.401] | 0.208 [0.206, 0.210] | 0.947 [0.943, 0.951] |
| +40% variables | 0.688 [0.680, 0.696] | 0.889 [0.871, 0.907] | 0.338 [0.327, 0.349] | 0.195 [0.192, 0.198] | 0.944 [0.936, 0.952] |
| +50% variables | 0.676 [0.665, 0.686] | 0.897 [0.871, 0.922] | 0.286 [0.278, 0.295] | 0.185 [0.182, 0.189] | 0.939 [0.926, 0.952] |
| +10% consecutive days | 0.692 [0.675, 0.709] | 0.819 [0.795, 0.844] | 0.516 [0.503, 0.529] | 0.234 [0.225, 0.243] | 0.940 [0.932, 0.949] |
| +20% consecutive days | 0.670 [0.644, 0.696] | 0.686 [0.658, 0.715] | 0.587 [0.570, 0.605] | 0.231 [0.223, 0.239] | 0.912 [0.905, 0.919] |
| +30% consecutive days | 0.649 [0.615, 0.683] | 0.561 [0.512, 0.611] | 0.656 [0.635, 0.677] | 0.228 [0.207, 0.249] | 0.892 [0.880, 0.904] |
| +40% consecutive days | 0.642 [0.607, 0.677] | 0.436 [0.367, 0.505] | 0.731 [0.709, 0.753] | 0.226 [0.198, 0.255] | 0.878 [0.865, 0.891] |
| +50% consecutive days | 0.642 [0.609, 0.675] | 0.329 [0.263, 0.395] | 0.802 [0.787, 0.816] | 0.230 [0.195, 0.264] | 0.869 [0.858, 0.879] |

For scattered days and for variables, AUROC stays near 0.70 through +30%, and at +50% it is 0.675 and 0.676. Consecutive days lower the load: at +20% the AUROC is 0.670, against 0.704 when those days are scattered, and 146 alarms turn off with none turning on. At +50% a consecutive block turns 386 alarms off.

## Contiguous gaps of 3, 7 and 14 days

One extra block, separate from the 10–50% ramp. On each of the 175 DB1 patients a single run of 3, 7 or 14 days is hidden, placed at random with seeds 101–105. Every one of the 1013 decisions still has a load.

| Gap | Posterior JS | Posterior MAE | Dominant | Threshold | Turns off | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 days | <0.001 | 0.001 [0.001, 0.002] | 0.995 [0.991, 0.998] | 0.995 [0.991, 0.999] | 3 [0, 6] | 0.703 [0.702, 0.704] | 0.876 [0.862, 0.890] | 0.452 [0.451, 0.454] |
| 7 days | <0.001 | 0.004 [0.003, 0.005] | 0.992 [0.987, 0.996] | 0.992 [0.987, 0.996] | 8 [5, 11] | 0.696 [0.692, 0.699] | 0.862 [0.847, 0.876] | 0.456 [0.455, 0.458] |
| 14 days | 0.001 | 0.010 [0.009, 0.011] | 0.960 [0.955, 0.965] | 0.983 [0.979, 0.987] | 17 [13, 21] | 0.688 [0.680, 0.696] | 0.852 [0.835, 0.869] | 0.466 [0.462, 0.470] |

Per-seed file: `results/missingness/Sensitivity_to_artificial_missingness/gaps_3_7_14_metrics_by_seed.csv`.
