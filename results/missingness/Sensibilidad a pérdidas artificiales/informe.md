# Sensitivity to artificial missingness

Masks, VQ-VAE a0 and LDA run on all 175 DB1 patients. Topics, pUF, the three-block load, F/U and the alarm use the 1013 decisions in `objetivo.csv`: 114 patients, 155 positive and 858 negative.

Every one of those 1013 decisions has a load. Six end after the encoded profile (22001 day 420, 31003 day 360, 31005 day 360, 41022 day 150, 51008 day 180, 71005 day 90) and take the last 30 profile days, the same overlap as the collapsed month. The baseline load matches the stored collapsed load, MAE 0.0019.

The threshold stays at 1.0596. At baseline, AUROC 0.703, sensitivity 0.884, specificity 0.452, PPV 0.226, NPV 0.956. Of the 1013 loads, 1006 meet the 15-day rule, and 113 of the 114 patients have at least one such point. The alarm metrics use all 1013 loads.

On the 175 patients the starting missingness is: fully empty days 1.4% (152 with none, 23 with at least one); variables inside a day that still has a measurement 35.7% (median 34.7%, tertiles 23.5% and 43.2%); the same count over the whole calendar is 36.1%.

Each masked value below is the mean of seeds 101–105. The bracket is the 95% confidence interval of that mean. The full tables and figures are in `informe.docx`. Per-seed file: `metricas_por_semilla.csv`.

| Added | AUROC | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| baseline | 0.703 | 0.884 | 0.452 | 0.226 | 0.956 |
| +10% days | 0.702 [0.698, 0.706] | 0.859 [0.838, 0.881] | 0.447 [0.442, 0.451] | 0.219 [0.214, 0.224] | 0.946 [0.938, 0.954] |
| +20% days | 0.709 [0.702, 0.716] | 0.859 [0.835, 0.884] | 0.430 [0.424, 0.436] | 0.214 [0.210, 0.218] | 0.944 [0.936, 0.953] |
| +30% days | 0.701 [0.691, 0.711] | 0.875 [0.868, 0.882] | 0.392 [0.383, 0.401] | 0.206 [0.204, 0.209] | 0.946 [0.943, 0.948] |
| +40% days | 0.686 [0.679, 0.693] | 0.871 [0.846, 0.896] | 0.385 [0.381, 0.389] | 0.204 [0.198, 0.209] | 0.943 [0.932, 0.954] |
| +50% days | 0.680 [0.660, 0.700] | 0.825 [0.786, 0.863] | 0.412 [0.396, 0.428] | 0.202 [0.192, 0.213] | 0.929 [0.912, 0.945] |
| +10% variables | 0.705 [0.698, 0.712] | 0.857 [0.846, 0.867] | 0.450 [0.445, 0.454] | 0.220 [0.218, 0.221] | 0.946 [0.942, 0.949] |
| +20% variables | 0.709 [0.704, 0.715] | 0.875 [0.866, 0.884] | 0.434 [0.426, 0.443] | 0.218 [0.215, 0.222] | 0.951 [0.947, 0.954] |
| +30% variables | 0.705 [0.700, 0.711] | 0.885 [0.870, 0.901] | 0.395 [0.388, 0.401] | 0.209 [0.206, 0.212] | 0.950 [0.944, 0.956] |
| +40% variables | 0.697 [0.689, 0.705] | 0.899 [0.869, 0.930] | 0.350 [0.343, 0.357] | 0.200 [0.194, 0.206] | 0.951 [0.936, 0.965] |
| +50% variables | 0.674 [0.662, 0.686] | 0.902 [0.883, 0.921] | 0.282 [0.266, 0.297] | 0.185 [0.179, 0.190] | 0.941 [0.928, 0.953] |
| +10% consecutive days | 0.692 [0.675, 0.709] | 0.818 [0.795, 0.841] | 0.516 [0.503, 0.529] | 0.234 [0.225, 0.243] | 0.940 [0.932, 0.948] |
| +20% consecutive days | 0.670 [0.644, 0.696] | 0.684 [0.654, 0.714] | 0.587 [0.570, 0.605] | 0.230 [0.223, 0.238] | 0.911 [0.905, 0.918] |
| +30% consecutive days | 0.648 [0.615, 0.682] | 0.560 [0.509, 0.611] | 0.656 [0.635, 0.677] | 0.227 [0.206, 0.249] | 0.892 [0.879, 0.904] |
| +40% consecutive days | 0.642 [0.607, 0.677] | 0.437 [0.371, 0.504] | 0.731 [0.709, 0.753] | 0.227 [0.199, 0.255] | 0.878 [0.866, 0.891] |
| +50% consecutive days | 0.641 [0.611, 0.672] | 0.326 [0.259, 0.394] | 0.801 [0.786, 0.816] | 0.228 [0.193, 0.263] | 0.868 [0.858, 0.879] |

For scattered days and for variables, AUROC stays near 0.70 through +30%, and at +50% it is 0.680 and 0.674. Consecutive days lower the load: at +20% the AUROC is 0.670, against 0.709 when those days are scattered, and 147 alarms turn off with none turning on. At +50% a consecutive block turns 386 alarms off.

The 15-day rule is separate from the load. At +50% scattered days, 833 of the 1006 points that met it are lost and 43 of the 113 patients lose every such point. With variables, 39 points and 1 patient. A consecutive block loses 730 points and 33 patients.

## Contiguous gaps of 3, 7 and 14 days

One extra block, separate from the 10–50% ramp. On each of the 175 DB1 patients a single run of 3, 7 or 14 calendar days is hidden, placed at random with seeds 101–105. The same VQ-VAE, LDA and threshold are kept. Two patients have a calendar shorter than 14 days, so that gap covers their whole span. Every one of the 1013 decisions still has a load.

The posterior is the mean topic distribution of those 1013 months. Its Jensen–Shannon divergence from the unmasked run is below 0.001 at 3 and 7 days and 0.001 at 14 days. The mean absolute difference of the six probabilities is 0.001, 0.004 and 0.010. Dominant-state agreement is the fraction of months whose winning topic stays the same. Threshold-crossing agreement is the fraction whose alarm stays on the same side of 1.0596. Availability is the 15-day rule: at baseline 1006 points meet it.

| Gap | Posterior JS | Posterior MAE | Dominant | Threshold | Turns off | Points lost | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 days | <0.001 | 0.001 [0.001, 0.002] | 0.995 [0.991, 0.998] | 0.995 [0.991, 0.999] | 3 [0, 6] | 0 [0, 1] | 0.703 [0.702, 0.704] | 0.876 [0.862, 0.890] | 0.452 [0.451, 0.454] |
| 7 days | <0.001 | 0.004 [0.003, 0.005] | 0.992 [0.987, 0.996] | 0.992 [0.987, 0.996] | 8 [5, 11] | 1 [0, 1] | 0.696 [0.692, 0.699] | 0.862 [0.847, 0.876] | 0.456 [0.455, 0.458] |
| 14 days | 0.001 | 0.010 [0.009, 0.011] | 0.960 [0.955, 0.965] | 0.983 [0.979, 0.987] | 17 [13, 21] | 3 [1, 4] | 0.688 [0.680, 0.696] | 0.852 [0.835, 0.869] | 0.466 [0.462, 0.470] |

Per-seed file: `results/missingness/huecos_3_7_14/metricas_por_semilla.csv`.
