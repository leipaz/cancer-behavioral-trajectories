# Cobertura real (punto 2A)

Mismos **1013** puntos de `objetivo.csv` (114 pacientes), horizonte 120 d. Cobertura = % de celdas variable-día **realmente observadas** (9 variables de entrada) sobre el calendario esperado de 90 d y 30 d que termina en la decisión, **antes** de imputar. Bandas SAP fijas: `<50%`, `50–80%`, `≥80%`. Misma alarma y corte `1.0596…`.

Variables (9): sleep_start, location_distance, location_time_home, sleep_duration, activity_walking, app_usage_total, location_clusters_count, steps_steps_total, practiced_sport.

Fuentes (mapa provisional): wearable = sleep_start, sleep_duration, activity_walking, steps_steps_total; smartphone = location_distance, location_time_home, location_clusters_count, app_usage_total, practiced_sport.

## Conteos por banda

```
 window_days   band  n_decisions  n_patients  mean_coverage
          90  >=80%          376          56       0.899760
          30  >=80%          369          57       0.906333
          30 50-80%          409          77       0.648583
          90 50-80%          422          75       0.646738
          30   <50%          235          52       0.314894
          90   <50%          215          41       0.328343
```

## Rendimiento por banda (IC bootstrap por paciente)

```
 window_days   band  n_patients  n_decisions  n_with_score  n_positive performance                AUROC          sensitivity          specificity                  PPV                  NPV
          90   <50%          41          215           215          33          ok 0.742 [0.594, 0.860] 0.939 [0.828, 1.000] 0.549 [0.362, 0.705] 0.274 [0.142, 0.434] 0.980 [0.938, 1.000]
          90 50-80%          75          422           422          59          ok 0.704 [0.585, 0.813] 0.864 [0.727, 0.982] 0.479 [0.363, 0.593] 0.212 [0.138, 0.297] 0.956 [0.902, 0.995]
          90  >=80%          56          376           376          63          ok 0.671 [0.546, 0.795] 0.873 [0.723, 0.983] 0.364 [0.220, 0.515] 0.217 [0.134, 0.335] 0.934 [0.865, 0.991]
          90    all         114         1013          1013         155          ok 0.703 [0.627, 0.780] 0.884 [0.809, 0.957] 0.452 [0.364, 0.540] 0.226 [0.168, 0.281] 0.956 [0.921, 0.983]
          30   <50%          52          235           235          36          ok 0.751 [0.607, 0.869] 0.917 [0.765, 1.000] 0.513 [0.365, 0.660] 0.254 [0.130, 0.400] 0.971 [0.904, 1.000]
          30 50-80%          77          409           409          59          ok 0.695 [0.585, 0.800] 0.881 [0.764, 0.983] 0.489 [0.356, 0.606] 0.225 [0.145, 0.314] 0.961 [0.910, 0.995]
          30  >=80%          57          369           369          60          ok 0.675 [0.553, 0.807] 0.867 [0.711, 0.981] 0.372 [0.234, 0.525] 0.211 [0.127, 0.329] 0.935 [0.869, 0.991]
          30    all         114         1013          1013         155          ok 0.703 [0.627, 0.780] 0.884 [0.809, 0.957] 0.452 [0.364, 0.540] 0.226 [0.168, 0.281] 0.956 [0.921, 0.983]
```

## Cobertura continua e interacción con el score

```
 window_days  n_decisions  n_patients  mean_coverage  spearman_load_coverage  spearman_load_coverage_lo  spearman_load_coverage_hi  spearman_coverage_outcome  spearman_coverage_outcome_lo  spearman_coverage_outcome_hi  AUROC_coverage_alone  AUROC_coverage_alone_lo  AUROC_coverage_alone_hi  AUROC_load  AUROC_load_plus_loadxcov  AUROC_load_plus_loadxcov_lo  AUROC_load_plus_loadxcov_hi
          90         1013         114          0.673                   0.094                     -0.079                      0.272                      0.002                        -0.106                         0.123                 0.502                    0.411                    0.602       0.703                     0.619                        0.538                        0.694
          30         1013         114          0.665                   0.085                     -0.101                      0.254                     -0.003                        -0.108                         0.111                 0.497                    0.414                    0.587       0.703                     0.619                        0.533                        0.704
```

Archivos: `coverage_by_decision.csv`, `performance_by_band.csv`, `availability_by_variable.csv`, `availability_by_center.csv`, `availability_by_tumor.csv`, `continuous_coverage_summary.csv`, `PROVENANCE.json`.

Pendiente (2B): comparador disponibilidad sola vs DCABP vs DCABP+disponibilidad, coordinado con el bloque de comparadores.