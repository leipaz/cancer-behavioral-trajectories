# Comparador disponibilidad vs DCABP (punto 2B)

Mismos **1013** puntos / horizonte **120 d** / solo **DB1**. Modelos ajustados en las 1013 decisiones y evaluados como la carga congelada (sin CV). IC = bootstrap por paciente; diferencia de AUROC emparejada frente a la congelada.

## Indicadores de disponibilidad (por decisión, ventana 90 d)

- `prop_absent_days_90`: proporción de días sin ninguna medida
- `n_gaps_ge_48h_90`: número de huecos ≥ 48 h (≥ 2 días vacíos seguidos)
- `days_since_last_valid_90`: días desde el último registro válido (hueco actual)
- `monitoring_interrupted_now`: 1 si ese hueco actual es ≥ 2 días (sin usar abandonos futuros)

## Emisión de predicción

```
 n_decisions  n_patients  n_dcabp  n_availability  n_combined  n_dcabp_only  n_availability_only  n_neither  n_both
        1013         114     1013            1013        1013             0                    0          0    1013
```

## Rendimiento

```
                  model  n_patients  n_scored  n_positive                AUROC              AUROC_diff          sensitivity          specificity                  PPV                  NPV  threshold
      availability_only         114      1013         155 0.586 [0.503, 0.662] -0.116 [-0.233, -0.012] 0.845 [0.748, 0.927] 0.287 [0.227, 0.348] 0.176 [0.129, 0.226] 0.911 [0.853, 0.958]   0.139591
      dcabp_load_frozen         114      1013         155 0.702 [0.627, 0.780]                       — 0.884 [0.809, 0.957] 0.451 [0.362, 0.539] 0.225 [0.168, 0.280] 0.956 [0.921, 0.983]   1.059638
            dcabp_logit         114      1013         155 0.702 [0.627, 0.780]    0.000 [0.000, 0.000] 0.884 [0.809, 0.957] 0.451 [0.362, 0.539] 0.225 [0.168, 0.280] 0.956 [0.921, 0.983]   0.101954
dcabp_plus_availability         114      1013         155 0.711 [0.637, 0.786]   0.008 [-0.003, 0.022] 0.794 [0.692, 0.890] 0.548 [0.464, 0.629] 0.241 [0.179, 0.304] 0.936 [0.896, 0.969]   0.127966
```

Archivos: `availability_indicators.csv`, `scores_by_decision.csv`, `performance_models.csv`, `emission_overlap.csv`, `coefficients_full_fit_descriptive.csv`, `PROVENANCE.json`.