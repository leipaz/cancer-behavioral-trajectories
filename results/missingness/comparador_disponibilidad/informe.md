# Comparador disponibilidad vs DCABP (punto 2B)

Mismos **1013** puntos / horizonte **120 d** / solo **DB1**. CV por paciente (GroupKFold 5). IC = bootstrap por paciente sobre scores OOF (o load congelado).

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
                      model  n_patients  n_scored  n_positive                AUROC          sensitivity          specificity                  PPV                  NPV  threshold
      availability_only_oof         114      1013         155 0.402 [0.320, 0.483] 0.923 [0.857, 0.977] 0.029 [0.011, 0.054] 0.147 [0.108, 0.189] 0.676 [0.389, 0.889]   0.152533
          dcabp_load_frozen         114      1013         155 0.703 [0.627, 0.780] 0.884 [0.809, 0.957] 0.452 [0.364, 0.540] 0.226 [0.168, 0.281] 0.956 [0.921, 0.983]   1.059638
            dcabp_logit_oof         114      1013         155 0.684 [0.607, 0.762] 0.845 [0.750, 0.939] 0.476 [0.386, 0.565] 0.225 [0.166, 0.284] 0.944 [0.902, 0.978]   0.109475
dcabp_plus_availability_oof         114      1013         155 0.677 [0.596, 0.752] 0.806 [0.705, 0.905] 0.505 [0.415, 0.589] 0.227 [0.167, 0.287] 0.935 [0.893, 0.969]   0.119912
```

Archivos: `availability_indicators.csv`, `scores_by_decision.csv`, `performance_models.csv`, `emission_overlap.csv`, `coefficients_full_fit_descriptive.csv`, `PROVENANCE.json`.