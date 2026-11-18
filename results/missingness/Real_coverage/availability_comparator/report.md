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
      availability_only_oof         114      1013         155 0.427 [0.344, 0.504] 0.806 [0.709, 0.886] 0.202 [0.146, 0.257] 0.154 [0.113, 0.202] 0.852 [0.750, 0.922]   0.156717
          dcabp_load_frozen         114      1013         155 0.703 [0.627, 0.780] 0.884 [0.809, 0.957] 0.452 [0.364, 0.540] 0.226 [0.168, 0.281] 0.956 [0.921, 0.983]   1.059638
            dcabp_logit_oof         114      1013         155 0.690 [0.615, 0.768] 0.774 [0.664, 0.881] 0.501 [0.408, 0.593] 0.219 [0.160, 0.277] 0.925 [0.877, 0.963]   0.116882
dcabp_plus_availability_oof         114      1013         155 0.677 [0.600, 0.758] 0.703 [0.578, 0.826] 0.533 [0.440, 0.625] 0.214 [0.155, 0.276] 0.909 [0.858, 0.952]   0.127797
```

Archivos: `availability_indicators.csv`, `scores_by_decision.csv`, `performance_models.csv`, `emission_overlap.csv`, `coefficients_full_fit_descriptive.csv`, `PROVENANCE.json`.