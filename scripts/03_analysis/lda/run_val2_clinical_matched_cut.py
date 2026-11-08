#!/usr/bin/env python3
"""Clinical alarm with the cut restricted to the refit's scale.

Reads the existing clinical scores and does not change them. For each outer
fold the Youden cut is chosen only on held-out decisions whose model has the
same number of unfavourable topics as that fold's refit. The refit scores of
the external patients are then compared with that cut. Writes
alarma_clinica_selcorte and leaves alarma_clinica untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_val2_anticipation import anticipation  # noqa: E402
from run_val2_fu import cut_rows, rendimiento, youden_cut  # noqa: E402
from run_val2_intervals import SEED, one_group  # noqa: E402

ROOT = _SCRIPT_DIR.parents[2]
SOURCE = ROOT / "results/lda/val_2/anidada_estratificada/alarma_clinica/decisiones.csv"
OUT = ROOT / "results/lda/val_2/anidada_estratificada/alarma_clinica_selcorte"


def intervals_of(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in frame.groupby(["outer_fold", "role"], sort=True):
        record = one_group(group, rng)
        record["outer_fold"], record["role"] = key
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    decisions = pd.read_csv(SOURCE)
    decisions["outcome"] = decisions["outcome"].astype(str)
    refit_count = (
        decisions.loc[decisions["role"] == "external_test", ["outer_fold", "n_U"]]
        .drop_duplicates()
        .set_index("outer_fold")["n_U"]
    )
    if refit_count.index.duplicated().any():
        raise SystemExit("un reajuste tiene más de un número de topics desfavorables")

    decisions["n_U_refit"] = decisions["outer_fold"].map(refit_count)
    decisions["entra_en_corte"] = (
        (decisions["role"] == "held_out") & (decisions["n_U"] == decisions["n_U_refit"])
    ).astype(int)

    cuts = []
    threshold_of = {}
    for outer_fold, frame in decisions.groupby("outer_fold"):
        matched = cut_rows(frame[frame["entra_en_corte"] == 1])
        found = youden_cut(matched["score"].to_numpy(float), matched["outcome"].astype(int).to_numpy())
        if found is None:
            raise SystemExit(f"vuelta {outer_fold} sin corte")
        found["outer_fold"] = int(outer_fold)
        found["n_U_refit"] = int(refit_count.loc[outer_fold])
        found["modelos"] = ",".join(sorted(matched["model"].unique()))
        found["tie_rule"] = "menor corte"
        found["seleccion"] = "held_out con el mismo n_U que el reajuste"
        found["comparison"] = "1013 every 30 days"
        found["map"] = "adjudicacion clinica"
        cuts.append(found)
        threshold_of[int(outer_fold)] = found["threshold"]

    decisions["threshold"] = decisions["outer_fold"].map(threshold_of)
    decisions["alarm"] = decisions["score"] >= decisions["threshold"]
    decisions.loc[~np.isfinite(decisions["score"]), "alarm"] = pd.NA

    reported = decisions[(decisions["role"] == "external_test") | (decisions["entra_en_corte"] == 1)].copy()
    metrics = rendimiento(reported, ["outer_fold", "role"])
    intervals = intervals_of(reported)
    merged = metrics.merge(intervals, on=["outer_fold", "role"], suffixes=("", "_i"))
    for name in ("sensitivity", "specificity", "PPV", "NPV", "AUROC"):
        gap = (merged[name] - merged[f"{name}_i"]).abs().max()
        if gap > 1e-9:
            raise SystemExit(f"{name} difiere del intervalo en {gap}")
    if (merged["patients_detected"] - merged["n_detected"]).abs().max() != 0:
        raise SystemExit("las pacientes detectadas no coinciden")

    by_patient, summary = anticipation(decisions, "outer_fold")
    finite = by_patient["first_tp_lead"].dropna()
    if (finite <= 0).any() or (finite > 120).any():
        raise SystemExit("una anticipación cae fuera de 1 a 120 días")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cuts).to_csv(OUT / "cortes.csv", index=False)
    decisions.to_csv(OUT / "decisiones.csv", index=False)
    metrics.to_csv(OUT / "rendimiento_por_vuelta.csv", index=False)
    intervals.to_csv(OUT / "intervalos_bootstrap.csv", index=False)
    by_patient.to_csv(OUT / "anticipacion_por_paciente.csv", index=False)
    summary.to_csv(OUT / "anticipacion_resumen.csv", index=False)

    external = metrics[metrics.role == "external_test"]
    print(pd.DataFrame(cuts)[["outer_fold", "n_U_refit", "threshold", "n_decisions", "sensitivity", "specificity", "youden", "modelos"]].to_string(index=False))
    print(external[["outer_fold", "n_decisions", "sensitivity", "specificity", "PPV", "AUROC", "patients_with_event", "patients_detected", "patients_false_alarm"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
