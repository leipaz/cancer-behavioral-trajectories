#!/usr/bin/env python3
"""Anticipation on the 1013 every-30-day landmarks.

For each external patient with an event, the lead is the event study day
minus the first decision day on which the frozen cut gives a true positive.
The event study day is Tiempo_obs_digital + 1. Held-out rows are not used:
a patient can have several inner scores there, and the external refit is the
one series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    (
        ROOT / "results/lda/val_2/anidada_estratificada/alarma/decisiones.csv",
        "outer_fold",
    ),
    (
        ROOT / "results/lda/val_2/loco/alarma/decisiones.csv",
        "left_out_centre",
    ),
)


def one_patient(group: pd.DataFrame) -> dict:
    ordered = group.sort_values("decision_day")
    event_day = float(ordered["t_digital"].iloc[0]) + 1.0
    hits = ordered[(ordered["outcome"].astype(str) == "1") & (ordered["alarm"] == True)]
    if len(hits) == 0:
        day = np.nan
        lead = np.nan
    else:
        day = float(hits["decision_day"].iloc[0])
        lead = event_day - day
    return {
        "id": int(ordered["id"].iloc[0]),
        "event_day": event_day,
        "first_tp_day": day,
        "first_tp_lead": lead,
        "n_tp": int(len(hits)),
    }


def anticipation(decisions: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    external = decisions[
        (decisions["role"] == "external_test") & (decisions["event"] == 1) & (decisions["in_comparison"] == 1)
    ]
    patients = []
    for (fold, pid), group in external.groupby([key, "id"], sort=True):
        row = one_patient(group)
        row[key] = int(fold)
        patients.append(row)
    by_patient = pd.DataFrame(patients)
    summary = []
    for fold, group in by_patient.groupby(key, sort=True):
        leads = group["first_tp_lead"].to_numpy(float)
        detected = leads[np.isfinite(leads)]
        summary.append({
            key: int(fold),
            "n_event": int(len(group)),
            "n_detected": int(len(detected)),
            "median_lead": float(np.median(detected)) if len(detected) else np.nan,
            "q25_lead": float(np.percentile(detected, 25)) if len(detected) else np.nan,
            "q75_lead": float(np.percentile(detected, 75)) if len(detected) else np.nan,
            "min_lead": float(np.min(detected)) if len(detected) else np.nan,
            "max_lead": float(np.max(detected)) if len(detected) else np.nan,
        })
    return by_patient, pd.DataFrame(summary)


def main() -> None:
    for path, key in SOURCES:
        decisions = pd.read_csv(path)
        decisions["alarm"] = decisions["alarm"].map({"True": True, "False": False, True: True, False: False})
        by_patient, summary = anticipation(decisions, key)
        folder = path.parent
        by_patient.to_csv(folder / "anticipacion_por_paciente.csv", index=False)
        summary.to_csv(folder / "anticipacion_resumen.csv", index=False)
        print(path.parent.parent.name)
        print(summary.to_string(index=False))
        if (by_patient["first_tp_lead"].dropna() > 120).any() or (by_patient["first_tp_lead"].dropna() <= 0).any():
            raise SystemExit("a lead falls outside 1 to 120 days")


if __name__ == "__main__":
    main()
