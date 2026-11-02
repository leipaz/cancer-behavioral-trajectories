#!/usr/bin/env python3
"""Alarm follow-up on digital time, for DB1, DB2 or DB3.

Tiempo_obs_digital = Tiempo_obs - (fecha_inicio_daily - fecha_inicio).days.
The subtraction removes only the days before the profile exists. When the
two start dates match, the digital time equals Tiempo_obs. A one-day start
shift changes the digital time by one day and by nothing else.
Day 1 is the first daily day, so the event falls on study day
Tiempo_obs_digital + 1. A landmark at study day T is

  positive    if evento = 1 and Tiempo_obs_digital + 1 <= T + horizon
  negative    if Tiempo_obs_digital + 1 >= T + horizon
  excluded    if T is on or after the event study day
  unresolved  otherwise

The stored time is never Tiempo_obs_digital + 1. The +1 is only the
study-day index used by the horizon.

    python scripts/03_analysis/alarm/digital_followup.py --db DB1
    python scripts/03_analysis/alarm/digital_followup.py --db DB2
    python scripts/03_analysis/alarm/digital_followup.py --db DB3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_DAYS = 120
WINDOW_MONTHS = 3
UNFAVOURABLE = (3, 4, 5)

ROOT = Path(__file__).resolve().parents[3]


def tiempo_obs_digital(tiempo_obs, fecha_inicio, fecha_inicio_daily) -> pd.Series:
    """Observation time after removing the gap before the first profile day.

    Blank when either start date is missing. The result is not Tiempo_obs + 1.
    """
    inicio = pd.to_datetime(fecha_inicio)
    diario = pd.to_datetime(fecha_inicio_daily)
    gap = (diario - inicio).dt.days
    out = pd.to_numeric(tiempo_obs, errors="coerce") - gap
    return out.where(inicio.notna() & diario.notna())


def label_landmark(day: float, t_digital: float, event: int, horizon: int = HORIZON_DAYS) -> str:
    """Label one decision day. ``t_digital`` is the date difference, not the study day."""
    if day >= t_digital + 1:
        return "drop"
    if event == 1 and t_digital + 1 <= day + horizon:
        return "1"
    if t_digital + 1 >= day + horizon:
        return "0"
    return "ne"


def collapsed_landmarks(
    monthly: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    horizon: int = HORIZON_DAYS,
    window: int = WINDOW_MONTHS,
) -> pd.DataFrame:
    """Last ``window`` monthly pUF values at each month end that the digital horizon can judge.

    ``monthly`` needs ``id``, ``month``, ``decision_day``, ``pUF``.
    ``patients`` needs ``id``, ``t_digital``, ``evento``. Months are taken in
    ``month`` order per patient. Only evaluable landmarks are returned.
    """
    need_m = {"id", "month", "decision_day", "pUF"}
    need_p = {"id", "t_digital", "evento"}
    missing = (need_m - set(monthly.columns)) | (need_p - set(patients.columns))
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    meta = patients.set_index("id")
    rows: list[dict] = []
    for pid, g in monthly.groupby("id"):
        if pid not in meta.index:
            continue
        g = g.sort_values("month")
        t_digital = float(meta.loc[pid, "t_digital"])
        event = int(meta.loc[pid, "evento"])
        if not np.isfinite(t_digital):
            continue
        pufs = g.pUF.to_numpy(float)
        days = g.decision_day.to_numpy(float)
        months = g.month.to_numpy(int)
        for i in range(window - 1, len(g)):
            block = pufs[i - window + 1 : i + 1]
            if np.isnan(block).any():
                continue
            day = float(days[i])
            outcome = label_landmark(day, t_digital, event, horizon)
            if outcome not in {"0", "1"}:
                continue
            score = float(block.sum())
            rows.append(
                {
                    "id": int(pid),
                    "month": int(months[i]),
                    "decision_day": day,
                    "W": window,
                    "H": horizon // 30,
                    "AUC_W": score,
                    "AUC_W_per_month": score / window,
                    "pUF": float(pufs[i]),
                    "target": int(outcome),
                    "Evento PD": event,
                    "t_digital": int(t_digital),
                }
            )
    return pd.DataFrame(rows).sort_values(["id", "decision_day"]).reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _puf_from_topics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pUF" not in out.columns:
        out["pUF"] = out[[f"topic{k}" for k in UNFAVOURABLE]].sum(axis=1)
    return out


def _first_block_start(db: str) -> pd.DataFrame:
    path = ROOT / "results" / "by_db" / db / "topics_bloques_rangos_long.csv"
    blocks = pd.read_csv(path)
    regular = blocks[blocks["es_ventana_regular"].astype(bool)]
    start = regular.groupby("id")["block_start"].min().rename("fecha_inicio_daily")
    return start.reset_index()


def load_patients(db: str) -> pd.DataFrame:
    """One row per patient. Digital time is Tiempo_obs minus the start gap."""
    if db == "DB1":
        audit = _read(ROOT / "results/by_db/DB1/178_audit_oct_2_2026_topics_revision_v2.xlsx")
        audit = audit.dropna(subset=["id"]).copy()
        rang = pd.read_csv(ROOT / "results/by_db/DB1/topics_meses_rangos_fechas.csv")
        patients = audit.merge(rang[["id", "fecha_inicio_daily"]], on="id", how="left")
        patients = patients.rename(
            columns={
                "Fecha_inicio_actualizada": "fecha_inicio",
                "Tiempo_obs_DB1": "tiempo_obs",
                "Evento_DB1": "evento",
            }
        )
    elif db == "DB2":
        clinical = _read(ROOT / "data/processed/clinical/DB2/fechas_cnio_oct_2_2026.xlsx")
        clinical = clinical.rename(
            columns={"S_RECORD_id": "id", "Inicio_datos_DB2": "fecha_inicio", "Evento_DB2": "evento"}
        )
        clinical["tiempo_obs"] = digital_days_from_end(clinical["Fin_datos_DB2"], clinical["fecha_inicio"])
        patients = clinical.merge(_first_block_start("DB2"), on="id", how="left")
    elif db == "DB3":
        clinical = _read(ROOT / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026.xlsx")
        clinical = clinical.rename(columns={"S_RECORD_id": "id", "Inicio_datos_DB3": "fecha_inicio"})
        clinical["tiempo_obs"] = digital_days_from_end(clinical["Fin_datos_DB3"], clinical["fecha_inicio"])
        patients = clinical.merge(_first_block_start("DB3"), on="id", how="left")
        patients["evento"] = np.nan
    else:
        raise ValueError(f"unknown database {db!r}; choose DB1, DB2 or DB3")
    patients["t_digital"] = tiempo_obs_digital(
        patients["tiempo_obs"], patients["fecha_inicio"], patients["fecha_inicio_daily"]
    )
    return patients


def digital_days_from_end(fecha_fin, fecha_inicio) -> pd.Series:
    """Observation time when the database stores the two dates and not Tiempo_obs."""
    return (pd.to_datetime(fecha_fin) - pd.to_datetime(fecha_inicio)).dt.days


def load_monthly(db: str) -> pd.DataFrame | None:
    if db == "DB1":
        path = ROOT / "results/alarm/runs/Evento_DB1_clock/paper_W3_H4/monthly_puf.csv"
        monthly = pd.read_csv(path)
        return monthly[["id", "month", "decision_day", "pUF"]]
    if db == "DB2":
        path = ROOT / "results/by_db/DB2/topics_mensuales_anclados_long.csv"
        monthly = _puf_from_topics(pd.read_csv(path))
        monthly = monthly.rename(columns={"mes_rel": "month"})
        return monthly[["id", "month", "decision_day", "pUF"]]
    return None


def run(db: str) -> Path:
    out = ROOT / "results/alarm/runs" / f"{db}_digital"
    out.mkdir(parents=True, exist_ok=True)
    patients = load_patients(db)
    keep = ["id", "fecha_inicio", "fecha_inicio_daily", "tiempo_obs", "t_digital", "evento"]
    patients[keep].to_csv(out / "tiempo_obs_digital.csv", index=False)
    monthly = load_monthly(db)
    if monthly is None or patients["evento"].isna().all():
        print(f"{db}: wrote digital time for {patients['t_digital'].notna().sum()} patients.")
        print(f"{db}: no event flag, so collapsed landmarks are not labelled.")
        print(f"→ {out}")
        return out
    ready = patients.dropna(subset=["t_digital", "evento"])
    landmarks = collapsed_landmarks(monthly, ready)
    landmarks.to_csv(out / "decision_points_collapsed.csv", index=False)
    n_pos = int((landmarks.target == 1).sum())
    n_neg = int((landmarks.target == 0).sum())
    print(f"{db}: {len(landmarks)} collapsed landmarks ({n_pos} positive, {n_neg} negative), {landmarks.id.nunique()} patients")
    print(f"→ {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapsed alarm landmarks on Tiempo_obs_digital")
    parser.add_argument("--db", required=True, choices=("DB1", "DB2", "DB3"))
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
