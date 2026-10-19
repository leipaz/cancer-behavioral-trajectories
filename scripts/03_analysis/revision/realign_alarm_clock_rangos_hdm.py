#!/usr/bin/env python3
"""Realign alarm landmarks: topic months start at inicio_rangos; Obs_time from Date_start_HDM.

Maps clinical times onto the sequence clock:
    delta = (inicio_rangos - Date_start_HDM).days
    t_evento_seq = Obs_time - delta
    follow_up_seq = Obs_time - delta

Then rebuilds monthly W=3 / H=4 decision points and metrics (paper θ=1.61 + Youden).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/03_analysis/alarm"))

from rolling_burden_alarm import (  # noqa: E402
    PAPER_THRESHOLD,
    build_decision_points,
    extract_monthly_puf_from_master,
    metrics_at_threshold,
    youden_optimal_threshold,
)

REV = PROJECT_ROOT / "data/revision"
OUT = PROJECT_ROOT / "results/alarm/clock_realign_rangos_hdm"
OUT.mkdir(parents=True, exist_ok=True)


def load_deltas() -> pd.DataFrame:
    path = REV / "comparacion_inicio_HDM_vs_rangos.csv"
    if path.exists():
        d = pd.read_csv(path)
        d["id"] = d["id"].astype(int)
        return d[["id", "d_rangos_vs_HDM", "date_change", "Obs_time", "PD_event"]].copy()

    clin = pd.read_excel(PROJECT_ROOT / "data/processed/clinical/Subjects_data.xlsx")
    clin = clin.rename(columns={"S_RECORD_id": "id"})
    clin["id"] = clin["id"].astype(int)
    clin["Date_start_HDM"] = pd.to_datetime(clin["Date_start_HDM"])
    clin["Obs_time"] = pd.to_numeric(clin["Obs_time"], errors="coerce")
    clin["PD_event"] = pd.to_numeric(clin["PD_event"], errors="coerce").fillna(0).astype(int)

    rangos = pd.read_excel(REV / "rangos_fechas.xlsx").rename(
        columns={"S_RECORD_id": "id", "Inicio_datos_DB1": "inicio_rangos"}
    )
    rangos["id"] = rangos["id"].astype(int)
    rangos["inicio_rangos"] = pd.to_datetime(rangos["inicio_rangos"])
    cc = next(c for c in rangos.columns if "change" in c.lower())
    rangos["date_change"] = rangos[cc].fillna(0).astype(int)

    m = rangos.merge(clin[["id", "Date_start_HDM", "Obs_time", "PD_event"]], on="id")
    m["d_rangos_vs_HDM"] = (m["inicio_rangos"] - m["Date_start_HDM"]).dt.days
    return m[["id", "d_rangos_vs_HDM", "date_change", "Obs_time", "PD_event"]]


def summarize(dp: pd.DataFrame, label: str) -> dict:
    paper = metrics_at_threshold(dp, PAPER_THRESHOLD)
    youden = youden_optimal_threshold(dp)
    return {
        "version": label,
        "n_landmarks": len(dp),
        "n_PD_plus": int(dp["target"].sum()),
        "n_patients": int(dp["id"].nunique()),
        "AUROC": paper.roc_auc,
        "paper_theta": PAPER_THRESHOLD,
        "paper_sens": paper.sensitivity,
        "paper_spec": paper.specificity,
        "paper_ppv": paper.ppv,
        "paper_npv": paper.npv,
        "youden_theta": youden.threshold,
        "youden_sens": youden.sensitivity,
        "youden_spec": youden.specificity,
    }


def align_puf(monthly: pd.DataFrame, deltas: pd.DataFrame, min_abs_delta: int = 0) -> pd.DataFrame:
    """Subtract delta from t_evento / follow_up so times are days since sequence start."""
    out = monthly.copy()
    dmap = deltas.set_index("id")["d_rangos_vs_HDM"].to_dict()
    delta = out["id"].map(dmap).fillna(0).astype(float)
    apply = delta.abs() >= min_abs_delta
    out["delta_inicio_rangos_hdm"] = delta
    out["t_evento_eb2_raw"] = out["t_evento_eb2"]
    out["follow_up_days_raw"] = out["follow_up_days"]
    out.loc[apply, "t_evento_eb2"] = out.loc[apply, "t_evento_eb2"] - delta[apply]
    out.loc[apply, "follow_up_days"] = out.loc[apply, "follow_up_days"] - delta[apply]
    return out


def per_id_landmark_delta(dp_base: pd.DataFrame, dp_aligned: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    b = dp_base.groupby("id").size().rename("n_lm_unaligned")
    a = dp_aligned.groupby("id").size().rename("n_lm_aligned")
    tb = dp_base.groupby("id")["target"].sum().rename("n_pos_unaligned")
    ta = dp_aligned.groupby("id")["target"].sum().rename("n_pos_aligned")
    out = (
        deltas.set_index("id")
        .join(b, how="left")
        .join(a, how="left")
        .join(tb, how="left")
        .join(ta, how="left")
        .fillna({"n_lm_unaligned": 0, "n_lm_aligned": 0, "n_pos_unaligned": 0, "n_pos_aligned": 0})
        .reset_index()
    )
    out["dn_landmarks"] = out["n_lm_aligned"] - out["n_lm_unaligned"]
    out["dn_pos"] = out["n_pos_aligned"] - out["n_pos_unaligned"]
    return out.sort_values("d_rangos_vs_HDM", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    master = pd.read_excel(PROJECT_ROOT / "data/processed/DCABPs/topics_probs.xlsx")
    monthly = extract_monthly_puf_from_master(master, root=PROJECT_ROOT)
    deltas = load_deltas()

    # Baseline revision (no clock shift)
    dp0 = build_decision_points(monthly, W=3, H=4)
    rows = [summarize(dp0, "revision_unaligned")]

    variants = [
        (0, "revision_aligned_all_deltas"),
        (1, "revision_aligned_|delta|>=1"),
        (7, "revision_aligned_|delta|>=7"),
        (15, "revision_aligned_|delta|>=15"),
    ]
    dp_aligned_all = None
    for thr, label in variants:
        aligned = align_puf(monthly, deltas, min_abs_delta=thr)
        # drop months whose event/FU became non-positive on sequence clock
        # (build_decision_points already requires t < t_ev)
        dp = build_decision_points(aligned, W=3, H=4)
        rows.append(summarize(dp, label))
        dp.to_csv(OUT / f"decision_points_W3_H4_{label}.csv", index=False)
        if thr == 0:
            dp_aligned_all = dp

    summary = pd.DataFrame(rows)
    # attach archived baselines for convenience
    summary.loc[len(summary)] = {
        "version": "pre_revision_archived",
        "n_landmarks": 1015,
        "n_PD_plus": 159,
        "n_patients": None,
        "AUROC": 0.704,
        "paper_theta": 1.61,
        "paper_sens": 0.704,
        "paper_spec": 0.626,
        "paper_ppv": None,
        "paper_npv": None,
        "youden_theta": None,
        "youden_sens": None,
        "youden_spec": None,
    }
    summary.to_csv(OUT / "alarm_metrics_clock_realign_summary.csv", index=False)

    per_id = per_id_landmark_delta(dp0, dp_aligned_all, deltas)
    per_id.to_csv(OUT / "landmarks_per_id_before_after_align.csv", index=False)

    # Focus table: |delta|>=7
    focus = per_id[per_id["d_rangos_vs_HDM"].abs() >= 7].copy()
    focus.to_csv(OUT / "landmarks_per_id_delta_ge7.csv", index=False)

    print("=== Alarm metrics (W=3, H=4, θ=1.61) ===")
    print(summary.to_string(index=False))
    print(f"\n=== IDs |delta|≥7 with landmark count change (n={len(focus)}) ===")
    cols = [
        "id",
        "d_rangos_vs_HDM",
        "date_change",
        "Obs_time",
        "PD_event",
        "n_lm_unaligned",
        "n_lm_aligned",
        "dn_landmarks",
        "n_pos_unaligned",
        "n_pos_aligned",
        "dn_pos",
    ]
    print(focus[cols].to_string(index=False))
    print(f"\nWrote → {OUT}")


if __name__ == "__main__":
    main()
