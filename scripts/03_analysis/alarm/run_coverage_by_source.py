#!/usr/bin/env python3
"""Point 2A — coverage by source (wearable vs smartphone) on the 1013 objetivo decisions.

Same 90- and 30-day windows, same daily table and filters as run_coverage_real_objetivo.py.
For each source, a day is observed when at least one of that source's variables is
observed. Reported per decision: share of observed days, number of gaps of >= 2
consecutive empty days, longest empty run, and cell coverage of the source's variables.
practiced_sport is derived from activity and belongs to neither source.

The denominator is the dense calendar window ending at the decision. The script checks
that every window lies inside the patient's digital monitoring period (first to last daily record).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_coverage_real_objetivo as cov  # noqa: E402

OUT = cov.OUT
OBJETIVO = cov.ROOT / "results/alarm/base_analysis/objetivo.csv"
SOURCES = {"wearable": cov.WEARABLE_COLS, "smartphone": cov.MOBILE_COLS}


def day_matrix(mat: dict, decision, n_days: int, last_day=None) -> np.ndarray:
    """Observed cells on the days the alarm used (see cov.window_slice)."""
    expected, _idx, _s, _e = cov.window_slice(mat["calendar"], decision, n_days, last_day)
    cal_index = {np.datetime64(d, "ns"): i for i, d in enumerate(mat["calendar"])}
    out = np.zeros((len(expected), len(cov.MEASURE_COLS)), dtype=bool)
    for t, day in enumerate(expected):
        j = cal_index.get(np.datetime64(day, "ns"))
        if j is not None:
            out[t] = mat["observed"][j]
    return out


def breakdowns(points: pd.DataFrame, mats: dict) -> None:
    """Same columns for variable, source, centre and tumour (90-day window).

    Unit coverage per decision: variable = share of days with that variable;
    source = share of that source's variable-day cells; centre and tumour = share of
    all nine variables x 90 days (the coverage of the bands). A gap is a run of at
    least 2 consecutive days with nothing observed for that unit.
    """
    w = 90
    tumor = pd.read_excel(cov.SUBJECTS).assign(id=lambda d: d["S_RECORD_id"].astype(int)).set_index("id")["Cohort"].to_dict()
    units = {f"variable:{v}": [v] for v in cov.MEASURE_COLS}
    units.update({f"source:{k}": v for k, v in SOURCES.items()})
    units["all:all"] = list(cov.MEASURE_COLS)
    recs = []
    for rec in points.itertuples(index=False):
        mat = mats[int(rec.id)]
        obs = day_matrix(mat, rec.decision_date, w, pd.Timestamp(rec.latest_input_date))
        base = {"id": int(rec.id), "study_day": int(rec.decision_study_day),
                "center": mat["service"].replace("PMP_", ""), "tumor": tumor.get(int(rec.id), "Unknown")}
        for unit, cols in units.items():
            sub = obs[:, [cov.MEASURE_COLS.index(c) for c in cols]]
            present = sub.any(axis=1)
            recs.append({**base, "unit": unit, "coverage": float(sub.mean()),
                         "days_observed": float(present.mean()),
                         "gap_ge_48h": cov.n_gaps_over_48h(present) > 0})
    per = pd.DataFrame(recs)

    def summarise(g: pd.DataFrame) -> dict:
        return {
            "n_patients": int(g["id"].nunique()),
            "n_decisions": int(len(g)),
            "mean_days_observed": float(g.days_observed.mean()),
            "mean_coverage": float(g.coverage.mean()),
            "median_coverage": float(g.coverage.median()),
            "p25": float(g.coverage.quantile(0.25)),
            "p75": float(g.coverage.quantile(0.75)),
            "pct_decisions_with_gap_ge_48h": float(g.gap_ge_48h.mean()),
        }

    out = []
    for unit, g in per[per.unit != "all:all"].groupby("unit", sort=False):
        level, name = unit.split(":")
        out.append({"level": level, "group": name, **summarise(g)})
    allv = per[per.unit == "all:all"]
    for level in ("center", "tumor"):
        for name, g in allv.groupby(level, sort=True):
            out.append({"level": level, "group": name, **summarise(g)})
    out.append({"level": "all", "group": "all", **summarise(allv)})
    pd.DataFrame(out).to_csv(OUT / "coverage_breakdowns.csv", index=False)


def main() -> None:
    points = pd.read_csv(OBJETIVO)
    points["decision_date"] = pd.to_datetime(points["decision_date"]).dt.normalize()
    if len(points) != 1013:
        raise SystemExit(f"objetivo has {len(points)} rows")
    daily = cov.load_daily_for_missingness(cov.DAILY)
    mats = cov.build_patient_matrices(daily)

    rows = []
    for rec in points.itertuples(index=False):
        mat = mats[int(rec.id)]
        for w in cov.WINDOWS:
            expected, _i, w_start, w_end = cov.window_slice(mat["calendar"], rec.decision_date, w, pd.Timestamp(rec.latest_input_date))
            # Digital monitoring period: first to last daily record of the patient.
            inside = bool(w_start >= mat["calendar"][0]) and bool(w_end <= mat["calendar"][-1])
            obs = day_matrix(mat, rec.decision_date, w, pd.Timestamp(rec.latest_input_date))
            for src, cols in SOURCES.items():
                sub = obs[:, [cov.MEASURE_COLS.index(c) for c in cols]]
                present = sub.any(axis=1)
                rows.append({
                    "id": int(rec.id),
                    "study_day": int(rec.decision_study_day),
                    "window_days": w,
                    "source": src,
                    "n_variables": len(cols),
                    "pct_days_observed": float(present.mean()),
                    "n_gaps_ge_48h": cov.n_gaps_over_48h(present),
                    "longest_empty_run_days": cov.longest_empty_run(present),
                    "cell_coverage": float(sub.mean()),
                    "window_inside_follow_up": bool(inside),
                })
    per = pd.DataFrame(rows)
    breakdowns(points, mats)
    if not per["window_inside_follow_up"].all():
        bad = per.loc[~per.window_inside_follow_up, ["id", "study_day"]].drop_duplicates()
        raise SystemExit(f"{len(bad)} windows fall outside follow-up")
    per.to_csv(OUT / "coverage_by_source_by_decision.csv", index=False)

    def q(x, p):
        return float(np.quantile(x, p))

    summ = []
    for (w, src), g in per.groupby(["window_days", "source"], sort=False):
        summ.append({
            "window_days": w,
            "source": src,
            "variables": ", ".join(SOURCES[src]),
            "n_decisions": int(len(g)),
            "n_patients": int(g["id"].nunique()),
            "pct_days_observed_mean": float(g.pct_days_observed.mean()),
            "pct_days_observed_median": float(g.pct_days_observed.median()),
            "pct_days_observed_p25": q(g.pct_days_observed, 0.25),
            "pct_days_observed_p75": q(g.pct_days_observed, 0.75),
            "cell_coverage_mean": float(g.cell_coverage.mean()),
            "n_gaps_ge_48h_mean": float(g.n_gaps_ge_48h.mean()),
            "pct_decisions_with_gap_ge_48h": float((g.n_gaps_ge_48h > 0).mean()),
            "longest_empty_run_median": float(g.longest_empty_run_days.median()),
            "longest_empty_run_p75": q(g.longest_empty_run_days, 0.75),
            "longest_empty_run_max": int(g.longest_empty_run_days.max()),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "coverage_by_source_summary.csv", index=False)

    prov_path = OUT / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    prov["coverage_by_source"] = {
        "script": "scripts/03_analysis/alarm/run_coverage_by_source.py",
        "sources": SOURCES,
        "day_observed": "at least one of the source's variables observed after the range filters",
        "gap": "run of >= 2 consecutive calendar days with no observation from that source",
        "denominator": "the calendar days the alarm used: 90 or 30 days ending at the decision, or, in the 6 decisions whose data end earlier, from the start of the first block to the last day with data; every window lies inside the patient's digital monitoring period (first to last daily record), not the clinical follow-up; no separate activation or end date per device is available",
        "outputs": ["coverage_by_source_by_decision.csv", "coverage_by_source_summary.csv", "coverage_breakdowns.csv"],
        "breakdowns": "coverage_breakdowns.csv: variable, source, centre and tumour with the same columns (90 d, full calendar denominator); unit coverage mean, median, IQR across decisions, and share of decisions with a gap of >= 2 days with nothing observed for that unit",
    }
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["variables"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
