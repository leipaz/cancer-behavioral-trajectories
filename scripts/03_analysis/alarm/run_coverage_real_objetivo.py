#!/usr/bin/env python3
"""Point 2A — real coverage on the 1013 objetivo decisions (SAP bands).

Coverage is measured on really observed cells before VQ-VAE imputation /
densification, on the expected calendar ending at each decision (90 d and
30 d). Bands are fixed a priori: <50%, 50–80%, ≥80%. The frozen alarm and
threshold are applied as-is inside each band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from sport_missing import SPORT_RULE, load_daily_for_missingness  # noqa: E402
from vqvae import constants as c  # noqa: E402

DAILY = ROOT / "data/daily_summaries/DB1/daily_summary_checked.csv"
OBJETIVO = ROOT / "results/lda/val_2/anidada_estratificada/objetivo.csv"
BASELINE = ROOT / "results/missingness/Sensitivity_to_artificial_missingness/alarm_scored_objetivo_baseline.csv"
SUBJECTS = ROOT / "data/processed/clinical/Subjects_data.xlsx"
OUT = ROOT / "results/missingness/Real_coverage"

MEASURE_COLS = list(c.CONTINUOUS_REAL_VALUED_COLS) + list(c.CONTINUOUS_POSITIVE_COLS) + ["practiced_sport"]
assert len(MEASURE_COLS) == 9

# Provisional provenance map until the SAP dictionary file is wired in.
# Sleep, steps and walking can come from the wearable or the phone.
# They are assigned to the wearable, the primary source. The phone is the
# fallback when the wearable has no record; that day-level split is not applied.
# The alternative (no heart rate and no SpO2, with steps or sleep, implies the
# phone) is not used. Location and app use are mobile. practiced_sport is
# derived from activity and is not a source of its own.
SOURCE_OF = {
    "sleep_start": "wearable (primary)",
    "sleep_duration": "wearable (primary)",
    "activity_walking": "wearable (primary)",
    "steps_steps_total": "wearable (primary)",
    "location_distance": "mobile",
    "location_time_home": "mobile",
    "location_clusters_count": "mobile",
    "app_usage_total": "mobile",
    "practiced_sport": "derived from activity",
}
SOURCE_MAP_NOTE = (
    "sleep_start, sleep_duration, activity_walking and steps_steps_total can come from the wearable or the phone. "
    "Assigned to the wearable as the primary source. Mobile is the fallback when the wearable has no record; "
    "that split is not applied day by day. The alternative rule (no heart rate and no SpO2, with steps or sleep present, "
    "implies the phone) is not used. Location and app use are mobile. practiced_sport is derived from activity, "
    "not counted in either source mean."
)
WEARABLE_COLS = [v for v, s in SOURCE_OF.items() if s.startswith("wearable")]
MOBILE_COLS = [v for v, s in SOURCE_OF.items() if s == "mobile"]

THR = 1.0596384140662845
BANDS = (
    ("<50%", -np.inf, 0.50),
    ("50-80%", 0.50, 0.80),
    (">=80%", 0.80, np.inf),
)
N_BOOT = 1000
BOOT_SEED = 20261116
WINDOWS = (90, 30)


def band_of(cov: float) -> str:
    if not np.isfinite(cov):
        return "NA"
    if cov < 0.50:
        return "<50%"
    if cov < 0.80:
        return "50-80%"
    return ">=80%"


def clasif(y, score) -> dict:
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV", "n_scored", "n_patients", "n_positive")}
    pred = s >= THR
    tp = int(((pred) & (y == 1)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())

    def r(a, b):
        return a / b if b else np.nan

    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    return {
        "AUROC": auc,
        "sensitivity": r(tp, tp + fn),
        "specificity": r(tn, tn + fp),
        "PPV": r(tp, tp + fp),
        "NPV": r(tn, tn + fn),
        "n_scored": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }


def patient_bootstrap_ci(frame: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Point metrics + percentile CI resampling patients (keeps all decisions of a patient)."""
    base = clasif(frame["y"], frame["load"])
    patients = frame["id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    keys = ("AUROC", "sensitivity", "specificity", "PPV", "NPV")
    store = {k: [] for k in keys}
    by = {pid: g for pid, g in frame.groupby("id", sort=False)}
    for _ in range(n_boot):
        draw = rng.choice(patients, size=len(patients), replace=True)
        parts = [by[pid] for pid in draw if pid in by]
        if not parts:
            continue
        boot = pd.concat(parts, ignore_index=True)
        m = clasif(boot["y"], boot["load"])
        for k in keys:
            if np.isfinite(m[k]):
                store[k].append(m[k])
    out = dict(base)
    out["n_patients"] = int(frame["id"].nunique())
    for k in keys:
        arr = np.asarray(store[k], float)
        if len(arr) >= 20:
            out[f"{k}_lo"] = float(np.quantile(arr, 0.025))
            out[f"{k}_hi"] = float(np.quantile(arr, 0.975))
        else:
            out[f"{k}_lo"] = np.nan
            out[f"{k}_hi"] = np.nan
    return out


def longest_empty_run(day_present: np.ndarray) -> int:
    best = cur = 0
    for flag in day_present:
        if not flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def n_gaps_over_48h(day_present: np.ndarray) -> int:
    """Count gaps of ≥2 consecutive empty calendar days (≈ >48 h without a day with data)."""
    n = 0
    cur = 0
    for flag in day_present:
        if not flag:
            cur += 1
        else:
            if cur >= 2:
                n += 1
            cur = 0
    if cur >= 2:
        n += 1
    return int(n)


def days_since_last_valid(day_present: np.ndarray) -> int:
    """Empty days at the end of the window since the last day with any measurement."""
    idx = np.flatnonzero(day_present)
    if len(idx) == 0:
        return int(len(day_present))
    return int(len(day_present) - 1 - idx[-1])


def build_patient_matrices(daily: pd.DataFrame) -> dict[int, dict]:
    """Full expected calendar first→last daily date; values before any imputation."""
    out = {}
    for uid, g in daily.groupby("user", sort=True):
        g = g.sort_values("date_time")
        dates = pd.to_datetime(g["date_time"]).dt.normalize()
        start, end = dates.min(), dates.max()
        calendar = pd.date_range(start, end, freq="D")
        indexed = g.copy()
        indexed["date_time"] = dates.to_numpy()
        indexed = indexed.drop_duplicates("date_time", keep="first").set_index("date_time")
        vals = indexed.reindex(calendar)[MEASURE_COLS]
        arr = vals.to_numpy(dtype=float)
        observed = ~np.isnan(arr)  # True = really present (real zeros count as present)
        service = str(g["service"].iloc[0]) if "service" in g.columns else ""
        out[int(uid)] = {
            "calendar": calendar.to_numpy(dtype="datetime64[ns]"),
            "observed": observed,  # (T, 9)
            "has_row": np.asarray(calendar.isin(indexed.index), dtype=bool),
            "service": service,
        }
    return out


def window_slice(calendar: np.ndarray, decision: np.datetime64, n_days: int):
    """Inclusive window of n_days ending at decision_date; keep holes in the calendar."""
    decision = np.datetime64(decision, "ns")
    start = decision - np.timedelta64(n_days - 1, "D")
    # indices on the patient calendar that fall inside [start, decision]
    idx = np.flatnonzero((calendar >= start) & (calendar <= decision))
    # Build a dense expected grid of n_days even if patient calendar is shorter
    expected = pd.date_range(pd.Timestamp(start), pd.Timestamp(decision), freq="D").to_numpy(dtype="datetime64[ns]")
    return expected, idx, start, decision


def coverage_for_window(obs: np.ndarray, calendar: np.ndarray, has_row: np.ndarray, decision, n_days: int) -> dict:
    expected, idx_on_cal, start, end = window_slice(calendar, decision, n_days)
    n_expected_cells = n_days * len(MEASURE_COLS)
    # Map expected days onto patient calendar (missing days = all-False)
    day_obs = np.zeros((n_days, len(MEASURE_COLS)), dtype=bool)
    day_row = np.zeros(n_days, dtype=bool)
    cal_index = {np.datetime64(d, "ns"): i for i, d in enumerate(calendar)}
    for t, day in enumerate(expected):
        j = cal_index.get(np.datetime64(day, "ns"))
        if j is not None:
            day_obs[t] = obs[j]
            day_row[t] = bool(has_row[j])
    n_obs_cells = int(day_obs.sum())
    cov = n_obs_cells / n_expected_cells if n_expected_cells else np.nan
    day_present = day_obs.any(axis=1)
    per_var = day_obs.mean(axis=0)
    wear = day_obs[:, [MEASURE_COLS.index(v) for v in WEARABLE_COLS]]
    phone = day_obs[:, [MEASURE_COLS.index(v) for v in MOBILE_COLS]]
    return {
        "n_days_window": n_days,
        "n_expected_cells": n_expected_cells,
        "n_observed_cells": n_obs_cells,
        "coverage": float(cov),
        "band": band_of(cov),
        "pct_days_with_data": float(day_present.mean()) if n_days else np.nan,
        "pct_days_with_row": float(day_row.mean()) if n_days else np.nan,
        "longest_empty_run_days": longest_empty_run(day_present),
        "n_gaps_ge_48h": n_gaps_over_48h(day_present),
        "days_since_last_valid": days_since_last_valid(day_present),
        "coverage_wearable": float(wear.mean()) if wear.size else np.nan,
        "coverage_mobile": float(phone.mean()) if phone.size else np.nan,
        **{f"cov_{col}": float(per_var[i]) for i, col in enumerate(MEASURE_COLS)},
        "window_start": str(pd.Timestamp(start).date()),
        "window_end": str(pd.Timestamp(end).date()),
    }


def fmt_ci(row: dict, key: str) -> str:
    v, lo, hi = row.get(key), row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if not np.isfinite(v):
        return "NA"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{v:.3f}"


def sport_coverage_on_row(decisions: pd.DataFrame) -> pd.Series:
    """Share of days-with-row on which practiced_sport is a real 0 or 1.

    Kept for checks. The coverage run applies the same rule before the bands.
    """
    from sport_missing import erroneous_sport_user_days

    keys = erroneous_sport_user_days(DAILY)
    bad = set(zip(keys["user"].astype(int), pd.to_datetime(keys["date_time"])))
    daily = pd.read_csv(DAILY, usecols=["user", "date_time", "practiced_sport"], low_memory=False)
    daily["user"] = daily["user"].astype(int)
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    daily["practiced_sport"] = pd.to_numeric(daily["practiced_sport"], errors="coerce")
    looked = daily.drop_duplicates(["user", "date_time"], keep="first").set_index(["user", "date_time"])

    rates = []
    for rec in decisions.itertuples(index=False):
        end = pd.Timestamp(rec.decision_date).normalize()
        days = pd.date_range(end - pd.Timedelta(days=89), end, freq="D")
        n_row = n_obs = 0
        uid = int(rec.id)
        for day in days:
            key = (uid, pd.Timestamp(day))
            if key not in looked.index:
                continue
            n_row += 1
            if key not in bad:
                n_obs += 1
        rates.append(n_obs / n_row if n_row else np.nan)
    return pd.Series(rates, index=decisions.index, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(OBJETIVO)
    points["id"] = points["id"].astype(int)
    points["decision_date"] = pd.to_datetime(points["decision_date"]).dt.normalize()
    points["y"] = points["outcome_120d"].astype(int)
    if len(points) != 1013 or points["id"].nunique() != 114:
        raise SystemExit(f"objetivo is {len(points)} rows / {points.id.nunique()} patients")

    scored = pd.read_csv(BASELINE)
    scored["id"] = scored["id"].astype(int)
    merge_keys = points[["id", "decision_study_day", "decision_date", "y", "event", "block"]].rename(
        columns={"decision_study_day": "study_day"}
    )
    panel = merge_keys.merge(scored, on=["id", "study_day", "y"], how="left", validate="one_to_one")
    if panel["load"].isna().any():
        raise SystemExit("baseline load missing for some objetivo decisions")

    subjects = pd.read_excel(SUBJECTS)
    subjects["id"] = subjects["S_RECORD_id"].astype(int)
    tumor = subjects.set_index("id")["Cohort"].to_dict()

    daily = load_daily_for_missingness(DAILY)
    mats = build_patient_matrices(daily)

    rows = []
    for rec in panel.itertuples(index=False):
        mat = mats.get(int(rec.id))
        if mat is None:
            raise SystemExit(f"patient {rec.id} absent from daily")
        base = {
            "id": int(rec.id),
            "study_day": int(rec.study_day),
            "decision_date": str(pd.Timestamp(rec.decision_date).date()),
            "y": int(rec.y),
            "event": int(rec.event) if pd.notna(rec.event) else np.nan,
            "block": int(rec.block) if pd.notna(getattr(rec, "block", np.nan)) else np.nan,
            "load": float(rec.load),
            "alarm": int(rec.alarm),
            "emitted": int(rec.emitted),
            "center": mat["service"],
            "tumor": tumor.get(int(rec.id), "Unknown"),
        }
        for w in WINDOWS:
            cov = coverage_for_window(mat["observed"], mat["calendar"], mat["has_row"], rec.decision_date, w)
            row = dict(base)
            row["window_days"] = w
            row.update(cov)
            rows.append(row)

    cov_df = pd.DataFrame(rows)
    cov_df.to_csv(OUT / "coverage_by_decision.csv", index=False)

    # Descriptive availability by variable / center / tumor (90 d window).
    # The sport-zero rule is already in the observed matrix, so every variable
    # uses the same days-with-a-row denominator.
    w90 = cov_df.loc[cov_df.window_days == 90].copy()
    days_with_row = w90["pct_days_with_row"].where(w90["pct_days_with_row"] > 0)
    var_rows = []
    for col in MEASURE_COLS:
        s = w90[f"cov_{col}"]
        on_row = s / days_with_row
        var_rows.append(
            {
                "variable": col,
                "source": SOURCE_OF[col],
                "mean_coverage": float(s.mean()),
                "median_coverage": float(s.median()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "mean_coverage_on_row": float(on_row.mean()),
                "median_coverage_on_row": float(on_row.median()),
                "p25_on_row": float(on_row.quantile(0.25)),
                "p75_on_row": float(on_row.quantile(0.75)),
                "mean_missing_on_row": float(1.0 - on_row.mean()),
            }
        )
    pd.DataFrame(var_rows).to_csv(OUT / "availability_by_variable.csv", index=False)

    def describe_group(frame: pd.DataFrame, key: str) -> pd.DataFrame:
        rows_g = []
        for name, g in frame.groupby(key, sort=True):
            rows_g.append(
                {
                    key: name,
                    "n_patients": int(g.id.nunique()),
                    "n_decisions": int(len(g)),
                    "mean_coverage_90": float(g.coverage.mean()),
                    "mean_pct_days_with_data": float(g.pct_days_with_data.mean()),
                    "mean_coverage_wearable": float(g.coverage_wearable.mean()),
                    "mean_coverage_mobile": float(g.coverage_mobile.mean()),
                }
            )
        return pd.DataFrame(rows_g)

    describe_group(w90, "center").to_csv(OUT / "availability_by_center.csv", index=False)
    describe_group(w90, "tumor").to_csv(OUT / "availability_by_tumor.csv", index=False)

    # Performance by SAP band (same alarm + cut)
    perf_rows = []
    for w in WINDOWS:
        sub = cov_df.loc[cov_df.window_days == w].copy()
        for band_name, _, _ in BANDS:
            g = sub.loc[sub.band == band_name]
            if g.empty:
                perf_rows.append(
                    {
                        "window_days": w,
                        "band": band_name,
                        "n_patients": 0,
                        "n_decisions": 0,
                        "n_with_score": 0,
                        "n_positive": 0,
                        "performance": "not_estimable",
                    }
                )
                continue
            with_score = g.loc[g.emitted == 1]
            meta = {
                "window_days": w,
                "band": band_name,
                "n_patients": int(g.id.nunique()),
                "n_decisions": int(len(g)),
                "n_with_score": int(len(with_score)),
                "n_positive": int((with_score.y == 1).sum()) if len(with_score) else 0,
                "n_negative": int((with_score.y == 0).sum()) if len(with_score) else 0,
                "n_distinct_progressions": int(with_score.loc[with_score.y == 1, "id"].nunique()) if len(with_score) else 0,
                "mean_coverage": float(g.coverage.mean()),
            }
            if len(with_score) < 5 or with_score.y.nunique() < 2:
                meta["performance"] = "not_estimable"
                perf_rows.append(meta)
                continue
            metrics = patient_bootstrap_ci(with_score)
            meta["performance"] = "ok"
            meta.update(metrics)
            perf_rows.append(meta)
        # overall reference
        overall = patient_bootstrap_ci(sub)
        overall.update(
            {
                "window_days": w,
                "band": "all",
                "n_decisions": int(len(sub)),
                "n_with_score": int((sub.emitted == 1).sum()),
                "performance": "ok",
                "mean_coverage": float(sub.coverage.mean()),
            }
        )
        perf_rows.append(overall)

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(OUT / "performance_by_band.csv", index=False)

    # Continuous coverage + interaction with score (descriptive + patient-bootstrapped correlations)
    cont_rows = []
    for w in WINDOWS:
        sub = cov_df.loc[(cov_df.window_days == w) & (cov_df.emitted == 1)].copy()
        # standardize for a simple interaction view: load and coverage
        load_z = (sub["load"] - sub["load"].mean()) / sub["load"].std(ddof=0)
        cov_z = (sub["coverage"] - sub["coverage"].mean()) / sub["coverage"].std(ddof=0)
        sub = sub.assign(load_z=load_z, cov_z=cov_z, load_x_cov=load_z * cov_z)

        def spearman_safe(a, b):
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            if len(a) < 5 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
                return np.nan
            return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

        base_corr_load_cov = spearman_safe(sub["load"], sub["coverage"])
        base_corr_cov_y = spearman_safe(sub["coverage"], sub["y"])
        # AUROC of coverage alone and of load (reference)
        auc_cov = float(roc_auc_score(sub["y"], sub["coverage"])) if sub["y"].nunique() > 1 else np.nan
        auc_load = float(roc_auc_score(sub["y"], sub["load"])) if sub["y"].nunique() > 1 else np.nan
        # crude interaction score: load_z + load_z*cov_z (coverage modulates load)
        auc_inter = float(roc_auc_score(sub["y"], sub["load_z"] + sub["load_x_cov"])) if sub["y"].nunique() > 1 else np.nan

        patients = sub["id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(BOOT_SEED + w)
        by = {pid: g for pid, g in sub.groupby("id", sort=False)}
        boot_load_cov, boot_cov_y, boot_auc_cov, boot_auc_load, boot_auc_inter = [], [], [], [], []
        for _ in range(N_BOOT):
            draw = rng.choice(patients, size=len(patients), replace=True)
            boot = pd.concat([by[pid] for pid in draw], ignore_index=True)
            boot_load_cov.append(spearman_safe(boot["load"], boot["coverage"]))
            boot_cov_y.append(spearman_safe(boot["coverage"], boot["y"]))
            if boot["y"].nunique() > 1:
                boot_auc_cov.append(float(roc_auc_score(boot["y"], boot["coverage"])))
                boot_auc_load.append(float(roc_auc_score(boot["y"], boot["load"])))
                lz = (boot["load"] - boot["load"].mean()) / (boot["load"].std(ddof=0) or 1.0)
                cz = (boot["coverage"] - boot["coverage"].mean()) / (boot["coverage"].std(ddof=0) or 1.0)
                boot_auc_inter.append(float(roc_auc_score(boot["y"], lz + lz * cz)))

        def ci(arr, point):
            arr = np.asarray([x for x in arr if np.isfinite(x)], float)
            if len(arr) < 20 or not np.isfinite(point):
                return point, np.nan, np.nan
            return point, float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

        c1 = ci(boot_load_cov, base_corr_load_cov)
        c2 = ci(boot_cov_y, base_corr_cov_y)
        c3 = ci(boot_auc_cov, auc_cov)
        c_load = ci(boot_auc_load, auc_load)
        c4 = ci(boot_auc_inter, auc_inter)
        cont_rows.append(
            {
                "window_days": w,
                "n_decisions": int(len(sub)),
                "n_patients": int(sub.id.nunique()),
                "mean_coverage": float(sub.coverage.mean()),
                "spearman_load_coverage": c1[0],
                "spearman_load_coverage_lo": c1[1],
                "spearman_load_coverage_hi": c1[2],
                "spearman_coverage_outcome": c2[0],
                "spearman_coverage_outcome_lo": c2[1],
                "spearman_coverage_outcome_hi": c2[2],
                "AUROC_coverage_alone": c3[0],
                "AUROC_coverage_alone_lo": c3[1],
                "AUROC_coverage_alone_hi": c3[2],
                "AUROC_load": c_load[0],
                "AUROC_load_lo": c_load[1],
                "AUROC_load_hi": c_load[2],
                "AUROC_load_plus_loadxcov": c4[0],
                "AUROC_load_plus_loadxcov_lo": c4[1],
                "AUROC_load_plus_loadxcov_hi": c4[2],
            }
        )
    cont = pd.DataFrame(cont_rows)
    cont.to_csv(OUT / "continuous_coverage_summary.csv", index=False)

    # Band counts for the report
    band_counts = (
        cov_df.groupby(["window_days", "band"], sort=False)
        .agg(n_decisions=("id", "size"), n_patients=("id", "nunique"), mean_coverage=("coverage", "mean"))
        .reset_index()
    )
    band_counts.to_csv(OUT / "band_counts.csv", index=False)

    provenance = {
        "objetivo": str(OBJETIVO),
        "n_decisions": 1013,
        "n_patients": 114,
        "daily": str(DAILY),
        "baseline_scores": str(BASELINE),
        "threshold": THR,
        "measure_cols": MEASURE_COLS,
        "source_map": SOURCE_OF,
        "source_map_note": SOURCE_MAP_NOTE,
        "source_means": (
            "coverage_wearable averages the four primary-wearable variables. "
            "coverage_mobile averages location and app use. practiced_sport is excluded from both."
        ),
        "bands": ["<50%", "50-80%", ">=80%"],
        "windows_days": list(WINDOWS),
        "expected_cells": {"90": 810, "30": 270},
        "calendar_rule": "Dense expected calendar of N days ending at decision_date; holes kept; not trimmed to last received record.",
        "sport_rule": SPORT_RULE,
        "zeros": "Observed zeros count as data after the range clip; values below the clip floor are blank. practiced_sport has no clip of its own. It is counted only when activity_walking survived [120, 15000] s. A stored sport value on a day whose walking was removed by that filter is missing. A stored 0 with raw walking and running both empty is already missing. This rule is applied before bands, the variable table, centre and tumour means, continuous coverage, and the availability indicators.",
        "variable_table_denominator": "Same 90-day windows. Variable coverage_on_row divides by days that have a daily row (pct_days_with_row), not by all 90 calendar days. cov_practiced_sport counts only days whose walking survived the activity filter.",
        "uncertainty": f"Patient-clustered bootstrap, n={N_BOOT}, seed={BOOT_SEED}",
        "comparator_block": "Not run here; coordinate with comparator pack (availability vs DCABP vs DCABP+availability).",
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))

    def as_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    show = perf.copy()
    for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV"):
        show[k] = show.apply(lambda r, key=k: fmt_ci(r, key), axis=1)
    cols = [
        "window_days",
        "band",
        "n_patients",
        "n_decisions",
        "n_with_score",
        "n_positive",
        "performance",
        "AUROC",
        "sensitivity",
        "specificity",
        "PPV",
        "NPV",
    ]

    # Short markdown report
    lines = [
        "# Cobertura real (punto 2A)",
        "",
        "Mismos **1013** puntos de `objetivo.csv` (114 pacientes), horizonte 120 d. "
        "Cobertura = % de celdas variable-día **realmente observadas** (9 variables de entrada) "
        "sobre el calendario esperado de 90 d y 30 d que termina en la decisión, **antes** de imputar. "
        "Bandas SAP fijas: `<50%`, `50–80%`, `≥80%`. Misma alarma y corte `1.0596…`.",
        "",
        f"Variables ({len(MEASURE_COLS)}): " + ", ".join(MEASURE_COLS) + ".",
        "",
        "Fuentes: wearable (primaria; el móvil es el respaldo, sin separar el día) = "
        + ", ".join(WEARABLE_COLS)
        + "; móvil = "
        + ", ".join(MOBILE_COLS)
        + "; practiced_sport deriva de activity y no entra en ninguna de las dos medias.",
        "",
        "## Conteos por banda",
        "",
        as_md(band_counts),
        "",
        "## Rendimiento por banda (IC bootstrap por paciente)",
        "",
        as_md(show[cols]),
        "",
        "## Cobertura continua e interacción con el score",
        "",
        as_md(cont.round(3)),
        "",
        "Archivos: `coverage_by_decision.csv`, `performance_by_band.csv`, "
        "`availability_by_variable.csv`, `availability_by_center.csv`, "
        "`availability_by_tumor.csv`, `continuous_coverage_summary.csv`, `PROVENANCE.json`.",
        "",
        "Pendiente (2B): comparador disponibilidad sola vs DCABP vs DCABP+disponibilidad, "
        "coordinado con el bloque de comparadores.",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("wrote", OUT)
    print(band_counts.to_string(index=False))
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
