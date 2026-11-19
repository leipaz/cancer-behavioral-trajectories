#!/usr/bin/env python3
"""Point 2B — missingness-only comparator on the 1013 objetivo decisions.

Three scores on the same 120-day landmarks (DB1 only):

1) availability alone
2) DCABP load (frozen alarm score)
3) DCABP load + availability indicators

Availability indicators (past-only, before imputation), primary window 90 d:
- proportion of absent days
- number of gaps ≥ 48 h (≥ 2 consecutive empty calendar days)
- days since last valid record (current gap at the decision)
- monitoring interrupted now (current gap ≥ 2 days)

The logistic models are fitted on the 1013 decisions and evaluated exactly as the
frozen load: fixed scores on the same decisions, no cross-validation, Youden cut on
those decisions, patient bootstrap, paired AUROC difference vs the frozen load.
No DB2/DB3 tuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
COVERAGE = ROOT / "results/missingness/Real_coverage/coverage_by_decision.csv"
OUT = ROOT / "results/missingness/Real_coverage/availability_comparator"

THR_LOAD = 1.0596384140662845
N_BOOT = 1000
BOOT_SEED = 20261116

# Prespecified availability feature set (90 d history + current gap).
AVAIL_FEATURES = [
    "prop_absent_days_90",
    "n_gaps_ge_48h_90",
    "days_since_last_valid_90",
    "monitoring_interrupted_now",
    "cell_coverage_90",
]
N_SPLITS = 5
CV_SEED = 20261116


def clasif(y, score, thr=None) -> dict:
    """thr: scalar or per-row array matching score."""
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if thr is None:
        thr_arr = np.full(len(s), 0.5)
    else:
        thr_arr = np.asarray(thr, float)
        if thr_arr.ndim == 0:
            thr_arr = np.full(len(s), float(thr_arr))
        else:
            thr_arr = thr_arr[ok]
    out = {
        "n_scored": int(len(y)),
        "n_positive": int((y == 1).sum()) if len(y) else 0,
        "n_negative": int((y == 0).sum()) if len(y) else 0,
        "AUROC": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "PPV": np.nan,
        "NPV": np.nan,
        "threshold": float(np.nanmedian(thr_arr)) if len(thr_arr) else np.nan,
    }
    if len(y) == 0:
        return out
    if len(np.unique(y)) > 1:
        out["AUROC"] = float(roc_auc_score(y, s))
    pred = s >= thr_arr

    def r(a, b):
        return a / b if b else np.nan

    tp = int(((pred) & (y == 1)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    out.update(
        {
            "sensitivity": r(tp, tp + fn),
            "specificity": r(tn, tn + fp),
            "PPV": r(tp, tp + fp),
            "NPV": r(tn, tn + fn),
        }
    )
    return out


def patient_bootstrap_ci(
    frame: pd.DataFrame,
    score_col: str,
    thr,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    thr_col: str | None = None,
) -> dict:
    if thr_col is not None:
        thr_arg = frame[thr_col].to_numpy()
    else:
        thr_arg = thr
    base = clasif(frame["y"], frame[score_col], thr=thr_arg)
    patients = frame["id"].drop_duplicates().to_numpy()
    by = {pid: g for pid, g in frame.groupby("id", sort=False)}
    rng = np.random.default_rng(seed)
    keys = ("AUROC", "sensitivity", "specificity", "PPV", "NPV")
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([by[pid] for pid in draw], ignore_index=True)
        thr_boot = boot[thr_col].to_numpy() if thr_col is not None else thr
        m = clasif(boot["y"], boot[score_col], thr=thr_boot)
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


def fmt_ci(row: dict, key: str) -> str:
    v, lo, hi = row.get(key), row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if not np.isfinite(v):
        return "NA"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{v:.3f}"


def fitted_logistic(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Logistic fitted on all decisions; score and Youden cut on the same decisions (as the frozen load)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    y = frame["y"].to_numpy(dtype=int)
    pipe.fit(frame[feature_cols].to_numpy(dtype=float), y)
    p = pipe.predict_proba(frame[feature_cols].to_numpy(dtype=float))[:, 1]
    return p, np.full(len(p), _youden_threshold(y, p))


def paired_auc_diff(frame: pd.DataFrame, score_col: str, ref_col: str = "load",
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """AUROC(score) - AUROC(frozen load) with the same resampled patients."""
    point = roc_auc_score(frame["y"], frame[score_col]) - roc_auc_score(frame["y"], frame[ref_col])
    patients = frame["id"].drop_duplicates().to_numpy()
    by = {pid: g for pid, g in frame.groupby("id", sort=False)}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        boot = pd.concat([by[pid] for pid in rng.choice(patients, size=len(patients), replace=True)], ignore_index=True)
        if boot["y"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(boot["y"], boot[score_col]) - roc_auc_score(boot["y"], boot[ref_col]))
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def patient_folds(frame: pd.DataFrame) -> np.ndarray:
    """One fixed patient-to-fold map, shared by every cross-validated model."""
    ids = np.sort(frame["id"].unique())
    rng = np.random.default_rng(CV_SEED)
    rng.shuffle(ids)
    fold_of = {pid: i % N_SPLITS for i, pid in enumerate(ids)}
    return frame["id"].map(fold_of).to_numpy()


def oof_logistic(frame: pd.DataFrame, feature_cols: list[str], folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probability of progression; Youden cut chosen on the training folds."""
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=int)
    oof = np.full(len(frame), np.nan)
    thr = np.full(len(frame), np.nan)
    for k in range(N_SPLITS):
        te, tr = folds == k, folds != k
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        if list(pipe.named_steps["clf"].classes_) != [0, 1]:
            raise RuntimeError("class order is not [no progression, progression]")
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        thr[te] = _youden_threshold(y[tr], pipe.predict_proba(X[tr])[:, 1])
    if np.isnan(oof).any():
        raise RuntimeError("out-of-fold scores incomplete")
    return oof, thr


def _youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity − 1 on the provided set."""
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(score)
    s = score[order]
    yy = y[order]
    # cumulative from high score end
    tp = np.cumsum(yy[::-1])
    fp = np.cumsum(1 - yy[::-1])
    P = max(int(yy.sum()), 1)
    N = max(int((1 - yy).sum()), 1)
    sens = tp / P
    spec = 1.0 - (fp / N)
    j = sens + spec - 1.0
    best = int(np.argmax(j))
    # threshold just below the selected high-score cut in reversed order
    thr_candidates = s[::-1]
    thr = float(thr_candidates[best])
    # Midpoint with next lower score when possible
    if best + 1 < len(thr_candidates):
        thr = 0.5 * (thr + float(thr_candidates[best + 1]))
    return thr


def build_indicator_table(cov: pd.DataFrame) -> pd.DataFrame:
    w90 = cov.loc[cov.window_days == 90].copy()
    w30 = cov.loc[cov.window_days == 30, ["id", "study_day", "pct_days_with_data", "n_gaps_ge_48h", "days_since_last_valid"]].copy()
    w30 = w30.rename(
        columns={
            "pct_days_with_data": "pct_days_with_data_30",
            "n_gaps_ge_48h": "n_gaps_ge_48h_30",
            "days_since_last_valid": "days_since_last_valid_30",
        }
    )
    panel = w90.merge(w30, on=["id", "study_day"], how="left", validate="one_to_one")
    panel["prop_absent_days_90"] = 1.0 - panel["pct_days_with_data"]
    panel["prop_absent_days_30"] = 1.0 - panel["pct_days_with_data_30"]
    panel["n_gaps_ge_48h_90"] = panel["n_gaps_ge_48h"]
    panel["days_since_last_valid_90"] = panel["days_since_last_valid"]
    # Known interruption *at the decision*: ongoing gap > 48 h (no future withdrawal used).
    panel["monitoring_interrupted_now"] = (panel["days_since_last_valid_90"] >= 2).astype(int)
    # Partial missingness: share of the 9 variables x 90 days really observed (same as section A).
    panel["cell_coverage_90"] = panel["coverage"]
    panel["can_emit_dcabp"] = (panel["emitted"] == 1) & panel["load"].notna()
    # Availability features are defined for every landmark with a 90 d window row.
    panel["can_emit_availability"] = panel[AVAIL_FEATURES].notna().all(axis=1)
    panel["can_emit_combined"] = panel["can_emit_dcabp"] & panel["can_emit_availability"]
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(COVERAGE)
    if cov.loc[cov.window_days == 90].shape[0] != 1013:
        raise SystemExit("expected 1013 rows in coverage 90 d window")

    panel = build_indicator_table(cov)
    panel = panel.sort_values(["id", "study_day"]).reset_index(drop=True)

    # Emission overlap
    emission = {
        "n_decisions": int(len(panel)),
        "n_patients": int(panel.id.nunique()),
        "n_dcabp": int(panel.can_emit_dcabp.sum()),
        "n_availability": int(panel.can_emit_availability.sum()),
        "n_combined": int(panel.can_emit_combined.sum()),
        "n_dcabp_only": int((panel.can_emit_dcabp & ~panel.can_emit_availability).sum()),
        "n_availability_only": int((panel.can_emit_availability & ~panel.can_emit_dcabp).sum()),
        "n_neither": int((~panel.can_emit_dcabp & ~panel.can_emit_availability).sum()),
        "n_both": int((panel.can_emit_dcabp & panel.can_emit_availability).sum()),
    }
    pd.DataFrame([emission]).to_csv(OUT / "emission_overlap.csv", index=False)

    # Scores
    panel["score_dcabp_load"] = panel["load"].astype(float)
    panel["score_dcabp_alarm"] = (panel["load"] >= THR_LOAD).astype(float)

    both = panel.loc[panel.can_emit_combined].copy()
    both["score_avail"], both["thr_avail"] = fitted_logistic(both, AVAIL_FEATURES)
    both["score_dcabp_logit"], both["thr_dcabp_logit"] = fitted_logistic(both, ["load"])
    both["score_combined"], both["thr_combined"] = fitted_logistic(both, ["load"] + AVAIL_FEATURES)

    # Reviewer's design: the same patient folds for every logistic, out-of-fold scores.
    folds = patient_folds(both)
    both["cv_fold"] = folds
    both["score_avail_cv"], both["thr_avail_cv"] = oof_logistic(both, AVAIL_FEATURES, folds)
    both["score_load_cv"], both["thr_load_cv"] = oof_logistic(both, ["load"], folds)
    both["score_combined_cv"], both["thr_combined_cv"] = oof_logistic(both, ["load"] + AVAIL_FEATURES, folds)

    # Persist per-decision scores (NaN where a method cannot emit)
    scores = panel[
        [
            "id",
            "study_day",
            "decision_date",
            "y",
            "load",
            "alarm",
            "prop_absent_days_90",
            "prop_absent_days_30",
            "n_gaps_ge_48h_90",
            "days_since_last_valid_90",
            "monitoring_interrupted_now",
            "can_emit_dcabp",
            "can_emit_availability",
            "can_emit_combined",
        ]
    ].merge(
        both[
            [
                "id",
                "study_day",
                "score_avail",
                "thr_avail",
                "score_dcabp_logit",
                "thr_dcabp_logit",
                "score_combined",
                "thr_combined",
                "cv_fold",
                "score_avail_cv",
                "score_load_cv",
                "score_combined_cv",
            ]
        ],
        on=["id", "study_day"],
        how="left",
    )
    scores["score_dcabp_load"] = panel["score_dcabp_load"]
    scores.to_csv(OUT / "scores_by_decision.csv", index=False)
    panel[AVAIL_FEATURES + ["prop_absent_days_30", "id", "study_day", "y", "load"]].to_csv(
        OUT / "availability_indicators.csv", index=False
    )

    # Performance table
    specs = [
        ("availability_only", "score_avail", None, "thr_avail", both),
        ("dcabp_load_frozen", "load", THR_LOAD, None, both),
        ("dcabp_logit", "score_dcabp_logit", None, "thr_dcabp_logit", both),
        ("dcabp_plus_availability", "score_combined", None, "thr_combined", both),
    ]

    rows = []
    for name, col, thr, thr_col, frame in specs:
        m = patient_bootstrap_ci(frame, col, thr=thr, thr_col=thr_col)
        m["model"] = name
        m["score_type"] = "fitted_prob_youden_same_decisions" if thr_col else "frozen_load_threshold"
        if col == "load":
            m["AUROC_diff_vs_frozen"] = m["AUROC_diff_lo"] = m["AUROC_diff_hi"] = np.nan
        else:
            m["AUROC_diff_vs_frozen"], m["AUROC_diff_lo"], m["AUROC_diff_hi"] = paired_auc_diff(frame, col)
        rows.append(m)

    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "performance_models.csv", index=False)

    cv_rows = []
    for name, col, thr_col, ref in [
        ("availability_only_cv", "score_avail_cv", "thr_avail_cv", None),
        ("load_logistic_cv", "score_load_cv", "thr_load_cv", None),
        ("load_plus_availability_cv", "score_combined_cv", "thr_combined_cv", "score_load_cv"),
    ]:
        m = patient_bootstrap_ci(both, col, thr=None, thr_col=thr_col)
        m["model"] = name
        if ref is None:
            m["AUROC_diff_vs_load_cv"] = m["AUROC_diff_lo"] = m["AUROC_diff_hi"] = np.nan
        else:
            m["AUROC_diff_vs_load_cv"], m["AUROC_diff_lo"], m["AUROC_diff_hi"] = paired_auc_diff(both, col, ref_col=ref)
        cv_rows.append(m)
    perf_cv = pd.DataFrame(cv_rows)
    perf_cv.to_csv(OUT / "performance_models_cv.csv", index=False)

    # Coefficients of the fitted models (the same fits that give the scores)
    def fit_coefs(cols):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
            ]
        )
        pipe.fit(both[cols], both["y"])
        clf = pipe.named_steps["clf"]
        return pd.DataFrame({"feature": cols, "coef_standardized": clf.coef_.ravel(), "intercept": clf.intercept_[0]})

    coefs = pd.concat(
        [
            fit_coefs(AVAIL_FEATURES).assign(model="availability_only"),
            fit_coefs(["load"]).assign(model="dcabp_logit"),
            fit_coefs(["load"] + AVAIL_FEATURES).assign(model="combined"),
        ],
        ignore_index=True,
    )
    coefs.to_csv(OUT / "coefficients_full_fit_descriptive.csv", index=False)

    provenance = {
        "n_decisions": 1013,
        "n_patients": 114,
        "horizon_days": 120,
        "source_coverage": str(COVERAGE),
        "availability_features": AVAIL_FEATURES,
        "feature_definitions": {
            "prop_absent_days_90": "1 - (days with ≥1 observed measure / 90 expected calendar days). practiced_sport counts only when activity_walking survived the 120–15000 s filter. A stored 0 with no raw walking and no raw running is not an observed measure.",
            "n_gaps_ge_48h_90": "count of runs of ≥2 consecutive empty calendar days in the 90 d window",
            "days_since_last_valid_90": "empty days at the end of the 90 d window since last day with any measure",
            "monitoring_interrupted_now": "1 if days_since_last_valid_90 ≥ 2 (ongoing >48 h gap at the decision); no future withdrawal flag",
        },
        "models": {
            "availability_only": "standardised logistic (L2, C=1) on the four indicators, fitted on the 1013 decisions; Youden cut on the same decisions",
            "dcabp_load_frozen": "frozen three-block DCABP load; classif at threshold 1.0596…",
            "dcabp_logit": "same logistic on load alone (ranks as the frozen load)",
            "dcabp_plus_availability": "same logistic on load + the four indicators",
        },
        "cross_validated_comparison": f"reviewer's design: availability, load and load + availability logistics with one shared patient-to-fold map ({N_SPLITS} folds, seed {CV_SEED}); out-of-fold probability of progression; Youden cut from the training folds; paired AUROC difference (load + availability minus load) with patient bootstrap. Output: performance_models_cv.csv",
        "valid_record": "a day with at least one of the nine input variables observed after the range filters",
        "evaluation": "as the frozen load: fixed scores on the same 1013 decisions, no cross-validation; patient bootstrap CIs; paired AUROC difference vs the frozen load",
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "patient"},
        "db2_db3": "not used",
        "note": "Coordinate with the broader comparator pack; this file is the missingness arm on the same 1013 points.",
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))

    # Report
    show = perf.copy()
    for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV"):
        show[k] = show.apply(lambda r, key=k: fmt_ci(r.to_dict(), key), axis=1)
    show["AUROC_diff"] = show.apply(lambda r: "—" if pd.isna(r["AUROC_diff_vs_frozen"]) else f"{r['AUROC_diff_vs_frozen']:.3f} [{r['AUROC_diff_lo']:.3f}, {r['AUROC_diff_hi']:.3f}]", axis=1)
    cols = ["model", "n_patients", "n_scored", "n_positive", "AUROC", "AUROC_diff", "sensitivity", "specificity", "PPV", "NPV", "threshold"]

    def as_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Comparador disponibilidad vs DCABP (punto 2B)",
        "",
        "Mismos **1013** puntos / horizonte **120 d** / solo **DB1**. "
        "Modelos ajustados en las 1013 decisiones y evaluados como la carga congelada (sin CV). IC = bootstrap por paciente; diferencia de AUROC emparejada frente a la congelada.",
        "",
        "## Indicadores de disponibilidad (por decisión, ventana 90 d)",
        "",
        "- `prop_absent_days_90`: proporción de días sin ninguna medida",
        "- `n_gaps_ge_48h_90`: número de huecos ≥ 48 h (≥ 2 días vacíos seguidos)",
        "- `days_since_last_valid_90`: días desde el último registro válido (hueco actual)",
        "- `monitoring_interrupted_now`: 1 si ese hueco actual es ≥ 2 días (sin usar abandonos futuros)",
        "",
        "## Emisión de predicción",
        "",
        as_md(pd.DataFrame([emission])),
        "",
        "## Rendimiento",
        "",
        as_md(show[cols]),
        "",
        "Archivos: `availability_indicators.csv`, `scores_by_decision.csv`, "
        "`performance_models.csv`, `emission_overlap.csv`, `coefficients_full_fit_descriptive.csv`, `PROVENANCE.json`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    (OUT / "PROVENANCE.md").write_text(
        "Missingness comparator (2B) on objetivo 1013.\n\n"
        f"Features: {AVAIL_FEATURES}.\n"
        "DCABP = frozen three-block load from the alarm baseline.\n"
        "No DB2/DB3. Models fitted on the 1013 decisions and evaluated as the frozen load (no CV); patient bootstrap CIs.\n"
        f"Script: scripts/03_analysis/alarm/run_missingness_comparator_objetivo.py\n"
    )

    print("wrote", OUT)
    print(pd.DataFrame([emission]).to_string(index=False))
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
