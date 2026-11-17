#!/usr/bin/env python3
"""Point 2B — missingness-only comparator on the 1013 objetivo decisions.

Three scores on the same 120-day landmarks (DB1 only, patient-grouped CV):

1) availability alone
2) DCABP load (frozen alarm score)
3) DCABP load + availability indicators

Availability indicators (past-only, before imputation), primary window 90 d:
- proportion of absent days
- number of gaps ≥ 48 h (≥ 2 consecutive empty calendar days)
- days since last valid record (current gap at the decision)
- monitoring interrupted now (current gap ≥ 2 days)

No DB2/DB3 tuning. Uncertainty uses patient-clustered bootstrap on OOF scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
COVERAGE = ROOT / "results/missingness/cobertura_real/coverage_by_decision.csv"
OUT = ROOT / "results/missingness/comparador_disponibilidad"

THR_LOAD = 1.0596384140662845
N_SPLITS = 5
CV_SEED = 20261116
N_BOOT = 1000
BOOT_SEED = 20261116

# Prespecified availability feature set (90 d history + current gap).
AVAIL_FEATURES = [
    "prop_absent_days_90",
    "n_gaps_ge_48h_90",
    "days_since_last_valid_90",
    "monitoring_interrupted_now",
]


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


def oof_logistic(frame: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Patient-grouped 5-fold OOF probability and fold-wise Youden threshold applied to test."""
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=int)
    groups = frame["id"].to_numpy(dtype=int)
    oof = np.full(len(frame), np.nan, dtype=float)
    thr_used = np.full(len(frame), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=N_SPLITS)
    rng = np.random.default_rng(CV_SEED)
    patient_order = frame["id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(patient_order)
    order_map = {pid: i for i, pid in enumerate(patient_order)}
    sort_idx = np.argsort([order_map[pid] for pid in groups])
    Xs, ys, gs = X[sort_idx], y[sort_idx], groups[sort_idx]
    oof_sorted = np.full(len(frame), np.nan, dtype=float)
    thr_sorted = np.full(len(frame), np.nan, dtype=float)

    for train_idx, test_idx in gkf.split(Xs, ys, gs):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                    ),
                ),
            ]
        )
        pipe.fit(Xs[train_idx], ys[train_idx])
        p_train = pipe.predict_proba(Xs[train_idx])[:, 1]
        p_test = pipe.predict_proba(Xs[test_idx])[:, 1]
        thr = _youden_threshold(ys[train_idx], p_train)
        oof_sorted[test_idx] = p_test
        thr_sorted[test_idx] = thr

    oof[sort_idx] = oof_sorted
    thr_used[sort_idx] = thr_sorted
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF incomplete for {label}")
    return oof, thr_used


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
    both["score_avail_oof"], both["thr_avail_oof"] = oof_logistic(both, AVAIL_FEATURES, "availability")
    both["score_dcabp_logit_oof"], both["thr_dcabp_logit_oof"] = oof_logistic(both, ["load"], "dcabp_logit")
    both["score_combined_oof"], both["thr_combined_oof"] = oof_logistic(
        both, ["load"] + AVAIL_FEATURES, "combined"
    )

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
                "score_avail_oof",
                "thr_avail_oof",
                "score_dcabp_logit_oof",
                "thr_dcabp_logit_oof",
                "score_combined_oof",
                "thr_combined_oof",
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
        ("availability_only_oof", "score_avail_oof", None, "thr_avail_oof", both),
        ("dcabp_load_frozen", "load", THR_LOAD, None, both),
        ("dcabp_logit_oof", "score_dcabp_logit_oof", None, "thr_dcabp_logit_oof", both),
        ("dcabp_plus_availability_oof", "score_combined_oof", None, "thr_combined_oof", both),
    ]

    rows = []
    for name, col, thr, thr_col, frame in specs:
        m = patient_bootstrap_ci(frame, col, thr=thr, thr_col=thr_col)
        m["model"] = name
        m["score_type"] = "oof_prob_youden_train" if thr_col else "frozen_load_threshold"
        rows.append(m)

    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "performance_models.csv", index=False)

    # Coefficient snapshot on full DB1 (descriptive; primary metrics are OOF)
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
            "prop_absent_days_90": "1 - (days with ≥1 observed measure / 90 expected calendar days)",
            "n_gaps_ge_48h_90": "count of runs of ≥2 consecutive empty calendar days in the 90 d window",
            "days_since_last_valid_90": "empty days at the end of the 90 d window since last day with any measure",
            "monitoring_interrupted_now": "1 if days_since_last_valid_90 ≥ 2 (ongoing >48 h gap at the decision); no future withdrawal flag",
        },
        "models": {
            "availability_only_oof": "logistic, patient GroupKFold 5, OOF probability; classif threshold = Youden on train fold",
            "dcabp_load_frozen": "frozen three-block DCABP load; classif at threshold 1.0596…",
            "dcabp_logit_oof": "logistic on load alone, same CV + Youden train threshold",
            "dcabp_plus_availability_oof": "logistic on load + availability features, same CV + Youden train threshold",
        },
        "cv": {"type": "GroupKFold", "n_splits": N_SPLITS, "seed": CV_SEED},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "patient"},
        "db2_db3": "not used",
        "note": "Coordinate with the broader comparator pack; this file is the missingness arm on the same 1013 points.",
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))

    # Report
    show = perf.copy()
    for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV"):
        show[k] = show.apply(lambda r, key=k: fmt_ci(r.to_dict(), key), axis=1)
    cols = ["model", "n_patients", "n_scored", "n_positive", "AUROC", "sensitivity", "specificity", "PPV", "NPV", "threshold"]

    def as_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Comparador disponibilidad vs DCABP (punto 2B)",
        "",
        "Mismos **1013** puntos / horizonte **120 d** / solo **DB1**. "
        "CV por paciente (GroupKFold 5). IC = bootstrap por paciente sobre scores OOF (o load congelado).",
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
    (OUT / "informe.md").write_text("\n".join(lines))
    (OUT / "PROVENANCE.md").write_text(
        "Missingness comparator (2B) on objetivo 1013.\n\n"
        f"Features: {AVAIL_FEATURES}.\n"
        "DCABP = frozen three-block load from the alarm baseline.\n"
        "No DB2/DB3. Patient GroupKFold OOF + patient bootstrap CIs.\n"
        f"Script: scripts/03_analysis/alarm/run_missingness_comparator_objetivo.py\n"
    )

    print("wrote", OUT)
    print(pd.DataFrame([emission]).to_string(index=False))
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
