#!/usr/bin/env python3
"""Re-run alarm system with audit DB1 event clock.

Day 0  = Fecha_inicio_actualizada
Event  = Evento_DB1 (1 = PD, 0 = censor)
t_ev/FU = Fecha_Fin_DB1_Actualizado - Fecha_inicio_actualizada

Reuses revision pUF series (topics already inferred); only re-labels landmarks.
Compares vs current revision (Subjects Obs_time) and pre-revision archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/03_analysis/alarm"))

from rolling_burden_alarm import (  # noqa: E402
    DAYS_PER_MONTH,
    PAPER_THRESHOLD,
    build_decision_points,
    extract_monthly_puf_from_master,
    metrics_at_threshold,
    youden_optimal_threshold,
)
from sliding_burden_alarm import (  # noqa: E402
    GRANULARITY_STEPS,
    build_collapsed_decision_points,
    build_monthly_collapsed_decision_points,
    build_sliding_decision_points,
    metrics_row_from_dp,
)

REV = PROJECT_ROOT / "data/revision"
TABLES = PROJECT_ROOT / "results/alarm/tables"
OUT = PROJECT_ROOT / "results/alarm/Evento_DB1_Fin"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = REV / "178 audit oct 2 2026 copia.xlsx"


def load_audit_clock() -> pd.DataFrame:
    audit = pd.read_excel(AUDIT)
    audit = audit[pd.to_numeric(audit["id"], errors="coerce").notna()].copy()
    audit["id"] = audit["id"].astype(int)
    audit["Evento_DB1"] = pd.to_numeric(audit["Evento_DB1"], errors="coerce")
    audit = audit[audit["Evento_DB1"].isin([0, 1])].copy()
    audit["Evento_DB1"] = audit["Evento_DB1"].astype(int)
    audit["Fecha_inicio_actualizada"] = pd.to_datetime(audit["Fecha_inicio_actualizada"])
    audit["Fecha_Fin_DB1_Actualizado"] = pd.to_datetime(audit["Fecha_Fin_DB1_Actualizado"])
    audit["t_evento_DB1"] = (
        audit["Fecha_Fin_DB1_Actualizado"] - audit["Fecha_inicio_actualizada"]
    ).dt.days.astype(float)
    return audit[
        [
            "id",
            "Evento_DB1",
            "Fecha_inicio_actualizada",
            "Fecha_Fin_DB1_Actualizado",
            "t_evento_DB1",
        ]
    ]


def apply_db1_clock(df: pd.DataFrame, clock: pd.DataFrame) -> pd.DataFrame:
    """Replace Evento PD / t_evento / follow_up with audit DB1 definition."""
    amap = clock.set_index("id")
    out = df.copy()
    out = out[out["id"].isin(amap.index)].copy()
    out["Evento PD"] = out["id"].map(amap["Evento_DB1"]).astype(int)
    out["t_evento_eb2"] = out["id"].map(amap["t_evento_DB1"]).astype(float)
    out["follow_up_days"] = out["id"].map(amap["t_evento_DB1"]).astype(float)
    return out


def paper_row(dp: pd.DataFrame, version: str) -> dict:
    paper = metrics_at_threshold(dp, PAPER_THRESHOLD, score_col="AUC_W" if "AUC_W" in dp.columns else "burden_score")
    score = "AUC_W" if "AUC_W" in dp.columns else "burden_score"
    youden = youden_optimal_threshold(dp, score_col=score)
    return {
        "version": version,
        "n_landmarks": len(dp),
        "n_positive_targets": int(dp["target"].sum()),
        "n_patients": int(dp["id"].nunique()),
        "AUROC": paper.roc_auc,
        "paper_theta": PAPER_THRESHOLD,
        "paper_sens": paper.sensitivity,
        "paper_spec": paper.specificity,
        "paper_ppv": paper.ppv,
        "paper_npv": paper.npv,
        "paper_TP": paper.tp,
        "paper_FN": paper.fn,
        "paper_FP": paper.fp,
        "paper_TN": paper.tn,
        "youden_theta": youden.threshold,
        "youden_sens": youden.sensitivity,
        "youden_spec": youden.specificity,
    }


def main() -> None:
    clock = load_audit_clock()
    clock.to_csv(OUT / "audit_Evento_DB1_clock.csv", index=False)

    # --- Monthly paper (W=3, H=4) ---
    master = pd.read_excel(PROJECT_ROOT / "data/processed/DCABPs/topics_probs.xlsx")
    monthly_subj = extract_monthly_puf_from_master(master, root=PROJECT_ROOT)
    monthly_db1 = apply_db1_clock(monthly_subj, clock)

    dp_subj = build_decision_points(monthly_subj, W=3, H=4)
    dp_db1 = build_decision_points(monthly_db1, W=3, H=4)
    dp_db1.to_csv(OUT / "decision_points_W3_H4_Evento_DB1_Fin.csv", index=False)
    monthly_db1.to_csv(OUT / "monthly_puf_Evento_DB1_Fin.csv", index=False)

    compare = pd.DataFrame(
        [
            paper_row(dp_subj, "revision_Subjects_Obs_time"),
            paper_row(dp_db1, "revision_Evento_DB1_Fin"),
            {
                "version": "pre_revision_archived",
                "n_landmarks": 1015,
                "n_positive_targets": 159,
                "n_patients": 115,
                "AUROC": 0.7042114853347439,
                "paper_theta": 1.61,
                "paper_sens": 0.7044025157232704,
                "paper_spec": 0.6261682242990654,
                "paper_ppv": 0.25925925925925924,
                "paper_npv": 0.9193825042881647,
                "paper_TP": 112,
                "paper_FN": 47,
                "paper_FP": 320,
                "paper_TN": 536,
                "youden_theta": 1.6107623935677098,
                "youden_sens": 0.7044025157232704,
                "youden_spec": 0.6261682242990654,
            },
        ]
    )
    compare.to_csv(OUT / "alarm_metrics_W3_H4_comparison.csv", index=False)

    # Landmark conservation
    ka = set(dp_subj["id"].astype(str) + "_m" + dp_subj["month"].astype(str))
    kb = set(dp_db1["id"].astype(str) + "_m" + dp_db1["month"].astype(str))
    cons = pd.DataFrame(
        [
            {
                "n_Subjects": len(ka),
                "n_Evento_DB1_Fin": len(kb),
                "n_conserved": len(ka & kb),
                "pct_conserved_of_Subjects": 100 * len(ka & kb) / len(ka),
                "n_lost": len(ka - kb),
                "n_gained": len(kb - ka),
            }
        ]
    )
    cons.to_csv(OUT / "landmarks_conservation.csv", index=False)

    # --- Granularity sweep @ W=90, H=120 (and paper grid subset) ---
    daily = apply_db1_clock(pd.read_csv(TABLES / "daily_sliding_puf.csv"), clock)
    daily.to_csv(OUT / "daily_sliding_puf_Evento_DB1_Fin.csv", index=False)

    collapsed = {}
    for name, block in (("weekly", 7), ("biweekly", 14)):
        path = TABLES / f"collapsed_puf_{name}.csv"
        if path.exists():
            collapsed[name] = apply_db1_clock(pd.read_csv(path), clock)
            collapsed[name].to_csv(OUT / f"collapsed_puf_{name}_Evento_DB1_Fin.csv", index=False)

    w_list = [60, 90, 120]
    h_list = [90, 120, 180]
    rows: list[dict] = []
    for w in w_list:
        for h in h_list:
            for gran in GRANULARITY_STEPS:
                dp = build_sliding_decision_points(
                    daily, granularity=gran, w_load_days=w, h_horizon_days=h
                )
                rows.append(metrics_row_from_dp(gran, w, h, dp))
            for name, cdf in collapsed.items():
                block = 7 if name == "weekly" else 14
                dp = build_collapsed_decision_points(
                    cdf,
                    block_size_days=block,
                    w_load_days=w,
                    h_horizon_days=h,
                    score="mean",
                    granularity=f"{name}_collapsed",
                )
                rows.append(metrics_row_from_dp(f"{name}_collapsed", w, h, dp))
            w_m = int(round(w / DAYS_PER_MONTH))
            h_m = int(round(h / DAYS_PER_MONTH))
            dp = build_monthly_collapsed_decision_points(monthly_db1, w_months=w_m, h_months=h_m)
            rows.append(metrics_row_from_dp("monthly_collapsed", w, h, dp))

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "granularity_sweep_summary_Evento_DB1_Fin.csv", index=False)

    # Focus slice W=90 H=120 vs previous revision sweep if present
    prev = TABLES / "granularity_sweep_summary.csv"
    focus = sweep[(sweep["w_load_days"] == 90) & (sweep["h_horizon_days"] == 120)].copy()
    focus["clock"] = "Evento_DB1_Fin"
    if prev.exists():
        old = pd.read_csv(prev)
        old_f = old[(old["w_load_days"] == 90) & (old["h_horizon_days"] == 120)].copy()
        old_f["clock"] = "Subjects_Obs_time"
        focus_cmp = pd.concat([old_f, focus], ignore_index=True)
        focus_cmp.to_csv(OUT / "granularity_W90_H120_comparison.csv", index=False)
    else:
        focus.to_csv(OUT / "granularity_W90_H120_comparison.csv", index=False)

    # Short markdown report
    md = OUT / "REPORT_Evento_DB1_Fin.md"
    md.write_text(
        "\n".join(
            [
                "# Alarm re-run — Evento_DB1 / Fecha_Fin_DB1",
                "",
                "Clock:",
                "- Day 0 = `Fecha_inicio_actualizada`",
                "- If `Evento_DB1=1`: PD date = `Fecha_Fin_DB1_Actualizado`",
                "- If `Evento_DB1=0`: censor at `Fecha_Fin_DB1_Actualizado`",
                "- `t_evento` = Fin − Inicio",
                "",
                "## Paper monthly W=3, H=4, θ=1.61",
                "",
                compare.to_markdown(index=False),
                "",
                "## Landmark conservation vs revision Subjects clock",
                "",
                cons.to_markdown(index=False),
                "",
                f"Outputs in `{OUT}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("=== Paper W=3 H=4 ===")
    print(compare.to_string(index=False))
    print("\n=== Conservation ===")
    print(cons.to_string(index=False))
    print("\n=== Granularity @ W=90 H=120 (Evento_DB1) ===")
    print(
        focus[
            ["granularity", "n_decision_points", "n_positive_targets", "roc_auc", "sensitivity", "specificity"]
        ].to_string(index=False)
    )
    print(f"\nWrote → {OUT}")
    print(f"Report → {md}")


if __name__ == "__main__":
    main()
