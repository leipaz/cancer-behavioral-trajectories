#!/usr/bin/env python3
"""Paper rolling-burden alarm (W=3 months, H=4 months) from a user clock BBDD.

Clock is the only clinical time source: fecha_inicio / fecha_fin / evento.
Subjects_data is not written and is not used as event/censor clock.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clock_schema import (  # noqa: E402
    apply_clock,
    clock_to_clinical,
    default_clock_path,
    load_clock,
    project_root_from,
    strip_master_clock_fields,
    write_provenance,
)
from rolling_burden_alarm import (  # noqa: E402
    PAPER_THRESHOLD,
    build_decision_points,
    default_master_table_path,
    extract_monthly_puf_from_master,
    metrics_at_threshold,
    youden_optimal_threshold,
)


def _metrics_row(dp: pd.DataFrame, *, W: int, H: int, theta: float) -> dict:
    score = "AUC_W" if "AUC_W" in dp.columns else "burden_score"
    paper = metrics_at_threshold(dp, theta, score_col=score)
    youden = youden_optimal_threshold(dp, score_col=score)
    return {
        "W_months": W,
        "H_months": H,
        "n_landmarks": len(dp),
        "n_positive_targets": int(dp["target"].sum()),
        "n_patients": int(dp["id"].nunique()),
        "AUROC": paper.roc_auc,
        "paper_theta": theta,
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


def run_paper_alarm(
    root: Path,
    clock_path: Path,
    *,
    master_path: Path | None = None,
    out_dir: Path | None = None,
    W: int = 3,
    H: int = 4,
    theta: float = PAPER_THRESHOLD,
    run_id: str | None = None,
) -> Path:
    clock = load_clock(clock_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir or (root / "results/alarm/runs" / run_id / "paper_W3_H4")
    out.mkdir(parents=True, exist_ok=True)

    master_path = master_path or default_master_table_path(root)
    master = strip_master_clock_fields(pd.read_excel(master_path))
    clinical = clock_to_clinical(clock)
    monthly = extract_monthly_puf_from_master(master, clinical=clinical, root=root)
    # Hard overwrite — never leave Subjects / leftover master times
    monthly = apply_clock(monthly, clock)

    dp = build_decision_points(monthly, W=W, H=H)
    metrics = pd.DataFrame([_metrics_row(dp, W=W, H=H, theta=theta)])

    monthly.to_csv(out / "monthly_puf.csv", index=False)
    dp.to_csv(out / f"decision_points_W{W}_H{H}.csv", index=False)
    metrics.to_csv(out / f"alarm_metrics_W{W}_H{H}.csv", index=False)

    write_provenance(
        out,
        clock_path=clock_path,
        clock=clock,
        run_kind=f"paper_alarm_W{W}_H{H}",
        extra={
            "master": str(master_path.resolve()),
            "W_months": W,
            "H_months": H,
            "theta": theta,
            "n_landmarks": len(dp),
            "n_patients": int(dp["id"].nunique()) if len(dp) else 0,
            "AUROC": float(metrics["AUROC"].iloc[0]) if len(metrics) else None,
            "out_dir": str(out.resolve()),
        },
    )

    # Convenience sync for legacy table path (paper landmarks)
    tables = root / "results/alarm/tables"
    tables.mkdir(parents=True, exist_ok=True)
    dp.to_csv(tables / "decision_points_W3_H4.csv", index=False)
    metrics.to_csv(tables / "alarm_metrics_W3_H4.csv", index=False)

    print(metrics.to_string(index=False))
    print(f"\nWrote → {out}")
    return out


def main() -> None:
    root = project_root_from()
    p = argparse.ArgumentParser(
        description="Paper alarm W=3 H=4 using only --clock BBDD as clinical time"
    )
    p.add_argument(
        "--clock",
        type=Path,
        default=None,
        help="Universal clock BBDD (xlsx/csv). Default: data/revision/clock_bbdd.xlsx",
    )
    p.add_argument("--master", type=Path, default=None, help="topics_probs.xlsx override")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--W", type=int, default=3)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--theta", type=float, default=PAPER_THRESHOLD)
    args = p.parse_args()

    clock_path = args.clock or default_clock_path(root)
    if not clock_path.exists():
        raise SystemExit(
            f"Clock BBDD not found: {clock_path}\n"
            "Build it with:\n"
            "  python scripts/03_analysis/alarm/clock_schema.py --from-audit-db1"
        )

    run_paper_alarm(
        root,
        clock_path,
        master_path=args.master,
        out_dir=args.out_dir,
        W=args.W,
        H=args.H,
        theta=args.theta,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
