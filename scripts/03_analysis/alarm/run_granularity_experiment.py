#!/usr/bin/env python3
"""Granularity-of-alarms experiment using the same user clock BBDD.

Separate from the paper baseline (``run_paper_alarm.py``). Applies ``--clock``
to daily / collapsed / monthly pUF series before the W×H sweep.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

_SCRIPT_DIR = Path(__file__).resolve().parent
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
    DAYS_PER_MONTH,
    default_master_table_path,
    extract_monthly_puf_from_master,
)
from sliding_burden_alarm import (  # noqa: E402
    GRANULARITY_STEPS,
    build_collapsed_decision_points,
    build_monthly_collapsed_decision_points,
    build_sliding_decision_points,
    metrics_row_from_dp,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_granularity_experiment(
    root: Path,
    clock_path: Path,
    *,
    out_dir: Path | None = None,
    w_list: list[int] | None = None,
    h_list: list[int] | None = None,
    run_id: str | None = None,
    rebuild_puf: bool = False,
    focus_w: int = 90,
    focus_h: int = 120,
) -> Path:
    clock = load_clock(clock_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir or (root / "results/alarm/runs" / run_id / "granularity")
    out.mkdir(parents=True, exist_ok=True)
    tables = root / "results/alarm/tables"
    tables.mkdir(parents=True, exist_ok=True)

    w_list = w_list or [60, 90, 120]
    h_list = h_list or [90, 120, 180]

    if rebuild_puf:
        day_csv = root / "data/revision/PD_cutoff_topics_revision_a0.csv"
        from build_daily_sliding_puf import main as build_daily_main
        from build_collapsed_puf import main as build_collapsed_main

        print("Rebuilding daily sliding pUF…")
        sys.argv = [
            "build_daily_sliding_puf.py",
            "--topics-day-csv",
            str(day_csv),
            "--out-csv",
            str(tables / "daily_sliding_puf.csv"),
        ]
        build_daily_main()
        print("Rebuilding collapsed weekly/biweekly pUF…")
        sys.argv = [
            "build_collapsed_puf.py",
            "--topics-day-csv",
            str(day_csv),
            "--block-size",
            "7",
            "14",
            "--out-dir",
            str(tables),
        ]
        build_collapsed_main()

    daily_path = tables / "daily_sliding_puf.csv"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing {daily_path}. Rebuild with --rebuild-puf or run revision pipeline step 5."
        )

    daily = apply_clock(pd.read_csv(daily_path), clock)
    daily.to_csv(out / "daily_sliding_puf_clock.csv", index=False)

    collapsed: dict[str, pd.DataFrame] = {}
    for name, block in (("weekly", 7), ("biweekly", 14)):
        path = tables / f"collapsed_puf_{name}.csv"
        if path.exists():
            collapsed[name] = apply_clock(pd.read_csv(path), clock)
            collapsed[name].to_csv(out / f"collapsed_puf_{name}_clock.csv", index=False)
        else:
            print(f"WARNING: missing {path}; skip {name}_collapsed")

    master = strip_master_clock_fields(pd.read_excel(default_master_table_path(root)))
    monthly = extract_monthly_puf_from_master(
        master, clinical=clock_to_clinical(clock), root=root
    )
    monthly = apply_clock(monthly, clock)
    monthly.to_csv(out / "monthly_puf_clock.csv", index=False)

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
            dp = build_monthly_collapsed_decision_points(
                monthly, w_months=w_m, h_months=h_m
            )
            rows.append(metrics_row_from_dp("monthly_collapsed", w, h, dp))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "granularity_sweep_summary.csv", index=False)

    focus = summary[
        (summary["w_load_days"] == focus_w) & (summary["h_horizon_days"] == focus_h)
    ].copy()
    focus.to_csv(out / f"granularity_W{focus_w}_H{focus_h}.csv", index=False)

    _plot_focus_roc(out, daily, monthly, collapsed, focus_w, focus_h)

    write_provenance(
        out,
        clock_path=clock_path,
        clock=clock,
        run_kind="granularity_experiment",
        extra={
            "w_days": w_list,
            "h_days": h_list,
            "focus_W": focus_w,
            "focus_H": focus_h,
            "n_summary_rows": len(summary),
            "out_dir": str(out.resolve()),
        },
    )

    print(f"\n=== Granularity @ W={focus_w} H={focus_h} ===")
    cols = [
        "granularity",
        "n_decision_points",
        "n_positive_targets",
        "roc_auc",
        "sensitivity",
        "specificity",
    ]
    print(focus[cols].to_string(index=False))
    print(f"\nWrote → {out}")
    return out


def _plot_focus_roc(
    out: Path,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    collapsed: dict[str, pd.DataFrame],
    w: int,
    h: int,
) -> None:
    series = []
    for gran in ("daily", "weekly", "biweekly", "monthly_sample"):
        dp = build_sliding_decision_points(daily, gran, w_load_days=w, h_horizon_days=h)
        label = f"{gran} (sampled)" if gran != "daily" else "daily"
        series.append((label, dp))
    for name, cdf in collapsed.items():
        block = 7 if name == "weekly" else 14
        label = f"{name} collapsed ({block} d)"
        dp = build_collapsed_decision_points(
            cdf, block, w_load_days=w, h_horizon_days=h, granularity=label
        )
        series.append((label, dp))
    dp_m = build_monthly_collapsed_decision_points(
        monthly,
        w_months=int(round(w / DAYS_PER_MONTH)),
        h_months=int(round(h / DAYS_PER_MONTH)),
    )
    series.append(("monthly collapsed (30 d)", dp_m))

    fig, ax = plt.subplots(figsize=(7.2, 6))
    for label, dp in series:
        if dp.empty or dp["target"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(dp["target"], dp["burden_score"])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.2f}, n={len(dp)})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR (1 – specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title(f"ROC by granularity (W={w} d, H={h} d) — user clock")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"roc_granularities_W{w}_H{h}.png", dpi=200)
    fig.savefig(out / f"roc_granularities_W{w}_H{h}.svg")
    plt.close(fig)


def main() -> None:
    root = project_root_from()
    p = argparse.ArgumentParser(
        description="Granularity alarm experiment with user --clock BBDD"
    )
    p.add_argument("--clock", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--w-days", type=str, default="60,90,120")
    p.add_argument("--h-days", type=str, default="90,120,180")
    p.add_argument("--rebuild-puf", action="store_true")
    p.add_argument("--focus-w", type=int, default=90)
    p.add_argument("--focus-h", type=int, default=120)
    args = p.parse_args()

    clock_path = args.clock or default_clock_path(root)
    if not clock_path.exists():
        raise SystemExit(
            f"Clock BBDD not found: {clock_path}\n"
            "Build with: python scripts/03_analysis/alarm/clock_schema.py --from-audit-db1"
        )

    run_granularity_experiment(
        root,
        clock_path,
        out_dir=args.out_dir,
        w_list=_parse_int_list(args.w_days),
        h_list=_parse_int_list(args.h_days),
        run_id=args.run_id,
        rebuild_puf=args.rebuild_puf,
        focus_w=args.focus_w,
        focus_h=args.focus_h,
    )


if __name__ == "__main__":
    main()
