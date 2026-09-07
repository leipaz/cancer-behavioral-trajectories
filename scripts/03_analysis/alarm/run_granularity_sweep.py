#!/usr/bin/env python3
"""Sweep alarm performance across temporal granularities and (W, H) grids.

Examples
--------
# Use existing daily / collapsed pUF tables (fast):
.venv/bin/python scripts/03_analysis/alarm/run_granularity_sweep.py --skip-lda

# Rebuild collapsed weekly/biweekly pUF then sweep:
.venv/bin/python scripts/03_analysis/alarm/run_granularity_sweep.py --rebuild-collapsed --skip-lda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rolling_burden_alarm import (  # noqa: E402
    DAYS_PER_MONTH,
    PAPER_THRESHOLD,
    default_clinical_path,
    default_master_table_path,
    extract_monthly_puf_from_master,
    metrics_at_threshold,
    project_root_from,
)
from sliding_burden_alarm import (  # noqa: E402
    COLLAPSED_BLOCK_DAYS,
    GRANULARITY_STEPS,
    build_collapsed_decision_points,
    build_monthly_collapsed_decision_points,
    build_sliding_decision_points,
    metrics_row_from_dp,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_sweep(
    root: Path,
    w_list: list[int],
    h_list: list[int],
    skip_lda: bool = True,
    rebuild_collapsed: bool = False,
    write_decision_points: bool = False,
) -> pd.DataFrame:
    tables = root / "results/alarm/tables"
    figs = root / "results/alarm/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    daily_path = tables / "daily_sliding_puf.csv"
    if not skip_lda or not daily_path.exists():
        from build_daily_sliding_puf import main as build_daily_main

        print("Building daily sliding pUF (LDA)…")
        sys.argv = ["build_daily_sliding_puf.py", "--out-csv", str(daily_path)]
        build_daily_main()

    if rebuild_collapsed:
        from build_collapsed_puf import main as build_collapsed_main

        print("Building collapsed pUF tables…")
        sys.argv = [
            "build_collapsed_puf.py",
            "--block-size",
            "7",
            "14",
            "--out-dir",
            str(tables),
        ]
        build_collapsed_main()

    daily_puf = pd.read_csv(daily_path)
    master = pd.read_excel(default_master_table_path(root))
    monthly_puf = extract_monthly_puf_from_master(master, root=root)

    collapsed = {}
    for name, block in (("weekly", 7), ("biweekly", 14)):
        path = tables / f"collapsed_puf_{name}.csv"
        if path.exists():
            collapsed[name] = pd.read_csv(path)
        else:
            print(f"WARNING: missing {path}; skip {name}_collapsed")

    rows: list[dict] = []
    for w in w_list:
        for h in h_list:
            for gran in GRANULARITY_STEPS:
                dp = build_sliding_decision_points(
                    daily_puf, granularity=gran, w_load_days=w, h_horizon_days=h
                )
                rows.append(metrics_row_from_dp(gran, w, h, dp))
                if write_decision_points and gran == "daily" and w == 90 and h == 120:
                    dp.to_csv(tables / "decision_points_daily_W90_H120.csv", index=False)

            for name, cdf in collapsed.items():
                gran = f"{name}_collapsed"
                block = 7 if name == "weekly" else 14
                dp = build_collapsed_decision_points(
                    cdf,
                    block_size_days=block,
                    w_load_days=w,
                    h_horizon_days=h,
                    score="mean",
                    granularity=gran,
                )
                rows.append(metrics_row_from_dp(gran, w, h, dp))

            w_months = int(round(w / DAYS_PER_MONTH))
            h_months = int(round(h / DAYS_PER_MONTH))
            dp = build_monthly_collapsed_decision_points(
                monthly_puf, w_months=w_months, h_months=h_months
            )
            rows.append(metrics_row_from_dp("monthly_collapsed", w, h, dp))

    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "granularity_sweep_summary.csv", index=False)
    print(f"Wrote {tables / 'granularity_sweep_summary.csv'} ({len(summary)} rows)")

    # Focus figures at paper-equivalent W≈90, H≈120
    plot_granularity_panel(
        root,
        daily_puf,
        monthly_puf,
        collapsed,
        w=90,
        h=120,
        summary=summary,
    )
    return summary


def plot_granularity_panel(
    root: Path,
    daily_puf: pd.DataFrame,
    monthly_puf: pd.DataFrame,
    collapsed: dict[str, pd.DataFrame],
    w: int,
    h: int,
    summary: pd.DataFrame,
) -> None:
    figs = root / "results/alarm/figures"
    figs.mkdir(parents=True, exist_ok=True)

    series = []
    for gran in ("daily", "weekly", "biweekly", "monthly_sample"):
        dp = build_sliding_decision_points(daily_puf, gran, w_load_days=w, h_horizon_days=h)
        series.append((f"{gran} (sampled)" if gran != "daily" else "daily", dp, "sampled"))

    for name, cdf in collapsed.items():
        block = 7 if name == "weekly" else 14
        label = f"{name} collapsed ({block} d)"
        dp = build_collapsed_decision_points(cdf, block, w_load_days=w, h_horizon_days=h, granularity=label)
        series.append((label, dp, "collapsed"))

    dp_m = build_monthly_collapsed_decision_points(
        monthly_puf,
        w_months=int(round(w / DAYS_PER_MONTH)),
        h_months=int(round(h / DAYS_PER_MONTH)),
    )
    series.append(("monthly collapsed (30 d)", dp_m, "collapsed"))

    # ROC overlay
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for label, dp, _kind in series:
        fpr, tpr, _ = roc_curve(dp["target"], dp["burden_score"])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.2f}, n={len(dp)})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR (1 – specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title(f"ROC by granularity (W={w} d, H={h} d)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / f"roc_granularities_W{w}_H{h}.png", dpi=200)
    fig.savefig(figs / f"roc_granularities_W{w}_H{h}.svg")
    plt.close(fig)

    # Summary panel
    focus = summary[(summary.w_load_days == w) & (summary.h_horizon_days == h)].copy()
    order = [
        "daily",
        "weekly",
        "biweekly",
        "monthly_sample",
        "weekly_collapsed",
        "biweekly_collapsed",
        "monthly_collapsed",
    ]
    focus["granularity"] = pd.Categorical(focus["granularity"], categories=order, ordered=True)
    focus = focus.sort_values("granularity").dropna(subset=["granularity"])
    labels = {
        "daily": "Daily",
        "weekly": "Weekly (sampled)",
        "biweekly": "Biweekly (sampled)",
        "monthly_sample": "Monthly (sampled)",
        "weekly_collapsed": "Weekly (collapsed 7 d)",
        "biweekly_collapsed": "Biweekly (collapsed 14 d)",
        "monthly_collapsed": "Monthly (collapsed 30 d)",
    }
    focus["label"] = focus["granularity"].map(labels)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.15]})
    ax = axes[0, 0]
    x = np.arange(len(focus))
    width = 0.2
    for i, col in enumerate(["sensitivity", "specificity", "PPV", "NPV"]):
        ax.bar(x + (i - 1.5) * width, focus[col], width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(f"Metrics at Youden threshold (W={w} d, H={h} d)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    colors = ["#4c72b0"] * 4 + ["#c44e52"] * 3
    bars = ax.bar(x, focus["roc_auc"], color=colors[: len(focus)])
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.62, max(0.74, float(focus["roc_auc"].max()) + 0.02))
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Discrimination by granularity")
    best_i = int(np.argmax(focus["roc_auc"].to_numpy()))
    for i, (bar, val) in enumerate(zip(bars, focus["roc_auc"])):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.2f}", ha="center", fontsize=8)
        if i == best_i:
            ax.scatter(
                bar.get_x() + bar.get_width() / 2,
                val + 0.012,
                marker="*",
                s=140,
                color="gold",
                edgecolors="black",
                zorder=5,
            )
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    ax.remove()
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 1, 2)
    best_label, best_dp, best_auc = None, None, -1.0
    for label, dp, _kind in series:
        fpr, tpr, _ = roc_curve(dp["target"], dp["burden_score"])
        curve_auc = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={curve_auc:.2f})")
        if curve_auc > best_auc:
            best_auc, best_label, best_dp = curve_auc, label, dp
    # Youden operating point of the best-AUC curve
    if best_dp is not None:
        from sliding_burden_alarm import summarize_decision_points

        best_m = summarize_decision_points(best_dp)
        ax.scatter(
            [1 - best_m["specificity"]],
            [best_m["sensitivity"]],
            color="gold",
            edgecolors="black",
            marker="*",
            s=180,
            zorder=5,
            label=(
                f"Best Youden: {best_label} "
                f"(AUC={best_auc:.3f}, θ={best_m['youden_threshold']:.2f})"
            ),
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR (1 – specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title(f"ROC curves by granularity (W={w} d, H={h} d)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.2)

    fig.suptitle("Alarm performance across temporal granularities", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(figs / f"summary_granularities_W{w}_H{h}.png", dpi=200)
    fig.savefig(figs / f"summary_granularities_W{w}_H{h}.svg")
    plt.close(fig)
    print(f"Wrote figures under {figs}")


def replicate_monthly_paper(root: Path, W: int = 3, H: int = 4) -> None:
    tables = root / "results/alarm/tables"
    figs = root / "results/alarm/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    master = pd.read_excel(default_master_table_path(root))
    monthly_puf = extract_monthly_puf_from_master(master, root=root)
    dp = build_monthly_collapsed_decision_points(monthly_puf, w_months=W, h_months=H)
    dp.to_csv(tables / f"decision_points_W{W}_H{H}.csv", index=False)

    paper = metrics_at_threshold(dp, PAPER_THRESHOLD, score_col="burden_score")
    from rolling_burden_alarm import youden_optimal_threshold

    # youden on AUC_W column name
    tmp = dp.rename(columns={"burden_score": "AUC_W"})
    youden = youden_optimal_threshold(tmp)
    summary = pd.DataFrame(
        [
            {
                "config": f"W={W}, H={H}",
                "n_decision_points": len(dp),
                "n_positive_targets": int(dp["target"].sum()),
                "roc_auc": paper.roc_auc,
                "youden_threshold_AUC_W": youden.threshold,
                "youden_threshold_mean_pUF": youden.threshold / W,
                "youden_sensitivity": youden.sensitivity,
                "youden_specificity": youden.specificity,
                "youden_ppv": youden.ppv,
                "youden_npv": youden.npv,
                "paper_threshold_AUC_W": PAPER_THRESHOLD,
                "paper_sensitivity": paper.sensitivity,
                "paper_specificity": paper.specificity,
                "paper_ppv": paper.ppv,
                "paper_npv": paper.npv,
                "paper_TP": paper.tp,
                "paper_FN": paper.fn,
                "paper_FP": paper.fp,
                "paper_TN": paper.tn,
            }
        ]
    )
    summary.to_csv(tables / f"alarm_metrics_W{W}_H{H}.csv", index=False)

    fpr, tpr, _ = roc_curve(dp["target"], dp["burden_score"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.scatter(
        [1 - paper.specificity],
        [paper.sensitivity],
        color="red",
        zorder=5,
        label=f"Paper θ={PAPER_THRESHOLD} (Sens={paper.sensitivity:.2f})",
    )
    ax.set_xlabel("False positive rate (1 – specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"Rolling-burden alarm (W={W} months, H={H} months)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / f"roc_W{W}_H{H}.png", dpi=200)
    fig.savefig(figs / f"roc_W{W}_H{H}.svg")
    plt.close(fig)
    print(f"Monthly replication saved (AUC={paper.roc_auc:.3f}, n={len(dp)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--w-days", type=str, default="60,90,120")
    parser.add_argument("--h-days", type=str, default="90,120,180")
    parser.add_argument("--skip-lda", action="store_true", help="Reuse daily_sliding_puf.csv")
    parser.add_argument("--rebuild-collapsed", action="store_true")
    parser.add_argument("--write-decision-points", action="store_true")
    parser.add_argument("--monthly-only", action="store_true")
    args = parser.parse_args()

    root = project_root_from(args.root)
    if args.monthly_only:
        replicate_monthly_paper(root)
        return

    run_sweep(
        root,
        w_list=_parse_int_list(args.w_days),
        h_list=_parse_int_list(args.h_days),
        skip_lda=args.skip_lda,
        rebuild_collapsed=args.rebuild_collapsed,
        write_decision_points=args.write_decision_points,
    )


if __name__ == "__main__":
    main()
