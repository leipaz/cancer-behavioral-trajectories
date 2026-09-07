#!/usr/bin/env python3
"""Missingness funnel by granularity + mini W×H experiment.

Outputs under ``results/alarm/``:
  tables/missingness_funnel.csv
  tables/wh_grid_auc.csv  (also refreshes granularity_sweep_summary if requested)
  figures/missingness_funnel.png
  figures/wh_auc_heatmaps.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rolling_burden_alarm import (  # noqa: E402
    DAYS_PER_MONTH,
    default_master_table_path,
    extract_monthly_puf_from_master,
    project_root_from,
)
from sliding_burden_alarm import (  # noqa: E402
    GRANULARITY_STEPS,
    add_rolling_burden,
    build_collapsed_decision_points,
    build_monthly_collapsed_decision_points,
    build_sliding_decision_points,
    metrics_row_from_dp,
)


def monthly_from_master(root: Path) -> pd.DataFrame:
    master = pd.read_excel(default_master_table_path(root))
    monthly = extract_monthly_puf_from_master(master, root=root)
    out = monthly.rename(columns={"month": "period"}).copy()
    out["block_size_days"] = 30
    out["decision_day"] = out["period"] * DAYS_PER_MONTH
    return out


def funnel_sliding(
    daily_puf: pd.DataFrame,
    granularity: str,
    w_load_days: int,
    h_horizon_days: int,
    sample_base_day: int = 30,
    min_periods: int = 1,
) -> dict:
    step = GRANULARITY_STEPS[granularity]
    scored = add_rolling_burden(daily_puf, w_load_days, min_periods=min_periods, score="mean")

    n_calendar = 0
    n_no_burden = 0
    n_after_exit = 0
    n_cens_short_fu = 0
    n_kept = 0
    n_pos = 0

    for _, g in scored.groupby("id"):
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        fu = float(g["follow_up_days"].iloc[0])
        days = g["study_day"].to_numpy(int)
        scores = g["burden_score"].to_numpy(float)
        on_grid = (days - sample_base_day) % step == 0
        for day, burden in zip(days[on_grid], scores[on_grid]):
            n_calendar += 1
            if np.isnan(burden):
                n_no_burden += 1
                continue
            t = float(day)
            if not (t < t_ev):
                n_after_exit += 1
                continue
            if event == 1:
                y = int(t_ev <= t + h_horizon_days)
                n_kept += 1
                n_pos += y
            else:
                if fu < t + h_horizon_days:
                    n_cens_short_fu += 1
                else:
                    n_kept += 1

    return _pack(
        granularity,
        w_load_days,
        h_horizon_days,
        n_calendar,
        n_no_burden,
        n_after_exit,
        n_cens_short_fu,
        n_kept,
        n_pos,
    )


def funnel_collapsed(
    collapsed_puf: pd.DataFrame,
    granularity: str,
    block_size_days: int,
    w_load_days: int,
    h_horizon_days: int,
) -> dict:
    w_periods = max(1, int(round(w_load_days / block_size_days)))
    h_days = max(1, int(round(h_horizon_days / block_size_days))) * block_size_days

    n_calendar = 0
    n_no_burden = 0
    n_after_exit = 0
    n_cens_short_fu = 0
    n_kept = 0
    n_pos = 0

    for _, g in collapsed_puf.groupby("id"):
        g = g.sort_values("period" if "period" in g.columns else "decision_day")
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        fu = float(g["follow_up_days"].iloc[0])
        vals = g["pUF"].to_numpy(float)
        days = g["decision_day"].to_numpy(float)
        for i in range(len(g)):
            n_calendar += 1
            start = i - w_periods + 1
            if start < 0 or np.isnan(vals[start : i + 1]).any():
                n_no_burden += 1
                continue
            t = float(days[i])
            if not (t < t_ev):
                n_after_exit += 1
                continue
            if event == 1:
                y = int(t_ev <= t + h_days)
                n_kept += 1
                n_pos += y
            else:
                if fu < t + h_days:
                    n_cens_short_fu += 1
                else:
                    n_kept += 1

    return _pack(
        granularity,
        w_load_days,
        h_horizon_days,
        n_calendar,
        n_no_burden,
        n_after_exit,
        n_cens_short_fu,
        n_kept,
        n_pos,
    )


def _pack(
    granularity: str,
    w: int,
    h: int,
    n_calendar: int,
    n_no_burden: int,
    n_after_exit: int,
    n_cens_short_fu: int,
    n_kept: int,
    n_pos: int,
) -> dict:
    lost = n_no_burden + n_after_exit + n_cens_short_fu
    return {
        "granularity": granularity,
        "w_load_days": w,
        "h_horizon_days": h,
        "n_calendar_landmarks": n_calendar,
        "n_drop_no_burden_or_incomplete_W": n_no_burden,
        "n_drop_after_exit_time": n_after_exit,
        "n_drop_censor_short_followup": n_cens_short_fu,
        "n_kept_adjudicable": n_kept,
        "n_positive_targets": n_pos,
        "frac_kept": n_kept / n_calendar if n_calendar else np.nan,
        "frac_lost_short_fu": n_cens_short_fu / n_calendar if n_calendar else np.nan,
        "frac_lost_no_burden": n_no_burden / n_calendar if n_calendar else np.nan,
        "frac_lost_after_exit": n_after_exit / n_calendar if n_calendar else np.nan,
        "n_lost_total": lost,
    }


def run_missingness(
    daily: pd.DataFrame,
    weekly_c: pd.DataFrame,
    biweekly_c: pd.DataFrame,
    monthly_c: pd.DataFrame,
    w_list: list[int],
    h_list: list[int],
) -> pd.DataFrame:
    rows = []
    for w in w_list:
        for h in h_list:
            for gran in GRANULARITY_STEPS:
                rows.append(funnel_sliding(daily, gran, w, h))
            rows.append(funnel_collapsed(weekly_c, "weekly_collapsed", 7, w, h))
            rows.append(funnel_collapsed(biweekly_c, "biweekly_collapsed", 14, w, h))
            # monthly collapsed uses sum AUC and month units; funnel uses same W-period rule
            rows.append(funnel_collapsed(monthly_c, "monthly_collapsed", 30, w, h))
    return pd.DataFrame(rows)


def run_wh_auc_grid(
    daily: pd.DataFrame,
    weekly_c: pd.DataFrame,
    biweekly_c: pd.DataFrame,
    monthly_puf_long: pd.DataFrame,
    w_list: list[int],
    h_list: list[int],
) -> pd.DataFrame:
    rows = []
    for w in w_list:
        for h in h_list:
            for gran in GRANULARITY_STEPS:
                dp = build_sliding_decision_points(daily, gran, w_load_days=w, h_horizon_days=h)
                rows.append(metrics_row_from_dp(gran, w, h, dp))
            for name, cdf, block in [
                ("weekly_collapsed", weekly_c, 7),
                ("biweekly_collapsed", biweekly_c, 14),
            ]:
                dp = build_collapsed_decision_points(
                    cdf, block, w_load_days=w, h_horizon_days=h, granularity=name
                )
                rows.append(metrics_row_from_dp(name, w, h, dp))
            dp = build_monthly_collapsed_decision_points(
                monthly_puf_long,
                w_months=max(1, int(round(w / DAYS_PER_MONTH))),
                h_months=max(1, int(round(h / DAYS_PER_MONTH))),
            )
            rows.append(metrics_row_from_dp("monthly_collapsed", w, h, dp))
    return pd.DataFrame(rows)


def plot_missingness(funnel: pd.DataFrame, out: Path, w: int = 90, h: int = 120) -> None:
    f = funnel[(funnel.w_load_days == w) & (funnel.h_horizon_days == h)].copy()
    order = [
        "daily",
        "weekly",
        "biweekly",
        "monthly_sample",
        "weekly_collapsed",
        "biweekly_collapsed",
        "monthly_collapsed",
    ]
    f["granularity"] = pd.Categorical(f["granularity"], categories=order, ordered=True)
    f = f.sort_values("granularity")

    labels = {
        "daily": "Daily",
        "weekly": "Weekly samp.",
        "biweekly": "Biweekly samp.",
        "monthly_sample": "Monthly samp.",
        "weekly_collapsed": "Weekly coll.",
        "biweekly_collapsed": "Biweekly coll.",
        "monthly_collapsed": "Monthly coll.",
    }
    x = np.arange(len(f))
    lab = [labels[g] for g in f["granularity"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    bottom = np.zeros(len(f))
    parts = [
        ("n_kept_adjudicable", "#2ca02c", "Kept (adjudicable)"),
        ("n_drop_censor_short_followup", "#ff7f0e", "Drop: censor short FU for H"),
        ("n_drop_after_exit_time", "#7f7f7f", "Drop: after exit/event time"),
        ("n_drop_no_burden_or_incomplete_W", "#d62728", "Drop: no burden / incomplete W"),
    ]
    for col, color, name in parts:
        vals = f[col].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=color, label=name, width=0.75)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Landmark counts")
    ax.set_title(f"Missingness funnel (W={w} d, H={h} d)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    width = 0.22
    for i, (col, color, name) in enumerate(
        [
            ("frac_kept", "#2ca02c", "Frac. kept"),
            ("frac_lost_short_fu", "#ff7f0e", "Frac. short FU"),
            ("frac_lost_no_burden", "#d62728", "Frac. no burden"),
        ]
    ):
        ax.bar(x + (i - 1) * width, f[col], width=width, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of calendar landmarks")
    ax.set_title("Loss composition (fractions)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def plot_wh_heatmaps(auc_grid: pd.DataFrame, out: Path) -> None:
    grans = [
        "daily",
        "weekly",
        "biweekly",
        "monthly_sample",
        "weekly_collapsed",
        "biweekly_collapsed",
        "monthly_collapsed",
    ]
    titles = {
        "daily": "Daily sampled",
        "weekly": "Weekly sampled",
        "biweekly": "Biweekly sampled",
        "monthly_sample": "Monthly sampled",
        "weekly_collapsed": "Weekly collapsed",
        "biweekly_collapsed": "Biweekly collapsed",
        "monthly_collapsed": "Monthly collapsed",
    }
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2))
    axes = axes.ravel()
    vmin = float(auc_grid["roc_auc"].min())
    vmax = float(auc_grid["roc_auc"].max())
    last_im = None
    for i, gran in enumerate(grans):
        ax = axes[i]
        sub = auc_grid[auc_grid.granularity == gran]
        pivot = sub.pivot(index="w_load_days", columns="h_horizon_days", values="roc_auc")
        pivot = pivot.sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(r) for r in pivot.index], fontsize=8)
        ax.set_xlabel("H (days)")
        ax.set_ylabel("W (days)")
        ax.set_title(titles[gran], fontsize=10)
        for yi, w in enumerate(pivot.index):
            for xi, h in enumerate(pivot.columns):
                val = pivot.loc[w, h]
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
    axes[-1].axis("off")
    fig.colorbar(last_im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="ROC-AUC")
    fig.suptitle("Mini experiment: ROC-AUC across lookback W × horizon H", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_wh_lines(auc_grid: pd.DataFrame, out: Path) -> None:
    """AUC vs H for each W, one panel per construction family."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    sampled = ["daily", "weekly", "biweekly", "monthly_sample"]
    collapsed = ["weekly_collapsed", "biweekly_collapsed", "monthly_collapsed"]
    colors = {
        "daily": "#1f77b4",
        "weekly": "#ff7f0e",
        "biweekly": "#2ca02c",
        "monthly_sample": "#9467bd",
        "weekly_collapsed": "#6a3d9a",
        "biweekly_collapsed": "#b15928",
        "monthly_collapsed": "#e31a1c",
    }
    for ax, family, title in [
        (axes[0], sampled, "Sampled family"),
        (axes[1], collapsed, "Collapsed family"),
    ]:
        for gran in family:
            for w, ls in [(60, ":"), (90, "-"), (120, "--")]:
                sub = auc_grid[(auc_grid.granularity == gran) & (auc_grid.w_load_days == w)].sort_values(
                    "h_horizon_days"
                )
                ax.plot(
                    sub["h_horizon_days"],
                    sub["roc_auc"],
                    ls=ls,
                    color=colors[gran],
                    marker="o",
                    ms=4,
                    lw=1.6,
                    label=f"{gran} W={w}",
                )
        ax.set_xlabel("Horizon H (days)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("AUC vs prediction horizon H (curves = W lookback)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--w-days", type=str, default="60,90,120")
    parser.add_argument("--h-days", type=str, default="90,120,180")
    args = parser.parse_args()

    root = project_root_from(args.root)
    tables = root / "results/alarm/tables"
    figs = root / "results/alarm/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    w_list = [int(x) for x in args.w_days.split(",") if x.strip()]
    h_list = [int(x) for x in args.h_days.split(",") if x.strip()]

    daily = pd.read_csv(tables / "daily_sliding_puf.csv")
    weekly_c = pd.read_csv(tables / "collapsed_puf_weekly.csv")
    biweekly_c = pd.read_csv(tables / "collapsed_puf_biweekly.csv")
    monthly_c = monthly_from_master(root)
    master = pd.read_excel(default_master_table_path(root))
    monthly_puf = extract_monthly_puf_from_master(master, root=root)

    print("Running missingness funnel…")
    funnel = run_missingness(daily, weekly_c, biweekly_c, monthly_c, w_list, h_list)
    funnel.to_csv(tables / "missingness_funnel.csv", index=False)
    plot_missingness(funnel, figs / "missingness_funnel", w=90, h=120)
    print(f"Wrote {tables / 'missingness_funnel.csv'}")

    # Focus print W90 H120
    focus = funnel[(funnel.w_load_days == 90) & (funnel.h_horizon_days == 120)]
    print(focus[
        [
            "granularity",
            "n_calendar_landmarks",
            "n_drop_no_burden_or_incomplete_W",
            "n_drop_after_exit_time",
            "n_drop_censor_short_followup",
            "n_kept_adjudicable",
            "frac_kept",
        ]
    ].to_string(index=False))

    print("Running W×H AUC grid…")
    auc_grid = run_wh_auc_grid(daily, weekly_c, biweekly_c, monthly_puf, w_list, h_list)
    auc_grid.to_csv(tables / "wh_grid_auc.csv", index=False)
    # keep summary in sync
    auc_grid.to_csv(tables / "granularity_sweep_summary.csv", index=False)
    plot_wh_heatmaps(auc_grid, figs / "wh_auc_heatmaps")
    plot_wh_lines(auc_grid, figs / "wh_auc_vs_horizon")
    print(f"Wrote {tables / 'wh_grid_auc.csv'} and heatmaps")

    # compact pivot for paper W around 90
    print("\nAUC pivot at each granularity (rows=W, cols=H):")
    for gran in ["daily", "weekly_collapsed", "monthly_collapsed"]:
        sub = auc_grid[auc_grid.granularity == gran]
        print(gran)
        print(sub.pivot(index="w_load_days", columns="h_horizon_days", values="roc_auc").round(3))


if __name__ == "__main__":
    main()
