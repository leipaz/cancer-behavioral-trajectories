#!/usr/bin/env python3
"""High-resolution figures for analysis delivery (summary + simplified trajectories).

Writes to ``results/alarm/analysis_figures/``:

* ``summary_granularities_W90_H120.{png,svg,pdf}`` — metrics table + AUROC
  points with patient-clustered bootstrap 95% CI + ROC (no winner star)
* ``trajectory_patient_{62004,41006}_W90_simplified.{png,svg,pdf}``
* ``metrics_table_W90_H120.csv`` / ``auc_ci_W90_H120.csv``

Example
-------
.venv/bin/python scripts/03_analysis/alarm/plot_analysis_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, roc_auc_score, roc_curve

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clock_schema import (  # noqa: E402
    apply_clock,
    clock_to_clinical,
    default_clock_path,
    load_clock,
    strip_master_clock_fields,
)
from rolling_burden_alarm import (  # noqa: E402
    DAYS_PER_MONTH,
    PAPER_THRESHOLD,
    default_master_table_path,
    extract_monthly_puf_from_master,
    metrics_at_threshold,
    project_root_from,
)
from sliding_burden_alarm import (  # noqa: E402
    add_rolling_burden,
    build_collapsed_decision_points,
    build_monthly_collapsed_decision_points,
    build_sliding_decision_points,
    metrics_row_from_dp,
)
from plot_granularity_trajectories import (  # noqa: E402
    add_collapsed_burden,
)

GRAN_ORDER = [
    "daily",
    "weekly",
    "biweekly",
    "monthly_sample",
    "weekly_collapsed",
    "biweekly_collapsed",
    "monthly_collapsed",
]
GRAN_LABELS = {
    "daily": "Daily",
    "weekly": "Weekly (sampled)",
    "biweekly": "Biweekly (sampled)",
    "monthly_sample": "Monthly (sampled)",
    "weekly_collapsed": "Weekly (collapsed 7 d)",
    "biweekly_collapsed": "Biweekly (collapsed 14 d)",
    "monthly_collapsed": "Monthly (collapsed 30 d)",
}
# Compact but complete labels for the metrics table (avoid clipping)
GRAN_LABELS_TABLE = {
    "daily": "Daily",
    "weekly": "Weekly · sampled",
    "biweekly": "Biweekly · sampled",
    "monthly_sample": "Monthly · sampled",
    "weekly_collapsed": "Weekly · collapsed",
    "biweekly_collapsed": "Biweekly · collapsed",
    "monthly_collapsed": "Monthly · collapsed",
}
# Collapsed = solid blues; sampled = dashed reds (easier to tell families apart on ROC)
SAMPLE_COLOR = "#c44e52"
COLLAPSE_COLOR = "#4c72b0"
SAMPLE_ROC_COLORS = ("#fc9272", "#ef3b2c", "#cb181d", "#99000d")
COLLAPSE_ROC_COLORS = ("#9ecae1", "#4292c6", "#08519c")
DPI = 400


def _clustered_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    cluster: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point AUC + percentile 95% CI with patient-level bootstrap."""
    y = np.asarray(y)
    score = np.asarray(score, float)
    cluster = np.asarray(cluster)
    point = float(roc_auc_score(y, score))
    ids = np.unique(cluster)
    by_id = {cid: np.flatnonzero(cluster == cid) for cid in ids}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([by_id[cid] for cid in draw])
        if len(idx) < 2:
            continue
        y_e, s_e = y[idx], score[idx]
        if len(np.unique(y_e)) < 2:
            continue
        boots.append(float(roc_auc_score(y_e, s_e)))
    if len(boots) < 100:
        return point, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def build_series(root: Path, w: int = 90, h: int = 120, clock_path: Path | None = None):
    """Build decision-point series aligned to the user clock BBDD."""
    tables = root / "results/alarm/tables"
    clock_path = clock_path or default_clock_path(root)
    if not clock_path.exists():
        raise FileNotFoundError(
            f"Clock BBDD required: {clock_path}. "
            "Build with: python scripts/03_analysis/alarm/clock_schema.py --from-audit-db1"
        )
    clock = load_clock(clock_path)
    print(f"Aligning all granularities to user clock BBDD → {clock_path}")

    run_g = root / "results/alarm/runs/Evento_DB1_clock/granularity"
    daily_src = run_g / "daily_sliding_puf_clock.csv"
    if not daily_src.exists():
        daily_src = tables / "daily_sliding_puf.csv"
    daily_puf = apply_clock(pd.read_csv(daily_src), clock)

    master = strip_master_clock_fields(pd.read_excel(default_master_table_path(root)))
    monthly_puf = extract_monthly_puf_from_master(
        master, clinical=clock_to_clinical(clock), root=root
    )
    monthly_puf = apply_clock(monthly_puf, clock)

    collapsed = {}
    for name in ("weekly", "biweekly"):
        cpath = run_g / f"collapsed_puf_{name}_clock.csv"
        if not cpath.exists():
            cpath = tables / f"collapsed_puf_{name}.csv"
        collapsed[name] = apply_clock(pd.read_csv(cpath), clock)

    series: list[tuple[str, str, pd.DataFrame]] = []
    for gran in ("daily", "weekly", "biweekly", "monthly_sample"):
        dp = build_sliding_decision_points(daily_puf, gran, w_load_days=w, h_horizon_days=h)
        series.append((gran, "sampled", dp))

    for name, cdf in collapsed.items():
        block = 7 if name == "weekly" else 14
        gran = f"{name}_collapsed"
        dp = build_collapsed_decision_points(
            cdf, block, w_load_days=w, h_horizon_days=h, score="mean", granularity=gran
        )
        series.append((gran, "collapsed", dp))

    # Prefer paper monthly landmarks already built with the same clock
    main_monthly = tables / "decision_points_W3_H4.csv"
    if w == 90 and h == 120 and main_monthly.exists():
        dp_m = pd.read_csv(main_monthly)
        if "burden_score" not in dp_m.columns and "AUC_W" in dp_m.columns:
            dp_m = dp_m.rename(columns={"AUC_W": "burden_score"})
        dp_m["granularity"] = "monthly_collapsed"
        print(f"  monthly_collapsed from {main_monthly.name} (n={len(dp_m)}, clock BBDD)")
    else:
        dp_m = build_monthly_collapsed_decision_points(
            monthly_puf,
            w_months=int(round(w / DAYS_PER_MONTH)),
            h_months=int(round(h / DAYS_PER_MONTH)),
        )
        print(f"  monthly_collapsed rebuilt with clock BBDD (n={len(dp_m)})")
    series.append(("monthly_collapsed", "collapsed", dp_m))
    return series, daily_puf, monthly_puf, collapsed, clock


def metrics_from_series(
    series: list[tuple[str, str, pd.DataFrame]], w: int, h: int
) -> pd.DataFrame:
    rows = [metrics_row_from_dp(gran, w, h, dp) for gran, _kind, dp in series]
    # Monthly collapsed: report operating θ (PAPER_THRESHOLD), not Youden (~1.06).
    for i, (gran, _kind, dp) in enumerate(series):
        if gran != "monthly_collapsed":
            continue
        score = "AUC_W" if "AUC_W" in dp.columns else "burden_score"
        m = metrics_at_threshold(dp, PAPER_THRESHOLD, score_col=score)
        rows[i].update(
            {
                "youden_threshold": float(PAPER_THRESHOLD),
                "sensitivity": m.sensitivity,
                "specificity": m.specificity,
                "PPV": m.ppv,
                "NPV": m.npv,
                "TP": m.tp,
                "FN": m.fn,
                "FP": m.fp,
                "TN": m.tn,
            }
        )
    focus = pd.DataFrame(rows)
    focus["granularity"] = pd.Categorical(focus["granularity"], categories=GRAN_ORDER, ordered=True)
    focus = focus.sort_values("granularity").dropna(subset=["granularity"])
    focus["label"] = focus["granularity"].astype(str).map(GRAN_LABELS)
    return focus


def plot_summary(
    series: list[tuple[str, str, pd.DataFrame]],
    focus: pd.DataFrame,
    auc_ci: pd.DataFrame,
    out_stem: Path,
    w: int,
    h: int,
) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 9.4), layout="constrained")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.2], width_ratios=[1.25, 1.0], hspace=0.28, wspace=0.18)

    # —— Top-left: metrics as table (exact values) ——
    ax_tab = fig.add_subplot(gs[0, 0])
    ax_tab.axis("off")
    display = focus[
        ["granularity", "n_decision_points", "roc_auc", "sensitivity", "specificity", "PPV", "NPV"]
    ].copy()
    display["label"] = display["granularity"].astype(str).map(GRAN_LABELS_TABLE)
    display["roc_auc"] = display["roc_auc"].map(lambda x: f"{x:.3f}")
    for c in ("sensitivity", "specificity", "PPV", "NPV"):
        display[c] = display[c].map(lambda x: f"{x:.3f}")
    display["n_decision_points"] = display["n_decision_points"].astype(int).astype(str)
    display = display[
        ["label", "n_decision_points", "roc_auc", "sensitivity", "specificity", "PPV", "NPV"]
    ].rename(
        columns={
            "label": "Granularity",
            "n_decision_points": "N",
            "roc_auc": "AUROC",
            "sensitivity": "Sens",
            "specificity": "Spec",
            "PPV": "PPV",
            "NPV": "NPV",
        }
    )
    # Give Granularity more width so "sampled"/"collapsed" are not clipped
    col_widths = [0.34, 0.10, 0.11, 0.11, 0.11, 0.11, 0.11]
    table = ax_tab.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        cell.set_edgecolor("#cccccc")
    ax_tab.set_title(
        f"Metrics at operating θ (monthly θ={PAPER_THRESHOLD:.2f}; else Youden; W={w} d, H={h} d)",
        fontsize=10,
        pad=8,
    )

    # —— Top-right: AUROC points + 95% CI ——
    ax_auc = fig.add_subplot(gs[0, 1])
    ci = auc_ci.set_index("granularity").reindex(GRAN_ORDER)
    x = np.arange(len(GRAN_ORDER))
    colors = [SAMPLE_COLOR] * 4 + [COLLAPSE_COLOR] * 3
    y = ci["auc"].to_numpy(float)
    yerr = np.vstack([y - ci["ci_low"].to_numpy(float), ci["ci_high"].to_numpy(float) - y])
    ax_auc.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        ms=8,
        lw=1.6,
        capsize=4,
        capthick=1.4,
        ecolor="#555555",
        color="none",
        zorder=3,
    )
    for i, (xi, yi, col) in enumerate(zip(x, y, colors)):
        ax_auc.scatter([xi], [yi], s=55, color=col, zorder=4, edgecolors="white", linewidths=0.6)
        ax_auc.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels([GRAN_LABELS[g] for g in GRAN_ORDER], rotation=25, ha="right", fontsize=8)
    ymin = max(0.55, float(np.nanmin(ci["ci_low"]) - 0.02))
    ymax = min(0.85, float(np.nanmax(ci["ci_high"]) + 0.03))
    ax_auc.set_ylim(ymin, ymax)
    ax_auc.set_ylabel("AUROC")
    ax_auc.set_title("Discrimination by granularity (95% CI)")
    ax_auc.grid(axis="y", alpha=0.25)
    # legend for construction family
    ax_auc.scatter([], [], color=SAMPLE_COLOR, s=45, label="Sampled")
    ax_auc.scatter([], [], color=COLLAPSE_COLOR, s=45, label="Collapsed")
    ax_auc.legend(fontsize=8, loc="lower right")

    # —— Bottom: ROC curves (no winner star) ——
    # Sampled = dashed (reds); collapsed = solid (blues)
    ax_roc = fig.add_subplot(gs[1, :])
    i_s = i_c = 0
    for gran, kind, dp in series:
        fpr, tpr, _ = roc_curve(dp["target"], dp["burden_score"])
        curve_auc = float(auc(fpr, tpr))
        if kind == "sampled":
            color = SAMPLE_ROC_COLORS[i_s % len(SAMPLE_ROC_COLORS)]
            ls = "--"
            i_s += 1
        else:
            color = COLLAPSE_ROC_COLORS[i_c % len(COLLAPSE_ROC_COLORS)]
            ls = "-"
            i_c += 1
        ax_roc.plot(
            fpr,
            tpr,
            lw=2.2 if kind == "collapsed" else 1.9,
            color=color,
            ls=ls,
            label=f"{GRAN_LABELS[gran]} (AUC={curve_auc:.3f})",
            alpha=0.95,
        )
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
    ax_roc.set_xlabel("FPR (1 − specificity)")
    ax_roc.set_ylabel("TPR (sensitivity)")
    ax_roc.set_title(f"ROC curves by granularity (W={w} d, H={h} d)")
    ax_roc.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax_roc.grid(alpha=0.2)

    fig.suptitle(
        "Alarm performance across temporal granularities",
        fontsize=13,
    )
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(out_stem.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_cap_day(daily_pid: pd.DataFrame, window: float = DAYS_PER_MONTH) -> float | None:
    """Last day to plot: end of last complete month of available data."""
    if daily_pid.empty:
        return None
    n = float(daily_pid["study_day"].max())
    if "n_days" in daily_pid.columns and daily_pid["n_days"].notna().any():
        n = float(daily_pid["n_days"].iloc[0])
    cap = float(int(np.floor(n / window)) * window)
    return cap if cap >= window else None


def plot_simplified_trajectory(
    pid: int,
    daily: pd.DataFrame,
    monthly_c: pd.DataFrame,
    W: int,
    theta_daily: float | None,
    theta_monthly_mean: float | None,
    out_stem: Path,
) -> None:
    g = daily[daily["id"] == pid].sort_values("study_day").copy()
    if g.empty:
        print(f"WARNING: no daily pUF for {pid}")
        return

    event = int(g["Evento PD"].iloc[0])
    t_ev = float(g["t_evento_eb2"].iloc[0])
    # Plots only: do not draw the incomplete trailing month (<30 d)
    cap = _plot_cap_day(g)
    if cap is not None:
        g = g[g["study_day"] <= cap].copy()
    scored = add_rolling_burden(g, w_load_days=W, min_periods=1, score="mean")
    mo = add_collapsed_burden(monthly_c[monthly_c["id"] == pid], 30, W, score="mean")
    if cap is not None and not mo.empty and "decision_day" in mo.columns:
        mo = mo[mo["decision_day"] <= cap].copy()

    days = scored["study_day"].to_numpy(int)
    puf = scored["pUF"].to_numpy(float)
    burden = scored["burden_score"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    # pUF traces (lighter)
    ax.plot(
        days,
        puf,
        color="#9ecae1",
        lw=1.0,
        alpha=0.85,
        label="Daily sliding pUF",
        zorder=1,
    )
    if not mo.empty:
        ax.plot(
            mo["decision_day"],
            mo["pUF"],
            color="#fc9272",
            lw=1.6,
            marker="o",
            markersize=4,
            alpha=0.95,
            label="Monthly collapsed pUF",
            zorder=2,
        )

    # Burden highlights (comparable mean scale)
    ax.plot(
        days,
        burden,
        color="#08519c",
        lw=2.6,
        label=f"Daily/sampled burden (mean, W={W} d)",
        zorder=3,
    )
    mo_ok = mo.dropna(subset=["burden_score"]) if not mo.empty else mo
    if len(mo_ok):
        ax.plot(
            mo_ok["decision_day"],
            mo_ok["burden_score"],
            color="#cb181d",
            lw=2.4,
            marker="o",
            markersize=5,
            label=f"Monthly collapsed burden (mean of {max(1, round(W/30))} months)",
            zorder=3,
        )

    # Construction-specific thresholds only (labelled provenance; not a shared line)
    if theta_daily is not None:
        ax.axhline(
            theta_daily,
            color="#08519c",
            ls=":",
            lw=1.3,
            alpha=0.85,
            label=f"Youden θ (daily sampled, mean) = {theta_daily:.3f}",
        )
    if theta_monthly_mean is not None:
        ax.axhline(
            theta_monthly_mean,
            color="#cb181d",
            ls=":",
            lw=1.3,
            alpha=0.85,
            label=f"Youden θ (monthly collapsed, mean) = {theta_monthly_mean:.3f}",
        )

    if event == 1 and np.isfinite(t_ev) and (cap is None or t_ev <= cap + 1e-9):
        ax.axvline(
            t_ev,
            color="black",
            ls="--",
            lw=1.4,
            label=f"Progression (day {int(t_ev)})",
        )

    status = f"PD, event day {int(t_ev)}" if event == 1 else "No PD"
    ax.set_ylim(0, 1)
    ax.set_xlabel("Study day")
    ax.set_ylabel("pUF / burden (mean scale)")
    ax.set_title(f"Patient {pid} ({status}) — sampled daily vs monthly collapsed")
    ax.legend(fontsize=8, loc="best", ncol=1, framealpha=0.92)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(out_stem.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--W", type=int, default=90)
    parser.add_argument("--H", type=int, default=120)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patients",
        type=str,
        default="62004,23003,31002,41006",
        help="Simplified trajectory patient ids",
    )
    parser.add_argument(
        "--clock",
        type=Path,
        default=None,
        help="User clock BBDD (default: data/revision/clock_bbdd.xlsx)",
    )
    args = parser.parse_args()

    root = project_root_from(args.root)
    out = root / "results/alarm/analysis_figures"
    out.mkdir(parents=True, exist_ok=True)

    print("Building decision-point series…")
    series, daily_puf, monthly_puf, _collapsed, clock = build_series(
        root, args.W, args.H, clock_path=args.clock
    )

    focus = metrics_from_series(series, args.W, args.H)
    focus.to_csv(out / f"metrics_table_W{args.W}_H{args.H}.csv", index=False)

    print(f"Bootstrap AUROC CIs (n_boot={args.n_boot}, patient-clustered)…")
    rows = []
    for gran, _kind, dp in series:
        point, lo, hi = _clustered_auc_ci(
            dp["target"].to_numpy(),
            dp["burden_score"].to_numpy(float),
            dp["id"].to_numpy(),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        rows.append(
            {
                "granularity": gran,
                "label": GRAN_LABELS[gran],
                "auc": point,
                "ci_low": lo,
                "ci_high": hi,
                "n_decision_points": len(dp),
                "n_patients": int(dp["id"].nunique()),
                "n_boot": args.n_boot,
                "ci_method": "patient-clustered bootstrap percentile 95%",
                "event_time_field": "clock_bbdd",
            }
        )
        print(f"  {gran}: {point:.3f} [{lo:.3f}, {hi:.3f}]")
    auc_ci = pd.DataFrame(rows)
    auc_ci.to_csv(out / f"auc_ci_W{args.W}_H{args.H}.csv", index=False)

    stem = out / f"summary_granularities_W{args.W}_H{args.H}"
    plot_summary(series, focus, auc_ci, stem, args.W, args.H)
    print(f"Wrote {stem}.png/.svg/.pdf")

    # Trajectories / mean-scale line: same operating θ as baseline (PAPER_THRESHOLD).
    theta_monthly_raw = float(PAPER_THRESHOLD)
    theta_monthly = theta_monthly_raw / 3.0
    theta_daily = theta_monthly
    print(
        f"Simplified thresholds (θ={theta_monthly_raw}): "
        f"daily={theta_daily:.4f} (mean); "
        f"monthly={theta_monthly:.4f} (mean)"
    )

    monthly_c = monthly_puf.copy()
    if "decision_day" not in monthly_c.columns and "month" in monthly_c.columns:
        monthly_c["decision_day"] = monthly_c["month"] * DAYS_PER_MONTH
        monthly_c["period"] = monthly_c["month"]

    patients = [int(x.strip()) for x in args.patients.split(",") if x.strip()]
    for pid in patients:
        tstem = out / f"trajectory_patient_{pid}_W{args.W}_simplified"
        plot_simplified_trajectory(
            pid,
            daily_puf,
            monthly_c,
            args.W,
            theta_daily=theta_daily,
            theta_monthly_mean=theta_monthly,
            out_stem=tstem,
        )
        print(f"Wrote {tstem}.png/.svg/.pdf")

    readme = out / "README.md"
    readme.write_text(
        f"""# Analysis figures (high resolution)

Delivery figures for the granularity summary and simplified patient trajectories.
These are the figures to show in the report (clean tables, **no** winner star).

| File | Content |
|------|---------|
| `summary_granularities_W{args.W}_H{args.H}.{{png,svg,pdf}}` | Metrics table + AUROC with patient-bootstrap 95% CI + ROC (**no** winner star) |
| `metrics_table_W{args.W}_H{args.H}.csv` | Exact Sens/Spec/PPV/NPV/AUROC |
| `auc_ci_W{args.W}_H{args.H}.csv` | AUROC point estimates and 95% CIs |
| `trajectory_patient_*_W{args.W}_simplified.*` | Simplified sampled vs monthly collapsed trajectories |

`results/alarm/figures/` holds working plots (ROC alone, W×H heatmaps, full trajectory panels). Prefer **analysis_figures** for the report summary panel.

### Notes

- **AUROC CI:** patient-clustered bootstrap ({args.n_boot} reps, seed={args.seed}).
- **Clinical clock:** user clock BBDD (`data/revision/clock_bbdd.xlsx`) for all granularities.
- Raster exports at **{DPI} dpi**.
- Regenerated by: `scripts/03_analysis/alarm/plot_analysis_figures.py --clock …`
""",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
