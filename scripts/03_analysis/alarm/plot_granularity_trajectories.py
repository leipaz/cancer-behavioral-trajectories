#!/usr/bin/env python3
"""Patient trajectories comparing sampled vs collapsed *alarm scores*.

Important distinctions
----------------------
* **Daily / sampled** — ``pUF(d)`` from a *sliding* 30-day LDA window.
  Burden = rolling mean of daily ``pUF`` over the last ``W`` days.
  Weekly/biweekly/monthly *sampled* only change the evaluation calendar
  (same burden series, fewer decision days).

* **Collapsed** — ``pUF(period)`` from *one* LDA on a non-overlapping
  block (7 / 14 / 30 d). The paper monthly ``pUF(m)`` **is** the 30-day
  collapsed series (from ``topics_probs.xlsx``). It is **not** the mean
  of daily sliding ``pUF`` inside the month.
  Burden = rolling mean of the last ``round(W/block)`` block ``pUF`` values
  (equivalent to paper ``AUC_W / W`` when block=30 and score=mean).

Example:
  .venv/bin/python scripts/03_analysis/alarm/plot_granularity_trajectories.py \\
      --patients 62004,23003,31002,41006 --W 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rolling_burden_alarm import (  # noqa: E402
    DAYS_PER_MONTH,
    PAPER_THRESHOLD,
    default_master_table_path,
    extract_monthly_puf_from_master,
    project_root_from,
)
from sliding_burden_alarm import add_rolling_burden  # noqa: E402

DEFAULT_PATIENTS = (62004, 23003, 31002, 41006)


def _event_label(event: int, t_ev: float) -> str:
    return f"PD, event day {int(t_ev)}" if event == 1 else "No PD"


def _step_xy(decision_days: np.ndarray, values: np.ndarray, block: int):
    xs, ys = [], []
    for t, v in zip(decision_days, values):
        if np.isnan(v):
            continue
        xs.extend([t - block, t, t])
        ys.extend([v, v, np.nan])
    return np.asarray(xs, float), np.asarray(ys, float)


def add_collapsed_burden(
    collapsed: pd.DataFrame,
    block_size_days: int,
    w_load_days: int,
    score: str = "mean",
) -> pd.DataFrame:
    """Rolling burden on a collapsed ``pUF`` series (paper-style for block=30)."""
    w_periods = max(1, int(round(w_load_days / block_size_days)))
    parts: list[pd.DataFrame] = []
    for _, g in collapsed.groupby("id"):
        g = g.sort_values("period" if "period" in g.columns else "decision_day").copy()
        roll = g["pUF"].rolling(w_periods, min_periods=w_periods)
        g["burden_score"] = roll.mean() if score == "mean" else roll.sum()
        g["w_periods"] = w_periods
        g["block_size_days"] = block_size_days
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else collapsed.copy()


def monthly_collapsed_from_master(root: Path) -> pd.DataFrame:
    """Paper monthly pUF (= 30-day collapsed LDA blocks)."""
    master = pd.read_excel(default_master_table_path(root))
    monthly = extract_monthly_puf_from_master(master, root=root)
    out = monthly.rename(columns={"month": "period"}).copy()
    out["block_size_days"] = 30
    return out


def plot_patient_granularity(
    pid: int,
    daily: pd.DataFrame,
    weekly_c: pd.DataFrame,
    biweekly_c: pd.DataFrame,
    monthly_c: pd.DataFrame,
    W: int,
    threshold: float,
    out_stem: Path,
) -> None:
    g = daily[daily["id"] == pid].sort_values("study_day").copy()
    if g.empty:
        print(f"WARNING: no daily pUF for {pid}")
        return

    event = int(g["Evento PD"].iloc[0])
    t_ev = float(g["t_evento_eb2"].iloc[0])
    scored = add_rolling_burden(g, w_load_days=W, min_periods=1, score="mean")

    wk = add_collapsed_burden(weekly_c[weekly_c["id"] == pid], 7, W)
    bi = add_collapsed_burden(biweekly_c[biweekly_c["id"] == pid], 14, W)
    mo = add_collapsed_burden(monthly_c[monthly_c["id"] == pid], 30, W)

    days = scored["study_day"].to_numpy(int)
    puf = scored["pUF"].to_numpy(float)
    burden = scored["burden_score"].to_numpy(float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.2]},
    )

    # ── 1) Burdens + key pUF curves (daily & monthly collapsed only) ──────
    ax = axes[0]
    # Paired families: daily = blues (pUF light / burden dark);
    #                  monthly = reds (pUF light / burden dark), continuous month→month
    ax.plot(days, puf, color="#9ecae1", lw=1.0, alpha=0.9, label="Daily sliding pUF", zorder=1)
    if not mo.empty:
        ax.plot(
            mo["decision_day"],
            mo["pUF"],
            color="#fc9272",
            lw=1.7,
            marker="o",
            markersize=4,
            alpha=0.95,
            label="Monthly collapsed pUF (paper)",
            zorder=2,
        )

    ax.plot(days, burden, color="#08519c", lw=2.3, label=f"Daily/sampled burden (W={W} d)", zorder=3)

    for step, marker, color, lab in [
        (7, "o", "#ff7f00", "Weekly sampled"),
        (14, "s", "#33a02c", "Biweekly sampled"),
        (30, "D", "#1b9e77", "Monthly sampled"),
    ]:
        mask = (days - 30) % step == 0
        ax.scatter(
            days[mask],
            burden[mask],
            marker=marker,
            s=30,
            color=color,
            edgecolor="white",
            linewidths=0.4,
            label=lab,
            zorder=4,
        )

    # weekly / biweekly burdens stay as block steps; monthly as continuous line
    for df, block, color, lab in [
        (wk, 7, "#6a3d9a", f"Weekly collapsed burden ({max(1, round(W/7))}×7 d)"),
        (bi, 14, "#b15928", f"Biweekly collapsed burden ({max(1, round(W/14))}×14 d)"),
    ]:
        if df.empty or df["burden_score"].isna().all():
            continue
        ok = df.dropna(subset=["burden_score"])
        xs, ys = _step_xy(ok["decision_day"].to_numpy(float), ok["burden_score"].to_numpy(float), block)
        ax.plot(xs, ys, color=color, lw=2.0, label=lab, zorder=3)

    mo_ok = mo.dropna(subset=["burden_score"]) if not mo.empty else mo
    if len(mo_ok):
        ax.plot(
            mo_ok["decision_day"],
            mo_ok["burden_score"],
            color="#cb181d",
            lw=2.2,
            marker="o",
            markersize=4.5,
            label=f"Monthly collapsed burden (= paper AUC/{max(1, round(W/30))})",
            zorder=3,
        )

    ax.axhline(threshold, color="red", ls="--", lw=1.2, label=f"Threshold θ/W≈{threshold:.2f}")
    if event == 1:
        ax.axvline(t_ev, color="black", ls="--", lw=1.3, label=f"Progression (day {int(t_ev)})")
    ax.set_ylabel("pUF / burden")
    ax.set_title(
        f"Patient {pid} ({_event_label(event, t_ev)}) — burdens + daily / monthly pUF"
    )
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1)

    # ── 2) Sliding pUF (underlying daily signal) ──────────────────────────
    ax = axes[1]
    ax.plot(days, puf, color="#6baed6", lw=1.2, label="Daily sliding pUF (LDA on t−29…t)")
    ax.fill_between(days, puf, color="#6baed6", alpha=0.2)
    ax.plot(days, burden, color="#08519c", lw=1.8, label=f"→ rolling mean burden W={W}d")
    for step, marker, color, lab in [
        (7, "o", "#ff7f00", "Weekly sampled days"),
        (14, "s", "#33a02c", "Biweekly sampled days"),
        (30, "D", "#1b9e77", "Monthly sampled days"),
    ]:
        mask = (days - 30) % step == 0
        ax.scatter(days[mask], burden[mask], marker=marker, s=24, color=color, zorder=3, label=lab)
    if event == 1:
        ax.axvline(t_ev, color="black", ls="--", lw=1.2)
    ax.axhline(threshold, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.set_ylabel("pUF / burden")
    ax.set_title("Simple/sliding path: one shared daily signal; sampling only thins decision days")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1)

    # ── 3) Collapsed raw pUF vs its own burden ────────────────────────────
    ax = axes[2]
    xmax = float(max(days.max(), t_ev if event else days.max()))
    for m0 in range(0, int(xmax) + 30, 60):
        ax.axvspan(m0, m0 + 30, color="#fde0dd", alpha=0.28, zorder=0)

    # raw block pUF (thin) + burden (thick) for each collapsed series
    series_spec = [
        (wk, 7, "#6a3d9a", "Weekly"),
        (bi, 14, "#b15928", "Biweekly"),
        (mo, 30, "#e31a1c", "Monthly (= paper pUF)"),
    ]
    for df, block, color, name in series_spec:
        if df.empty:
            continue
        xs, ys = _step_xy(df["decision_day"].to_numpy(float), df["pUF"].to_numpy(float), block)
        ax.plot(xs, ys, color=color, lw=1.0, alpha=0.45, label=f"{name} block pUF")
        ok = df.dropna(subset=["burden_score"])
        if ok.empty:
            continue
        xs, ys = _step_xy(ok["decision_day"].to_numpy(float), ok["burden_score"].to_numpy(float), block)
        ax.plot(xs, ys, color=color, lw=2.2, label=f"{name} burden")
        if block == 30:
            for i, (t, v) in enumerate(zip(ok["decision_day"], ok["burden_score"]), start=1):
                ax.scatter([t], [v], marker="^", s=36, color=color, zorder=4)
                ax.text(t, min(0.98, float(v) + 0.04), f"M{i}", fontsize=7, ha="center", color=color)

    ax.axhline(threshold, color="red", ls="--", lw=1.0)
    if event == 1:
        ax.axvline(t_ev, color="black", ls="--", lw=1.2)
    ax.set_ylabel("Collapsed pUF / burden")
    ax.set_xlabel("Study day")
    ax.set_title(
        "Collapsed path: new pUF per non-overlapping block → own rolling burden "
        f"(W≈{W} d). Paper monthly pUF = red block series."
    )
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_panel(
    patients: list[int],
    daily: pd.DataFrame,
    weekly_c: pd.DataFrame,
    biweekly_c: pd.DataFrame,
    monthly_c: pd.DataFrame,
    W: int,
    threshold: float,
    out_stem: Path,
) -> None:
    n = len(patients)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 3.0 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, pid in zip(axes, patients):
        g = daily[daily["id"] == pid].sort_values("study_day")
        if g.empty:
            ax.set_title(f"Patient {pid} — no data")
            continue
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        scored = add_rolling_burden(g, w_load_days=W, min_periods=1, score="mean")
        days = scored["study_day"].to_numpy(int)
        puf = scored["pUF"].to_numpy(float)
        burden = scored["burden_score"].to_numpy(float)

        # Underlying pUF (light) — only daily + monthly collapsed to avoid clutter
        # Paired color families: daily blues / monthly reds (pUF light, burden dark)
        ax.plot(
            days,
            puf,
            color="#9ecae1",
            lw=0.9,
            alpha=0.85,
            label="Daily sliding pUF",
            zorder=1,
        )
        mo_raw = monthly_c[monthly_c["id"] == pid].sort_values("decision_day")
        if len(mo_raw):
            # continuous polyline across months (same continuity feel as daily)
            ax.plot(
                mo_raw["decision_day"],
                mo_raw["pUF"],
                color="#fc9272",
                lw=1.5,
                marker="o",
                markersize=3.5,
                alpha=0.95,
                label="Monthly collapsed pUF (paper)",
                zorder=2,
            )

        # Burdens (strong)
        ax.plot(days, burden, color="#08519c", lw=2.1, label="Daily/sampled burden", zorder=3)

        for df0, block, color, lab in [
            (weekly_c, 7, "#6a3d9a", "Weekly collapsed burden"),
            (biweekly_c, 14, "#b15928", "Biweekly collapsed burden"),
        ]:
            df = add_collapsed_burden(df0[df0["id"] == pid], block, W)
            ok = df.dropna(subset=["burden_score"])
            if ok.empty:
                continue
            xs, ys = _step_xy(ok["decision_day"].to_numpy(float), ok["burden_score"].to_numpy(float), block)
            ax.plot(xs, ys, color=color, lw=1.9, label=lab, zorder=3)

        # Monthly burden: continuous line month→month (paired with monthly pUF)
        mo_b = add_collapsed_burden(monthly_c[monthly_c["id"] == pid], 30, W)
        mo_ok = mo_b.dropna(subset=["burden_score"])
        if len(mo_ok):
            ax.plot(
                mo_ok["decision_day"],
                mo_ok["burden_score"],
                color="#cb181d",
                lw=2.1,
                marker="o",
                markersize=4,
                label="Monthly collapsed burden",
                zorder=3,
            )

        for step, marker, color, lab in [
            (7, "o", "#ff7f00", "Weekly sampled"),
            (14, "s", "#33a02c", "Biweekly sampled"),
            (30, "D", "#1b9e77", "Monthly sampled"),
        ]:
            mask = (days - 30) % step == 0
            ax.scatter(days[mask], burden[mask], marker=marker, s=18, color=color, zorder=4, label=lab)

        ax.axhline(threshold, color="red", ls="--", lw=1.0)
        if event == 1:
            ax.axvline(t_ev, color="black", ls="--", lw=1.1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("pUF / burden")
        ax.set_title(f"Patient {pid} ({_event_label(event, t_ev)})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, loc="best", ncol=3)

    axes[-1].set_xlabel("Study day")
    fig.suptitle(
        "Burdens by construction + daily / monthly-collapsed pUF "
        "(sampled share daily burden; collapsed recomputes its own)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--patients",
        type=str,
        default=",".join(str(p) for p in DEFAULT_PATIENTS),
    )
    parser.add_argument("--W", type=int, default=90, help="Lookback in days")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PAPER_THRESHOLD / 3,
        help="Mean-pUF threshold (default paper θ/W = 1.61/3)",
    )
    args = parser.parse_args()

    root = project_root_from(args.root)
    tables = root / "results/alarm/tables"
    figs = root / "results/alarm/figures"
    patients = [int(x.strip()) for x in args.patients.split(",") if x.strip()]

    daily = pd.read_csv(tables / "daily_sliding_puf.csv")
    weekly_c = pd.read_csv(tables / "collapsed_puf_weekly.csv")
    biweekly_c = pd.read_csv(tables / "collapsed_puf_biweekly.csv")
    monthly_c = monthly_collapsed_from_master(root)

    for pid in patients:
        stem = figs / f"trajectory_patient_{pid}_W{args.W}"
        plot_patient_granularity(
            pid, daily, weekly_c, biweekly_c, monthly_c, args.W, args.threshold, stem
        )
        print(f"Wrote {stem}.png/.svg")

    panel = figs / f"trajectories_panel_W{args.W}"
    plot_panel(patients, daily, weekly_c, biweekly_c, monthly_c, args.W, args.threshold, panel)
    print(f"Wrote {panel}.png/.svg")


if __name__ == "__main__":
    main()
