#!/usr/bin/env python3
"""Paper-style individual trajectories: monthly pUF + rolling AUC_W.

Matches Figure 5 (Patients and methods alarm):
  - pUF(m) = P(topic 3+4+5) in month m
  - AUC_W(m) = sum of last W monthly pUF values
  - threshold θ = 1.61 (shown scaled as θ/W when normalize_auc=True)

Example:
  .venv/bin/python scripts/03_analysis/alarm/plot_patient_trajectories.py \\
      --patients 62004,23003,31002,41006
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
    build_decision_points,
    default_clinical_path,
    default_master_table_path,
    extract_monthly_puf_from_master,
    project_root_from,
)


DEFAULT_PATIENTS = (62004, 23003, 31002, 41006)


def patient_month_series(monthly_puf: pd.DataFrame, pid: int, W: int = 3) -> pd.DataFrame | None:
    g = monthly_puf[monthly_puf["id"] == pid].sort_values("month")
    if g.empty:
        return None
    t_ev = float(g["t_evento_eb2"].iloc[0])
    fu = float(g["follow_up_days"].iloc[0]) if "follow_up_days" in g.columns else t_ev
    max_by_fu = int(np.floor(fu / DAYS_PER_MONTH))
    g = g[g["month"] <= max_by_fu].copy()
    if g.empty:
        return None

    pufs = g["pUF"].to_numpy(dtype=float)
    months = g["month"].to_numpy(dtype=int)
    auc = np.full(len(g), np.nan, dtype=float)
    for i, m in enumerate(months):
        if i + 1 < W:
            continue
        window = pufs[i - W + 1 : i + 1]
        # require contiguous months
        if months[i] - months[i - W + 1] != W - 1:
            continue
        if np.isnan(window).any():
            continue
        auc[i] = float(window.sum())

    out = pd.DataFrame({"month": months, "pUF": pufs, "AUC": auc})
    out["Evento PD"] = int(g["Evento PD"].iloc[0])
    out["t_evento_eb2"] = t_ev
    out["follow_up_days"] = fu
    return out


def alarm_landmarks(decision_points: pd.DataFrame, pid: int, threshold: float) -> dict:
    dp = decision_points[decision_points["id"] == pid].copy()
    if dp.empty:
        return {"first_alarm_m": np.nan, "first_TP_alarm_m": np.nan}
    dp["alarm"] = dp["AUC_W"] >= threshold
    dp["TP"] = dp["alarm"] & (dp["target"] == 1)
    first_alarm = dp.loc[dp["alarm"], "month"].min() if dp["alarm"].any() else np.nan
    first_tp = dp.loc[dp["TP"], "month"].min() if dp["TP"].any() else np.nan
    return {"first_alarm_m": first_alarm, "first_TP_alarm_m": first_tp}


def cohort_label(clinical: pd.DataFrame | None, pid: int, event: int) -> str:
    if clinical is not None and "Cohort" in clinical.columns:
        id_col = "S_RECORD_id" if "S_RECORD_id" in clinical.columns else "id"
        hit = clinical.loc[clinical[id_col] == pid, "Cohort"]
        if len(hit):
            return str(hit.iloc[0])
    return "PD" if event == 1 else "No PD"


def plot_patient(
    series: pd.DataFrame,
    pid: int,
    W: int,
    H: int,
    threshold: float,
    landmarks: dict,
    cohort: str,
    out_path: Path,
    normalize_auc: bool = True,
) -> None:
    ss = series.dropna(subset=["pUF"]).copy()
    event = int(ss["Evento PD"].iloc[0])
    t_ev = float(ss["t_evento_eb2"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ss["month"], ss["pUF"], marker="o", lw=1.8, label="pUF (unfavourable hypertopic)")

    if normalize_auc:
        ax.plot(
            ss["month"],
            ss["AUC"] / W,
            marker="o",
            lw=1.8,
            label=f"Rolling AUC/W (W={W})",
        )
        ax.axhline(
            threshold / W,
            linestyle="--",
            color="C3",
            label=f"Threshold θ={threshold:.2f} (scaled θ/W={threshold / W:.2f})",
        )
        ax.set_ylabel("Probability / normalized AUC")
        ax.set_ylim(0, 1)
    else:
        ax2 = ax.twinx()
        ax2.plot(ss["month"], ss["AUC"], marker="o", color="C1", label=f"Rolling AUC (W={W})")
        ax2.axhline(threshold, linestyle="--", color="C3")
        ax.set_ylabel("pUF")
        ax2.set_ylabel("Rolling AUC")
        ax.set_ylim(0, 1)

    fa = landmarks.get("first_alarm_m", np.nan)
    fta = landmarks.get("first_TP_alarm_m", np.nan)
    if pd.notna(fa):
        ax.axvline(int(fa), linestyle="--", color="C2", label=f"First alarm (m={int(fa)})")
    if pd.notna(fta):
        ax.axvline(int(fta), linestyle=":", color="C4", label=f"First TP alarm (m={int(fta)})")
    if event == 1:
        ax.axvline(
            t_ev / DAYS_PER_MONTH,
            linestyle="-.",
            color="red",
            label=f"PD at day {t_ev:.0f}",
        )

    ax.set_title(f"Patient {pid} ({cohort}) — rolling-AUC alarm (H={H} months)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_panel(
    series_by_pid: dict[int, pd.DataFrame],
    landmarks_by_pid: dict[int, dict],
    cohorts: dict[int, str],
    W: int,
    H: int,
    threshold: float,
    out_path: Path,
) -> None:
    pids = list(series_by_pid.keys())
    n = len(pids)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, pid in zip(axes, pids):
        ss = series_by_pid[pid].dropna(subset=["pUF"]).copy()
        event = int(ss["Evento PD"].iloc[0])
        t_ev = float(ss["t_evento_eb2"].iloc[0])
        ax.plot(ss["month"], ss["pUF"], marker="o", lw=1.6, label="pUF")
        ax.plot(ss["month"], ss["AUC"] / W, marker="o", lw=1.6, label=f"AUC/{W}")
        ax.axhline(threshold / W, linestyle="--", color="C3", label=f"θ/{W}={threshold / W:.2f}")
        lm = landmarks_by_pid[pid]
        if pd.notna(lm.get("first_alarm_m", np.nan)):
            ax.axvline(int(lm["first_alarm_m"]), linestyle="--", color="C2", alpha=0.8)
        if event == 1:
            ax.axvline(t_ev / DAYS_PER_MONTH, linestyle="-.", color="red", alpha=0.9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("pUF / AUC/W")
        ax.set_title(f"Patient {pid} ({cohorts[pid]})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    axes[-1].set_xlabel("Month")
    fig.suptitle(
        f"Paper-style trajectories — monthly pUF & rolling AUC (W={W}, H={H}, θ={threshold})",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--patients",
        type=str,
        default=",".join(str(p) for p in DEFAULT_PATIENTS),
        help="Comma-separated patient ids",
    )
    parser.add_argument("--W", type=int, default=3)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=PAPER_THRESHOLD)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    root = project_root_from(args.root)
    figs = root / "results/alarm/figures"
    tables = root / "results/alarm/tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    patients = [int(x.strip()) for x in args.patients.split(",") if x.strip()]
    master = pd.read_excel(default_master_table_path(root))
    monthly = extract_monthly_puf_from_master(master, root=root)
    dp = build_decision_points(monthly, W=args.W, H=args.H)

    clinical = None
    clin_path = default_clinical_path(root)
    if clin_path.exists():
        clinical = pd.read_excel(clin_path)
        clinical.columns = [str(c).strip() for c in clinical.columns]

    series_by_pid: dict[int, pd.DataFrame] = {}
    landmarks_by_pid: dict[int, dict] = {}
    cohorts: dict[int, str] = {}

    for pid in patients:
        series = patient_month_series(monthly, pid, W=args.W)
        if series is None:
            print(f"WARNING: no monthly pUF for patient {pid}")
            continue
        lm = alarm_landmarks(dp, pid, args.threshold)
        cohort = cohort_label(clinical, pid, int(series["Evento PD"].iloc[0]))
        series_by_pid[pid] = series
        landmarks_by_pid[pid] = lm
        cohorts[pid] = cohort

        out = figs / f"trajectory_patient_{pid}_paper_W{args.W}_H{args.H}"
        plot_patient(
            series,
            pid,
            args.W,
            args.H,
            args.threshold,
            lm,
            cohort,
            out,
            normalize_auc=not args.no_normalize,
        )
        series.assign(id=pid).to_csv(tables / f"trajectory_series_{pid}_W{args.W}.csv", index=False)
        print(f"Wrote {out}.png/.svg  | first_alarm={lm['first_alarm_m']} first_TP={lm['first_TP_alarm_m']}")

    if series_by_pid:
        panel = figs / f"trajectories_panel_paper_W{args.W}_H{args.H}"
        plot_panel(series_by_pid, landmarks_by_pid, cohorts, args.W, args.H, args.threshold, panel)
        print(f"Wrote {panel}.png/.svg")


if __name__ == "__main__":
    main()
