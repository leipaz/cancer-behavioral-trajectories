#!/usr/bin/env python3
"""Orchestrate the rolling-burden alarm analysis pipeline.

Downstream of LDA / DCABP monthly topics. Typical flow:

1. Monthly paper replication (W=3 months, H=4 months)
2. Granularity sweep (sampled vs collapsed; W×H grid) using existing pUF tables
3. Missingness funnel + W×H AUC heatmaps
4. Optional patient trajectories

Examples
--------
# Fast path (reuse tables under results/alarm/tables/)
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda

# Paper monthly only
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --monthly-only

# Rebuild collapsed pUF then full sweep
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --rebuild-collapsed --skip-lda

# Include trajectory figures for example patients
.venv/bin/python scripts/03_analysis/alarm/run_alarm_pipeline.py --skip-lda --with-trajectories
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _run(script: str, *extra: str, label: str) -> None:
    cmd = [sys.executable, str(_SCRIPT_DIR / script), *extra]
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rolling-burden alarm pipeline (monthly + granularity)."
    )
    parser.add_argument(
        "--monthly-only",
        action="store_true",
        help="Only replicate manuscript monthly collapsed alarm (W=3, H=4).",
    )
    parser.add_argument(
        "--skip-lda",
        action="store_true",
        help="Reuse daily_sliding_puf.csv / collapsed tables (do not rebuild LDA windows).",
    )
    parser.add_argument(
        "--rebuild-collapsed",
        action="store_true",
        help="Rebuild weekly/biweekly collapsed pUF before the sweep.",
    )
    parser.add_argument(
        "--skip-missingness",
        action="store_true",
        help="Skip missingness funnel and W×H grid plots.",
    )
    parser.add_argument(
        "--with-trajectories",
        action="store_true",
        help="Plot paper + granularity trajectories for example patients.",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default="62004,23003,31002,41006",
        help="Comma-separated patient IDs for --with-trajectories.",
    )
    parser.add_argument("--w-days", type=str, default="60,90,120")
    parser.add_argument("--h-days", type=str, default="90,120,180")
    args = parser.parse_args()

    if args.monthly_only:
        _run(
            "run_granularity_sweep.py",
            "--monthly-only",
            label="Monthly paper replication (W=3 m, H=4 m)",
        )
        print("\nAlarm pipeline completed (monthly-only).")
        return

    sweep_args: list[str] = ["--w-days", args.w_days, "--h-days", args.h_days]
    if args.skip_lda:
        sweep_args.append("--skip-lda")
    if args.rebuild_collapsed:
        sweep_args.append("--rebuild-collapsed")

    _run(
        "run_granularity_sweep.py",
        "--monthly-only",
        label="1/4 Monthly paper replication (W=3 m, H=4 m)",
    )
    _run(
        "run_granularity_sweep.py",
        *sweep_args,
        label="2/4 Granularity × (W, H) sweep",
    )

    if not args.skip_missingness:
        _run(
            "analyze_missingness_and_wh_grid.py",
            "--w-days",
            args.w_days,
            "--h-days",
            args.h_days,
            label="3/4 Missingness funnel + W×H AUC grid",
        )
    else:
        print("\nSkipping: missingness / W×H grid")

    if args.with_trajectories:
        _run(
            "plot_patient_trajectories.py",
            "--patients",
            args.patients,
            label="4/4 Paper-style patient trajectories",
        )
        _run(
            "plot_granularity_trajectories.py",
            "--patients",
            args.patients,
            "--W",
            "90",
            label="4b/4 Granularity comparison trajectories",
        )
    else:
        print("\nSkipping: trajectories (pass --with-trajectories to enable)")

    print("\nAlarm pipeline completed.")
    print("Report: results/alarm/granularidad_alarma.html")


if __name__ == "__main__":
    main()
