#!/usr/bin/env python3
"""Orchestrate rolling-burden alarms from a user clock BBDD.

Preferred flow (clock is the only clinical time source):

1. Paper baseline W=3 / H=4  → ``run_paper_alarm.py --clock …``
2. Optional granularity experiment → ``run_granularity_experiment.py --clock …``

Legacy Subjects-based sweep remains available with ``--legacy-subjects-clock``.

Examples
--------
# Paper only (default clock_bbdd.xlsx)
python scripts/03_analysis/alarm/run_alarm_pipeline.py --monthly-only

# Paper + granularity
python scripts/03_analysis/alarm/run_alarm_pipeline.py --with-granularity --skip-lda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _run(script: str, *extra: str, label: str) -> None:
    cmd = [sys.executable, str(_SCRIPT_DIR / script), *extra]
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rolling-burden alarm pipeline (clock BBDD → paper ± granularity)."
    )
    parser.add_argument(
        "--clock",
        type=Path,
        default=None,
        help="Universal clock BBDD (default: data/revision/clock_bbdd.xlsx)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Output folder under results/alarm/runs/",
    )
    parser.add_argument(
        "--monthly-only",
        action="store_true",
        help="Only paper monthly collapsed alarm (W=3, H=4).",
    )
    parser.add_argument(
        "--with-granularity",
        action="store_true",
        help="Also run the granularity experiment (same --clock).",
    )
    parser.add_argument(
        "--skip-lda",
        action="store_true",
        help="Reuse daily_sliding_puf.csv / collapsed tables (granularity path).",
    )
    parser.add_argument(
        "--rebuild-puf",
        action="store_true",
        help="Rebuild daily/collapsed pUF before granularity.",
    )
    parser.add_argument(
        "--legacy-subjects-clock",
        action="store_true",
        help="Use old Subjects Obs_time sweep (not recommended).",
    )
    parser.add_argument(
        "--rebuild-collapsed",
        action="store_true",
        help="(legacy) Rebuild weekly/biweekly collapsed pUF before the sweep.",
    )
    parser.add_argument(
        "--skip-missingness",
        action="store_true",
        help="(legacy) Skip missingness funnel and W×H grid plots.",
    )
    parser.add_argument(
        "--with-trajectories",
        action="store_true",
        help="(legacy) Plot paper + granularity trajectories for example patients.",
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

    if args.legacy_subjects_clock:
        _legacy_main(args)
        return

    from clock_schema import default_clock_path, project_root_from

    root = project_root_from()
    clock_path = args.clock or default_clock_path(root)
    if not clock_path.exists():
        raise SystemExit(
            f"Clock BBDD not found: {clock_path}\n"
            "Build with: python scripts/03_analysis/alarm/clock_schema.py --from-audit-db1"
        )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    _run(
        "run_paper_alarm.py",
        "--clock",
        str(clock_path),
        "--run-id",
        run_id,
        label="1/2 Paper alarm W=3 H=4 (user clock BBDD)",
    )

    if args.monthly_only and not args.with_granularity:
        print(f"\nAlarm pipeline completed (paper only). Run id: {run_id}")
        print(f"Outputs: results/alarm/runs/{run_id}/paper_W3_H4/")
        return

    if args.with_granularity or not args.monthly_only:
        # Default path without --monthly-only still runs paper only unless
        # --with-granularity is set (keeps experiments separate).
        if not args.with_granularity:
            print(
                "\nSkipping granularity (pass --with-granularity). "
                f"Paper outputs: results/alarm/runs/{run_id}/paper_W3_H4/"
            )
            return
        gran_extra = [
            "--clock",
            str(clock_path),
            "--run-id",
            run_id,
            "--w-days",
            args.w_days,
            "--h-days",
            args.h_days,
        ]
        if args.rebuild_puf or not args.skip_lda:
            # rebuild only when explicitly asked; --skip-lda keeps existing tables
            if args.rebuild_puf:
                gran_extra.append("--rebuild-puf")
        _run(
            "run_granularity_experiment.py",
            *gran_extra,
            label="2/2 Granularity experiment (user clock BBDD)",
        )

    print(f"\nAlarm pipeline completed. Run id: {run_id}")
    print(f"Outputs under results/alarm/runs/{run_id}/")


def _legacy_main(args: argparse.Namespace) -> None:
    """Previous Subjects Obs_time orchestration (kept for comparison)."""
    if args.monthly_only:
        _run(
            "run_granularity_sweep.py",
            "--monthly-only",
            label="(legacy) Monthly paper replication (W=3 m, H=4 m)",
        )
        print("\nAlarm pipeline completed (legacy monthly-only).")
        return

    sweep_args: list[str] = ["--w-days", args.w_days, "--h-days", args.h_days]
    if args.skip_lda:
        sweep_args.append("--skip-lda")
    if args.rebuild_collapsed:
        sweep_args.append("--rebuild-collapsed")

    _run(
        "run_granularity_sweep.py",
        "--monthly-only",
        label="1/4 (legacy) Monthly paper replication (W=3 m, H=4 m)",
    )
    _run(
        "run_granularity_sweep.py",
        *sweep_args,
        label="2/4 (legacy) Granularity × (W, H) sweep",
    )

    if not args.skip_missingness:
        _run(
            "analyze_missingness_and_wh_grid.py",
            "--w-days",
            args.w_days,
            "--h-days",
            args.h_days,
            label="3/4 (legacy) Missingness funnel + W×H AUC grid",
        )
    else:
        print("\nSkipping: missingness / W×H grid")

    if args.with_trajectories:
        _run(
            "plot_patient_trajectories.py",
            "--patients",
            args.patients,
            label="4/4 (legacy) Paper-style patient trajectories",
        )
        _run(
            "plot_granularity_trajectories.py",
            "--patients",
            args.patients,
            "--W",
            "90",
            label="4b/4 (legacy) Granularity comparison trajectories",
        )
    else:
        print("\nSkipping: trajectories (pass --with-trajectories to enable)")

    print("\nAlarm pipeline completed (legacy Subjects clock).")


if __name__ == "__main__":
    main()
