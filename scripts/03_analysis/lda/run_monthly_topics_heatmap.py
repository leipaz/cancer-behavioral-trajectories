#!/usr/bin/env python3
"""Build monthly topic table (optional) and plot predominant-topics heatmap."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly predominant topics heatmap pipeline (notebook 6_1)."
    )
    parser.add_argument(
        "--rebuild-table",
        action="store_true",
        help="Regenerate monthly CSV from per-day topics + LDA model",
    )
    parser.add_argument("--skip-plot", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.rebuild_table:
        print(f"\n{'=' * 60}\nBuild monthly predominant topics table\n{'=' * 60}")
        subprocess.run(
            [sys.executable, str(_SCRIPT_DIR / "build_monthly_predominant_topics.py"), *unknown],
            check=True,
        )

    if not args.skip_plot:
        print(f"\n{'=' * 60}\nPlot monthly topics heatmap\n{'=' * 60}")
        subprocess.run(
            [sys.executable, str(_SCRIPT_DIR / "plot_monthly_topics_heatmap.py"), *unknown],
            check=True,
        )

    print("\nMonthly topics heatmap pipeline completed.")


if __name__ == "__main__":
    main()
