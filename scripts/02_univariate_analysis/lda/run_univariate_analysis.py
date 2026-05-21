#!/usr/bin/env python3
"""Run the full univariate analysis pipeline (prepare → plots → summary table)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    "prepare_cohorts.py",
    "plot_density_comparison.py",
    "plot_histogram_comparison.py",
    "export_summary_table.py",
]


def main() -> None:
    for step in STEPS:
        script = _SCRIPT_DIR / step
        print(f"\n{'=' * 60}\nRunning {step}\n{'=' * 60}")
        subprocess.run([sys.executable, str(script)], check=True)
    print("\nUnivariate analysis pipeline completed.")


if __name__ == "__main__":
    main()
