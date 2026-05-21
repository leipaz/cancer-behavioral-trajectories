#!/usr/bin/env python3
"""Decode LDA top-10 profiles and generate decoded-profile figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("decode_profiles.py", "Map profile tokens to decoded behavioral features"),
    ("plot_decoded_profiles.py", "Decoded profile figures"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode LDA top-10 profiles and plot behavioral features."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write CSV tables, no figures",
    )
    args, unknown = parser.parse_known_args()

    for script, label in STEPS:
        if args.skip_plots and script == "plot_decoded_profiles.py":
            print(f"Skipping: {label}")
            continue

        cmd = [sys.executable, str(_SCRIPT_DIR / script), *unknown]
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        subprocess.run(cmd, check=True)

    print("\nDecode profiles pipeline completed.")


if __name__ == "__main__":
    main()
