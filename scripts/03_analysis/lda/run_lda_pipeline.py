#!/usr/bin/env python3
"""
Full LDA pipeline: train model (optional) → assign topics → merge clinical → plots.

Skips training if ``--skip-train`` and model files already exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("train_lda_from_profiles.py", "Train LDA from profiles PKL"),
    ("assign_lda_topics.py", "Assign topics and top-terms plots"),
    ("merge_clinical_topics.py", "Merge with clinical covariates"),
    ("plot_lda_results.py", "Event vs topic figures"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LDA analysis pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Use existing model/dict")
    parser.add_argument("--passes", type=int, default=None, help="Override LDA passes")
    args, unknown = parser.parse_known_args()

    for script, label in STEPS:
        if args.skip_train and script == "train_lda_from_profiles.py":
            print(f"Skipping: {label}")
            continue

        cmd = [sys.executable, str(_SCRIPT_DIR / script), *unknown]
        if script == "train_lda_from_profiles.py" and args.passes is not None:
            cmd.extend(["--passes", str(args.passes)])

        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        subprocess.run(cmd, check=True)

    print("\nLDA pipeline completed.")


if __name__ == "__main__":
    main()
