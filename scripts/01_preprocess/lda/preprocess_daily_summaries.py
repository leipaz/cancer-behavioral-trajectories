#!/usr/bin/env python3
"""
Unify and clean the daily summary dataset only (not full LDA / VQ-VAE).

Reads ``daily_summary_PMP_lock_2025_09_11_with_extra_garmin.csv``,
cleans heart-rate sentinels and invalid sleep_start values, and writes
``daily_summary_eb2prod_dic25_enriched_filtered.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/raw/daily_summary_PMP_lock_2025_09_11_with_extra_garmin.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/raw/daily_summary_eb2prod_dic25_enriched_filtered.csv"
)

COLS_HR = ["heart_rate_hr_mean", "heart_rate_hr_min", "heart_rate_hr_max"]
SLEEP_START_MAX_SECONDS = 24 * 60 * 60  # 86,400


def preprocess_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Apply HR and sleep_start cleaning from the analysis notebook."""
    out = df.copy()

    present_hr = [c for c in COLS_HR if c in out.columns]
    if present_hr:
        out[present_hr] = out[present_hr].replace(-1.0, np.nan)

    if "sleep_start" in out.columns:
        out["sleep_start"] = out["sleep_start"].mask(
            out["sleep_start"] > SLEEP_START_MAX_SECONDS,
            np.nan,
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean daily summary CSV (HR -1 → NaN, sleep_start > 24h → NaN)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV (default: data/raw/...PMP_lock...garmin.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV (default: data/raw/...enriched_filtered.csv)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    df_clean = preprocess_daily_summary(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output, index=False)

    print("Preprocessing completed.")
    print(f"  Input:  {args.input} ({len(df):,} rows)")
    print(f"  Output: {args.output}")

    if "heart_rate_hr_mean" in df_clean.columns:
        print(f"  Min heart_rate_hr_mean: {df_clean['heart_rate_hr_mean'].min()}")
    if "sleep_start" in df_clean.columns:
        print(f"  Max sleep_start: {df_clean['sleep_start'].max()}")


if __name__ == "__main__":
    main()
