#!/usr/bin/env python3
"""Export full univariate summary table (mean, median IQR, min, max, variance, p-values)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    COHORT_WINDOW1,
    COHORT_WINDOW2,
    SEL_COLS,
    TABLE_SUMMARY_FULL,
    columns_present,
)
from stats_utils import (  # noqa: E402
    build_summary_table,
    compute_mann_whitney_p_values,
    filter_by_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export univariate summary CSV.")
    parser.add_argument("--cohort-v1", type=Path, default=COHORT_WINDOW1)
    parser.add_argument("--cohort-v2", type=Path, default=COHORT_WINDOW2)
    parser.add_argument("--output", type=Path, default=TABLE_SUMMARY_FULL)
    args = parser.parse_args()

    df_first = pd.read_csv(args.cohort_v1, low_memory=False)
    df_previous = pd.read_csv(args.cohort_v2, low_memory=False)
    variables = columns_present(df_first.columns, SEL_COLS)

    df_first_f = filter_by_response(df_first)
    df_previous_f = filter_by_response(df_previous)
    p_first, p_previous = compute_mann_whitney_p_values(
        df_first, df_previous, variables
    )

    summary = build_summary_table(
        df_first_f,
        df_previous_f,
        variables,
        p_first,
        p_previous,
        include_min_max=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
