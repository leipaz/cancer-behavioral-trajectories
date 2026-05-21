#!/usr/bin/env python3
"""Histogram comparison (discrete counts) for E.PD vs no-E.PD across windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    COHORT_WINDOW1,
    COHORT_WINDOW2,
    FIGURE_HISTOGRAM,
    GRUPO_EPD,
    GRUPO_NO_EPD,
    LABEL_EPD,
    LABEL_NO_EPD,
    SEL_COLS,
    columns_present,
)
from stats_utils import filter_by_response  # noqa: E402


def _early_pd_str(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).replace(
        {str(GRUPO_EPD): LABEL_EPD, str(GRUPO_NO_EPD): LABEL_NO_EPD}
    )
    return mapped


def plot_histogram_comparison(
    df_first: pd.DataFrame,
    df_previous: pd.DataFrame,
    variables: list[str],
    output: Path,
    bins_count: int = 15,
) -> None:
    df_first = filter_by_response(df_first)
    df_previous = filter_by_response(df_previous)

    color_epd = "red"
    color_no_epd = "green"
    color_map = {LABEL_EPD: color_epd, LABEL_NO_EPD: color_no_epd}

    plt.figure(figsize=(18, len(variables) * 3.5))

    for i, var in enumerate(variables):
        plt.subplot(len(variables), 1, i + 1)

        plot_first = df_first.copy()
        plot_prev = df_previous.copy()
        plot_first["Early PD_str"] = _early_pd_str(plot_first["Early PD"])
        plot_prev["Early PD_str"] = _early_pd_str(plot_prev["Early PD"])

        sns.histplot(
            data=plot_prev,
            x=var,
            hue="Early PD_str",
            stat="count",
            common_norm=False,
            bins=bins_count,
            palette=color_map,
            kde=False,
            fill=True,
            alpha=0.3,
            edgecolor=None,
            linewidth=0,
            legend=(i == 0),
        )
        sns.histplot(
            data=plot_first,
            x=var,
            hue="Early PD_str",
            stat="count",
            common_norm=False,
            bins=bins_count,
            palette=color_map,
            kde=False,
            element="step",
            fill=False,
            linewidth=2,
            legend=False,
        )

        plt.title(var, fontsize=15)
        plt.xlabel(var, fontsize=13)
        plt.ylabel("Count (Absolute Frequency)", fontsize=13)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color=color_epd, linewidth=2, linestyle="-", label=f"{LABEL_EPD} (First 15 Days)"),
                plt.Rectangle((0, 0), 1, 1, fc=color_epd, alpha=0.3, label=f"{LABEL_EPD} (Previous 15 Days)"),
                plt.Line2D([0], [0], color=color_no_epd, linewidth=2, linestyle="-", label=f"{LABEL_NO_EPD} (First 15 Days)"),
                plt.Rectangle((0, 0), 1, 1, fc=color_no_epd, alpha=0.3, label=f"{LABEL_NO_EPD} (Previous 15 Days)"),
            ],
            fontsize=12,
            loc="best",
        )

    plt.suptitle(
        "Comparison of Variable Distributions: First 15 Days vs Previous 15 Days (E.PD vs no-E.PD)",
        fontsize=20,
        fontweight="bold",
        y=1.005,
    )
    plt.subplots_adjust(hspace=0.4, top=0.95)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histogram univariate comparison.")
    parser.add_argument("--cohort-v1", type=Path, default=COHORT_WINDOW1)
    parser.add_argument("--cohort-v2", type=Path, default=COHORT_WINDOW2)
    parser.add_argument("--output", type=Path, default=FIGURE_HISTOGRAM)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    df_first = pd.read_csv(args.cohort_v1, low_memory=False)
    df_previous = pd.read_csv(args.cohort_v2, low_memory=False)
    variables = columns_present(df_first.columns, SEL_COLS)

    plot_histogram_comparison(
        df_first, df_previous, variables, args.output, bins_count=args.bins
    )


if __name__ == "__main__":
    main()
