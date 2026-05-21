#!/usr/bin/env python3
"""KDE comparison of E.PD vs no-E.PD (first vs previous 15-day windows)."""

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
    FIGURE_DENSITY,
    LABEL_EPD,
    LABEL_NO_EPD,
    SEL_COLS,
    columns_present,
)
from stats_utils import compute_mann_whitney_p_values, filter_by_response  # noqa: E402


def plot_density_comparison(
    df_first: pd.DataFrame,
    df_previous: pd.DataFrame,
    variables: list[str],
    output: Path,
) -> None:
    df_first = filter_by_response(df_first)
    df_previous = filter_by_response(df_previous)
    p_first, p_previous = compute_mann_whitney_p_values(df_first, df_previous, variables)

    palette = {LABEL_EPD: "red", LABEL_NO_EPD: "green"}
    plt.figure(figsize=(20, len(variables) * 3))

    for i, var in enumerate(variables):
        plt.subplot(len(variables), 1, i + 1)

        sns.kdeplot(
            data=df_first,
            x=var,
            hue="Respuesta",
            fill=False,
            common_norm=False,
            linestyle="-",
            linewidth=2,
            palette=palette,
            hue_order=[LABEL_EPD, LABEL_NO_EPD],
            legend=False,
        )
        sns.kdeplot(
            data=df_previous,
            x=var,
            hue="Respuesta",
            fill=False,
            common_norm=False,
            linestyle="--",
            linewidth=2,
            palette=palette,
            hue_order=[LABEL_EPD, LABEL_NO_EPD],
            legend=False,
        )

        plt.title(
            f"{var}\n"
            f"(p-value First 15 days: {p_first[var]:.3e}, "
            f"p-value Previous 15 days: {p_previous[var]:.3e})",
            fontsize=15,
        )
        plt.xlabel(var, fontsize=13)
        plt.ylabel("Density", fontsize=13)
        plt.legend(
            handles=[
                plt.Line2D([], [], color="red", linestyle="-", linewidth=2, label=f"{LABEL_EPD} (First 15 days)"),
                plt.Line2D([], [], color="red", linestyle="--", linewidth=2, label=f"{LABEL_EPD} (Previous 15 days)"),
                plt.Line2D([], [], color="green", linestyle="-", linewidth=2, label=f"{LABEL_NO_EPD} (First 15 days)"),
                plt.Line2D([], [], color="green", linestyle="--", linewidth=2, label=f"{LABEL_NO_EPD} (Previous 15 days)"),
            ],
            loc="upper right",
            fontsize=12,
        )

    plt.suptitle(
        "Comparison of Variable Distributions: First 15 Days vs Previous 15 Days (E.PD vs no-E.PD)",
        fontsize=20,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KDE univariate comparison.")
    parser.add_argument("--cohort-v1", type=Path, default=COHORT_WINDOW1)
    parser.add_argument("--cohort-v2", type=Path, default=COHORT_WINDOW2)
    parser.add_argument("--output", type=Path, default=FIGURE_DENSITY)
    args = parser.parse_args()

    df_first = pd.read_csv(args.cohort_v1, low_memory=False)
    df_previous = pd.read_csv(args.cohort_v2, low_memory=False)
    variables = columns_present(df_first.columns, SEL_COLS)

    plot_density_comparison(df_first, df_previous, variables, args.output)


if __name__ == "__main__":
    main()
