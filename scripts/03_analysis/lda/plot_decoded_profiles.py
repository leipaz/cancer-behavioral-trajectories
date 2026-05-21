#!/usr/bin/env python3
"""Figures for decoded LDA top-10 profiles (notebook 5_decodificar_perfiles)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    BEHAVIORAL_FEATURE_COLS,
    DECODED_PROFILES_SCALED_CSV,
    FIGURE_DECODED_FEATURES_BOXPLOT,
    FIGURE_FINAL_PATTERN_GRID,
    FIGURE_PATTERN_0_INDIVIDUAL,
    FIGURE_PATTERN_3_INDIVIDUAL,
    PATTERN_BAR_DEFAULT_PALETTE,
    PATTERN_BAR_PALETTE,
    PATTERN_INDIVIDUAL_TOPICS,
    RESULTS_DECODED_TOPIC_PLOTS_DIR,
    TOPIC_COLORS,
)


def _plot_topic_barpanel(
    df_melted: pd.DataFrame,
    topic_id: int,
    ax: plt.Axes,
    *,
    ylabel: str,
    title_fontsize: int = 18,
    tick_labelsize: int = 10,
    legend_fontsize: str = "x-small",
) -> None:
    current_palette = PATTERN_BAR_PALETTE.get(topic_id, PATTERN_BAR_DEFAULT_PALETTE)
    sns.barplot(
        data=df_melted,
        x="Feature",
        y="Z-score",
        hue="profile",
        palette=current_palette,
        ax=ax,
    )
    ax.set_title(f"Pattern {topic_id}", fontsize=title_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=tick_labelsize)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="-")
    ax.legend(title="Day-Type ID", loc="best", fontsize=legend_fontsize, ncol=2)


def plot_final_pattern_analysis_grid(
    df_scaled: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_dir: Path | None = None,
    grid_output: Path | None = None,
    individual_topics: tuple[int, ...] = PATTERN_INDIVIDUAL_TOPICS,
) -> None:
    """
    Final manuscript grid: z-scored features per profile, per pattern.

    Matches ``5_decodificar_perfiles.ipynb`` (grid SVG + individual SVG for 0 and 3).
    """
    feature_cols = feature_cols or BEHAVIORAL_FEATURE_COLS
    save_dir = save_dir or RESULTS_DECODED_TOPIC_PLOTS_DIR
    grid_output = grid_output or FIGURE_FINAL_PATTERN_GRID
    save_dir.mkdir(parents=True, exist_ok=True)

    topics = sorted(df_scaled["topic"].unique())
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15), sharey=True)
    axes = axes.flatten()

    for i, topic_id in enumerate(topics):
        ax = axes[i]
        df_topic = df_scaled[df_scaled["topic"] == topic_id]
        df_melted = df_topic.melt(
            id_vars=["profile"],
            value_vars=feature_cols,
            var_name="Feature",
            value_name="Z-score",
        )

        ylabel = "Standardized Value (Z-score)" if i % 3 == 0 else ""
        _plot_topic_barpanel(
            df_melted,
            topic_id,
            ax,
            ylabel=ylabel,
        )

        if topic_id in individual_topics:
            fig_sub = plt.figure(figsize=(8, 5))
            ax_sub = fig_sub.add_subplot(111)
            _plot_topic_barpanel(
                df_melted,
                topic_id,
                ax_sub,
                ylabel="Standardized Value (Z-score)",
                title_fontsize=14,
                tick_labelsize=8,
                legend_fontsize="xx-small",
            )
            if topic_id == 0:
                sub_path = FIGURE_PATTERN_0_INDIVIDUAL
            elif topic_id == 3:
                sub_path = FIGURE_PATTERN_3_INDIVIDUAL
            else:
                sub_path = save_dir / f"pattern_{topic_id}_individual.svg"
            fig_sub.savefig(sub_path, format="svg", bbox_inches="tight")
            print(f"Saved individual plot: {sub_path}")
            plt.close(fig_sub)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Predominant Day-Types per Pattern: Clinical Feature Deviations",
        fontsize=26,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(grid_output, format="svg", bbox_inches="tight")
    print(f"Saved main grid plot: {grid_output}")
    plt.close(fig)


def plot_features_boxplot_by_pattern(
    df_scaled: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """Boxplot of z-scored behavioral features across patterns."""
    feature_cols = feature_cols or BEHAVIORAL_FEATURE_COLS
    df_melted = df_scaled.melt(
        id_vars=["topic"],
        value_vars=feature_cols,
        var_name="Feature",
        value_name="Value",
    )
    df_plot = df_melted.rename(columns={"topic": "Pattern"})

    fig, ax = plt.subplots(figsize=(24, 12))
    sns.boxplot(
        data=df_plot,
        x="Feature",
        y="Value",
        hue="Pattern",
        palette=TOPIC_COLORS,
        showfliers=False,
        ax=ax,
    )
    ax.set_title(
        "Distribution of Behavioral Features by Pattern",
        fontsize=24,
        fontweight="bold",
    )
    ax.set_xlabel("Behavioral Features", fontsize=18)
    ax.set_ylabel("Standardized Value (Z-score)", fontsize=18)
    ax.legend(
        title="Pattern",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=14,
        title_fontsize=16,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decoded LDA profile features.")
    parser.add_argument(
        "--decoded-scaled-csv",
        type=Path,
        default=DECODED_PROFILES_SCALED_CSV,
        help="Z-scored decoded profiles CSV",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=RESULTS_DECODED_TOPIC_PLOTS_DIR,
        help="Directory for final pattern grid SVG outputs",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=FIGURE_FINAL_PATTERN_GRID,
    )
    parser.add_argument(
        "--boxplot-output",
        type=Path,
        default=FIGURE_DECODED_FEATURES_BOXPLOT,
    )
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-boxplot", action="store_true")
    args = parser.parse_args()

    if not args.skip_grid:
        if not args.decoded_scaled_csv.is_file():
            raise FileNotFoundError(
                f"Missing {args.decoded_scaled_csv}; run decode_profiles.py"
            )
        df_scaled = pd.read_csv(args.decoded_scaled_csv)
        plot_final_pattern_analysis_grid(
            df_scaled,
            save_dir=args.save_dir,
            grid_output=args.grid_output,
        )

    if not args.skip_boxplot:
        if not args.decoded_scaled_csv.is_file():
            raise FileNotFoundError(
                f"Missing {args.decoded_scaled_csv}; run decode_profiles.py"
            )
        df_scaled = pd.read_csv(args.decoded_scaled_csv)
        plot_features_boxplot_by_pattern(
            df_scaled, output_path=args.boxplot_output
        )


if __name__ == "__main__":
    main()
