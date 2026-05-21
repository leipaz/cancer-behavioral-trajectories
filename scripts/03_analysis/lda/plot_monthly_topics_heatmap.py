#!/usr/bin/env python3
"""
Heatmap of predominant LDA topics per month per patient.

Extracted from ``notebooks/6_1_entropy_variability_cleaned_dec25.ipynb`` (cell 32).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    FIGURE_MONTHLY_TOPICS_HEATMAP,
    FIGURE_MONTHLY_TOPICS_HEATMAP_SVG,
    HEATMAP_TOPIC_COLORS,
    resolve_month_predom_csv,
)
from monthly_topics_utils import (  # noqa: E402
    sort_patients_for_heatmap,
    topic_month_columns,
)


def plot_monthly_topics_heatmap(
    ordered_df: pd.DataFrame,
    output_png: Path | None = None,
    output_svg: Path | None = None,
    topic_colors: dict[int, str] | None = None,
) -> plt.Figure:
    topic_cols = topic_month_columns(ordered_df)
    if not topic_cols:
        raise ValueError("No topic_predominante_mes* columns found in input data.")

    ordered_df_sorted = sort_patients_for_heatmap(ordered_df, topic_cols)

    plot_df_clean = (
        ordered_df_sorted[["id"] + topic_cols]
        .set_index("id")
        .replace({pd.NA: np.nan})
        .apply(pd.to_numeric, errors="coerce")
    )

    topic_colors = topic_colors or HEATMAP_TOPIC_COLORS
    color_list = [topic_colors[i] for i in sorted(topic_colors)]
    cmap = ListedColormap(color_list)

    fig, ax = plt.subplots(figsize=(12, len(plot_df_clean) * 0.15 + 1))

    sns.heatmap(
        plot_df_clean,
        cmap=cmap,
        vmin=0,
        vmax=len(color_list) - 1,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Predominant Topic"},
        mask=plot_df_clean.isna(),
        ax=ax,
    )

    for i, row in ordered_df_sorted.iterrows():
        if row["Event"] == 1:
            if pd.isna(row.get("t_evento_eb2")):
                continue
            mes_evento = int(row["t_evento_eb2"] // 30)
            if 0 <= mes_evento < len(topic_cols):
                ax.add_patch(
                    plt.Rectangle(
                        (mes_evento, i),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                        linestyle="-",
                    )
                )

    n_no_pd = len(ordered_df_sorted[ordered_df_sorted["Event"] == 0])
    ax.axhline(n_no_pd, color="black", linewidth=2.5)

    ax.set_title("Predominant topics per month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Patient ID")
    plt.tight_layout()

    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_png}")
    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, format="svg", bbox_inches="tight")
        print(f"Saved: {output_svg}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot monthly predominant LDA topics heatmap."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Monthly predominant topics table (id, Event, t_evento_eb2, mes*)",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURE_MONTHLY_TOPICS_HEATMAP,
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=FIGURE_MONTHLY_TOPICS_HEATMAP_SVG,
    )
    parser.add_argument("--no-svg", action="store_true")
    args = parser.parse_args()

    input_csv = args.input_csv or resolve_month_predom_csv()
    ordered_df = pd.read_csv(input_csv)
    ordered_df = ordered_df.dropna(subset=["topic_predominante_mes1"])

    fig = plot_monthly_topics_heatmap(
        ordered_df,
        output_png=args.output_png,
        output_svg=None if args.no_svg else args.output_svg,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
