#!/usr/bin/env python3
"""Plot LDA topic summaries vs clinical event (PD vs no-PD)."""

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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_NUM_TOPICS,
    FIGURE_AVG_TOPIC_BY_EVENT,
    FIGURE_TOPICS_BY_EVENT,
    MERGED_TOPICS_CLINICAL_CSV,
)


def plot_topics_by_event(
    df: pd.DataFrame,
    event_col: str = "PD_event",
    n_topic: int = 6,
    npatients: int | None = None,
    output_path: Path | None = None,
) -> None:
    df = df.copy()
    df[event_col] = df[event_col].astype(float)
    df_event_0 = df[df[event_col] == 0.0]
    df_event_1 = df[df[event_col] == 1.0]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.countplot(
        data=df_event_0, x="predominant_topic", hue="predominant_topic",
        ax=axs[0], palette="Blues_d", legend=False,
    )
    axs[0].set_title(f"Predominant Topic: no-PD (n={len(df_event_0)})", fontweight="bold")
    axs[0].set_xlabel("Topic Index")
    axs[0].set_ylabel("Number of Patients")
    axs[0].grid(axis="y", linestyle="--", alpha=0.6)

    sns.countplot(
        data=df_event_1, x="predominant_topic", hue="predominant_topic",
        ax=axs[1], palette="Reds_d", legend=False,
    )
    axs[1].set_title(f"Predominant Topic: PD (n={len(df_event_1)})", fontweight="bold")
    axs[1].set_xlabel("Topic Index")
    axs[1].set_ylabel("")
    axs[1].grid(axis="y", linestyle="--", alpha=0.6)

    title = f"Comparison of Predominant Topics by {event_col}"
    if npatients is not None:
        title += f"\n({npatients} patients, {n_topic} topics)"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def plot_avg_topic_distribution_by_event(
    df: pd.DataFrame,
    topic_columns: list[str],
    group_col: str = "PD_event",
    n_topic: int = 6,
    npatients: int | None = None,
    output_path: Path | None = None,
) -> None:
    df_avg = df.groupby(group_col)[topic_columns].mean()
    categories = sorted(df_avg.index.unique())
    x_pos = np.arange(len(topic_columns))
    bar_width = 0.35
    colors = ["skyblue", "salmon", "lightgreen", "orange"]

    plt.figure(figsize=(12, 6))
    for i, cat in enumerate(categories):
        offset = -bar_width / 2 if i == 0 else bar_width / 2
        if cat in [0, 1, 0.0, 1.0]:
            label = "no-PD" if cat in (0, 0.0) else "PD"
        else:
            label = f"Group {cat}"
        plt.bar(
            x_pos + offset,
            df_avg.loc[cat],
            width=bar_width,
            label=label,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Topics")
    plt.ylabel("Average Probability")
    title = f"Average Topic Distribution by {group_col} ({n_topic} topics)"
    if npatients is not None:
        title += f" (n={npatients} patients)"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(x_pos, topic_columns, rotation=45)
    plt.legend(title=group_col)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LDA vs clinical event.")
    parser.add_argument("--merged-csv", type=Path, default=MERGED_TOPICS_CLINICAL_CSV)
    parser.add_argument("--event-col", default="PD_event")
    parser.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS)
    parser.add_argument("--topics-by-event-out", type=Path, default=FIGURE_TOPICS_BY_EVENT)
    parser.add_argument("--avg-topic-out", type=Path, default=FIGURE_AVG_TOPIC_BY_EVENT)
    args = parser.parse_args()

    df = pd.read_csv(args.merged_csv)
    topic_columns = [f"topic_{i}" for i in range(args.num_topics)]
    n_patients = df["id"].nunique() if "id" in df.columns else len(df)

    plot_topics_by_event(
        df,
        event_col=args.event_col,
        n_topic=args.num_topics,
        npatients=n_patients,
        output_path=args.topics_by_event_out,
    )
    plot_avg_topic_distribution_by_event(
        df,
        topic_columns=topic_columns,
        group_col=args.event_col,
        n_topic=args.num_topics,
        npatients=n_patients,
        output_path=args.avg_topic_out,
    )


if __name__ == "__main__":
    main()
