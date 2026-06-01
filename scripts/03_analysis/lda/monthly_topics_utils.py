"""Helpers for monthly predominant LDA topics (notebook 6_1)."""

from __future__ import annotations

import ast
from typing import List, Tuple

import pandas as pd
from gensim.models import LdaModel


def parse_embedding_ids(value) -> list:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def get_topic_distributions_and_predominant_topics_relaxedwindow(
    df: pd.DataFrame,
    patient_id,
    lda_model: LdaModel,
    dictionary,
    window_size: int = 30,
    min_embeddings: int = 15,
    verbose: bool = False,
) -> Tuple[List[List[float]], List[int]]:
    """
    Sliding 30-day windows; incomplete final block uses last ``window_size`` days.
    Patients with fewer than ``min_embeddings`` days are skipped.
    """
    row = df[df["id"] == patient_id]
    if row.empty:
        if verbose:
            print(f"[-] ID {patient_id}: not found in dataframe.")
        return [], []

    embedding_ids = parse_embedding_ids(row.iloc[0]["embedding_ids"])
    n_embeddings = len(embedding_ids)

    if n_embeddings < min_embeddings:
        if verbose:
            print(
                f"[-] ID {patient_id}: skipped ({n_embeddings} embeddings, "
                f"min {min_embeddings})."
            )
        return [], []

    topic_distributions: List[List[float]] = []
    predominant_topics: List[int] = []

    if min_embeddings <= n_embeddings < window_size:
        window = embedding_ids
        bow = dictionary.doc2bow([str(tok) for tok in window])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        topic_distributions.append(probs)
        predominant_topics.append(max(topics, key=lambda x: x[1])[0])
        return topic_distributions, predominant_topics

    for start in range(0, n_embeddings, window_size):
        end = start + window_size
        if end > n_embeddings:
            window = embedding_ids[-window_size:]
        else:
            window = embedding_ids[start:end]

        bow = dictionary.doc2bow([str(tok) for tok in window])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        topic_distributions.append(probs)
        predominant_topics.append(max(topics, key=lambda x: x[1])[0])

        if end > n_embeddings:
            break

    return topic_distributions, predominant_topics


def build_monthly_predominant_table(
    df_day: pd.DataFrame,
    lda_model: LdaModel,
    dictionary,
    window_size: int = 30,
    min_embeddings: int = 15,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per patient with ``topic_predominante_mesN`` columns."""
    patient_ids = df_day["id"].unique()
    topic_blocks_by_patient: dict = {}
    max_blocks = 0

    for pid in patient_ids:
        _, predominant_topics = (
            get_topic_distributions_and_predominant_topics_relaxedwindow(
                df_day,
                pid,
                lda_model,
                dictionary,
                window_size=window_size,
                min_embeddings=min_embeddings,
                verbose=verbose,
            )
        )
        topic_blocks_by_patient[pid] = predominant_topics
        max_blocks = max(max_blocks, len(predominant_topics))

    rows = []
    for pid in patient_ids:
        base_info = (
            df_day.loc[df_day["id"] == pid, ["pd_event", "pd_event_diff"]]
            .iloc[0]
            .to_dict()
        )
        row = {"id": pid, **base_info}
        topics = topic_blocks_by_patient[pid]
        for i in range(max_blocks):
            colname = f"topic_predominante_mes{i + 1}"
            row[colname] = topics[i] if i < len(topics) else pd.NA
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.loc[result_df["pd_event"] == 0, "pd_event_diff"] = pd.NA
    result_df = result_df.rename(
        columns={"pd_event": "Event", "pd_event_diff": "t_evento_eb2"}
    )

    df_no_event = result_df[result_df["Event"] == 0]
    df_with_event = result_df[result_df["Event"] == 1]
    ordered_df = pd.concat([df_no_event, df_with_event], ignore_index=True)
    return ordered_df.dropna(subset=["topic_predominante_mes1"])


def get_topic_probabilities_daily_windows(
    df: pd.DataFrame,
    patient_id,
    lda_model: LdaModel,
    dictionary,
    window_size: int = 30,
    min_embeddings: int = 30,
    step: int = 1,
) -> pd.DataFrame:
    """
    Sliding ``window_size``-day windows advancing ``step`` days (default: daily).
    Returns one row per window with topic_0..topic_{K-1} probabilities.
    """
    row = df[df["id"] == patient_id]
    if row.empty:
        return pd.DataFrame()

    embedding_ids = parse_embedding_ids(row.iloc[0]["embedding_ids"])
    n_embeddings = len(embedding_ids)
    num_topics = lda_model.num_topics

    if n_embeddings < min_embeddings:
        return pd.DataFrame()

    rows: list[dict] = []

    if n_embeddings < window_size:
        windows = [(0, embedding_ids)]
    else:
        windows = [
            (start, embedding_ids[start : start + window_size])
            for start in range(0, n_embeddings - window_size + 1, step)
        ]

    for start, window in windows:
        bow = dictionary.doc2bow([str(tok) for tok in window])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = {i: 0.0 for i in range(num_topics)}
        for topic_id, prob in topics:
            probs[topic_id] = prob
        record = {f"topic_{i}": probs[i] for i in range(num_topics)}
        record["time_window_day"] = start
        rows.append(record)

    return pd.DataFrame(rows)


def topic_month_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("topic_predominante_mes")]


def sort_patients_for_heatmap(df: pd.DataFrame, topic_cols: list[str]) -> pd.DataFrame:
    """Event 0 on top; within each group, more available months first."""
    temp_df = df.copy()
    temp_df["available_months"] = temp_df[topic_cols].notna().sum(axis=1)
    return (
        temp_df.dropna(subset=topic_cols, how="all")
        .sort_values(by=["Event", "available_months"], ascending=[True, False])
        .reset_index(drop=True)
    )
