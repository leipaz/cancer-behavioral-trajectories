#!/usr/bin/env python3
"""Merge LDA topic assignments with clinical covariates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    CLINICAL_COLS,
    MERGED_TOPICS_CLINICAL_CSV,
    PATIENT_TOPICS_CLINICAL_CSV,
    USER_EMBEDDINGS_CSV,
    USER_TOPIC_CLUSTER_CSV,
    resolve_clinical_xlsx,
)
from lda_utils import harmonize_id_column, parse_embedding_ids  # noqa: E402


def merge_patient_clinical(
    result_df: pd.DataFrame,
    df_profiles: pd.DataFrame,
    bbdd_clinica: pd.DataFrame,
) -> pd.DataFrame:
    result_df = harmonize_id_column(result_df)
    df_profiles = harmonize_id_column(df_profiles)
    bbdd_clinica = harmonize_id_column(bbdd_clinica)

    df_profiles = df_profiles.copy()
    df_profiles["embedding_ids"] = df_profiles["embedding_ids"].apply(parse_embedding_ids)

    df_master = pd.merge(
        result_df,
        df_profiles[["id", "embedding_ids"]],
        on="id",
        how="inner",
    )

    cols_validas = [c for c in CLINICAL_COLS if c in bbdd_clinica.columns]
    df_final = pd.merge(df_master, bbdd_clinica[cols_validas], on="id", how="inner")
    return df_final


def merge_topics_with_clinical_wide(
    user_topics: pd.DataFrame,
    bbdd_clinica: pd.DataFrame,
) -> pd.DataFrame:
    """Notebook merge: full topic columns + full clinical table on id."""
    df_day = harmonize_id_column(user_topics.copy())
    df_res = harmonize_id_column(bbdd_clinica.copy())
    return pd.merge(df_day, df_res, on="id", how="inner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LDA outputs with clinical data.")
    parser.add_argument("--user-topics", type=Path, default=USER_TOPIC_CLUSTER_CSV)
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--clinical-xlsx", type=Path, default=None)
    parser.add_argument("--patient-out", type=Path, default=PATIENT_TOPICS_CLINICAL_CSV)
    parser.add_argument("--merged-out", type=Path, default=MERGED_TOPICS_CLINICAL_CSV)
    args = parser.parse_args()

    clinical_path = args.clinical_xlsx or resolve_clinical_xlsx()
    bbdd_clinica = pd.read_excel(clinical_path)
    bbdd_clinica.columns = bbdd_clinica.columns.str.strip()

    user_topics = pd.read_csv(args.user_topics)
    df_profiles = pd.read_csv(args.profiles_csv)

    patient_df = merge_patient_clinical(user_topics, df_profiles, bbdd_clinica)
    args.patient_out.parent.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(args.patient_out, index=False)
    print(f"Saved patient-level merge: {args.patient_out} ({len(patient_df)} rows)")

    merged_df = merge_topics_with_clinical_wide(user_topics, bbdd_clinica)
    merged_df.to_csv(args.merged_out, index=False)
    print(f"Saved wide merge: {args.merged_out} ({len(merged_df)} rows, {merged_df['id'].nunique()} ids)")

    if "PD_event" in patient_df.columns:
        print(patient_df["PD_event"].value_counts())


if __name__ == "__main__":
    main()
