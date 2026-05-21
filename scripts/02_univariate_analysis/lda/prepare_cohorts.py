#!/usr/bin/env python3
"""
Load daily summaries and matching table, QC ID overlap, build 15-day window cohorts.
"""

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
    DEFAULT_DAILY_ENRICHED,
    DEFAULT_DAILY_FILTERED,
    FECHA_COL_EB2,
    ID_COL,
    RESPONSE_MAP,
    WINDOW_DATE_COLS,
    columns_present,
    resolve_matching_path,
)


def harmonize_ids(
    df_daily: pd.DataFrame, df_matching: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_daily = df_daily.rename(columns={"user": ID_COL}, errors="ignore")
    df_matching = df_matching.rename(columns={"S_RECORD_id": ID_COL}, errors="ignore")

    df_daily[ID_COL] = pd.to_numeric(df_daily[ID_COL], errors="coerce").astype("Int64")
    df_matching[ID_COL] = pd.to_numeric(df_matching[ID_COL], errors="coerce").astype(
        "Int64"
    )
    return df_daily, df_matching


def report_id_overlap(df_daily: pd.DataFrame, df_matching: pd.DataFrame) -> list:
    ids_daily = set(df_daily[ID_COL].dropna().unique())
    ids_matching = set(df_matching[ID_COL].dropna().unique())
    common = sorted(ids_daily.intersection(ids_matching))

    print(f"Total unique IDs in daily summary: {len(ids_daily)}")
    print(f"Total unique IDs in matching table: {len(ids_matching)}")
    print(f"Total common IDs (intersection): {len(common)}")
    return common


def build_window_cohorts(
    df_daily: pd.DataFrame,
    df_matching: pd.DataFrame,
    sel_cols: list[str],
    path_v1: Path,
    path_v2: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_daily = df_daily.copy()
    if "user" in df_daily.columns:
        df_daily = df_daily.rename(columns={"user": ID_COL})
    if "S_RECORD_id" in df_matching.columns:
        df_matching = df_matching.rename(columns={"S_RECORD_id": ID_COL})

    df_daily[FECHA_COL_EB2] = pd.to_datetime(df_daily[FECHA_COL_EB2], errors="coerce")
    for col in WINDOW_DATE_COLS:
        df_matching[col] = pd.to_datetime(df_matching[col], errors="coerce")

    merged = pd.merge(
        df_daily,
        df_matching[["ID", "match_set_id", "Early PD"] + WINDOW_DATE_COLS],
        on=ID_COL,
        how="inner",
    )
    merged["Respuesta"] = merged["Early PD"].map(RESPONSE_MAP)

    mask_v1 = (merged[FECHA_COL_EB2] >= merged["Fecha_entrada_v1"]) & (
        merged[FECHA_COL_EB2] <= merged["Fecha_final_simulada_v1"]
    )
    mask_v2 = (merged[FECHA_COL_EB2] >= merged["Fecha_entrada_v2"]) & (
        merged[FECHA_COL_EB2] <= merged["Fecha_final_simulada_v2"]
    )

    base_cols = ["ID", FECHA_COL_EB2, "Early PD", "Respuesta"]
    df_first = (
        merged.loc[mask_v1, base_cols + sel_cols]
        .drop_duplicates()
        .sort_values(by=["ID", FECHA_COL_EB2])
        .copy()
    )
    df_previous = (
        merged.loc[mask_v2, base_cols + ["match_set_id"] + sel_cols]
        .drop_duplicates()
        .sort_values(by=["ID", FECHA_COL_EB2])
        .copy()
    )

    path_v1.parent.mkdir(parents=True, exist_ok=True)
    df_first.to_csv(path_v1, index=False)
    df_previous.to_csv(path_v2, index=False)

    print(f"Window 1 cohort: {df_first[ID_COL].nunique()} unique IDs -> {path_v1}")
    print(f"Window 2 cohort: {df_previous[ID_COL].nunique()} unique IDs -> {path_v2}")
    return df_first, df_previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare univariate analysis cohorts.")
    parser.add_argument("--daily-filtered", type=Path, default=DEFAULT_DAILY_FILTERED)
    parser.add_argument("--daily-enriched", type=Path, default=DEFAULT_DAILY_ENRICHED)
    parser.add_argument("--matching", type=Path, default=None)
    parser.add_argument("--cohort-v1-out", type=Path, default=COHORT_WINDOW1)
    parser.add_argument("--cohort-v2-out", type=Path, default=COHORT_WINDOW2)
    args = parser.parse_args()

    matching_path = args.matching or resolve_matching_path()

    df_filtered = pd.read_csv(args.daily_filtered, low_memory=False)
    df_enriched = pd.read_csv(args.daily_enriched, low_memory=False)
    df_matching = pd.read_csv(matching_path)

    df_filtered, df_matching = harmonize_ids(df_filtered, df_matching)
    report_id_overlap(df_filtered, df_matching)

    sel_cols = columns_present(df_enriched.columns)
    print(f"Using {len(sel_cols)} metric columns present in enriched daily summary.")

    build_window_cohorts(
        df_enriched,
        df_matching,
        sel_cols,
        args.cohort_v1_out,
        args.cohort_v2_out,
    )


if __name__ == "__main__":
    main()
