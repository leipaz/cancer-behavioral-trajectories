"""Shared statistics helpers for univariate comparison plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from config import GRUPO_EPD, GRUPO_NO_EPD, LABEL_EPD, LABEL_NO_EPD


def filter_by_response(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["Respuesta"]).copy()


def compute_mann_whitney_p_values(
    df_first: pd.DataFrame,
    df_previous: pd.DataFrame,
    variables: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return p-values for first vs previous 15-day windows."""
    first = filter_by_response(df_first)
    previous = filter_by_response(df_previous)

    p_first: dict[str, float] = {}
    p_previous: dict[str, float] = {}

    for var in variables:
        first[var] = pd.to_numeric(first[var], errors="coerce")
        previous[var] = pd.to_numeric(previous[var], errors="coerce")

        epd_prev = previous.loc[previous["Respuesta"] == LABEL_EPD, var].dropna()
        no_epd_prev = previous.loc[previous["Respuesta"] == LABEL_NO_EPD, var].dropna()
        epd_first = first.loc[first["Respuesta"] == LABEL_EPD, var].dropna()
        no_epd_first = first.loc[first["Respuesta"] == LABEL_NO_EPD, var].dropna()

        if len(epd_prev) > 0 and len(no_epd_prev) > 0:
            _, p_previous[var] = mannwhitneyu(
                epd_prev, no_epd_prev, alternative="two-sided"
            )
        else:
            p_previous[var] = np.nan

        if len(epd_first) > 0 and len(no_epd_first) > 0:
            _, p_first[var] = mannwhitneyu(
                epd_first, no_epd_first, alternative="two-sided"
            )
        else:
            p_first[var] = np.nan

    return p_first, p_previous


def build_summary_table(
    df_first: pd.DataFrame,
    df_previous: pd.DataFrame,
    variables: list[str],
    p_first: dict[str, float],
    p_previous: dict[str, float],
    include_min_max: bool = True,
) -> pd.DataFrame:
    """Build summary statistics table (E.PD vs no-E.PD, both windows)."""
    rows: list[dict] = []

    for var in variables:
        for period_name, df_period, p_map in [
            ("First 15 days", df_first, p_first),
            ("Previous 15 days", df_previous, p_previous),
        ]:
            for group_label, group_value in [
                (LABEL_EPD, GRUPO_EPD),
                (LABEL_NO_EPD, GRUPO_NO_EPD),
            ]:
                values = pd.to_numeric(
                    df_period.loc[df_period["Early PD"] == group_value, var],
                    errors="coerce",
                ).dropna()

                if len(values) > 0:
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    row = {
                        "Variable": var,
                        "Period": period_name,
                        "Group": group_label,
                        "N": len(values),
                        "Mean": round(float(np.mean(values)), 3),
                        "Median [Q1–Q3]": (
                            f"{np.median(values):.3f} [{q1:.3f}–{q3:.3f}]"
                        ),
                        "Variance": round(float(np.var(values, ddof=1)), 3),
                        "p-value": p_map.get(var, np.nan),
                    }
                    if include_min_max:
                        row["Min"] = round(float(np.min(values)), 3)
                        row["Max"] = round(float(np.max(values)), 3)
                else:
                    row = {
                        "Variable": var,
                        "Period": period_name,
                        "Group": group_label,
                        "N": 0,
                        "Mean": np.nan,
                        "Median [Q1–Q3]": "NaN",
                        "Variance": np.nan,
                        "p-value": p_map.get(var, np.nan),
                    }
                    if include_min_max:
                        row["Min"] = np.nan
                        row["Max"] = np.nan

                rows.append(row)

    summary = pd.DataFrame(rows)
    base_cols = [
        "Variable",
        "Period",
        "Group",
        "N",
        "Mean",
        "Median [Q1–Q3]",
    ]
    if include_min_max:
        base_cols += ["Min", "Max"]
    base_cols += ["Variance", "p-value"]

    summary = summary[base_cols]
    summary["p-value"] = summary["p-value"].apply(
        lambda x: f"{x:.2e}" if pd.notnull(x) else "NaN"
    )
    return summary
