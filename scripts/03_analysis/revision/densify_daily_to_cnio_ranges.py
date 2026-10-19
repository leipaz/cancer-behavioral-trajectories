#!/usr/bin/env python3
"""Expand daily model-input to full CNIO calendar per patient (gap days = NaN).

VQ-VAE ``build_selected_test_sequences`` only densifies min→max *observed*.
This script first inserts all days in [Inicio_datos_DB1, Fin_datos_DB1] so that
zero-imputed VQ-VAE codes cover the official CNIO window (same idea as the old
Obs_time-length sequences).

Writes:
  data/revision/oncology_daily_summary_model_input_rangos_cnio_filled.csv
  data/revision/densify_cnio_fill_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REV = PROJECT_ROOT / "data/revision"

DEFAULT_DAILY = REV / "oncology_daily_summary_model_input_rangos.csv"
DEFAULT_RANGOS = REV / "rangos_fechas.xlsx"
DEFAULT_OUT = REV / "oncology_daily_summary_model_input_rangos_cnio_filled.csv"
DEFAULT_SUMMARY = REV / "densify_cnio_fill_summary.csv"


def densify_to_cnio(
    daily: pd.DataFrame,
    rangos: pd.DataFrame,
    include_users_without_daily: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.copy()
    daily["user"] = pd.to_numeric(daily["user"], errors="coerce")
    daily["date_time"] = pd.to_datetime(daily["date_time"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["user", "date_time"]).drop_duplicates(
        ["user", "date_time"], keep="first"
    )

    rangos = rangos.rename(
        columns={
            "S_RECORD_id": "user",
            "Inicio_datos_DB1": "inicio_cnio",
            "Fin_datos_DB1": "fin_cnio",
        }
    )
    rangos["user"] = pd.to_numeric(rangos["user"], errors="coerce")
    rangos["inicio_cnio"] = pd.to_datetime(rangos["inicio_cnio"], errors="coerce").dt.normalize()
    rangos["fin_cnio"] = pd.to_datetime(rangos["fin_cnio"], errors="coerce").dt.normalize()
    rangos = rangos.dropna(subset=["user", "inicio_cnio", "fin_cnio"]).drop_duplicates("user")

    feature_cols = [c for c in daily.columns if c not in ("user", "date_time")]
    users_daily = set(daily["user"].astype(int))
    users_rangos = set(rangos["user"].astype(int))
    if include_users_without_daily:
        target_users = sorted(users_rangos)
    else:
        target_users = sorted(users_rangos & users_daily)

    pieces: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for uid in target_users:
        rr = rangos.loc[rangos["user"] == uid].iloc[0]
        ini, fin = rr["inicio_cnio"], rr["fin_cnio"]
        if fin < ini:
            continue
        full = pd.date_range(ini, fin, freq="D")
        expected = len(full)

        sub = daily.loc[daily["user"] == uid].set_index("date_time")
        n_before = len(sub)
        # keep observed days even if slightly outside CNIO? clip to CNIO for model window
        sub = sub.reindex(full)
        sub["user"] = uid
        # weekend always from calendar (same as VQ-VAE densify)
        if "weekend" in sub.columns or "weekend" in feature_cols:
            sub["weekend"] = (sub.index.dayofweek >= 5).astype(np.uint8)
        out = sub.reset_index().rename(columns={"index": "date_time"})
        # ensure column order
        cols = ["user", "date_time"] + [c for c in feature_cols if c in out.columns]
        # add any missing feature cols as NaN
        for c in feature_cols:
            if c not in out.columns:
                out[c] = np.nan
        out = out[["user", "date_time"] + feature_cols]
        n_after = len(out)
        n_obs = int(out[feature_cols].notna().any(axis=1).sum()) if feature_cols else 0
        # more precise: a day is observed if any non-weekend feature is present
        feat_no_wk = [c for c in feature_cols if c != "weekend"]
        if feat_no_wk:
            n_obs = int(out[feat_no_wk].notna().any(axis=1).sum())
        n_filled = n_after - n_obs

        pieces.append(out)
        summary_rows.append(
            {
                "user": int(uid),
                "inicio_cnio": ini.date().isoformat(),
                "fin_cnio": fin.date().isoformat(),
                "dias_cnio_esperados": expected,
                "dias_observados_en_cnio": n_obs,
                "dias_rellenados_nan": n_filled,
                "dias_en_daily_antes": n_before,
                "pct_observado": round(100.0 * n_obs / expected, 2) if expected else np.nan,
                "tenia_daily": int(uid) in users_daily,
            }
        )

    filled = pd.concat(pieces, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["dias_rellenados_nan", "user"], ascending=[False, True]
    )
    return filled, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--rangos", type=Path, default=DEFAULT_RANGOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--only-with-daily",
        action="store_true",
        help="Skip CNIO users with zero rows in daily (default: include them as all-NaN).",
    )
    args = parser.parse_args()

    daily = pd.read_csv(args.daily, low_memory=False)
    rangos = pd.read_excel(args.rangos)
    filled, summary = densify_to_cnio(
        daily, rangos, include_users_without_daily=not args.only_with_daily
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(args.out, index=False)
    summary.to_csv(args.summary, index=False)

    print(f"Input daily rows:     {len(daily):,}")
    print(f"Output filled rows:   {len(filled):,}")
    print(f"Users:                {filled['user'].nunique()}")
    print(f"Days rellenados NaN:  {int(summary['dias_rellenados_nan'].sum()):,}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")
    print("\nTop users by days filled:")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
