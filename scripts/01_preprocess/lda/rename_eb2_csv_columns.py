#!/usr/bin/env python3
"""Rename eB2 CSV columns to English and overwrite the same file."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EB2_CSV = (
    PROJECT_ROOT / "data/processed/lda/PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv"
)

COLUMN_RENAME = {
    "Fecha_entrada_HDM": "date_start_hdm",
    "Evento PD": "pd_event",
    "Tiempo Obs Final": "obs_time_final",
    "diferencia": "time_difference",
    "Evento PD diff": "pd_event_diff",
    "early PD": "early_pd",
    "primer_registro_EB2": "first_eb2_record",
}


def main() -> None:
    df = pd.read_csv(EB2_CSV)
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns})
    df.to_csv(EB2_CSV, index=False)
    print(f"Saved {len(df)} rows -> {EB2_CSV}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
