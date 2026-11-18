"""Corpus composition and HDO identifier overlap for the pretraining provenance.

Counts participants in user_splits and days in the May 2026 eB2 extract.
An observed day has at least one of the eight continuous variables inside the
frozen clip. The vqvae column is reported as stored. It is not treated as
evidence that a participant was left out of VQ-VAE training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.vqvae import constants as c
from scripts.vqvae.data import replace_out_of_bounds_with_nan
from scripts.vqvae.run_external_pretraining_adequacy import (
    HDO_CSV,
    MENTAL_CSV,
    SPLITS_CSV,
    program_of_service,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "vq-vae" / "training_provenance"
CONTINUOUS = [
    "sleep_start",
    "location_distance",
    "location_time_home",
    "sleep_duration",
    "activity_walking",
    "app_usage_total",
    "location_clusters_count",
    "steps_steps_total",
]


def day_counts(path: Path) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(path, usecols=lambda col: col in set(c.COLS), chunksize=250_000, low_memory=False):
        chunk = replace_out_of_bounds_with_nan(chunk)
        chunk["user"] = chunk["user"].astype(int)
        chunk["day"] = pd.to_datetime(chunk["date_time"]).dt.normalize()
        chunk["observed"] = chunk[CONTINUOUS].notna().any(axis=1)
        grouped = chunk.groupby(["user", "day"], sort=False).agg(
            raw_rows=("day", "size"),
            observed=("observed", "max"),
        )
        pieces.append(grouped)
    counts = pd.concat(pieces)
    counts = counts.groupby(level=[0, 1]).agg(raw_rows=("raw_rows", "sum"), observed=("observed", "max"))
    by_user = counts.reset_index(level=1).groupby(level=0).agg(
        raw_rows=("raw_rows", "sum"),
        observed_days=("observed", "sum"),
        first_day=("day", "min"),
        last_day=("day", "max"),
    )
    return by_user


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS_CSV)
    splits["user"] = splits["user"].astype(int)
    splits["program"] = splits["service"].map(program_of_service)
    print("Counting eB2 days", flush=True)
    mental = day_counts(MENTAL_CSV)
    print("Counting HDO days", flush=True)
    hdo = day_counts(HDO_CSV)
    hdo_ids = set(hdo.index.astype(int))

    joined = splits.merge(mental, left_on="user", right_index=True, how="left")
    joined["in_daily"] = joined["raw_rows"].notna()
    joined["hdo_id_overlap"] = joined["user"].isin(hdo_ids)
    joined.to_csv(OUT / "extract_participants.csv", index=False)

    rows = []
    for (program, service), sub in joined.groupby(["program", "service"], sort=True):
        rows.append(
            {
                "program": program,
                "service": service,
                "participants": int(len(sub)),
                "vqvae_1": int(sub["vqvae"].eq(1).sum()),
                "vqvae_0": int(sub["vqvae"].eq(0).sum()),
                "in_daily": int(sub["in_daily"].sum()),
                "raw_rows": int(sub["raw_rows"].fillna(0).sum()),
                "observed_days": int(sub["observed_days"].fillna(0).sum()),
                "first_day": sub["first_day"].min(),
                "last_day": sub["last_day"].max(),
                "ids_also_in_hdo": int(sub["hdo_id_overlap"].sum()),
            }
        )
    by_service = pd.DataFrame(rows)
    by_service.to_csv(OUT / "composition_by_service.csv", index=False)

    program = (
        by_service.groupby("program", sort=True)
        .agg(
            participants=("participants", "sum"),
            vqvae_1=("vqvae_1", "sum"),
            vqvae_0=("vqvae_0", "sum"),
            in_daily=("in_daily", "sum"),
            raw_rows=("raw_rows", "sum"),
            observed_days=("observed_days", "sum"),
            first_day=("first_day", "min"),
            last_day=("last_day", "max"),
            ids_also_in_hdo=("ids_also_in_hdo", "sum"),
        )
        .reset_index()
    )
    program.to_csv(OUT / "composition_by_program.csv", index=False)

    hdo_summary = {
        "participants": int(hdo.shape[0]),
        "raw_rows": int(hdo["raw_rows"].sum()),
        "observed_days": int(hdo["observed_days"].sum()),
        "first_day": str(pd.Timestamp(hdo["first_day"].min()).date()),
        "last_day": str(pd.Timestamp(hdo["last_day"].max()).date()),
        "ids_in_user_splits": int(splits["user"].isin(hdo_ids).sum()),
        "ids_in_eb2_daily": int(len(hdo_ids & set(mental.index.astype(int)))),
    }
    (OUT / "hdo_overlap.json").write_text(json.dumps(hdo_summary, indent=2), encoding="utf-8")
    print(program.to_string(index=False))
    print(hdo_summary)


if __name__ == "__main__":
    main()
