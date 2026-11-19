"""practiced_sport enters only when filtered walking does.

The flag is derived. A stored value, 0 or 1, counts only when activity_walking
survived the range filter of 120–15000 seconds. Walking outside that range is
already missing, and the sport flag on that day is missing too. A stored 0
with no raw walking and no raw running is missing for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vqvae.data import aggregate_duplicate_user_days, load_daily_summary

SPORT_RULE = "practiced_sport_counts_only_when_walking_survives_120_15000"
ACTIVITY_COLS = ("activity_walking", "activity_running")


def erroneous_sport_user_days(path: Path) -> pd.DataFrame:
    """User-days whose stored 0 has no raw walking and no raw running."""
    raw = pd.read_csv(
        path,
        usecols=["user", "date_time", "practiced_sport", *ACTIVITY_COLS],
        low_memory=False,
    )
    raw["user"] = pd.to_numeric(raw["user"], errors="coerce").astype("int64")
    raw["date_time"] = pd.to_datetime(raw["date_time"]).dt.normalize()
    for col in ("practiced_sport", *ACTIVITY_COLS):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    grouped = raw.groupby(["user", "date_time"], sort=False)
    sport = grouped["practiced_sport"].max()
    has_activity = False
    for col in ACTIVITY_COLS:
        has_activity = has_activity | (grouped[col].count() > 0)
    erroneous = (sport == 0) & ~has_activity
    return erroneous[erroneous].reset_index()[["user", "date_time"]]


def apply_erroneous_sport_missing(daily: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Set those stored zeros to NaN. Other cells are unchanged."""
    out = daily.copy()
    out["user"] = pd.to_numeric(out["user"], errors="coerce").astype("int64")
    out["date_time"] = pd.to_datetime(out["date_time"]).dt.normalize()
    keys = erroneous_sport_user_days(path)
    keys["erroneous_sport_zero"] = True
    marked = out.merge(keys, on=["user", "date_time"], how="left")
    if len(marked) != len(out):
        raise RuntimeError("sport-zero merge changed the number of daily rows")
    sport = pd.to_numeric(marked["practiced_sport"], errors="coerce").to_numpy(dtype=float)
    hit = marked["erroneous_sport_zero"].eq(True).to_numpy() & (sport == 0)
    sport[hit] = np.nan
    out["practiced_sport"] = sport
    return out


def blank_sport_unless_filtered_walking(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop sport where activity_walking did not survive [120, 15000] s."""
    out = daily.copy()
    walking = pd.to_numeric(out["activity_walking"], errors="coerce").to_numpy(dtype=float)
    sport = pd.to_numeric(out["practiced_sport"], errors="coerce").to_numpy(dtype=float)
    sport[np.isnan(walking)] = np.nan
    out["practiced_sport"] = sport
    return out


def load_daily_for_missingness(path: Path) -> pd.DataFrame:
    """Daily table after the activity filter, with sport gated on that walking."""
    daily = aggregate_duplicate_user_days(load_daily_summary(path))
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    daily = apply_erroneous_sport_missing(daily, path)
    return blank_sport_unless_filtered_walking(daily)


ALARM_SPORT_RULE = "practiced_sport_walking_gate_but_stored_ones_kept"


def load_daily_for_alarm(path: Path) -> pd.DataFrame:
    """Daily table for anything that feeds the frozen alarm.

    Same as load_daily_for_missingness, except that a stored 1 is never blanked.
    A stored 0 without filtered walking is missing, and a0 reads it as 0 again,
    so the VQ-VAE input, and the baseline alarm, match the frozen pipeline.
    """
    daily = aggregate_duplicate_user_days(load_daily_summary(path))
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    daily = apply_erroneous_sport_missing(daily, path)
    stored = pd.to_numeric(daily["practiced_sport"], errors="coerce").to_numpy(dtype=float)
    out = blank_sport_unless_filtered_walking(daily)
    sport = out["practiced_sport"].to_numpy(dtype=float)
    sport[stored == 1] = 1.0
    out["practiced_sport"] = sport
    return out
