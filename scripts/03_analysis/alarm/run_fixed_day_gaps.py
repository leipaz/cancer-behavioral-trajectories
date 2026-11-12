#!/usr/bin/env python3
"""Contiguous gaps of 3, 7 and 14 days on the existing DB1 missingness run.

Does not change the 10–50% ramp. Masks walk all 175 patients. Alarm metrics
use the same 1013 objetivo decisions and the stored unmasked reference.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parents[2]))
sys.path.insert(0, str(_SCRIPT_DIR))

import run_missingness_ramp as ramp  # noqa: E402
import run_sensitivity_db1_objetivo as base  # noqa: E402
from gensim.models import LdaModel  # noqa: E402
from vqvae import constants as c  # noqa: E402
from vqvae.data import aggregate_duplicate_user_days, load_daily_summary, load_scaler_params  # noqa: E402
from vqvae.inference import load_models  # noqa: E402

ROOT = ramp.ROOT
PRIOR = ROOT / "results/missingness/ramp_db1_175"
OUT = ROOT / "results/missingness/huecos_3_7_14"
LENGTHS = (3, 7, 14)
SEEDS = ramp.SEEDS


def mask_fixed_gap(frame: pd.DataFrame, length: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide one block of exactly ``length`` calendar days, or the whole span if shorter."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    placements = []
    for uid, g in out.groupby("user", sort=True):
        dates = pd.to_datetime(g["date_time"]).dt.normalize()
        start, end = dates.min(), dates.max()
        n_cal = int((end - start).days + 1)
        use = min(n_cal, int(length))
        offset = int(rng.integers(0, n_cal - use + 1))
        gap_start = start + pd.Timedelta(days=offset)
        gap_end = gap_start + pd.Timedelta(days=use - 1)
        hit = g.index[(dates >= gap_start).to_numpy() & (dates <= gap_end).to_numpy()]
        out.loc[hit, ramp.MEASURE_COLS] = np.nan
        placements.append(
            {
                "id": int(uid),
                "requested_days": int(length),
                "gap_calendar_days": use,
                "span_calendar_days": n_cal,
                "shorter_than_requested": int(n_cal < length),
                "gap_start": str(gap_start.date()),
                "gap_end": str(gap_end.date()),
            }
        )
    return out, pd.DataFrame(placements)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "landmarks").mkdir(parents=True, exist_ok=True)
    points = base.landmarks_of()
    reference = pd.read_csv(PRIOR / "landmarks" / "reference.csv")
    if len(reference) != 1013 or reference["load"].isna().any():
        raise SystemExit("the stored reference does not score all 1013 decisions")
    ref_counts = np.load(PRIOR / "reference_code_counts.npy")
    ref_topic = np.load(PRIOR / "reference_topic_mean.npy")

    daily = aggregate_duplicate_user_days(load_daily_summary(ramp.DAILY))
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    if int(daily["user"].nunique()) != 175:
        raise SystemExit(f"expected 175 DB1 patients, found {daily['user'].nunique()}")
    base_observed = ramp.observed_dates(daily)

    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")["a0"]
    scaler = load_scaler_params(c.DEFAULT_SCALER)
    lda = LdaModel.load(str(ramp.LDA_PATH))
    dictionary = ramp.Dictionary_load()

    rows = []
    for length in LENGTHS:
        for seed in SEEDS:
            name = f"gap_{length}d_s{seed}"
            dest = OUT / "landmarks" / f"{name}.csv"
            started = time.time()
            if dest.exists():
                scored = pd.read_csv(dest)
                print("skip", name, flush=True)
            else:
                masked, placements = mask_fixed_gap(daily, length, seed)
                placements.to_csv(OUT / f"{name}_placement.csv", index=False)
                encoded = ramp.encode(masked, model, scaler)
                scored = ramp.score_landmarks(encoded, ramp.observed_dates(masked), points, lda, dictionary)
                if len(scored) != 1013 or int(scored.emitted.sum()) != 1013 or scored["load"].isna().any():
                    raise SystemExit(f"{name}: {int(scored.emitted.sum())} loads of {len(scored)}")
                scored.to_csv(dest, index=False)
                np.save(OUT / f"{name}_code_counts.npy", ramp.code_counts(encoded, base_observed))
                np.save(OUT / f"{name}_topic_mean.npy", base.topic_mean_last_block(encoded, points, lda, dictionary))
                ramp.missingness_table(masked).to_csv(OUT / f"{name}_missingness.csv", index=False)
                print(f"{name} {time.time() - started:.1f}s shorter={int(placements.shorter_than_requested.sum())}", flush=True)
            scored = pd.read_csv(dest)
            miss = pd.read_csv(OUT / f"{name}_missingness.csv")
            counts = np.load(OUT / f"{name}_code_counts.npy")
            topics = np.load(OUT / f"{name}_topic_mean.npy")
            summary = ramp.summarize(name, "fixed_gap", float(length), seed, scored, reference, counts, ref_counts, topics, ref_topic, miss)
            summary["topic_cosine"] = base.dominant_cosine(scored, reference)
            summary["gap_days"] = length
            rows.append(summary)
            print(
                f"  {name} JS={summary['topic_js']:.3f} dominant={summary['dominant_agreement']:.3f} "
                f"alarm={summary['alarm_agreement']:.3f} AUROC={summary['emitted_AUROC']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(OUT / "metricas_por_semilla.csv", index=False)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
