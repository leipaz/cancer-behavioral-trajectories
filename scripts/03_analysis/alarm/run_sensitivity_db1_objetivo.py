#!/usr/bin/env python3
"""Missingness ramp on all DB1 patients, alarm on the 1013 objetivo decisions.

Masks walk every DB1 patient. A decision past the encoded profile keeps its
day and uses the last 30 profile days, the collapsed-month overlap. Every
objetivo decision is required to receive a load.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parents[2]))
sys.path.insert(0, str(_SCRIPT_DIR))

import run_missingness_ramp as ramp  # noqa: E402
from gensim.models import LdaModel  # noqa: E402
from vqvae import constants as c  # noqa: E402
from vqvae.data import aggregate_duplicate_user_days, load_daily_summary, load_scaler_params  # noqa: E402
from vqvae.inference import load_models  # noqa: E402

ROOT = ramp.ROOT
OUT = ROOT / "results/missingness/ramp_db1_175"
OBJETIVO = ROOT / "results/lda/val_2/anidada_estratificada/objetivo.csv"
RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
SEEDS = ramp.SEEDS


def landmarks_of() -> pd.DataFrame:
    frame = pd.read_csv(OBJETIVO)
    frame["id"] = frame["id"].astype(int)
    frame["study_day"] = frame["decision_study_day"].astype(int)
    frame["y"] = frame["outcome_120d"].astype(int)
    if len(frame) != 1013 or frame["id"].nunique() != 114:
        raise SystemExit(f"objetivo is {len(frame)} rows, {frame.id.nunique()} patients")
    return frame[["id", "study_day", "y"]]


def fixed_calendar_variable_missing(frame: pd.DataFrame) -> float:
    rates = []
    for _, g in frame.groupby("user", sort=True):
        dates = pd.to_datetime(g["date_time"]).dt.normalize()
        calendar = pd.date_range(dates.min(), dates.max(), freq="D")
        indexed = g.copy()
        indexed["date_time"] = dates.to_numpy()
        indexed = indexed.drop_duplicates("date_time", keep="first").set_index("date_time")
        vals = indexed.reindex(calendar)[ramp.MEASURE_COLS]
        rates.append(float(vals.isna().to_numpy().mean()) if len(vals) else np.nan)
    return float(np.nanmean(rates))


def topic_mean_last_block(encoded, landmarks, lda, dictionary) -> np.ndarray:
    cache: dict = {}
    acc = []
    for rec in landmarks.itertuples(index=False):
        seq = encoded.get(int(rec.id))
        if seq is None:
            continue
        n_codes = len(seq["codes"])
        stop = int(rec.study_day)
        if stop > n_codes:
            stop = n_codes
        if stop < ramp.WINDOW:
            continue
        acc.append(ramp.topic_probs(seq["codes"][stop - ramp.WINDOW : stop], lda, dictionary, cache))
    return np.mean(acc, axis=0) if acc else np.full(6, np.nan)


def dominant_cosine(scored: pd.DataFrame, reference: pd.DataFrame) -> float:
    def dist(frame: pd.DataFrame) -> np.ndarray:
        counts = np.bincount(frame.dominant.to_numpy(int), minlength=6).astype(float)
        return counts / counts.sum()

    a = dist(reference)
    b = dist(scored)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "landmarks").mkdir(parents=True, exist_ok=True)
    points = landmarks_of()
    daily = aggregate_duplicate_user_days(load_daily_summary(ramp.DAILY))
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    n_patients = int(daily["user"].nunique())
    if n_patients != 175:
        raise SystemExit(f"expected 175 DB1 patients, found {n_patients}")
    missing = sorted(set(points.id) - set(daily["user"].astype(int)))
    if missing:
        raise SystemExit(f"landmark patients absent from daily: {missing}")

    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")["a0"]
    scaler = load_scaler_params(c.DEFAULT_SCALER)
    lda = LdaModel.load(str(ramp.LDA_PATH))
    dictionary = ramp.Dictionary_load()

    def run_one(name: str, kind: str | None, rate: float, seed: int):
        dest = OUT / "landmarks" / f"{name}.csv"
        if dest.exists():
            scored = pd.read_csv(dest)
            print("skip", name, flush=True)
            return scored
        started = time.time()
        if kind is None:
            masked = daily
        elif kind == "days":
            masked = ramp.mask_extra_days(daily, rate, seed)
        elif kind == "cells":
            masked = ramp.mask_extra_cells(daily, rate, seed)
        else:
            masked, _placements = ramp.mask_consecutive_gap(daily, rate, seed)
        encoded = ramp.encode(masked, model, scaler)
        observed = ramp.observed_dates(masked)
        scored = ramp.score_landmarks(encoded, observed, points, lda, dictionary)
        if len(scored) != 1013 or int(scored.emitted.sum()) != 1013 or scored["load"].isna().any():
            raise SystemExit(f"{name}: {int(scored.emitted.sum())} loads of {len(scored)}")
        scored.to_csv(dest, index=False)
        counts = ramp.code_counts(encoded, ramp.observed_dates(daily))
        topics = topic_mean_last_block(encoded, points, lda, dictionary)
        np.save(OUT / f"{name}_code_counts.npy", counts)
        np.save(OUT / f"{name}_topic_mean.npy", topics)
        miss = ramp.missingness_table(masked)
        miss.to_csv(OUT / f"{name}_missingness.csv", index=False)
        pd.DataFrame([{
            "mean_variable_missing_fixed_calendar": fixed_calendar_variable_missing(masked),
        }]).to_csv(OUT / f"{name}_fixed_calendar.csv", index=False)
        print(f"{name} {time.time() - started:.1f}s", flush=True)
        return scored

    reference = run_one("reference", None, 0.0, 0)
    stored = pd.read_csv(ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv")
    stored["id"] = stored["id"].astype(int)
    stored["study_day"] = stored["decision_day"].astype(int)
    chk = reference.merge(stored[["id", "study_day", "AUC_W"]], on=["id", "study_day"], how="left")
    mae = float(np.mean(np.abs(chk["load"] - chk["AUC_W"])))
    print(f"reference load MAE vs collapsed AUC_W = {mae:.4f} on {chk.AUC_W.notna().sum()} points", flush=True)
    (OUT / "reference_vs_collapsed.json").write_text(json.dumps({
        "n_with_load": 1013,
        "mae_vs_collapsed_AUC_W": mae,
        "n_db1_patients": n_patients,
    }, indent=2), encoding="utf-8")

    jobs = []
    for rate in RATES:
        for seed in SEEDS:
            jobs.append((f"days_plus_{int(rate * 100)}_s{seed}", "days", rate, seed, "extra_days"))
            jobs.append((f"variables_plus_{int(rate * 100)}_s{seed}", "cells", rate, seed, "extra_variables"))
            jobs.append((f"consecutive_plus_{int(rate * 100)}_s{seed}", "gap", rate, seed, "consecutive_days"))

    rows = []
    ref_counts = np.load(OUT / "reference_code_counts.npy")
    ref_topic = np.load(OUT / "reference_topic_mean.npy")
    for name, kind, rate, seed, mode in [("reference", None, 0.0, 0, "initial")] + jobs:
        if name != "reference":
            run_one(name, kind, rate, seed)
        scored = pd.read_csv(OUT / "landmarks" / f"{name}.csv")
        miss = pd.read_csv(OUT / f"{name}_missingness.csv")
        fixed = pd.read_csv(OUT / f"{name}_fixed_calendar.csv")
        counts = np.load(OUT / f"{name}_code_counts.npy")
        topics = np.load(OUT / f"{name}_topic_mean.npy")
        summary = ramp.summarize(name, mode, rate, seed, scored, reference, counts, ref_counts, topics, ref_topic, miss)
        summary["topic_cosine"] = dominant_cosine(scored, reference)
        summary["n_db1_patients"] = n_patients
        summary["mean_variable_missing_fixed_calendar"] = float(fixed.mean_variable_missing_fixed_calendar.iloc[0])
        rows.append(summary)
        print(f"  {name} AUROC={summary['emitted_AUROC']:.3f} scored={summary['emitted_n_scored']}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "metricas_por_semilla.csv", index=False)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
