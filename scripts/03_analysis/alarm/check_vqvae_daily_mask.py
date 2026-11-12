#!/usr/bin/env python3
"""Check the daily -> VQ-VAE a0 path, then a mask on those same variables.

The mask is applied on the daily table, only on the measurements the VQ-VAE
reads. Dates stay. The usual loader (densify, scale, zero-fill) runs afterwards
and the fixed a0 weights assign one code per calendar day.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from vqvae import constants as c  # noqa: E402
from vqvae.data import (  # noqa: E402
    DailySummaryDataset,
    collate_full_sequences,
    load_daily_summary,
    load_scaler_params,
)
from vqvae.inference import load_models  # noqa: E402

DAILY = ROOT / "data/daily_summaries/DB1/daily_summary_checked.csv"
LANDMARKS = ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv"
STORED = ROOT / "data/processed/lda/user_embeddings_from_pkl.csv"
OUT = ROOT / "results/missingness/procedure_check"

# Measurements the mask is allowed to hide. Weekend is not one of them:
# the loader overwrites it from the calendar date.
MEASURE_COLS = list(c.CONTINUOUS_REAL_VALUED_COLS) + list(c.CONTINUOUS_POSITIVE_COLS) + ["practiced_sport"]


def encode(frame: pd.DataFrame, model) -> dict[int, dict]:
    scaler = load_scaler_params(c.DEFAULT_SCALER)
    dataset = DailySummaryDataset(frame, scaler, missingness_mode="none", missing_rate=0.0)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_full_sequences, num_workers=0
    )
    batches = list(loader)
    max_len = max(int(b["lengths"].max().item()) for b in batches)
    out: dict[int, dict] = {}
    for batch in batches:
        inp = batch["input"]["signal_imp"].float()
        msk = batch["input"]["mask_signal"].float()
        inp = torch.nn.functional.pad(inp, (0, max_len - inp.size(2)))
        msk = torch.nn.functional.pad(msk, (0, max_len - msk.size(2)))
        inp[msk == 0] = 0
        with torch.no_grad():
            # a0 does not read the mask channel; missing values are already 0.
            _, _, indices, _ = model(inp, msk)
        inds = indices.cpu().numpy()
        lengths = batch["lengths"].cpu().numpy()
        users = batch["users"].cpu().numpy()
        for i, uid in enumerate(users):
            length = int(lengths[i])
            dates = pd.to_datetime(pd.Series(batch["dates"][i])).dt.normalize()
            out[int(uid)] = {
                "codes": inds[i, :length].astype(np.int16),
                "first": pd.Timestamp(dates.iloc[0]),
                "last": pd.Timestamp(dates.iloc[-1]),
                "n": length,
            }
    return out


def mask_days(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, labels in out.groupby("user", sort=True).groups.items():
        labels = np.asarray(list(labels))
        k = int(np.round(rate * len(labels)))
        if k <= 0:
            continue
        chosen = rng.choice(labels, size=k, replace=False)
        out.loc[chosen, MEASURE_COLS] = np.nan
    return out


def mask_cells(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, labels in out.groupby("user", sort=True).groups.items():
        labels = np.asarray(list(labels))
        block = np.array(out.loc[labels, MEASURE_COLS].to_numpy(dtype=float), copy=True, order="C")
        valid = np.flatnonzero(~np.isnan(block))
        k = int(np.round(rate * valid.size))
        if k <= 0:
            continue
        pick = rng.choice(valid, size=min(k, valid.size), replace=False)
        rows, cols = np.unravel_index(pick, block.shape)
        block[rows, cols] = np.nan
        out.loc[labels, MEASURE_COLS] = block
    return out


def mask_first_day(frame: pd.DataFrame, user_id: int) -> pd.DataFrame:
    out = frame.copy()
    rows = out.index[out["user"] == user_id]
    first = out.loc[rows, "date_time"].idxmin()
    out.loc[first, MEASURE_COLS] = np.nan
    return out


def spans(frame: pd.DataFrame) -> dict[int, tuple[pd.Timestamp, pd.Timestamp, int]]:
    got = {}
    for uid, g in frame.groupby("user", sort=True):
        d = pd.to_datetime(g["date_time"])
        got[int(uid)] = (pd.Timestamp(d.min()).normalize(), pd.Timestamp(d.max()).normalize(), int(len(g)))
    return got


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_cols = list(pd.read_csv(DAILY, nrows=0).columns)
    daily = load_daily_summary(DAILY)
    landmarks = pd.read_csv(LANDMARKS)
    keep = set(landmarks["id"].astype(int))
    daily = daily[daily["user"].astype(int).isin(keep)].reset_index(drop=True)
    origin = spans(daily)

    models = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")
    model = models["a0"]
    ref = encode(daily, model)
    ref_again = encode(daily, model)

    day_masked = mask_days(daily, 0.20, 101)
    cell_masked = mask_cells(daily, 0.20, 101)
    probe = min(origin)
    first_masked = mask_first_day(daily, probe)

    enc_day = encode(day_masked, model)
    enc_cell = encode(cell_masked, model)
    enc_first = encode(first_masked, model)

    stored = {}
    if STORED.exists():
        emb = pd.read_csv(STORED)
        for row in emb.itertuples(index=False):
            stored[int(row.user_id)] = np.asarray(json.loads(row.embedding_ids), dtype=np.int16)

    rows = []
    n_ref_identical = 0
    n_stored_equal_len = 0
    n_stored_exact = 0
    stored_agree = []
    for uid, (first, last, n_rec) in origin.items():
        r = ref[uid]
        same_run = bool(np.array_equal(r["codes"], ref_again[uid]["codes"]))
        n_ref_identical += int(same_run)
        span_ok = r["first"] == first and r["last"] == last and r["n"] == (last - first).days + 1
        day_ok = enc_day[uid]["first"] == first and enc_day[uid]["last"] == last and enc_day[uid]["n"] == r["n"]
        cell_ok = enc_cell[uid]["first"] == first and enc_cell[uid]["last"] == last and enc_cell[uid]["n"] == r["n"]
        first_ok = enc_first[uid]["first"] == first and enc_first[uid]["last"] == last and enc_first[uid]["n"] == r["n"]
        st = stored.get(uid)
        if st is not None and len(st) == r["n"]:
            n_stored_equal_len += 1
            agree = float(np.mean(st == r["codes"]))
            stored_agree.append(agree)
            n_stored_exact += int(agree == 1.0)
        else:
            agree = np.nan
        rows.append(
            {
                "id": uid,
                "first_date": str(first.date()),
                "last_date": str(last.date()),
                "n_recorded_days": n_rec,
                "n_calendar_days": r["n"],
                "span_matches_calendar": int(span_ok),
                "rerun_codes_identical": int(same_run),
                "daymask_calendar_unchanged": int(day_ok),
                "cellmask_calendar_unchanged": int(cell_ok),
                "firstday_calendar_unchanged": int(first_ok),
                "daymask_code_changed_frac": float(np.mean(enc_day[uid]["codes"] != r["codes"])),
                "cellmask_code_changed_frac": float(np.mean(enc_cell[uid]["codes"] != r["codes"])),
                "firstday_code_changed_at_origin": int(enc_first[uid]["codes"][0] != r["codes"][0]),
                "stored_length": int(len(st)) if st is not None else 0,
                "stored_code_agreement": agree,
            }
        )

    table = pd.DataFrame(rows).sort_values("id")
    table.to_csv(OUT / "patient_check.csv", index=False)

    # Show that the mask touches only VQ-VAE measurements and leaves the date.
    raw_probe = daily[daily["user"] == probe].sort_values("date_time").iloc[0]
    hid_probe = first_masked[first_masked["user"] == probe].sort_values("date_time").iloc[0]
    probe_row = {
        "id": probe,
        "date_time": str(pd.Timestamp(raw_probe["date_time"]).date()),
        "date_unchanged": bool(raw_probe["date_time"] == hid_probe["date_time"]),
        "service_unchanged": bool(raw_probe["service"] == hid_probe["service"]),
    }
    for col in c.FEATURE_COLS:
        probe_row[f"raw_{col}"] = raw_probe[col]
        probe_row[f"masked_{col}"] = hid_probe[col]
    pd.DataFrame([probe_row]).to_csv(OUT / "first_day_mask_example.csv", index=False)

    summary = {
        "daily": str(DAILY.relative_to(ROOT)),
        "model": "models/vq-vae/vqvae_a0.pt",
        "scaler": str(c.DEFAULT_SCALER.relative_to(ROOT)),
        "n_raw_columns": len(raw_cols),
        "columns_kept": c.COLS,
        "columns_dropped": [col for col in raw_cols if col not in c.COLS],
        "vqvae_feature_order": c.FEATURE_COLS,
        "masked_columns": MEASURE_COLS,
        "not_masked": ["user", "date_time", "service", "weekend"],
        "weekend_rule": "recomputed from the calendar date inside build_selected_test_sequences",
        "imputation": "after scaling, NaN becomes 0 in signal_imp; a0 does not receive the mask channel",
        "n_patients": int(len(origin)),
        "rerun_identical_patients": n_ref_identical,
        "daymask_calendar_ok": int(table["daymask_calendar_unchanged"].all()),
        "cellmask_calendar_ok": int(table["cellmask_calendar_unchanged"].all()),
        "firstday_calendar_ok": int(table["firstday_calendar_unchanged"].all()),
        "daymask_median_code_change": float(table["daymask_code_changed_frac"].median()),
        "cellmask_median_code_change": float(table["cellmask_code_changed_frac"].median()),
        "firstday_origin_code_changed_n": int(table["firstday_code_changed_at_origin"].sum()),
        "stored_embeddings": str(STORED.relative_to(ROOT)),
        "stored_equal_length_n": n_stored_equal_len,
        "stored_exact_n": n_stored_exact,
        "stored_median_agreement": float(np.median(stored_agree)) if stored_agree else None,
        "probe_patient_first_day": probe,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
