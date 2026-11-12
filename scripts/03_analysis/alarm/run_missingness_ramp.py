#!/usr/bin/env python3
"""Raise missingness above the missingness already present in the daily table.

The start is the daily file as the VQ-VAE sees it. Two separate ramps then hide
an extra 10, 20, 30 or 40 percent of what is still observed:

- whole days: days that still have at least one measurement
- variables: cells that still have a value

Dates, the first day and the last day stay. VQ-VAE a0, the LDA, the F/U split
and the alarm threshold stay fixed. Each ramp is repeated with registered seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import LdaModel
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from vqvae import constants as c  # noqa: E402
from vqvae.data import (  # noqa: E402
    DailySummaryDataset,
    aggregate_duplicate_user_days,
    collate_full_sequences,
    load_daily_summary,
    load_scaler_params,
)
from vqvae.inference import load_models  # noqa: E402

DAILY = ROOT / "data/daily_summaries/DB1/daily_summary_checked.csv"
LANDMARKS = ROOT / "results/lda/val_2/anidada_estratificada/objetivo.csv"
LDA_PATH = ROOT / "models/lda/LDA_model_vdec25.gensim"
DICT_PATH = ROOT / "models/lda/dictionary_LDA_vdec25.dict"
OUT = ROOT / "results/missingness/ramp"

MEASURE_COLS = list(c.CONTINUOUS_REAL_VALUED_COLS) + list(c.CONTINUOUS_POSITIVE_COLS) + ["practiced_sport"]
WINDOW = 30
MIN_OBS = 15
UNFAV = (3, 4, 5)
THR_SUM = 1.0596384140662845
RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
SEEDS = (101, 102, 103, 104, 105)
GAP_SEED = 20261102


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in frame.groupby("user", sort=True):
        dates = pd.to_datetime(g["date_time"]).dt.normalize()
        start, end = dates.min(), dates.max()
        calendar = pd.date_range(start, end, freq="D")
        indexed = g.copy()
        indexed["date_time"] = dates.to_numpy()
        indexed = indexed.drop_duplicates("date_time", keep="first").set_index("date_time")
        vals = indexed.reindex(calendar)[MEASURE_COLS]
        present = vals.notna().any(axis=1)
        n_cal = int(len(calendar))
        n_days = int(present.sum())
        if n_days:
            observed = vals.loc[present.to_numpy()]
            n_cells = int(observed.size)
            n_empty = int(observed.isna().to_numpy().sum())
            var_miss = n_empty / n_cells
        else:
            n_cells = 0
            n_empty = 0
            var_miss = np.nan
        rows.append(
            {
                "id": int(uid),
                "first_date": str(start.date()),
                "last_date": str(end.date()),
                "n_calendar_days": n_cal,
                "n_days_with_measurement": n_days,
                "day_missing_rate": 1.0 - n_days / n_cal if n_cal else np.nan,
                "n_cells_on_observed_days": n_cells,
                "n_empty_cells_on_observed_days": n_empty,
                "variable_missing_rate": var_miss,
                "service": g["service"].dropna().mode().iloc[0] if g["service"].notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def encode(frame: pd.DataFrame, model, scaler) -> dict[int, dict]:
    dataset = DailySummaryDataset(frame, scaler, missingness_mode="none", missing_rate=0.0)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_full_sequences, num_workers=0)
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
            _, _, indices, _ = model(inp, msk)
        inds = indices.cpu().numpy()
        lengths = batch["lengths"].cpu().numpy()
        users = batch["users"].cpu().numpy()
        for i, uid in enumerate(users):
            length = int(lengths[i])
            days = pd.to_datetime(pd.Series(batch["dates"][i])).dt.normalize()
            out[int(uid)] = {
                "codes": inds[i, :length].astype(np.int16),
                "dates": days.to_numpy(dtype="datetime64[D]"),
            }
    return out


def observed_dates(frame: pd.DataFrame) -> dict[int, set]:
    ok = frame[MEASURE_COLS].notna().any(axis=1)
    found: dict[int, set] = {}
    use = frame.loc[ok, ["user", "date_time"]]
    for uid, g in use.groupby("user", sort=True):
        found[int(uid)] = set(pd.to_datetime(g["date_time"]).dt.normalize().to_numpy(dtype="datetime64[D]"))
    return found


def mask_extra_days(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Hide an extra fraction of days that currently have a measurement."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    present = out[MEASURE_COLS].notna().any(axis=1)
    for _, g in out.groupby("user", sort=True):
        labels = g.index.to_numpy()
        eligible = labels[present.loc[labels].to_numpy()]
        k = int(np.round(rate * len(eligible)))
        if k <= 0:
            continue
        chosen = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
        out.loc[chosen, MEASURE_COLS] = np.nan
    return out


def mask_extra_cells(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Hide an extra fraction of measurement cells that currently have a value."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, g in out.groupby("user", sort=True):
        labels = g.index.to_numpy()
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


def mask_consecutive_gap(frame: pd.DataFrame, span_frac: float, seed: int) -> pd.DataFrame:
    """One extra scenario: a single run of consecutive calendar days."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    placements = []
    for uid, g in out.groupby("user", sort=True):
        dates = pd.to_datetime(g["date_time"]).dt.normalize()
        start, end = dates.min(), dates.max()
        n_cal = int((end - start).days + 1)
        length = min(n_cal, max(7, int(np.round(span_frac * n_cal))))
        offset = int(rng.integers(0, n_cal - length + 1))
        gap_start = start + pd.Timedelta(days=offset)
        gap_end = gap_start + pd.Timedelta(days=length - 1)
        hit = g.index[(dates >= gap_start).to_numpy() & (dates <= gap_end).to_numpy()]
        out.loc[hit, MEASURE_COLS] = np.nan
        placements.append(
            {
                "id": int(uid),
                "gap_start": str(gap_start.date()),
                "gap_end": str(gap_end.date()),
                "gap_calendar_days": length,
                "span_calendar_days": n_cal,
            }
        )
    return out, pd.DataFrame(placements)


def topic_probs(codes: np.ndarray, lda: LdaModel, dictionary, cache: dict) -> np.ndarray:
    key = tuple(int(x) for x in codes)
    hit = cache.get(key)
    if hit is not None:
        return hit
    bow = dictionary.doc2bow([str(tok) for tok in key])
    probs = np.zeros(6, dtype=float)
    for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
        if int(tid) < 6:
            probs[int(tid)] = float(p)
    cache[key] = probs
    return probs


def score_landmarks(encoded, observed, landmarks, lda, dictionary) -> pd.DataFrame:
    cache: dict = {}
    rows = []
    for rec in landmarks.itertuples(index=False):
        uid = int(rec.id)
        end = int(rec.study_day)
        seq = encoded.get(uid)
        if seq is None:
            rows.append(_empty_point(uid, end, int(rec.y)))
            continue
        codes = seq["codes"]
        dates = seq["dates"]
        seen = observed.get(uid, set())
        obs = np.fromiter((d in seen for d in dates), dtype=bool, count=len(dates))
        blocks = []
        n_obs = []
        ok = True
        n_codes = len(codes)
        for shift in (60, 30, 0):
            # A decision past the profile keeps its day. Its block is the last
            # 30 profile days, the same overlap the collapsed month uses.
            stop = end - shift
            if stop > n_codes:
                stop = n_codes
            if stop < WINDOW:
                ok = False
                break
            sl = slice(stop - WINDOW, stop)
            blocks.append(topic_probs(codes[sl], lda, dictionary, cache))
            n_obs.append(int(obs[sl].sum()))
        if not ok:
            rows.append(_empty_point(uid, end, int(rec.y)))
            continue
        pufs = [float(b[list(UNFAV)].sum()) for b in blocks]
        dom = int(np.argmax(blocks[2]))
        rows.append(
            {
                "id": uid,
                "study_day": end,
                "y": int(rec.y),
                "pUF_m2": pufs[0],
                "pUF_m1": pufs[1],
                "pUF_m0": pufs[2],
                "load": float(sum(pufs)),
                "dominant": dom,
                "fu": "U" if dom in UNFAV else "F",
                "alarm": int(sum(pufs) >= THR_SUM),
                "n_obs_m2": n_obs[0],
                "n_obs_m1": n_obs[1],
                "n_obs_m0": n_obs[2],
                "valid": int(min(n_obs) >= MIN_OBS),
                "emitted": 1,
            }
        )
    return pd.DataFrame(rows)


def _empty_point(uid: int, end: int, y: int) -> dict:
    return {
        "id": uid,
        "study_day": end,
        "y": y,
        "pUF_m2": np.nan,
        "pUF_m1": np.nan,
        "pUF_m0": np.nan,
        "load": np.nan,
        "dominant": -1,
        "fu": "",
        "alarm": -1,
        "n_obs_m2": 0,
        "n_obs_m1": 0,
        "n_obs_m0": 0,
        "valid": 0,
        "emitted": 0,
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log(a[keep] / np.maximum(b[keep], 1e-300))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def code_counts(encoded, base_observed) -> np.ndarray:
    counts = np.zeros(256, dtype=float)
    for uid, seq in encoded.items():
        seen = base_observed.get(uid, set())
        keep = np.fromiter((d in seen for d in seq["dates"]), dtype=bool, count=len(seq["dates"]))
        if keep.any():
            counts += np.bincount(seq["codes"][keep], minlength=256)
    return counts


def clasif(y, score) -> dict:
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    pred = s >= THR_SUM
    tp = int(((pred) & (y == 1)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())

    def r(a, b):
        return a / b if b else np.nan

    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    return {
        "AUROC": auc,
        "sensitivity": r(tp, tp + fn),
        "specificity": r(tn, tn + fp),
        "PPV": r(tp, tp + fp),
        "NPV": r(tn, tn + fn),
        "n_scored": int(len(y)),
    }


def summarize(name, mode, rate, seed, scored, reference, counts, ref_counts, topic_mean, ref_topic, miss_now) -> dict:
    both = scored.merge(reference, on=["id", "study_day", "y"], suffixes=("", "_ref"))
    emitted = both[(both.emitted == 1) & (both.emitted_ref == 1)]
    out = {
        "scenario": name,
        "mode": mode,
        "added_rate": rate,
        "seed": seed,
        "n_landmarks": int(len(both)),
        "n_emitted": int(len(emitted)),
        "mean_day_missing": float(miss_now["day_missing_rate"].mean()),
        "mean_variable_missing": float(miss_now["variable_missing_rate"].mean()),
    }
    if len(emitted) < 2:
        return out
    p_ref = np.concatenate([emitted.pUF_m2_ref, emitted.pUF_m1_ref, emitted.pUF_m0_ref])
    p_new = np.concatenate([emitted.pUF_m2, emitted.pUF_m1, emitted.pUF_m0])
    r_puf, _ = pearsonr(p_ref, p_new)
    r_load, _ = pearsonr(emitted.load_ref, emitted["load"])
    alarm_ref = emitted.alarm_ref.to_numpy()
    alarm_new = emitted.alarm.to_numpy()
    on = int(((alarm_ref == 0) & (alarm_new == 1)).sum())
    off = int(((alarm_ref == 1) & (alarm_new == 0)).sum())
    still = emitted[(emitted.valid_ref == 1) & (emitted.valid == 1)]
    lost_points = int(((emitted.valid_ref == 1) & (emitted.valid == 0)).sum())
    ref_patients = set(emitted.loc[emitted.valid_ref == 1, "id"])
    new_patients = set(emitted.loc[emitted.valid == 1, "id"])
    lost_patients = len(ref_patients - new_patients)
    perf_emitted = clasif(emitted.y, emitted["load"])
    perf_valid = clasif(still.y, still["load"]) if len(still) else {k: np.nan for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV", "n_scored")}
    out.update(
        {
            "pUF_pearson": float(r_puf),
            "pUF_mae": float(np.mean(np.abs(p_new - p_ref))),
            "load_pearson": float(r_load),
            "load_mae": float(np.mean(np.abs(emitted["load"] - emitted.load_ref))),
            "dominant_agreement": float(np.mean(emitted.dominant == emitted.dominant_ref)),
            "fu_agreement": float(np.mean(emitted.fu == emitted.fu_ref)),
            "alarm_agreement": float(np.mean(alarm_new == alarm_ref)),
            "flip_off_to_on": on,
            "flip_on_to_off": off,
            "n_points_lost_valid": lost_points,
            "n_patients_lost_all_valid": lost_patients,
            "n_patients_ref_valid": len(ref_patients),
            "daytype_js": js_divergence(counts, ref_counts),
            "topic_js": js_divergence(topic_mean, ref_topic),
            "topic_mae": float(np.mean(np.abs(topic_mean - ref_topic))),
        }
    )
    for prefix, block in (("emitted", perf_emitted), ("valid", perf_valid)):
        for key, value in block.items():
            out[f"{prefix}_{key}"] = value
    return out


def topic_mean_at_landmarks(scored_topics_holder: np.ndarray) -> np.ndarray:
    return scored_topics_holder.mean(axis=0)


def decision_topic_mean(encoded, landmarks, lda, dictionary) -> np.ndarray:
    cache: dict = {}
    acc = []
    for rec in landmarks.itertuples(index=False):
        seq = encoded.get(int(rec.id))
        end = int(rec.study_day)
        if seq is None or end < WINDOW or end > len(seq["codes"]):
            continue
        acc.append(topic_probs(seq["codes"][end - WINDOW : end], lda, dictionary, cache))
    return np.mean(acc, axis=0) if acc else np.full(6, np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = aggregate_duplicate_user_days(load_daily_summary(DAILY))
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    landmarks = pd.read_csv(LANDMARKS)
    landmarks["id"] = landmarks["id"].astype(int)
    day_col = "decision_study_day" if "decision_study_day" in landmarks.columns else "decision_day"
    label_col = "outcome_120d" if "outcome_120d" in landmarks.columns else "target"
    landmarks["study_day"] = landmarks[day_col].astype(int)
    landmarks["y"] = landmarks[label_col].astype(int)
    keep = set(landmarks["id"])
    missing_ids = sorted(keep - set(daily["user"].astype(int)))
    daily = daily[daily["user"].astype(int).isin(keep)].reset_index(drop=True)
    landmarks = landmarks[landmarks["id"].isin(set(daily["user"].astype(int)))].reset_index(drop=True)

    base_miss = missingness_table(daily)
    base_miss.to_csv(OUT / "initial_missingness_by_patient.csv", index=False)
    q33, q66 = base_miss["variable_missing_rate"].quantile([1 / 3, 2 / 3])
    base_miss["variable_missing_band"] = pd.cut(
        base_miss["variable_missing_rate"],
        [-0.01, float(q33), float(q66), 1.01],
        labels=["lower_third", "middle_third", "upper_third"],
    )
    base_miss["any_fully_missing_day"] = (base_miss["day_missing_rate"] > 0).astype(int)
    base_miss.to_csv(OUT / "initial_missingness_by_patient.csv", index=False)

    design = {
        "daily": str(DAILY.relative_to(ROOT)),
        "population": "the 1013 landmarks in objetivo.csv, 114 patients",
        "n_patients": int(daily["user"].nunique()),
        "n_landmarks": int(len(landmarks)),
        "landmark_ids_absent_from_daily": missing_ids,
        "selection": "not chosen by response to masking",
        "services": base_miss["service"].value_counts().to_dict(),
        "day_missing_rate_mean": float(base_miss.day_missing_rate.mean()),
        "day_missing_rate_median": float(base_miss.day_missing_rate.median()),
        "variable_missing_rate_mean": float(base_miss.variable_missing_rate.mean()),
        "variable_missing_rate_median": float(base_miss.variable_missing_rate.median()),
        "variable_missing_tertile_cuts": [float(q33), float(q66)],
        "patients_with_a_fully_missing_day": int((base_miss.day_missing_rate > 0).sum()),
        "added_rates": list(RATES),
        "added_rate_definition": "fraction of units that are still observed at the initial missingness",
        "seeds": list(SEEDS),
        "consecutive_days": {
            "rates": list(RATES),
            "seeds": list(SEEDS),
            "definition": "one contiguous block covering this fraction of each patient's calendar, at least 7 days",
        },
        "masked_columns": MEASURE_COLS,
        "models_fixed": {
            "vqvae": "models/vq-vae/vqvae_a0.pt",
            "lda": str(LDA_PATH.relative_to(ROOT)),
            "unfavourable_topics": list(UNFAV),
            "threshold_sum": THR_SUM,
        },
        "valid_score": f"each of the three 30-day blocks still has at least {MIN_OBS} days with a measurement",
        "comparison": "each ramp versus the same pipeline run on the initial missingness",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(json.dumps({k: design[k] for k in (
        "n_patients", "n_landmarks", "day_missing_rate_mean", "variable_missing_rate_mean"
    )}, indent=2), flush=True)

    summary_path = OUT / "scenario_summary.csv"
    done = set(pd.read_csv(summary_path)["scenario"]) if summary_path.exists() else set()
    ref_path = OUT / "reference_landmarks.csv"
    counts_path = OUT / "reference_code_counts.npy"

    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")["a0"]
    scaler = load_scaler_params(c.DEFAULT_SCALER)
    lda = LdaModel.load(str(LDA_PATH))
    dictionary = Dictionary_load()

    if ref_path.exists() and counts_path.exists():
        reference = pd.read_csv(ref_path)
        ref_counts = np.load(counts_path)
        ref_topic = np.load(OUT / "reference_topic_mean.npy")
    else:
        print("encoding reference", flush=True)
        ref_enc = encode(daily, model, scaler)
        ref_obs = observed_dates(daily)
        _assert_calendar(ref_enc, base_miss)
        reference = score_landmarks(ref_enc, ref_obs, landmarks, lda, dictionary)
        reference.to_csv(ref_path, index=False)
        ref_counts = code_counts(ref_enc, ref_obs)
        np.save(counts_path, ref_counts)
        ref_topic = decision_topic_mean(ref_enc, landmarks, lda, dictionary)
        np.save(OUT / "reference_topic_mean.npy", ref_topic)
        stored = pd.read_csv(ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv")
        stored["id"] = stored["id"].astype(int)
        stored["study_day"] = stored["decision_day"].astype(int)
        stored = stored[["id", "study_day", "AUC_W"]].rename(columns={"AUC_W": "load_stored"})
        chk = reference.merge(stored, on=["id", "study_day"])
        have = chk[chk.emitted == 1]
        mae = float(np.mean(np.abs(have["load"] - have.load_stored))) if len(have) else np.nan
        (OUT / "reference_vs_stored.json").write_text(json.dumps({
            "n_emitted": int(len(have)),
            "load_mae_vs_stored_AUC_W": mae,
            "valid_points": int(reference.valid.sum()),
            "valid_patients": int(reference.loc[reference.valid == 1, "id"].nunique()),
        }, indent=2), encoding="utf-8")
        print(f"reference vs stored AUC_W MAE={mae:.4f}", flush=True)

    jobs = [("reference", "initial", 0.0, 0, daily, None)]
    for rate in RATES:
        for seed in SEEDS:
            jobs.append((f"days_plus_{int(rate*100)}_s{seed}", "extra_days", rate, seed, None, "days"))
            jobs.append((f"variables_plus_{int(rate*100)}_s{seed}", "extra_variables", rate, seed, None, "cells"))
            jobs.append((f"consecutive_plus_{int(rate*100)}_s{seed}", "consecutive_days", rate, seed, None, "gap"))

    base_obs = observed_dates(daily)
    for name, mode, rate, seed, frame, kind in jobs:
        if name in done:
            print("skip", name, flush=True)
            continue
        if kind is None:
            # The reference row records the starting metrics against itself.
            miss_now = base_miss
            row = summarize(name, mode, rate, seed, reference, reference, ref_counts, ref_counts, ref_topic, ref_topic, miss_now)
        else:
            print("run", name, flush=True)
            if kind == "days":
                masked = mask_extra_days(daily, rate, seed)
                placements = None
            elif kind == "cells":
                masked = mask_extra_cells(daily, rate, seed)
                placements = None
            else:
                masked, placements = mask_consecutive_gap(daily, rate, seed)
                gap_dir = OUT / "consecutive_gap_placement"
                gap_dir.mkdir(parents=True, exist_ok=True)
                placements.to_csv(gap_dir / f"{name}.csv", index=False)
            enc = encode(masked, model, scaler)
            _assert_calendar(enc, base_miss)
            obs = observed_dates(masked)
            scored = score_landmarks(enc, obs, landmarks, lda, dictionary)
            counts = code_counts(enc, base_obs)
            topics = decision_topic_mean(enc, landmarks, lda, dictionary)
            miss_now = missingness_table(masked)
            row = summarize(name, mode, rate, seed, scored, reference, counts, ref_counts, topics, ref_topic, miss_now)
            scored.to_csv(OUT / "landmarks" / f"{name}.csv", index=False)
        _append_summary(summary_path, row)
        done.add(name)
        print(
            f"  {name} day_miss={row['mean_day_missing']:.3f} var_miss={row['mean_variable_missing']:.3f} "
            f"load_r={row.get('load_pearson', float('nan')):.3f}",
            flush=True,
        )


def Dictionary_load():
    from gensim.corpora import Dictionary
    return Dictionary.load(str(DICT_PATH))


def _assert_calendar(encoded, base_miss: pd.DataFrame) -> None:
    expect = {
        int(r.id): (np.datetime64(r.first_date), np.datetime64(r.last_date), int(r.n_calendar_days))
        for r in base_miss.itertuples(index=False)
    }
    for uid, seq in encoded.items():
        first, last, n = expect[uid]
        if seq["dates"][0] != first or seq["dates"][-1] != last or len(seq["dates"]) != n:
            raise AssertionError(f"calendar moved for {uid}")


def _append_summary(path: Path, row: dict) -> None:
    (OUT / "landmarks").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
