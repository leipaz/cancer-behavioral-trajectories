"""Adequacy of the external pretraining for HDO.

The VQ-VAE a0, its scaler and the 6-topic LDA stay frozen. The comparison
uses the Table S3 names Eating disorders, Cognitive impairment, Suicide
prevention and Depression / Anxiety disorders. The patient count is the
published count or lower. HDO is DB1 and was not part of the fit.

Reconstruction error uses only observed continuous components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
from gensim import corpora
from gensim.models import LdaModel

from . import constants as c
from .data import (
    apply_scaler,
    build_selected_test_sequences,
    load_scaler_params,
    replace_out_of_bounds_with_nan,
)
from .inference import inverse_transform_signal, load_models

ROOT = c.PROJECT_ROOT
OUT = ROOT / "results" / "vq-vae" / "adequacy_external_pretraining"
MENTAL_CSV = ROOT / "data" / "daily_summaries" / "daily_summary_eb2prod_may26.csv"
SPLITS_CSV = ROOT / "data" / "daily_summaries" / "daily_mental" / "user_splits.csv"
HDO_CSV = ROOT / "data" / "daily_summaries" / "DB1" / "daily_summary_checked.csv"
SUBJECTS_XLSX = ROOT / "data" / "processed" / "clinical" / "Subjects_data.xlsx"
LDA_PATH = ROOT / "models" / "lda" / "LDA_model_vdec25.gensim"
DICT_PATH = ROOT / "models" / "lda" / "dictionary_LDA_vdec25.dict"

WINDOW = 30
N_CONT = 8
N_CODES = 256
FEATURE_LABELS = {
    "sleep_start": "Sleep onset",
    "location_distance": "Distance",
    "location_time_home": "Time at home",
    "sleep_duration": "Sleep duration",
    "activity_walking": "Walking",
    "app_usage_total": "App use",
    "location_clusters_count": "Places",
    "steps_steps_total": "Steps",
}
CONT_FEATURES = c.FEATURE_COLS[:N_CONT]
TUMOR_LABEL = {"Breast": "Breast", "Lung": "Lung", "Colorectal": "Colorectal"}
CENTER_LABEL = {
    "PMP_IDIBELL": "ICO Bellvitge",
    "PMP_CHUAC": "CHUAC",
    "PMP_FUENLABRADA": "Fuenlabrada",
    "PMP_HUVM": "Virgen Macarena",
    "PMP_PRINCESA": "La Princesa",
    "PMP_BALEARES": "Son Espases",
    "PMP_NAVARRA": "Navarra",
    "PMP_VALENCIA": "Clínico Valencia",
    "PMP_CACERES": "San Pedro de Alcántara",
}
COHORT_LABEL = {
    "hdo": "HDO",
    "eating": "Eating disorders",
    "cognitive": "Cognitive impairment",
    "suicide": "Suicide prevention",
    "tmc": "Depression / Anxiety disorders",
}
COHORT_ORDER = ["hdo", "eating", "cognitive", "suicide", "tmc"]
COLORS = {
    "hdo": "#c45c26",
    "eating": "#2f6f8f",
    "cognitive": "#3d8b6e",
    "suicide": "#7a6a9a",
    "tmc": "#b0893e",
}


def program_of_service(service: str) -> str:
    if service == "adalmed" or service.startswith("ita_"):
        return "eating"
    if service.startswith("DC_"):
        return "cognitive"
    if service == "das":
        return "das"
    if service in {"afsp", "FJD_smartcrisis_2_0"}:
        return "suicide"
    if service == "TMC":
        return "tmc"
    if service == "SJD":
        return "sjd"
    if service.startswith("GM_"):
        return "cancer_gm"
    if service == "DM":
        return "dm"
    if service.startswith("PMP_"):
        return "eb2_pmp"
    if service == "cnio_caspar":
        return "cnio"
    return "other"


TABLE_S3_CAP = {
    "eating": 2464,
    "cognitive": 182 + 15,
    "suicide": 2555,
    "tmc": 149,
}


def cap_to_table_s3(reference_ids: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Keep at most the published Table S3 count.

    A larger extract is cut by first observed day, then user id. A smaller
    extract is kept whole, as a subset of the published count.
    """
    first_day = daily.groupby("user")["date_time"].min()
    kept = []
    for program, sub in reference_ids.groupby("program", sort=False):
        cap = TABLE_S3_CAP[program]
        ordered = sub.copy()
        ordered["first_day"] = ordered["user"].map(first_day)
        ordered = ordered.dropna(subset=["first_day"]).sort_values(["first_day", "user"])
        if len(ordered) > cap:
            ordered = ordered.iloc[:cap]
        kept.append(ordered.drop(columns=["first_day"]))
        print(f"  {program}: {len(ordered)} of {len(sub)} (Table S3 {cap})", flush=True)
    return pd.concat(kept, ignore_index=True)


def build_roster(splits: pd.DataFrame) -> pd.DataFrame:
    """One row per program. The reference is mental-health and vqvae == 1."""
    meta = [
        ("eating", "Eating disorders", "Eating disorders", 2464, True, False,
         "Eating disorder 1 (ED1, ita_*) and Eating disorder 2 (ED2, adalmed). Table S3 has one eating-disorders row. The comparison uses patients with vqvae=1."),
        ("cognitive", "Cognitive impairment", "Prevention of cognitive impairment; Cognitive impairment follow-up", 197, True, False,
         "Code CI. Services DC_vizcaya, DC_huelva, DC_madrid, DC_lleida, DC_alicante, DC_barcelona and DC_albacete. Table S3 splits prevention (182) and follow-up (15). This extract does not. The comparison uses patients with vqvae=1."),
        ("das", "das", "", None, False, False,
         "Code das, service das. Table S3 does not use this name, so das is not one of the four comparison cohorts."),
        ("suicide", "Suicide prevention", "Suicide prevention", 2555, True, False,
         "Code SR. Services afsp and FJD_smartcrisis_2_0. The comparison uses the Table S3 name. Every patient in the extract is marked vqvae=0. The comparison uses the patients present, not the published 2,555."),
        ("tmc", "Depression / Anxiety disorders", "Depression / Anxiety disorders", 149, True, False,
         "Service TMC only. The extract has 140 patients, below the published 149, so all of them enter as a subset. SJD is not added: TMC and SJD together would be 181, above 149."),
        ("cancer_gm", "Gregorio Marañón", "Cancer Patients", 93, False, False,
         "Code GM. Services GM_colon and GM_MCEI. Table S3 lists Cancer Patients. This is not the HDO study and not one of the four comparison cohorts."),
        ("dm", "DM", "HIV Patients", 24, False, False,
         "Code DM, service DM. Table S3 lists HIV Patients. Not one of the four comparison cohorts."),
        ("eb2_pmp", "PMP", "", None, False, False,
         "Code PMP. Services PMP_IDIBELL, PMP_CHUAC, PMP_FUENLABRADA, PMP_HUVM, PMP_PRINCESA, PMP_BALEARES, PMP_NAVARRA, PMP_VALENCIA and PMP_CACERES. These identifiers are not the DB1 HDO patients and they are not CNIO. Table S3 does not use this name."),
        ("cnio", "CNIO", "", None, False, False,
         "Code CNIO, service cnio_caspar. CNIO did not enter the VQ-VAE training. Not one of the four comparison cohorts."),
        ("hdo", "HDO", "", None, True, True,
         "DB1 patients. No identifier is in user_splits. They were not part of the VQ-VAE fit."),
    ]
    rows = []
    for program, label, table_name, table_n, in_reference, is_hdo, note in meta:
        if program == "hdo":
            n_all = n1 = n0 = np.nan
            services = ""
        else:
            sub = splits[splits["program"] == program]
            n_all = int(sub["user"].nunique())
            n1 = int(sub.loc[sub["vqvae"] == 1, "user"].nunique())
            n0 = int(sub.loc[sub["vqvae"] == 0, "user"].nunique())
            services = " ".join(sorted(sub["service"].unique()))
        rows.append(
            {
                "program": program,
                "label": label,
                "table_s3": table_name,
                "table_s3_n": table_n,
                "services": services,
                "n_users_in_splits": n_all,
                "n_vqvae_1": n1,
                "n_vqvae_0": n0,
                "in_reference": in_reference and program != "hdo",
                "formed_part_of_vqvae_train": program in {"eating", "cognitive", "das", "suicide", "tmc"}
                or (program not in {"hdo", "cnio"} and isinstance(n1, (int, np.integer)) and n1 > 0),
                "is_hdo": is_hdo,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def load_subjects() -> pd.DataFrame:
    subjects = pd.read_excel(SUBJECTS_XLSX)
    subjects["id"] = subjects["S_RECORD_id"].astype(int)
    subjects["tumor"] = subjects["Cohort"].map(TUMOR_LABEL).fillna(subjects["Cohort"])
    subjects["age"] = pd.to_numeric(subjects["Age"], errors="coerce")
    return subjects[["id", "tumor", "age"]]


def prepare_sequences(frame: pd.DataFrame, scaler: dict) -> dict[int, dict]:
    seq = build_selected_test_sequences(frame).reset_index(drop=True)
    scaled = apply_scaler(seq, scaler)
    features = c.FEATURE_COLS
    packed = {}
    for user, sample in seq.groupby("user", sort=False):
        positions = sample.index.to_numpy()
        raw = sample[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        sc = scaled.iloc[positions][features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        dates = pd.to_datetime(sample["date_time"]).to_numpy()
        observed = np.isfinite(raw[:, :N_CONT]).any(axis=1)
        packed[int(user)] = {"raw": raw, "scaled": sc, "dates": dates, "observed": observed}
    return packed


def quantize_and_decode(model, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """batch is (B, F, T). Returns recon, winner, distance, margin, winner probability, each flat over B*T except recon."""
    quantizer = model.quantizer
    encoded = model.encoder(batch, None)
    flat = encoded.permute(0, 2, 1).reshape(-1, quantizer.embed_dim)
    embed = quantizer.embed
    dist2 = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ embed + embed.pow(2).sum(0, keepdim=True)
    eu = torch.sqrt(dist2.clamp_min(0) + quantizer.eps)
    top2 = torch.topk(eu, k=2, largest=False, dim=1)
    winner = top2.indices[:, 0]
    distance = top2.values[:, 0]
    margin = top2.values[:, 1] - top2.values[:, 0]
    winner_prob = torch.softmax(-eu, dim=1).gather(1, winner.unsqueeze(1)).squeeze(1)
    chosen = embed[:, winner].T.reshape(batch.shape[0], batch.shape[2], quantizer.embed_dim)
    recon = model.decoder(chosen.permute(0, 2, 1), None)
    return recon, winner, distance, margin, winner_prob


def accumulate_feature(stats: dict, feature: int, original: np.ndarray, reconstructed: np.ndarray, observed: np.ndarray) -> None:
    if observed.sum() == 0:
        return
    x = original[observed]
    y = reconstructed[observed]
    ae = np.abs(x - y)
    stats["n"][feature] += ae.size
    stats["sum_ae"][feature] += ae.sum()
    stats["sum_x"][feature] += x.sum()
    stats["sum_y"][feature] += y.sum()
    stats["sum_xx"][feature] += np.square(x).sum()
    stats["sum_yy"][feature] += np.square(y).sum()
    stats["sum_xy"][feature] += (x * y).sum()


def new_feature_stats() -> dict:
    return {key: np.zeros(N_CONT, dtype=np.float64) for key in ("n", "sum_ae", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")}


def corr_from_stats(stats: dict, feature: int) -> float:
    n = stats["n"][feature]
    if n < 3:
        return np.nan
    sx, sy = stats["sum_x"][feature], stats["sum_y"][feature]
    ssx = stats["sum_xx"][feature] - sx * sx / n
    ssy = stats["sum_yy"][feature] - sy * sy / n
    sxy = stats["sum_xy"][feature] - sx * sy / n
    if ssx <= 0 or ssy <= 0:
        return np.nan
    return float(sxy / np.sqrt(ssx * ssy))


def code_topic_map(dictionary, lda) -> np.ndarray:
    mapping = np.full(N_CODES, -1, dtype=np.int16)
    topics = lda.get_topics()
    for token, index in dictionary.token2id.items():
        mapping[int(token)] = int(np.argmax(topics[:, index]))
    return mapping


def window_topics(codes: np.ndarray, dictionary, lda) -> np.ndarray:
    bows = []
    for start in range(0, len(codes) - WINDOW + 1, WINDOW):
        bow = dictionary.doc2bow([str(int(code)) for code in codes[start : start + WINDOW]])
        if bow:
            bows.append(bow)
    if not bows:
        return np.zeros((0, lda.num_topics))
    gamma, _ = lda.inference(bows)
    totals = gamma.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return gamma / totals


def entropy_and_mass(codes: np.ndarray) -> dict:
    if len(codes) == 0:
        return {"n_days": 0, "n_codes": 0, "entropy": np.nan, "effective_n": np.nan, "top1_share": np.nan, "top10_share": np.nan, "top1_code": -1}
    counts = np.bincount(codes, minlength=N_CODES).astype(np.float64)
    total = counts.sum()
    prob = counts[counts > 0] / total
    entropy = float(-(prob * np.log(prob)).sum())
    order = np.argsort(counts)[::-1]
    return {
        "n_days": int(total),
        "n_codes": int((counts > 0).sum()),
        "entropy": entropy,
        "effective_n": float(np.exp(entropy)),
        "top1_share": float(counts[order[0]] / total),
        "top10_share": float(counts[order[:10]].sum() / total),
        "top1_code": int(order[0]),
    }


def encode_cohort(
    name: str,
    frame: pd.DataFrame,
    scaler: dict,
    model,
    dictionary,
    lda,
    code_topic: np.ndarray,
    device: str,
    batch_size: int,
    limit: int | None,
) -> dict:
    users = list(pd.unique(frame["user"].astype(int)))
    if limit is not None:
        users = users[:limit]
    frame = frame[frame["user"].isin(users)]
    center = (
        frame.groupby("user")["service"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size else "")
        .to_dict()
    )
    stats = new_feature_stats()
    orig_ae = [[] for _ in range(N_CONT)]
    code_counts = np.zeros(N_CODES, dtype=np.int64)
    patient_rows = []
    window_rows = []
    distances = []
    hdo_days = {}
    chunk = 150
    for start in range(0, len(users), chunk):
        chunk_users = users[start : start + chunk]
        packed = prepare_sequences(frame[frame["user"].isin(chunk_users)], scaler)
        items = [(uid, packed[uid]) for uid in chunk_users if uid in packed]
        items.sort(key=lambda item: item[1]["scaled"].shape[0])
        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start : batch_start + batch_size]
            lengths = [item[1]["scaled"].shape[0] for item in batch_items]
            max_len = max(lengths)
            inp = np.zeros((len(batch_items), 10, max_len), dtype=np.float32)
            scaled_full = []
            raw_full = []
            for i, (_, sample) in enumerate(batch_items):
                scaled = sample["scaled"].T
                finite = np.isfinite(scaled)
                inp[i, :, : scaled.shape[1]] = np.where(finite, scaled, 0.0)
                scaled_full.append(scaled)
                raw_full.append(sample["raw"].T)
            with torch.inference_mode():
                recon, winner, distance, margin, winner_prob = quantize_and_decode(
                    model, torch.from_numpy(inp).to(device)
                )
            recon_np = recon.cpu().numpy()
            winner_np = winner.cpu().numpy().reshape(len(batch_items), max_len)
            distance_np = distance.cpu().numpy().reshape(len(batch_items), max_len)
            margin_np = margin.cpu().numpy().reshape(len(batch_items), max_len)
            prob_np = winner_prob.cpu().numpy().reshape(len(batch_items), max_len)
            for i, (uid, sample) in enumerate(batch_items):
                length = lengths[i]
                observed = sample["observed"]
                scaled = scaled_full[i][:, :length]
                raw = raw_full[i][:, :length]
                recon_i = recon_np[i, :, :length]
                finite = np.isfinite(scaled[:N_CONT])
                patient_stats = new_feature_stats()
                recon_orig = inverse_transform_signal(recon_i, scaler)
                for feature in range(N_CONT):
                    mask = finite[feature]
                    accumulate_feature(stats, feature, scaled[feature], recon_i[feature], mask)
                    accumulate_feature(patient_stats, feature, scaled[feature], recon_i[feature], mask)
                    if mask.any() and np.isfinite(recon_orig[feature, mask]).all() and np.isfinite(raw[feature, mask]).all():
                        orig_ae[feature].append(np.abs(raw[feature, mask] - recon_orig[feature, mask]).astype(np.float32))
                day_ae = np.full(length, np.nan, dtype=np.float64)
                day_n = finite.sum(axis=0)
                for t in np.flatnonzero(observed):
                    if day_n[t] == 0:
                        continue
                    day_ae[t] = np.abs(scaled[:N_CONT, t] - recon_i[:N_CONT, t])[finite[:, t]].mean()
                keep = observed & np.isfinite(day_ae)
                codes = winner_np[i, :length][keep].astype(np.int16)
                eu = distance_np[i, :length][keep].astype(np.float32)
                mar = margin_np[i, :length][keep].astype(np.float32)
                pwin = prob_np[i, :length][keep].astype(np.float32)
                dates = sample["dates"][keep]
                vocab = entropy_and_mass(codes)
                code_counts += np.bincount(codes, minlength=N_CODES)
                distances.append(eu)
                windows = window_topics(codes, dictionary, lda)
                if len(windows):
                    mean_window = windows.mean(axis=0)
                    for w_index, probs in enumerate(windows):
                        window_rows.append(
                            {
                                "cohort": name,
                                "user": uid,
                                "window": w_index,
                                "n_days": WINDOW,
                                **{f"topic_{k}": float(probs[k]) for k in range(6)},
                            }
                        )
                else:
                    mean_window = np.full(6, np.nan)
                n_cells = int(finite[:, keep].sum()) if keep.any() else 0
                patient_rows.append(
                    {
                        "cohort": name,
                        "user": uid,
                        "center": CENTER_LABEL.get(center.get(uid, ""), center.get(uid, "")),
                        "center_code": center.get(uid, ""),
                        "n_calendar_days": int(length),
                        "n_observed_days": int(keep.sum()),
                        "n_observed_cells": n_cells,
                        "coverage": float(finite.sum() / (length * N_CONT)) if length else np.nan,
                        "mae_scaled": float(patient_stats["sum_ae"].sum() / patient_stats["n"].sum()) if patient_stats["n"].sum() else np.nan,
                        "median_eu": float(np.median(eu)) if len(eu) else np.nan,
                        "median_margin": float(np.median(mar)) if len(mar) else np.nan,
                        "median_winner_prob": float(np.median(pwin)) if len(pwin) else np.nan,
                        "n_windows": int(len(windows)),
                        **{f"topic_{k}": float(mean_window[k]) for k in range(6)},
                        **vocab,
                    }
                )
                if name == "hdo":
                    hdo_days[uid] = {
                        "dates": dates,
                        "ae": day_ae[keep],
                        "n_obs": day_n[keep],
                        "code": codes,
                        "eu": eu,
                        "margin": mar,
                        "prob": pwin,
                    }
        print(f"  {name}: {min(start + chunk, len(users))}/{len(users)} patients", flush=True)
    return {
        "patients": pd.DataFrame(patient_rows),
        "windows": pd.DataFrame(window_rows),
        "stats": stats,
        "orig_ae": orig_ae,
        "code_counts": code_counts,
        "distances": np.concatenate(distances) if distances else np.array([]),
        "hdo_days": hdo_days,
    }


def feature_table(cohort: str, stats: dict, orig_ae: list) -> pd.DataFrame:
    rows = []
    for feature, name in enumerate(CONT_FEATURES):
        chunks = orig_ae[feature]
        original = np.concatenate(chunks) if chunks else np.array([])
        rows.append(
            {
                "cohort": cohort,
                "feature": name,
                "label": FEATURE_LABELS[name],
                "n_observed": int(stats["n"][feature]),
                "mae_scaled": float(stats["sum_ae"][feature] / stats["n"][feature]) if stats["n"][feature] else np.nan,
                "corr_scaled": corr_from_stats(stats, feature),
                "median_ae_original": float(np.median(original)) if len(original) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pooled_mae(stats: dict) -> float:
    n = stats["n"].sum()
    return float(stats["sum_ae"].sum() / n) if n else np.nan


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    mid = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def duration_band(n_days: int) -> str:
    if n_days <= 0:
        return "no observed day"
    if n_days < 30:
        return "1–29 days"
    if n_days < 90:
        return "30–89 days"
    if n_days < 180:
        return "90–179 days"
    return "180 days or more"


def summarize(results: dict[str, dict], subjects: pd.DataFrame) -> dict[str, pd.DataFrame]:
    patients = pd.concat([results[name]["patients"] for name in results], ignore_index=True)
    hdo_ids = set(patients.loc[patients["cohort"] == "hdo", "user"].astype(int))
    clinical = subjects[subjects["id"].isin(hdo_ids)]
    patients = patients.merge(clinical, left_on="user", right_on="id", how="left")
    patients["tumor"] = patients["tumor"].fillna("")
    patients["formed_part_of_vqvae_train"] = patients["cohort"] != "hdo"
    patients["duration_band"] = patients["n_observed_days"].map(duration_band)

    features = pd.concat(
        [feature_table(name, results[name]["stats"], results[name]["orig_ae"]) for name in results],
        ignore_index=True,
    )
    quant_rows = []
    vocab_rows = []
    for name, result in results.items():
        dist = result["distances"]
        counts = result["code_counts"].astype(np.float64)
        total = counts.sum()
        prob = counts[counts > 0] / total if total else np.array([])
        entropy = float(-(prob * np.log(prob)).sum()) if len(prob) else np.nan
        order = np.argsort(counts)[::-1]
        quant_rows.append(
            {
                "cohort": name,
                "n_days": int(dist.size),
                "eu_p10": float(np.percentile(dist, 10)) if dist.size else np.nan,
                "eu_p50": float(np.percentile(dist, 50)) if dist.size else np.nan,
                "eu_p90": float(np.percentile(dist, 90)) if dist.size else np.nan,
            }
        )
        vocab_rows.append(
            {
                "cohort": name,
                "n_patients": int((patients["cohort"] == name).sum()),
                "n_days": int(total),
                "n_codes": int((counts > 0).sum()),
                "entropy": entropy,
                "effective_n": float(np.exp(entropy)) if np.isfinite(entropy) else np.nan,
                "top1_code": int(order[0]) if total else -1,
                "top1_share": float(counts[order[0]] / total) if total else np.nan,
                "top10_share": float(counts[order[:10]].sum() / total) if total else np.nan,
            }
        )
    quant = pd.DataFrame(quant_rows)
    patient_quant = patients.groupby("cohort")[["median_eu", "median_margin", "median_winner_prob"]].median().reset_index()
    quant = quant.merge(patient_quant, on="cohort")
    vocab = pd.DataFrame(vocab_rows)

    windows = pd.concat([results[name]["windows"] for name in results if len(results[name]["windows"])], ignore_index=True)
    topic_cols = [f"topic_{k}" for k in range(6)]
    topic_cohort = windows.groupby("cohort")[topic_cols].mean().reset_index()
    topic_cohort["n_windows"] = windows.groupby("cohort").size().values
    topic_cohort["n_patients"] = windows.groupby("cohort")["user"].nunique().values
    patient_topic = patients[patients["n_windows"] > 0].groupby("cohort")[topic_cols].mean().reset_index()
    patient_topic = patient_topic.rename(columns={col: f"patient_mean_{col}" for col in topic_cols})
    topic_cohort = topic_cohort.merge(patient_topic, on="cohort")
    hdo_mix = topic_cohort.loc[topic_cohort["cohort"] == "hdo", topic_cols]
    if len(hdo_mix):
        hdo_vec = hdo_mix.iloc[0].to_numpy()
        topic_cohort["js_vs_hdo"] = [
            0.0 if cohort == "hdo" else js_divergence(hdo_vec, row[topic_cols].to_numpy())
            for cohort, row in topic_cohort.set_index("cohort").iterrows()
        ]
    else:
        topic_cohort["js_vs_hdo"] = np.nan

    context_rows = []
    for name, sub in patients.groupby("cohort"):
        context_rows.append(
            {
                "cohort": name,
                "n_patients": int(len(sub)),
                "n_observed_days": int(sub["n_observed_days"].sum()),
                "median_observed_days": float(sub["n_observed_days"].median()),
                "median_calendar_days": float(sub["n_calendar_days"].median()),
                "median_coverage": float(sub["coverage"].median()),
                "median_age": float(sub["age"].median()) if sub["age"].notna().any() else np.nan,
                "n_with_age": int(sub["age"].notna().sum()),
            }
        )
    context = pd.DataFrame(context_rows)

    hdo = patients[patients["cohort"] == "hdo"].copy()
    by_tumor_recon = (
        hdo.groupby("tumor")
        .agg(n_patients=("user", "nunique"), mae_scaled=("mae_scaled", "median"), median_eu=("median_eu", "median"), median_coverage=("coverage", "median"))
        .reset_index()
    )
    by_center_recon = (
        hdo.groupby("center")
        .agg(n_patients=("user", "nunique"), mae_scaled=("mae_scaled", "median"), median_eu=("median_eu", "median"), median_coverage=("coverage", "median"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
    hdo_windows = windows[windows["cohort"] == "hdo"].merge(hdo[["user", "tumor", "age"]], on="user", how="left")
    by_tumor_topic = hdo_windows.groupby("tumor")[topic_cols].mean().reset_index()
    by_tumor_topic["n_windows"] = hdo_windows.groupby("tumor").size().values
    aged = hdo[hdo["age"].notna() & (hdo["n_windows"] > 0)].copy()
    if len(aged) >= 9:
        aged["age_tertile"] = pd.qcut(aged["age"], 3, duplicates="drop")
        rows = []
        for _, group in aged.groupby("age_tertile", observed=True):
            row = {column: float(group[column].mean()) for column in topic_cols}
            row["age_tertile"] = f"{int(group['age'].min())}–{int(group['age'].max())}"
            row["n_patients"] = int(len(group))
            row["median_age"] = float(group["age"].median())
            row["median_coverage"] = float(group["coverage"].median())
            rows.append(row)
        age_topic = pd.DataFrame(rows)
    else:
        age_topic = pd.DataFrame()

    vocab_band = (
        patients.groupby(["cohort", "duration_band"])
        .agg(
            n_patients=("user", "nunique"),
            median_effective_n=("effective_n", "median"),
            median_top1_share=("top1_share", "median"),
            median_n_codes=("n_codes", "median"),
            median_n_days=("n_observed_days", "median"),
        )
        .reset_index()
    )
    return {
        "patients": patients,
        "features": features,
        "quant": quant,
        "vocab": vocab,
        "vocab_band": vocab_band,
        "topics": topic_cohort,
        "context": context,
        "by_tumor_recon": by_tumor_recon,
        "by_center_recon": by_center_recon,
        "by_tumor_topic": by_tumor_topic,
        "by_age_topic": age_topic,
        "windows": windows,
    }


def select_examples(patients: pd.DataFrame) -> pd.DataFrame:
    pool = patients[(patients["cohort"] == "hdo") & (patients["n_observed_days"] >= WINDOW)].copy()
    taken = set()
    rows = []

    def take(reason: str, ordered: pd.DataFrame) -> None:
        for user in ordered["user"]:
            if int(user) not in taken:
                row = pool.loc[pool["user"] == user].iloc[0].to_dict()
                row["reason"] = reason
                rows.append(row)
                taken.add(int(user))
                return

    z = pd.DataFrame(index=pool.index)
    for col in ("mae_scaled", "coverage", "effective_n", "median_eu"):
        sd = pool[col].std(ddof=0)
        z[col] = 0 if sd == 0 else (pool[col] - pool[col].median()).abs() / sd
    representative = pool.assign(_score=z.sum(axis=1)).sort_values("_score")
    take("Closest to the HDO median in error, coverage, effective number of codes and distance to the code", representative)
    take("Lowest coverage among follow-ups of at least 30 observed days", pool.sort_values(["coverage", "n_observed_days"]))
    high_cov = pool[pool["coverage"] >= pool["coverage"].quantile(0.75)]
    take("Highest reconstruction error among high coverage (75th percentile or above)", high_cov.sort_values("mae_scaled", ascending=False))
    long = pool[pool["n_observed_days"] >= 90]
    take("Highest concentration on a single code among follow-ups of at least 90 days", long.sort_values("top1_share", ascending=False))
    take("Highest effective number of codes among follow-ups of at least 90 days", long.sort_values("effective_n", ascending=False))
    covered = pool[pool["coverage"] >= pool["coverage"].median()]
    take("Highest quantization distance among coverage at or above the median", covered.sort_values("median_eu", ascending=False))
    return pd.DataFrame(rows)


def example_days(selected: pd.DataFrame, hdo_days: dict) -> pd.DataFrame:
    rows = []
    for rec in selected.itertuples(index=False):
        days = hdo_days[int(rec.user)]
        eligible = np.flatnonzero(days["n_obs"] >= 4)
        if len(eligible) == 0:
            eligible = np.flatnonzero(days["n_obs"] >= 1)
        order = eligible[np.argsort(days["ae"][eligible])]
        picks = {"Lowest-error day": order[0], "Median-error day": order[len(order) // 2], "Highest-error day": order[-1]}
        for role, index in picks.items():
            rows.append(
                {
                    "user": int(rec.user),
                    "reason": rec.reason,
                    "day_role": role,
                    "date": str(pd.Timestamp(days["dates"][index]).date()),
                    "code": int(days["code"][index]),
                    "eu_dist": float(days["eu"][index]),
                    "margin": float(days["margin"][index]),
                    "winner_prob": float(days["prob"][index]),
                    "mae_scaled_day": float(days["ae"][index]),
                    "n_observed_features": int(days["n_obs"][index]),
                }
            )
    return pd.DataFrame(rows)


def fmt(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def fmt_int(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{int(round(value)):,}"


def write_informe(roster: pd.DataFrame, tables: dict, examples: pd.DataFrame, days: pd.DataFrame, path: Path, limit: int | None) -> None:
    patients = tables["patients"]
    features = tables["features"]
    context = tables["context"].set_index("cohort")
    topics = tables["topics"].set_index("cohort")
    vocab = tables["vocab"].set_index("cohort")
    quant = tables["quant"].set_index("cohort")
    lines = []
    lines.append("# Adequacy of the external pretraining for HDO")
    lines.append("")
    if limit is not None:
        lines.append(f"This run is limited to {limit} patients per cohort. It is not the full count.")
        lines.append("")
    lines.append("The VQ-VAE a0, its scaler and the 6-topic LDA stay frozen. Nothing is refit. The aim is to see how the model represents HDO patients and whether that reading changes when the external corpus is inspected. A good reconstruction, or a different topic mix across cohorts, does not by itself show absence of bias or clinical relevance.")
    lines.append("")
    lines.append("## Cohorts")
    lines.append("")
    lines.append("The comparison is with these psychiatry cohorts from the VQ-VAE training: Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders. HDO is the study cohort. These four cohorts are training patients. The extract does not contain patients from these programs who were held out of the VQ-VAE fit, so the reference is identified as training data.")
    lines.append("")
    lines.append("Age and coverage are reported because they can contribute to differences in topic occupation. Age is in the HDO clinical file. It is not in the mental-health daily or in `user_splits`, so age is not compared across cohorts.")
    lines.append("")
    lines.append("| Cohort | Patients | Observed days | Median days | Median coverage | Median age |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in COHORT_ORDER:
        if name not in context.index:
            continue
        row = context.loc[name]
        age = fmt(row.median_age, 0) if row.n_with_age else "not available"
        lines.append(
            f"| {COHORT_LABEL[name]} | {fmt_int(row.n_patients)} | {fmt_int(row.n_observed_days)} | {fmt(row.median_observed_days, 0)} | {fmt(row.median_coverage, 2)} | {age} |"
        )
    lines.append("")
    zero_counts = patients.loc[patients["n_observed_days"] <= 0].groupby("cohort").size()
    zero_bits = []
    for name in COHORT_ORDER:
        if name not in zero_counts.index or int(zero_counts[name]) <= 0:
            continue
        n_zero = int(zero_counts[name])
        noun = "patient" if n_zero == 1 else "patients"
        zero_bits.append(f"{fmt_int(n_zero)} {COHORT_LABEL[name]} {noun}")
    lines.append(
        "Coverage is the share of observed continuous cells over each patient's densified calendar (8 variables × days from the first to the last day). Weekend is excluded. "
        + ((", ".join(zero_bits[:-1]) + " and " + zero_bits[-1] if len(zero_bits) > 1 else zero_bits[0]) + " have no day with a continuous variable inside range. They do not enter the error, the topics or the vocabulary." if zero_bits else "")
    )
    lines.append("")
    lines.append("## 1. Reconstruction and quantization")
    lines.append("")
    lines.append("The same frozen preprocessing: clipping, `log1p` on the positive variables and the `vqvae_a0` scaler. Error is computed only on continuous components that were actually observed. Weekend is excluded. Quantization distance is the Euclidean distance from the continuous embedding to the nearest code, on days with at least one observed continuous variable. A zero-filled gap stays inside the sequence the encoder sees; it does not enter the error.")
    lines.append("")
    lines.append("| Cohort | Scaled MAE | Median distance (p10–p90) | Median margin per patient | Median probability of the code |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in COHORT_ORDER:
        if name not in quant.index:
            continue
        sub = features[features["cohort"] == name]
        weights = sub["n_observed"].to_numpy()
        values = sub["mae_scaled"].to_numpy()
        ok = (weights > 0) & np.isfinite(values)
        mae = float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan
        q = quant.loc[name]
        lines.append(
            f"| {COHORT_LABEL[name]} | {fmt(mae, 2)} | {fmt(q.eu_p50, 1)} ({fmt(q.eu_p10, 1)}–{fmt(q.eu_p90, 1)}) | {fmt(q.median_margin, 2)} | {fmt(q.median_winner_prob, 2)} |"
        )
    lines.append("")
    lines.append("Mean absolute error in the scaled space, by variable. In parentheses, the median absolute error in original units.")
    lines.append("")
    header = "| Variable | " + " | ".join(COHORT_LABEL[name] for name in COHORT_ORDER if name in set(features["cohort"])) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for name in COHORT_ORDER if name in set(features["cohort"])) + " |")
    present = [name for name in COHORT_ORDER if name in set(features["cohort"])]
    for feature in CONT_FEATURES:
        cells = []
        for name in present:
            row = features[(features["cohort"] == name) & (features["feature"] == feature)].iloc[0]
            cells.append(f"{fmt(row.mae_scaled, 2)} ({fmt(row.median_ae_original, 0)})")
        lines.append(f"| {FEATURE_LABELS[feature]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Correlation, in the scaled space, between the observed value and the reconstruction.")
    lines.append("")
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in present) + " |")
    for feature in CONT_FEATURES:
        cells = []
        for name in present:
            row = features[(features["cohort"] == name) & (features["feature"] == feature)].iloc[0]
            cells.append(fmt(row.corr_scaled, 2))
        lines.append(f"| {FEATURE_LABELS[feature]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("HDO by tumor. The median is across patients, not across cells.")
    lines.append("")
    lines.append("| Tumor | Patients | Median MAE | Median distance | Median coverage |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for rec in tables["by_tumor_recon"].itertuples(index=False):
        tumor = rec.tumor or "no record"
        lines.append(f"| {tumor} | {fmt_int(rec.n_patients)} | {fmt(rec.mae_scaled, 2)} | {fmt(rec.median_eu, 1)} | {fmt(rec.median_coverage, 2)} |")
    lines.append("")
    lines.append("HDO by center.")
    lines.append("")
    lines.append("| Center | Patients | Median MAE | Median distance | Median coverage |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for rec in tables["by_center_recon"].itertuples(index=False):
        lines.append(f"| {rec.center} | {fmt_int(rec.n_patients)} | {fmt(rec.mae_scaled, 2)} | {fmt(rec.median_eu, 1)} | {fmt(rec.median_coverage, 2)} |")
    lines.append("")
    lines.append("Patient-level error is in `reconstruction_by_patient.csv`. A similar MAE across cohorts means the observed day is recovered with a similar residual. It does not show that the representation is free of bias.")
    lines.append("")
    lines.append("## 2. Topic occupation")
    lines.append("")
    lines.append("The same frozen LDA. Each document is a window of 30 days with at least one observed continuous variable, without overlap. The windows have the same length. A patient who does not reach 30 observed days is left out of this section. In mental health the topics are read as T0–T5. The oncological prognostic meaning of favorable or unfavorable is not transferred to them.")
    lines.append("")
    lines.append("The comparison is descriptive. The distributions do not need to differ for the representation to be considered valid. Age is not available for the external reference. Coverage and duration are, in the cohort table: a longer follow-up or a different coverage can change the mix even when the model is the same.")
    lines.append("")
    lines.append("Mean occupation of the windows. Each window has equal weight.")
    lines.append("")
    lines.append("| Cohort | Windows | Patients | T0 | T1 | T2 | T3 | T4 | T5 | JS versus HDO |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in COHORT_ORDER:
        if name not in topics.index:
            continue
        row = topics.loc[name]
        cells = " | ".join(fmt(row[f"topic_{k}"], 2) for k in range(6))
        lines.append(f"| {COHORT_LABEL[name]} | {fmt_int(row.n_windows)} | {fmt_int(row.n_patients)} | {cells} | {fmt(row.js_vs_hdo, 3)} |")
    lines.append("")
    lines.append("HDO by tumor, the same mean of 30-day windows.")
    lines.append("")
    lines.append("| Tumor | Windows | T0 | T1 | T2 | T3 | T4 | T5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rec in tables["by_tumor_topic"].itertuples(index=False):
        cells = " | ".join(fmt(getattr(rec, f"topic_{k}"), 2) for k in range(6))
        lines.append(f"| {rec.tumor or 'no record'} | {fmt_int(rec.n_windows)} | {cells} |")
    lines.append("")
    if len(tables["by_age_topic"]):
        lines.append("HDO by age tertile, only patients with at least one window and age in the clinical file. This remains descriptive.")
        lines.append("")
        lines.append("| Age tertile | Patients | Median age | Median coverage | T0 | T1 | T2 | T3 | T4 | T5 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rec in tables["by_age_topic"].itertuples(index=False):
            cells = " | ".join(fmt(getattr(rec, f"topic_{k}"), 2) for k in range(6))
            lines.append(f"| {rec.age_tertile} | {fmt_int(rec.n_patients)} | {fmt(rec.median_age, 0)} | {fmt(rec.median_coverage, 2)} | {cells} |")
        lines.append("")
    lines.append("## 3. Vocabulary use")
    lines.append("")
    lines.append("Codes from the 256-entry codebook used on observed days. The effective number is the exponential of the entropy. It is used to see whether the assignment collapses or one code takes over a follow-up as an artifact. Uniform occupation of the 256 codes is not required, and neither is a large number of codes per patient.")
    lines.append("")
    lines.append("| Cohort | Codes used | Effective number | Mass of the most frequent | Mass of the top 10 | Most frequent code |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in COHORT_ORDER:
        if name not in vocab.index:
            continue
        row = vocab.loc[name]
        lines.append(
            f"| {COHORT_LABEL[name]} | {fmt_int(row.n_codes)} / 256 | {fmt(row.effective_n, 0)} | {fmt(100 * row.top1_share, 1)} % | {fmt(100 * row.top10_share, 1)} % | {fmt_int(row.top1_code)} |"
        )
    lines.append("")
    lines.append("By patient, within bands of observed duration. The median effective number and the share of the most used code. A short band accumulates fewer codes without that being a collapse.")
    lines.append("")
    lines.append("| Cohort | Duration | Patients | Median days | Median codes | Median effective number | Share of the top code |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    band_order = ["1–29 days", "30–89 days", "90–179 days", "180 days or more"]
    band = tables["vocab_band"]
    for name in COHORT_ORDER:
        for label in band_order:
            sub = band[(band["cohort"] == name) & (band["duration_band"] == label)]
            if sub.empty:
                continue
            rec = sub.iloc[0]
            lines.append(
                f"| {COHORT_LABEL[name]} | {label} | {fmt_int(rec.n_patients)} | {fmt(rec.median_n_days, 0)} | {fmt(rec.median_n_codes, 0)} | {fmt(rec.median_effective_n, 1)} | {fmt(100 * rec.median_top1_share, 1)} % |"
            )
    lines.append("")
    append_day_occupation(lines, tables.get("day_occupation"))
    lines.append("## 4. Individual examples")
    lines.append("")
    lines.append("The five examples are HDO patients with at least 30 observed days. The criterion is written on each row. If a patient has already been selected, the next one is taken. The day is the VQ-VAE representation: code, distance, margin over the second code and that day's error. The window is the LDA summary: the T0–T5 mix of that patient's 30-day windows. They are not the same object.")
    lines.append("")
    lines.append("| Patient | Tumor | Center | Criterion | Days | Coverage | MAE | Effective number | Share of the top code | Median distance |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rec in examples.itertuples(index=False):
        lines.append(
            f"| {int(rec.user)} | {rec.tumor or '—'} | {rec.center or '—'} | {rec.reason} | {fmt_int(rec.n_observed_days)} | {fmt(rec.coverage, 2)} | {fmt(rec.mae_scaled, 2)} | {fmt(rec.effective_n, 1)} | {fmt(100 * rec.top1_share, 1)} % | {fmt(rec.median_eu, 1)} |"
        )
    lines.append("")
    sparse_users = sorted(int(user) for user, n_obs in days.groupby("user")["n_observed_features"].max().items() if n_obs < 4)
    sparse_note = ""
    if sparse_users:
        listed = ", ".join(str(user) for user in sparse_users)
        sparse_note = f" {listed} has no such day: those days are chosen among days with any observed variable, and each contributes a single one. This is the low-coverage case."
    lines.append("Days chosen within each patient: lowest error, median error and highest error. When days with at least four observed continuous variables exist, the choice is made among them." + sparse_note)
    lines.append("")
    lines.append("| Patient | Day | Date | Code | Distance | Margin | Day error | Observed variables |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for rec in days.itertuples(index=False):
        lines.append(
            f"| {int(rec.user)} | {rec.day_role} | {rec.date} | {int(rec.code)} | {fmt(rec.eu_dist, 2)} | {fmt(rec.margin, 2)} | {fmt(rec.mae_scaled_day, 2)} | {int(rec.n_observed_features)} |"
        )
    lines.append("")
    lines.append("Mean LDA-window mix for these patients. This summarizes the sequence. It is not the code of the day.")
    lines.append("")
    lines.append("| Patient | Windows | T0 | T1 | T2 | T3 | T4 | T5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rec in examples.itertuples(index=False):
        cells = " | ".join(fmt(getattr(rec, f"topic_{k}"), 2) for k in range(6))
        lines.append(f"| {int(rec.user)} | {fmt_int(rec.n_windows)} | {cells} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("The comparison is with Eating disorders, Cognitive impairment, Suicide prevention and Depression / Anxiety disorders, psychiatry cohorts from the VQ-VAE training. HDO is the study cohort. The observed day is reconstructed on both sides, with the error and quantization distance in the tables. The 30-day windows need not occupy the topics in the same way; any difference is read together with coverage and duration, and without age in the external cohort. The vocabulary is inspected for collapse or a code that takes over a follow-up, not to require that all 256 codes appear. The examples separate the quantized day from the LDA summary.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_docx(markdown_path: Path, docx_path: Path) -> Path:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    figures = {
        "1. Reconstruction and quantization": ("fig_reconstruction_mae.png", "Figure 1. Scaled reconstruction error by variable and cohort."),
        "2. Topic occupation": ("fig_topic_occupation.png", "Figure 2. Mean topic occupation of the 30-day windows."),
        "3. Vocabulary use": ("fig_vocabulary_vs_duration.png", "Figure 3. Effective number of codes against observed duration."),
        "Days covered by the most used day-types": (
            "fig_daytype_coverage.png",
            "Figure 4. Share of observed days covered by each cohort's own most used day-types.",
        ),
        "The 25 day-types with the highest mean share": (
            "fig_daytype_top25.png",
            "Figure 5. Share of observed days (%) in the 25 day-types with the highest mean occupation. The number beside each code is how many cohorts place it in their own top 25.",
        ),
    }
    blocks = []
    paragraph = []
    table_rows = []

    def flush_paragraph() -> None:
        text = " ".join(part.strip() for part in paragraph).strip()
        paragraph.clear()
        if text:
            blocks.append(("p", text))

    def flush_table() -> None:
        if table_rows:
            blocks.append(("table", [row[:] for row in table_rows]))
            table_rows.clear()

    for raw in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and set("".join(cells)) <= set("-: "):
                continue
            table_rows.append(cells)
            continue
        flush_table()
        if line.startswith("# "):
            flush_paragraph()
            blocks.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            flush_paragraph()
            blocks.append(("h2", line[3:].strip()))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    flush_table()

    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(1.3)
    section.right_margin = Cm(1.3)
    section.top_margin = Cm(1.3)
    section.bottom_margin = Cm(1.3)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x1C, 0x1B, 0x19)

    def shade(cell, color: str) -> None:
        props = cell._tc.get_or_add_tcPr()
        fill = OxmlElement("w:shd")
        fill.set(qn("w:fill"), color)
        fill.set(qn("w:val"), "clear")
        props.append(fill)

    def write_runs(paragraph_obj, text: str, size: float, bold: bool = False, color: str | None = None) -> None:
        parts = text.split("`")
        for index, part in enumerate(parts):
            if not part:
                continue
            run = paragraph_obj.add_run(part)
            run.bold = bold
            run.font.size = Pt(size)
            ink = RGBColor(0x1C, 0x1B, 0x19)
            if color:
                ink = RGBColor(int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
            run.font.color.rgb = ink
            run.font.name = "Consolas" if index % 2 else "Calibri"

    def add_figure(heading: str) -> None:
        filename, caption = figures[heading]
        image = markdown_path.parent / filename
        if not image.exists():
            return
        picture = document.add_paragraph()
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.space_before = Pt(8)
        picture.paragraph_format.space_after = Pt(2)
        picture.add_run().add_picture(str(image), width=Cm(17.8))
        cap = document.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10)
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0x1C, 0x1B, 0x19)

    pending = None
    for kind, payload in blocks:
        if kind in {"h1", "h2"} and pending:
            add_figure(pending)
            pending = None
        if kind == "h1":
            heading = document.add_paragraph()
            heading.paragraph_format.space_after = Pt(8)
            write_runs(heading, payload, 18, bold=True)
        elif kind == "h2":
            heading = document.add_paragraph()
            heading.paragraph_format.space_before = Pt(12)
            heading.paragraph_format.space_after = Pt(6)
            write_runs(heading, payload, 14, bold=True)
            if payload in figures:
                pending = payload
        elif kind == "p":
            body = document.add_paragraph()
            body.paragraph_format.space_after = Pt(6)
            body.paragraph_format.space_before = Pt(0)
            write_runs(body, payload, 11)
        else:
            rows = payload
            width = max(len(row) for row in rows)
            table = document.add_table(rows=len(rows), cols=width)
            table.style = "Table Grid"
            table.autofit = True
            table_props = table._tbl.tblPr
            table_width = OxmlElement("w:tblW")
            table_width.set(qn("w:w"), "5000")
            table_width.set(qn("w:type"), "pct")
            table_props.append(table_width)
            margins = OxmlElement("w:tblCellMar")
            for edge, value in (("top", "40"), ("bottom", "40"), ("left", "50"), ("right", "50")):
                node = OxmlElement(f"w:{edge}")
                node.set(qn("w:w"), value)
                node.set(qn("w:type"), "dxa")
                margins.append(node)
            table_props.append(margins)
            font_size = 7 if width >= 8 else 8
            for r_index, row in enumerate(rows):
                for c_index in range(width):
                    cell = table.cell(r_index, c_index)
                    cell.text = ""
                    text = row[c_index] if c_index < len(row) else ""
                    para = cell.paragraphs[0]
                    para.paragraph_format.space_before = Pt(1)
                    para.paragraph_format.space_after = Pt(1)
                    write_runs(para, text, font_size, bold=(r_index == 0), color="FFFFFF" if r_index == 0 else None)
                    shade(cell, "2F5D50" if r_index == 0 else ("F4F1EA" if r_index % 2 == 0 else "FFFFFF"))
            document.add_paragraph().paragraph_format.space_after = Pt(4)
    if pending:
        add_figure(pending)

    try:
        document.save(docx_path)
        return docx_path
    except PermissionError:
        fallback = docx_path.with_name(docx_path.stem + "_en.docx")
        document.save(fallback)
        return fallback


def day_occupation_frame(code_counts: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in COHORT_ORDER:
        counts = np.asarray(code_counts[name])
        total = int(counts.sum())
        order = np.argsort(counts)[::-1]
        for rank, code in enumerate(order, start=1):
            rows.append(
                {
                    "cohort": name,
                    "label": COHORT_LABEL[name],
                    "code": int(code),
                    "rank": rank,
                    "days": int(counts[code]),
                    "share": float(counts[code] / total) if total else 0.0,
                    "n_days": total,
                }
            )
    return pd.DataFrame(rows)


def _coverage(frame: pd.DataFrame, cohort: str, n_codes: int) -> float:
    return 100 * float(frame.loc[(frame["cohort"] == cohort) & (frame["rank"] <= n_codes), "share"].sum())


def append_day_occupation(lines: list[str], frame: pd.DataFrame | None) -> None:
    if frame is None or frame.empty:
        return
    present = [name for name in COHORT_ORDER if name in set(frame["cohort"])]
    top5 = {name: _coverage(frame, name, 5) for name in present}
    top25 = {name: _coverage(frame, name, 25) for name in present}
    top50 = {name: _coverage(frame, name, 50) for name in present}
    low5 = min(top5, key=top5.get)
    high5 = max(top5, key=top5.get)
    low25 = min(top25, key=top25.get)
    high25 = max(top25, key=top25.get)
    high50 = max(top50, key=top50.get)
    lines.append("## Days covered by the most used day-types")
    lines.append("")
    lines.append(
        "Each observed day counts once. "
        f"The five most used codes cover {fmt(top5[low5], 1)}% of {COHORT_LABEL[low5]} days and {fmt(top5[high5], 1)}% of {COHORT_LABEL[high5]} days. "
        f"Each cohort's own top 25 cover {fmt(top25[low25], 1)}% to {fmt(top25[high25], 1)}% of its observed days. "
        f"At 50 codes, the highest share is {fmt(top50[high50], 1)}%."
    )
    lines.append("")
    wide = frame.pivot(index="code", columns="cohort", values="share").reindex(columns=present).fillna(0.0)
    codes = wide.mean(axis=1).nlargest(25).index
    shared = wide.loc[codes, present] * 100
    covered = ", ".join(f"{fmt(float(shared[name].sum()), 1)}% of {COHORT_LABEL[name]} days" for name in present)
    flat = shared.stack()
    code, cohort = flat.idxmax()
    own_top = {name: set(frame.loc[(frame["cohort"] == name) & (frame["rank"] <= 25), "code"].astype(int)) for name in present}
    in_all = [int(item) for item in codes if all(int(item) in own_top[name] for name in present)]
    all_codes = ", ".join(str(item) for item in in_all) if in_all else "none"
    lines.append("## The 25 day-types with the highest mean share")
    lines.append("")
    lines.append(
        "These 25 codes are ranked by the mean share across the five cohorts, so each cohort has the same weight. "
        f"Together they cover {covered}. "
        f"The largest single share is code {int(code)} in {COHORT_LABEL[cohort]}, {fmt(float(flat.max()), 1)}% of that cohort's days. "
        f"Codes in the own top 25 of every cohort: {all_codes}."
    )
    lines.append("")


def plot_day_occupation(frame: pd.DataFrame, path: Path) -> None:
    present = [name for name in COHORT_ORDER if name in set(frame["cohort"])]
    cuts = [5, 10, 25, 50]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(cuts))
    width = 0.8 / len(present)
    for index, name in enumerate(present):
        values = [_coverage(frame, name, n_codes) for n_codes in cuts]
        ax.bar(x + index * width, values, width=width, label=COHORT_LABEL[name], color=COLORS[name])
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels([f"Top {n_codes}" for n_codes in cuts])
    ax.set_ylabel("Share of observed days (%)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path / "fig_daytype_coverage.png", dpi=140)
    plt.close(fig)

    wide = frame.pivot(index="code", columns="cohort", values="share").reindex(columns=present).fillna(0.0)
    codes = list(wide.mean(axis=1).nlargest(25).index)
    matrix = wide.loc[codes, present].to_numpy() * 100
    own_top = {name: set(frame.loc[(frame["cohort"] == name) & (frame["rank"] <= 25), "code"].astype(int)) for name in present}
    tick_labels = [f"{int(code)} ({sum(int(code) in own_top[name] for name in present)})" for code in codes]
    fig, ax = plt.subplots(figsize=(8.6, 8.8))
    cmap = LinearSegmentedColormap.from_list("occupation", ["#F7F4EE", "#2F5D50"])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(3.1, float(matrix.max())))
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([COHORT_LABEL[name] for name in present], rotation=30, ha="right")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Day-type (cohorts in whose top 25 it appears)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7, color="white" if value >= 1.8 else "#1C1B19")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of observed days (%)")
    fig.tight_layout()
    fig.savefig(path / "fig_daytype_top25.png", dpi=140)
    plt.close(fig)


def plot_all(tables: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    features = tables["features"]
    present = [name for name in COHORT_ORDER if name in set(features["cohort"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(CONT_FEATURES))
    width = 0.8 / len(present)
    for i, name in enumerate(present):
        sub = features[features["cohort"] == name].set_index("feature").loc[CONT_FEATURES]
        ax.bar(x + i * width, sub["mae_scaled"], width=width, label=COHORT_LABEL[name], color=COLORS[name])
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in CONT_FEATURES], rotation=30, ha="right")
    ax.set_ylabel("MAE in the scaled space")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path / "fig_reconstruction_mae.png", dpi=140)
    plt.close(fig)

    topics = tables["topics"].set_index("cohort")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    width = 0.8 / len(present)
    for i, name in enumerate(present):
        if name not in topics.index:
            continue
        vals = [topics.loc[name, f"topic_{k}"] for k in range(6)]
        ax.bar(x + i * width, vals, width=width, label=COHORT_LABEL[name], color=COLORS[name])
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels([f"T{k}" for k in range(6)])
    ax.set_ylabel("Mean share of the 30-day windows")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path / "fig_topic_occupation.png", dpi=140)
    plt.close(fig)

    patients = tables["patients"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in present:
        sub = patients[patients["cohort"] == name]
        ax.scatter(sub["n_observed_days"], sub["effective_n"], s=12, alpha=0.35, label=COHORT_LABEL[name], color=COLORS[name])
    ax.set_xlabel("Observed days")
    ax.set_ylabel("Effective number of codes")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path / "fig_vocabulary_vs_duration.png", dpi=140)
    plt.close(fig)
    if "day_occupation" in tables and tables["day_occupation"] is not None and len(tables["day_occupation"]):
        plot_day_occupation(tables["day_occupation"], path)


def read_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in set(c.COLS), low_memory=False)
    missing = sorted(set(c.COLS) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    frame = frame[c.COLS].copy()
    frame["date_time"] = pd.to_datetime(frame["date_time"])
    frame = replace_out_of_bounds_with_nan(frame)
    return frame.sort_values(["user", "date_time"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adequacy of the external pretraining for HDO.")
    parser.add_argument("--limit-users", type=int, default=None, help="Limit each cohort. Used for a trial run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(SPLITS_CSV)
    splits["program"] = splits["service"].map(program_of_service)
    roster = build_roster(splits)
    roster.to_csv(OUT / "reference_cohorts.csv", index=False)
    in_subset = splits["program"].isin(["eating", "cognitive"]) & splits["vqvae"].eq(1)
    in_extract = splits["program"].isin(["suicide", "tmc"])
    reference_ids = splits.loc[in_subset | in_extract, ["user", "program", "service"]].copy()
    reference_ids["user"] = reference_ids["user"].astype(int)

    scaler = load_scaler_params(c.DEFAULT_SCALER)
    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device=args.device)["a0"]
    dictionary = corpora.Dictionary.load(str(DICT_PATH))
    lda = LdaModel.load(str(LDA_PATH))
    code_topic = code_topic_map(dictionary, lda)
    subjects = load_subjects()

    print("Reading the mental-health daily", flush=True)
    mental = read_daily(MENTAL_CSV)
    mental["user"] = mental["user"].astype(int)
    print("Capping comparison cohorts at the Table S3 counts", flush=True)
    reference_ids = cap_to_table_s3(reference_ids, mental)
    print(f"Comparison patients: {reference_ids.user.nunique()}", flush=True)
    print("Reading HDO", flush=True)
    hdo = read_daily(HDO_CSV)
    hdo["user"] = hdo["user"].astype(int)
    hdo_ids = set(hdo["user"].unique())
    overlap = int(reference_ids["user"].isin(hdo_ids).sum())
    split_overlap = int(splits["user"].isin(hdo_ids).sum())

    frames = {"hdo": hdo}
    for program, ids in reference_ids.groupby("program")["user"]:
        frames[program] = mental[mental["user"].isin(set(ids))]

    results = {}
    for name in COHORT_ORDER:
        print(f"Encoding {name}", flush=True)
        results[name] = encode_cohort(
            name,
            frames[name],
            scaler,
            model,
            dictionary,
            lda,
            code_topic,
            args.device,
            args.batch_size,
            args.limit_users,
        )
        if name == "hdo" and args.limit_users is None:
            n_days = int(results["hdo"]["patients"]["n_observed_days"].sum())
            n_rows = int(len(frames["hdo"]))
            if n_days > n_rows:
                raise SystemExit(
                    f"HDO has {n_days} observed days and the daily file has {n_rows} rows. "
                    "Patient sequences were mixed."
                )
    tables = summarize(results, subjects)
    tables["day_occupation"] = day_occupation_frame({name: results[name]["code_counts"] for name in results})
    tables["day_occupation"].to_csv(OUT / "daytype_occupation_by_cohort.csv", index=False)
    examples = select_examples(tables["patients"])
    days = example_days(examples, results["hdo"]["hdo_days"])
    tables["patients"].drop(columns=[col for col in ("id",) if col in tables["patients"].columns]).to_csv(OUT / "reconstruction_by_patient.csv", index=False)
    tables["features"].to_csv(OUT / "reconstruction_by_feature.csv", index=False)
    tables["quant"].to_csv(OUT / "quantization_by_cohort.csv", index=False)
    tables["vocab"].to_csv(OUT / "vocabulary_by_cohort.csv", index=False)
    tables["vocab_band"].loc[lambda frame: frame["duration_band"] != "no observed day"].to_csv(OUT / "vocabulary_by_duration.csv", index=False)
    tables["topics"].to_csv(OUT / "topic_occupation_by_cohort.csv", index=False)
    tables["context"].to_csv(OUT / "context_age_coverage.csv", index=False)
    tables["by_tumor_recon"].to_csv(OUT / "hdo_reconstruction_by_tumor.csv", index=False)
    tables["by_center_recon"].to_csv(OUT / "hdo_reconstruction_by_center.csv", index=False)
    tables["by_tumor_topic"].to_csv(OUT / "hdo_topics_by_tumor.csv", index=False)
    tables["by_age_topic"].to_csv(OUT / "hdo_topics_by_age.csv", index=False)
    examples.to_csv(OUT / "example_selection.csv", index=False)
    days.to_csv(OUT / "example_days.csv", index=False)
    plot_all(tables, OUT)
    write_informe(roster, tables, examples, days, OUT / "report.md", args.limit_users)
    provenance = {
        "vqvae": "models/vq-vae/vqvae_a0.pt",
        "scaler": "models/vq-vae/preprocessing_scaler_vqvae.npz",
        "lda": "models/lda/LDA_model_vdec25.gensim",
        "dictionary": "models/lda/dictionary_LDA_vdec25.dict",
        "mental_daily": str(MENTAL_CSV),
        "hdo_daily": str(HDO_CSV),
        "splits": str(SPLITS_CSV),
        "window_observed_days": WINDOW,
        "vqvae_training_mental_cohorts": "eB2 mental-health training cohorts; HDO excluded",
        "reference_rule": "Table S3 names only: Eating disorders 2464, Cognitive impairment 197 (prevention 182 + follow-up 15), Suicide prevention 1091 of 2555, Depression / Anxiety disorders 140 of 149. Counts are at most the published N. A larger extract keeps the earliest first observed day, then the smaller user id.",
        "hdo_in_vqvae_train": False,
        "hdo_ids_in_user_splits": split_overlap,
        "hdo_ids_in_reference": overlap,
        "limit_users": args.limit_users,
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    written = write_docx(OUT / "report.md", OUT / "report.docx")
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Word: {written}", flush=True)


if __name__ == "__main__":
    main()
