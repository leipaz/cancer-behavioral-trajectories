#!/usr/bin/env python3
"""How the frozen external VQ-VAE represents HDO, against mental patients it did not fit.

Four descriptive pieces, each with a limited claim:

1. Reconstruction and quantization on components that were really observed,
   with the frozen scaler. The mental arm is the held-out split
   (user_splits.vqvae == 0), not the fit set. HDO is also split by tumour
   and centre.
2. Topic occupation of the frozen LDA on complete 30-day windows labeled
   T0–T5 from the first densified day. Those labels are positional. They
   are not the oncological F/U.
3. Codebook use: unique codes, concentration, entropy and effective number,
   by cohort and patient, next to the number of observed days.
4. Individual examples chosen by a fixed rule (coverage, code diversity,
   reconstruction error), with the VQ-VAE day kept separate from the LDA
   window.

Nothing here is a test of clinical relevance or of the absence of bias.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim import corpora
from gensim.models import LdaModel
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from vqvae import constants as c  # noqa: E402
from vqvae.data import DailySummaryDataset, replace_out_of_bounds_with_nan  # noqa: E402
from vqvae.inference import inverse_transform_signal, load_models  # noqa: E402
from vqvae.sample_cohort_profiles import COHORT_GROUPS  # noqa: E402

MENTAL_CSV = Path(
    "/lustre/uc3m/gts_c3_cluster_1-12/lpaz/Transformer_TimeEmotion/data/daily_sum/"
    "daily_summary_eb2prod_may26.csv"
)
SPLITS_CSV = ROOT / "data/daily_summaries/daily_mental/user_splits.csv"
HDO_CSV = ROOT / "data/daily_summaries/DB1/daily_summary_checked.csv"
VALSET_CSV = ROOT / "data/daily_summaries/DB3/daily_summary_checked.csv"
VALSET_CLINICAL = ROOT / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026.xlsx"
# SAP 4.3 sensitivity subset. 21003 is on the 48-person list and has no digital row here.
VALSET_SENSITIVITY_EXCLUDE = {21003, 91021, 31017}
SUBJECTS_XLSX = ROOT / "data/processed/clinical/Subjects_data.xlsx"
PARTITIONS_CSV = ROOT / "results/lda/val_2/anidada_estratificada/partitions.csv"
LDA_PATH = ROOT / "models/lda/LDA_model_vdec25.gensim"
DICT_PATH = ROOT / "models/lda/dictionary_LDA_vdec25.dict"
OUT = ROOT / "results/revision/mental_vs_hdo"

WINDOW = 30
N_WINDOWS = 6
N_TOPICS = 6
MIN_DAYS_EXAMPLE = 30
# practiced_sport is a binary flag. Its absolute error is not on the same
# scale as the continuous features, so it is reported on its own and is not
# folded into the patient MAE.
ERROR_NAMES = list(c.CONTINUOUS_REAL_VALUED_COLS) + list(c.CONTINUOUS_POSITIVE_COLS)
SENSOR_NAMES = ERROR_NAMES + ["practiced_sport"]
SENSOR_IDX = [c.FEATURE_COLS.index(name) for name in SENSOR_NAMES]
ERROR_IDX = [c.FEATURE_COLS.index(name) for name in ERROR_NAMES]
# Oncology services that sit in the mental extract. They are not the external
# mental reference.
EXCLUDED_MENTAL_SERVICES = {"cnio_caspar"}
EXCLUDED_MENTAL_PREFIXES = ("PMP_",)

HOSPITAL_SHORT = {
    "CHUAC": "CHUAC",
    "Hospital Clínico Unversitario de Valencia": "Clínico Valencia",
    "Hospital San Pedro de Alcántara": "San Pedro",
    "Hospital Universitario Son Espases": "Son Espases",
    "Hospital Universitario de Fuenlabrada": "Fuenlabrada",
    "Hospital Unversitario Virgen de la Macarena": "Macarena",
    "Hospital Unversitario de La Princesa": "La Princesa",
    "Hospital de Navarra": "Navarra",
    "ICO Bellvitge": "ICO Bellvitge",
}

COLORS = {"mental_held_out": "#3d5a80", "HDO": "#c44536", "VAL-SET": "#2a9d8f"}
COHORT_ORDER = ("mental_held_out", "HDO", "VAL-SET")


def service_group(service: str) -> str:
    for code, spec in COHORT_GROUPS.items():
        if service in spec["services"]:
            return code
    return "other"


def is_excluded_service(service: str) -> bool:
    if service in EXCLUDED_MENTAL_SERVICES:
        return True
    return any(str(service).startswith(prefix) for prefix in EXCLUDED_MENTAL_PREFIXES)


def prepare_frame(path: Path, users: set[int] | None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in set(c.COLS))
    missing = sorted(set(c.COLS) - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    frame = frame[c.COLS].copy()
    frame["user"] = frame["user"].astype(int)
    if users is not None:
        frame = frame[frame["user"].isin(users)]
    frame["date_time"] = pd.to_datetime(frame["date_time"])
    frame = replace_out_of_bounds_with_nan(frame)
    return frame.sort_values(["user", "date_time"]).reset_index(drop=True)


def scaled_groups(frame: pd.DataFrame, scaler: dict) -> dict[int, pd.DataFrame]:
    dataset = DailySummaryDataset(frame, scaler)
    grouped = {}
    for uid, sample in dataset.dataset.groupby("user", sort=False):
        grouped[int(uid)] = sample.reset_index(drop=True)
    return grouped


def arrays_for(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    signal_df = sample.drop(columns=c.UNINFORMATIVE)
    signal_df = signal_df[c.FEATURE_COLS]
    signal = signal_df.to_numpy(dtype=np.float64)
    mask = (~np.isnan(signal)).astype(np.uint8)
    signal_imp = np.nan_to_num(signal, nan=0.0).astype(np.float32)
    return signal, signal_imp, mask, sample["date_time"]


@torch.no_grad()
def forward_fast(model, signal_imp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decoded (F, T), code index (T,), L2 distance to the chosen code (T,)."""
    x = torch.from_numpy(signal_imp.T.copy()).float().unsqueeze(0)
    encoded = model.encoder(x, None)
    flat = encoded.permute(0, 2, 1).reshape(-1, encoded.shape[1])
    embed = model.quantizer.embed
    dist2 = (
        flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ embed + embed.pow(2).sum(0, keepdim=True)
    ).clamp_min(0)
    index = dist2.argmin(dim=1)
    chosen = embed.t().index_select(0, index)
    distance = (chosen - flat).pow(2).sum(1).sqrt()
    decoded = model.decoder(chosen.T.unsqueeze(0), None).squeeze(0).cpu().numpy()
    return decoded, index.cpu().numpy().astype(np.int16), distance.cpu().numpy().astype(np.float32)


@torch.no_grad()
def assert_matches_stock_forward(model, signal_imp: np.ndarray) -> None:
    decoded, index, _ = forward_fast(model, signal_imp)
    x = torch.from_numpy(np.nan_to_num(signal_imp, nan=0.0).T.copy()).float().unsqueeze(0)
    stock, _, stock_index, _ = model(x, None)
    stock_index = stock_index.squeeze(0).cpu().numpy()
    if not np.array_equal(index, stock_index):
        raise RuntimeError("Fast quantizer indices do not match vqvae_a0.forward.")
    if not np.allclose(decoded, stock.squeeze(0).cpu().numpy(), atol=1e-5, rtol=1e-5):
        raise RuntimeError("Fast decoder does not match vqvae_a0.forward.")


def entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def top_share(counts: np.ndarray, k: int) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float(np.sort(counts)[::-1][:k].sum() / total)


def iqr(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "n": int(s.shape[0]),
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }


def topic_vector(lda, dictionary, codes: np.ndarray) -> tuple[np.ndarray, int]:
    tokens = [str(int(code)) for code in codes]
    known = [token for token in tokens if token in dictionary.token2id]
    n_oov = len(tokens) - len(known)
    if not known:
        return np.full(N_TOPICS, np.nan), n_oov
    bow = dictionary.doc2bow(known)
    pairs = lda.get_document_topics(bow, minimum_probability=0.0)
    probs = np.zeros(N_TOPICS, dtype=float)
    for topic_id, prob in pairs:
        if 0 <= int(topic_id) < N_TOPICS:
            probs[int(topic_id)] = float(prob)
    return probs, n_oov


def load_hdo_meta() -> pd.DataFrame:
    subjects = pd.read_excel(SUBJECTS_XLSX)
    subjects = subjects.dropna(subset=["S_RECORD_id"]).copy()
    subjects["id"] = subjects["S_RECORD_id"].astype(int)
    meta = subjects[["id", "Cohort", "Age"]].drop_duplicates("id").rename(columns={"Cohort": "tumor", "Age": "age"})

    parts = pd.read_csv(PARTITIONS_CSV, usecols=["id", "hospital"])
    parts["id"] = parts["id"].astype(int)
    parts = parts.drop_duplicates("id")
    parts["digit"] = parts["id"].astype(str).str[0]
    digit_name = (
        parts.groupby("digit")["hospital"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )
    meta["digit"] = meta["id"].astype(str).str[0]
    hospital = meta["id"].map(parts.set_index("id")["hospital"])
    hospital = hospital.fillna(meta["digit"].map(digit_name))
    meta["center"] = hospital.map(lambda name: HOSPITAL_SHORT.get(name, name) if pd.notna(name) else "unlisted")
    return meta.drop(columns=["digit"])


def held_out_users() -> tuple[pd.DataFrame, dict]:
    splits = pd.read_csv(SPLITS_CSV)
    splits["user"] = splits["user"].astype(int)
    splits["service"] = splits["service"].astype(str)
    flagged = splits[splits["vqvae"] == 0].copy()
    flagged["excluded_service"] = flagged["service"].map(is_excluded_service)
    kept = flagged[~flagged["excluded_service"]].drop_duplicates("user").copy()
    kept["clinical_group"] = kept["service"].map(service_group)
    note = {
        "split_file": str(SPLITS_CSV.relative_to(ROOT)),
        "rule": "user_splits.vqvae == 0, then drop cnio_caspar and PMP_* (oncology services in the same extract)",
        "n_flagged_vqvae_0": int((splits["vqvae"] == 0).sum()),
        "n_flagged_vqvae_1_not_used": int((splits["vqvae"] == 1).sum()),
        "n_excluded_oncology_service": int(flagged["excluded_service"].sum()),
        "fit_set_used_as_reference": False,
    }
    return kept, note


def run_cohort(name: str, groups: dict[int, pd.DataFrame], model, meta: pd.DataFrame | None) -> dict:
    rows = []
    feature_rows = []
    window_rows = []
    day_store = {}
    checked = False
    for n_done, (uid, sample) in enumerate(groups.items(), start=1):
        signal, signal_imp, mask, dates = arrays_for(sample)
        if not checked:
            # Same input prefix for both paths. The convolution is position-wise
            # given its input, so a short prefix is enough to check the quantizer.
            assert_matches_stock_forward(model, signal_imp[: min(48, signal_imp.shape[0])])
            checked = True
            print(f"  {name}: fast path matches vqvae_a0.forward", flush=True)
        decoded, codes, quant = forward_fast(model, signal_imp)
        sensor_obs = mask[:, SENSOR_IDX].any(axis=1)
        span = int(signal.shape[0])
        n_sensor_days = int(sensor_obs.sum())

        abs_scaled = np.abs(decoded - signal.T)
        original = inverse_transform_signal(signal.T, model_scaler)
        recon_original = inverse_transform_signal(decoded, model_scaler)
        abs_original = np.abs(recon_original - original)

        per_feature = []
        for feat_i, feat_name in zip(SENSOR_IDX, SENSOR_NAMES):
            observed = mask[:, feat_i].astype(bool)
            n_obs = int(observed.sum())
            mae_s = float(abs_scaled[feat_i, observed].mean()) if n_obs else np.nan
            mae_o = float(abs_original[feat_i, observed].mean()) if n_obs else np.nan
            med_o = float(np.median(abs_original[feat_i, observed])) if n_obs else np.nan
            level = float(np.median(signal[observed, feat_i])) if n_obs else np.nan
            feature_rows.append(
                {
                    "cohort": name,
                    "id": uid,
                    "feature": feat_name,
                    "n_observed": n_obs,
                    "mae_scaled": mae_s,
                    "mae_original": mae_o,
                    "median_ae_original": med_o,
                    "median_scaled": level,
                }
            )
            if n_obs and feat_name in ERROR_NAMES:
                per_feature.append(mae_s)

        day_mae = np.full(span, np.nan, dtype=np.float32)
        for t in np.flatnonzero(sensor_obs):
            cols = mask[t, SENSOR_IDX].astype(bool)
            feat_at = [SENSOR_IDX[i] for i, flag in enumerate(cols) if flag]
            day_mae[t] = float(np.mean(abs_scaled[feat_at, t]))

        observed_codes = codes[sensor_obs]
        counts = np.bincount(observed_codes, minlength=256).astype(np.int64) if n_sensor_days else np.zeros(256, dtype=np.int64)
        gap = ~sensor_obs
        gap_counts = np.bincount(codes[gap], minlength=256).astype(np.int64) if gap.any() else np.zeros(256, dtype=np.int64)
        h = entropy_bits(counts)
        n_eff = float(2.0 ** h) if np.isfinite(h) else np.nan

        info = {"id": uid, "tumor": np.nan, "center": np.nan, "age": np.nan, "clinical_group": np.nan, "service": np.nan}
        if meta is not None and uid in meta.index:
            info.update(meta.loc[uid].to_dict())

        quant_obs = quant[sensor_obs]
        observed_cells = mask[:, SENSOR_IDX].T.astype(bool)
        micro = float(abs_scaled[SENSOR_IDX][observed_cells].mean()) if n_sensor_days else np.nan
        rows.append(
            {
                "cohort": name,
                "id": uid,
                "service": info.get("service", np.nan),
                "clinical_group": info.get("clinical_group", np.nan),
                "tumor": info.get("tumor", np.nan),
                "center": info.get("center", np.nan),
                "age": info.get("age", np.nan),
                "span_days": span,
                "n_sensor_days": n_sensor_days,
                "coverage": n_sensor_days / span if span else np.nan,
                "n_features_with_obs": int(len(per_feature)),
                "mae_scaled": float(np.mean(per_feature)) if per_feature else np.nan,
                "mae_scaled_micro": micro,
                "quant_distance": float(quant_obs.mean()) if n_sensor_days else np.nan,
                "n_unique_codes": int((counts > 0).sum()),
                "top1_share": top_share(counts, 1),
                "top5_share": top_share(counts, 5),
                "entropy_bits": h,
                "n_effective": n_eff,
                "gap_days": int(gap.sum()),
                "gap_top_code": int(gap_counts.argmax()) if gap.any() else np.nan,
                "gap_top_share": top_share(gap_counts, 1),
                "observed_top_code": int(counts.argmax()) if n_sensor_days else np.nan,
                "sap_sensitivity_exclude": int(uid) in VALSET_SENSITIVITY_EXCLUDE,
            }
        )

        for w in range(N_WINDOWS):
            start, end = w * WINDOW, (w + 1) * WINDOW
            if end > span:
                break
            probs, n_oov = topic_vector(lda_model, lda_dictionary, codes[start:end])
            row = {
                "cohort": name,
                "id": uid,
                "window": f"T{w}",
                "window_index": w,
                "n_days": WINDOW,
                "n_sensor_days_in_window": int(sensor_obs[start:end].sum()),
                "sensor_fraction": float(sensor_obs[start:end].mean()),
                "n_oov_codes": n_oov,
            }
            for topic in range(N_TOPICS):
                row[f"topic_{topic}"] = float(probs[topic])
            finite = probs[np.isfinite(probs)]
            positive = finite[finite > 0]
            row["topic_entropy_bits"] = (
                float(-(positive * np.log2(positive)).sum()) if positive.size else np.nan
            )
            window_rows.append(row)

        day_store[uid] = {
            "date": pd.to_datetime(dates).to_numpy(),
            "code": codes,
            "quant": quant,
            "day_mae": day_mae,
            "sensor_obs": sensor_obs.astype(np.uint8),
        }
        if n_done % 100 == 0:
            print(f"  {name}: {n_done}/{len(groups)}", flush=True)

    print(f"  {name}: {len(groups)} patients", flush=True)
    return {
        "patients": pd.DataFrame(rows),
        "features": pd.DataFrame(feature_rows),
        "windows": pd.DataFrame(window_rows),
        "days": day_store,
    }


def stratified_mae(patients: pd.DataFrame, features: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    hdo = patients[patients["cohort"] == "HDO"]
    for key, sub in hdo.groupby(by, dropna=False):
        ids = set(sub["id"])
        block = {"group": by, "level": key if pd.notna(key) else "missing"}
        block.update({f"mae_{k}": v for k, v in iqr(sub["mae_scaled"]).items()})
        block.update({f"quant_{k}": v for k, v in iqr(sub["quant_distance"]).items()})
        block["n_patients"] = int(len(sub))
        feat = features[(features["cohort"] == "HDO") & (features["id"].isin(ids))]
        for feature, fsub in feat.groupby("feature"):
            block[f"mae_scaled_{feature}"] = float(fsub["mae_scaled"].median())
        rows.append(block)
    return pd.DataFrame(rows)


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, feature), sub in features.groupby(["cohort", "feature"]):
        w = sub["n_observed"].to_numpy()
        mae = sub["mae_scaled"].to_numpy()
        ok = np.isfinite(mae) & (w > 0)
        pooled = float(np.average(mae[ok], weights=w[ok])) if ok.any() else np.nan
        orig = sub["mae_original"].to_numpy()
        ok_o = np.isfinite(orig) & (w > 0)
        rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "n_patients": int(ok.sum()),
                "n_observed": int(w.sum()),
                "in_patient_mae": feature in ERROR_NAMES,
                "mae_scaled_pooled": pooled,
                "mae_scaled_patient_median": float(np.nanmedian(mae)),
                "mae_original_pooled": float(np.average(orig[ok_o], weights=w[ok_o])) if ok_o.any() else np.nan,
                "median_ae_original_patient_median": float(sub["median_ae_original"].median()),
            }
        )
    return pd.DataFrame(rows)


def feature_levels(features: pd.DataFrame) -> pd.DataFrame:
    """Patient-median of the observed scaled value, then the cohort median of those."""
    rows = []
    use = features[features["feature"].isin(ERROR_NAMES)]
    for (cohort, feature), sub in use.groupby(["cohort", "feature"]):
        level = sub["median_scaled"].dropna()
        rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "n_patients": int(len(level)),
                "scaled_level_median": float(level.median()) if len(level) else np.nan,
                "scaled_level_q25": float(level.quantile(0.25)) if len(level) else np.nan,
                "scaled_level_q75": float(level.quantile(0.75)) if len(level) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_presence(features: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Share of the recorded span on which each variable is present after clipping."""
    span = patients.set_index(["cohort", "id"])["span_days"]
    rows = []
    for (cohort, feature), sub in features.groupby(["cohort", "feature"]):
        joined = sub.join(span, on=["cohort", "id"])
        rate = joined["n_observed"] / joined["span_days"]
        rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "n_patients_in_cohort": int(patients.loc[patients.cohort == cohort, "id"].nunique()),
                "n_patients_with_any": int((sub["n_observed"] > 0).sum()),
                "presence_median": float(rate.median()),
                "presence_q25": float(rate.quantile(0.25)),
                "presence_q75": float(rate.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(patients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in patients.groupby("cohort"):
        age = iqr(sub["age"])
        cov = iqr(sub["coverage"])
        days = iqr(sub["n_sensor_days"])
        span = iqr(sub["span_days"])
        rows.append(
            {
                "cohort": cohort,
                "n_patients": int(len(sub)),
                "age_n": age["n"],
                "age_median": age["median"],
                "age_q25": age["q25"],
                "age_q75": age["q75"],
                "coverage_median": cov["median"],
                "coverage_q25": cov["q25"],
                "coverage_q75": cov["q75"],
                "sensor_days_median": days["median"],
                "sensor_days_q25": days["q25"],
                "sensor_days_q75": days["q75"],
                "span_days_median": span["median"],
                "span_days_q25": span["q25"],
                "span_days_q75": span["q75"],
            }
        )
    return pd.DataFrame(rows)


def vocabulary_table(patients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in patients.groupby("cohort"):
        use = sub[sub["n_sensor_days"] > 0]
        rho, pval = (np.nan, np.nan)
        if len(use) >= 5:
            rho, pval = spearmanr(use["n_sensor_days"], use["n_effective"])
        collapse = use[(use["n_sensor_days"] >= MIN_DAYS_EXAMPLE) & (use["top1_share"] >= 0.8)]
        same_artifact = use[use["observed_top_code"] == use["gap_top_code"]]
        rows.append(
            {
                "cohort": cohort,
                "n_patients": int(len(use)),
                "n_unique_median": float(use["n_unique_codes"].median()),
                "top1_share_median": float(use["top1_share"].median()),
                "top5_share_median": float(use["top5_share"].median()),
                "entropy_bits_median": float(use["entropy_bits"].median()),
                "n_effective_median": float(use["n_effective"].median()),
                "n_effective_q25": float(use["n_effective"].quantile(0.25)),
                "n_effective_q75": float(use["n_effective"].quantile(0.75)),
                "spearman_effective_vs_observed_days": float(rho),
                "spearman_p": float(pval),
                "n_top1_share_ge_0.8_and_days_ge_30": int(len(collapse)),
                "n_observed_top_equals_gap_top": int(len(same_artifact)),
                "gap_top_code_mode": float(use["gap_top_code"].mode().iloc[0]) if use["gap_top_code"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def topic_summary(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if windows.empty:
        return pd.DataFrame()
    for (cohort, window), sub in windows.groupby(["cohort", "window"]):
        row = {
            "cohort": cohort,
            "window": window,
            "n_patients": int(len(sub)),
            "sensor_fraction_median": float(sub["sensor_fraction"].median()),
            "oov_median": float(sub["n_oov_codes"].median()),
            "topic_entropy_bits_median": float(sub["topic_entropy_bits"].median()),
        }
        for topic in range(N_TOPICS):
            row[f"topic_{topic}"] = float(sub[f"topic_{topic}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_examples(patients: pd.DataFrame) -> pd.DataFrame:
    """Fixed rule, applied separately inside each cohort. Earlier roles keep the patient."""
    chosen = []
    for cohort, sub in patients.groupby("cohort"):
        pool = sub[sub["n_sensor_days"] >= MIN_DAYS_EXAMPLE].copy()
        used: set[int] = set()

        def take(role: str, frame: pd.DataFrame, how: str) -> None:
            frame = frame[~frame["id"].isin(used)]
            if frame.empty:
                chosen.append({"cohort": cohort, "role": role, "id": np.nan, "reason": "no patient met the rule"})
                return
            if how == "max":
                row = frame.sort_values(["mae_scaled", "id"], ascending=[False, True]).iloc[0]
            elif how == "min":
                row = frame.sort_values(["mae_scaled", "id"], ascending=[True, True]).iloc[0]
            elif how == "max_eff":
                row = frame.sort_values(["n_effective", "id"], ascending=[False, True]).iloc[0]
            elif how == "min_eff":
                row = frame.sort_values(["n_effective", "id"], ascending=[True, True]).iloc[0]
            elif how == "nearest_mae":
                target = float(frame["mae_scaled"].median())
                frame = frame.assign(_dist=(frame["mae_scaled"] - target).abs())
                row = frame.sort_values(["_dist", "id"]).iloc[0]
            else:
                raise ValueError(how)
            used.add(int(row["id"]))
            chosen.append(
                {
                    "cohort": cohort,
                    "role": role,
                    "id": int(row["id"]),
                    "mae_scaled": float(row["mae_scaled"]),
                    "coverage": float(row["coverage"]),
                    "n_sensor_days": int(row["n_sensor_days"]),
                    "n_effective": float(row["n_effective"]),
                    "top1_share": float(row["top1_share"]),
                    "quant_distance": float(row["quant_distance"]),
                    "tumor": row.get("tumor", np.nan),
                    "center": row.get("center", np.nan),
                    "clinical_group": row.get("clinical_group", np.nan),
                }
            )

        eligible = pool[np.isfinite(pool["mae_scaled"])]
        take("error_high", eligible, "max")
        take("error_low", eligible, "min")
        take("diversity_low", eligible[eligible["n_sensor_days"] >= 60], "min_eff")
        take("diversity_high", eligible, "max_eff")
        take("coverage_low", eligible[eligible["coverage"] < 0.50], "nearest_mae")
        take("coverage_high", eligible[eligible["coverage"] >= 0.80], "nearest_mae")
    return pd.DataFrame(chosen)


def write_example(row: pd.Series, days: dict, windows: pd.DataFrame, folder: Path) -> None:
    uid = int(row["id"])
    store = days[uid]
    day = pd.DataFrame(
        {
            "date": store["date"],
            "code": store["code"],
            "quant_distance": store["quant"],
            "mae_scaled_day": store["day_mae"],
            "sensor_observed": store["sensor_obs"],
        }
    )
    day.insert(0, "id", uid)
    day.insert(0, "cohort", row["cohort"])
    day.insert(0, "role", row["role"])
    stem = f"{row['cohort']}_{row['role']}_{uid}"
    day.to_csv(folder / f"{stem}_days.csv", index=False)
    win = windows[(windows["cohort"] == row["cohort"]) & (windows["id"] == uid)].copy()
    win.insert(0, "role", row["role"])
    win.to_csv(folder / f"{stem}_topics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), gridspec_kw={"height_ratios": [1.3, 1]})
    show = min(len(day), 180)
    t = np.arange(show)
    ax = axes[0]
    ax.scatter(
        t[day["sensor_observed"].to_numpy()[:show] == 1],
        day["code"].to_numpy()[:show][day["sensor_observed"].to_numpy()[:show] == 1],
        s=8,
        c=COLORS[row["cohort"]],
        label="sensor day",
    )
    ax.scatter(
        t[day["sensor_observed"].to_numpy()[:show] == 0],
        day["code"].to_numpy()[:show][day["sensor_observed"].to_numpy()[:show] == 0],
        s=8,
        c="#b0b0b0",
        label="no sensor",
    )
    ax.set_ylabel("VQ-VAE code")
    ax.set_title(
        f"{row['cohort']} {row['role']}  id {uid}  — day codes (first {show} densified days)\n"
        f"coverage {row['coverage']:.2f}   MAE {row['mae_scaled']:.3f}   "
        f"effective codes {row['n_effective']:.1f}"
    )
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, day["mae_scaled_day"].to_numpy()[:show], color="#222222", lw=0.8, alpha=0.8)
    ax2.set_ylabel("Day MAE (scaled, observed features)")

    ax = axes[1]
    if win.empty:
        ax.text(0.5, 0.5, "No complete 30-day window", ha="center", va="center")
        ax.set_axis_off()
    else:
        mat = win.sort_values("window_index")[[f"topic_{k}" for k in range(N_TOPICS)]].to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanmax(mat))))
        ax.set_yticks(range(len(win)))
        ax.set_yticklabels(win.sort_values("window_index")["window"].tolist())
        ax.set_xticks(range(N_TOPICS))
        ax.set_xticklabels([f"topic {k}" for k in range(N_TOPICS)])
        ax.set_title("LDA on complete 30-day windows from day 0 (not clinical F/U)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="probability")
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=140)
    plt.close(fig)


def make_figures(patients, features, topics, vocab_patients) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = feature_summary(features)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    features_order = ERROR_NAMES
    y = np.arange(len(features_order))
    height = 0.24
    offsets = np.linspace(-height, height, len(COHORT_ORDER))
    for offset, cohort in zip(offsets, COHORT_ORDER):
        sub = summary[summary["cohort"] == cohort].set_index("feature").loc[features_order]
        ax.barh(y + offset, sub["mae_scaled_pooled"], height=height, color=COLORS[cohort], label=cohort)
    ax.set_yticks(y)
    ax.set_yticklabels(features_order)
    ax.set_xlabel("Pooled MAE in the frozen scaled space (observed cells only)")
    ax.set_title("Reconstruction by variable")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "reconstruction_by_feature.png", dpi=150)
    plt.close(fig)

    hdo = patients[patients["cohort"] == "HDO"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, col, title in (
        (axes[0], "tumor", "HDO reconstruction by tumour"),
        (axes[1], "center", "HDO reconstruction by centre"),
    ):
        levels = [k for k, _ in sorted(hdo.groupby(col), key=lambda kv: -len(kv[1]))]
        data = [hdo.loc[hdo[col] == level, "mae_scaled"].dropna() for level in levels]
        ax.boxplot(data, vert=False, showfliers=False)
        ax.set_yticks(range(1, len(levels) + 1))
        ax.set_yticklabels([f"{level} (n={int((hdo[col] == level).sum())})" for level in levels])
        ax.set_xlabel("Patient MAE (mean of per-feature scaled MAEs)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_dir / "hdo_mae_by_tumor_center.png", dpi=150)
    plt.close(fig)

    if not topics.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
        mats = []
        prepared = []
        for cohort in COHORT_ORDER:
            sub = topics[topics["cohort"] == cohort].copy()
            sub["window_index"] = sub["window"].str[1:].astype(int)
            sub = sub.sort_values("window_index")
            prepared.append(sub)
            mats.append(sub[[f"topic_{k}" for k in range(N_TOPICS)]].to_numpy())
        peak = max(0.35, max(float(np.nanmax(mat)) for mat in mats if mat.size))
        for ax, cohort, sub, mat in zip(axes, COHORT_ORDER, prepared, mats):
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=peak)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(
                [f"{w} (n={n})" for w, n in zip(sub["window"], sub["n_patients"])]
            )
            ax.set_xticks(range(N_TOPICS))
            ax.set_xticklabels([str(k) for k in range(N_TOPICS)])
            ax.set_xlabel("Topic")
            ax.set_title(cohort)
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="mean probability")
        fig.suptitle("Frozen LDA on positional 30-day windows (not F/U)", y=1.02)
        fig.savefig(fig_dir / "topic_occupation_T0_T5.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for cohort, sub in vocab_patients.groupby("cohort"):
        ax.scatter(
            sub["n_sensor_days"],
            sub["n_effective"],
            s=12,
            alpha=0.45,
            c=COLORS[cohort],
            label=cohort,
            linewidths=0,
        )
    ax.set_xlabel("Observed sensor days")
    ax.set_ylabel("Effective number of codes (2^entropy)")
    ax.set_title("Vocabulary versus observed duration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "vocabulary_vs_duration.png", dpi=150)
    plt.close(fig)


def corpus_archive(hdo_ids: set[int], valset_ids: set[int]) -> dict:
    """SAP 17.4 archive: who was in the VQ-VAE fit, on which variables, and the id check."""
    splits = pd.read_csv(SPLITS_CSV)
    splits["user"] = splits["user"].astype(int)
    splits["service"] = splits["service"].astype(str)
    fit = splits[splits["vqvae"] == 1]
    oncology = fit["service"].eq("cnio_caspar") | fit["service"].str.startswith(("PMP_", "GM_"))
    mental_ids = set(splits["user"])
    fit_ids = set(fit["user"])
    date_min = date_max = None
    n_rows = 0
    for chunk in pd.read_csv(MENTAL_CSV, usecols=["user", "date_time"], chunksize=400_000):
        chunk = chunk[chunk["user"].astype(int).isin(fit_ids)]
        if chunk.empty:
            continue
        dates = pd.to_datetime(chunk["date_time"], errors="coerce").dropna()
        if dates.empty:
            continue
        n_rows += int(len(dates))
        lo, hi = dates.min(), dates.max()
        date_min = lo if date_min is None or lo < date_min else date_min
        date_max = hi if date_max is None or hi > date_max else date_max
    return {
        "source_csv": str(MENTAL_CSV),
        "split_file": str(SPLITS_CSV.relative_to(ROOT)),
        "input_variables": list(c.FEATURE_COLS),
        "objective": (
            "vqvae_a0 reconstructs the scaled inputs and quantises the encoder state. "
            "Its forward pass takes no psychiatric label. user_splits stores phq9, suicide "
            "and whodas in separate columns from the vqvae membership flag."
        ),
        "n_fit": int(len(fit)),
        "n_fit_oncology_service": int(oncology.sum()),
        "fit_oncology_services": {
            str(name): int(count)
            for name, count in fit.loc[oncology, "service"].value_counts().items()
        },
        "fit_date_min": None if date_min is None else str(pd.Timestamp(date_min).date()),
        "fit_date_max": None if date_max is None else str(pd.Timestamp(date_max).date()),
        "fit_rows": n_rows,
        "id_overlap_with_D178": [int(i) for i in sorted(mental_ids & set(hdo_ids))],
        "id_overlap_with_VALSET": [int(i) for i in sorted(mental_ids & set(valset_ids))],
        "id_check_limit": (
            "Pretraining ids are eb2 user ids. D178 and VAL-SET ids are S_RECORD_id. "
            "A zero intersection does not show that the same person is absent: this "
            "repository has no crosswalk. The fit set does include users from PMP and GM services."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    global model_scaler, lda_model, lda_dictionary

    print("Loading frozen models", flush=True)
    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")["a0"]
    model.eval()
    model_scaler = {key: value for key, value in np.load(c.DEFAULT_SCALER).items()}
    lda_model = LdaModel.load(str(LDA_PATH))
    lda_dictionary = corpora.Dictionary.load(str(DICT_PATH))
    missing_codes = sorted(set(map(str, range(256))) - set(lda_dictionary.token2id))

    kept, split_note = held_out_users()
    print(f"Mental held-out candidates: {len(kept)}", flush=True)
    mental = prepare_frame(MENTAL_CSV, set(kept["user"]))
    present = set(mental["user"].unique())
    kept = kept[kept["user"].isin(present)].copy()
    print(f"Mental held-out with daily rows: {len(kept)}", flush=True)
    mental_groups = scaled_groups(mental, model_scaler)
    mental_meta = kept.set_index("user")[["service", "clinical_group"]]

    hdo_meta = load_hdo_meta().set_index("id")
    hdo = prepare_frame(HDO_CSV, set(hdo_meta.index))
    hdo_ids = sorted(set(hdo["user"].unique()) & set(hdo_meta.index))
    print(f"HDO patients in the checked daily and the subject table: {len(hdo_ids)}", flush=True)
    hdo_groups = scaled_groups(hdo[hdo["user"].isin(hdo_ids)], model_scaler)

    val_clin = pd.read_excel(VALSET_CLINICAL).dropna(subset=["S_RECORD_id"])
    val_ids = set(val_clin["S_RECORD_id"].astype(int))
    val = prepare_frame(VALSET_CSV, val_ids)
    val_present = sorted(set(val["user"].unique()) & val_ids)
    print(f"VAL-SET with daily rows: {len(val_present)} of {len(val_ids)}", flush=True)
    val_groups = scaled_groups(val[val["user"].isin(val_present)], model_scaler)
    val_service = (
        val[val["user"].isin(val_present)]
        .groupby("user")["service"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
    val_meta = pd.DataFrame(
        {
            "service": val_service,
            "center": val_service.astype(str).str.replace(r"^PMP_", "", regex=True),
            "clinical_group": "VAL-SET",
        }
    )

    mental_result = run_cohort("mental_held_out", mental_groups, model, mental_meta)
    hdo_result = run_cohort("HDO", hdo_groups, model, hdo_meta)
    val_result = run_cohort("VAL-SET", val_groups, model, val_meta)

    patients = pd.concat(
        [mental_result["patients"], hdo_result["patients"], val_result["patients"]],
        ignore_index=True,
    )
    features = pd.concat(
        [mental_result["features"], hdo_result["features"], val_result["features"]],
        ignore_index=True,
    )
    windows = pd.concat(
        [mental_result["windows"], hdo_result["windows"], val_result["windows"]],
        ignore_index=True,
    )

    patients.to_csv(OUT / "patient_summary.csv", index=False)
    features.to_csv(OUT / "reconstruction_by_patient_feature.csv", index=False)
    feature_summary(features).to_csv(OUT / "reconstruction_by_feature.csv", index=False)
    stratified_mae(patients, features, "tumor").to_csv(OUT / "hdo_by_tumor.csv", index=False)
    stratified_mae(patients, features, "center").to_csv(OUT / "hdo_by_center.csv", index=False)
    coverage_table(patients).to_csv(OUT / "coverage_and_age.csv", index=False)
    feature_presence(features, patients).to_csv(OUT / "feature_presence.csv", index=False)
    feature_levels(features).to_csv(OUT / "feature_levels_scaled.csv", index=False)
    kept.groupby(["clinical_group", "service"]).size().reset_index(name="n_patients").to_csv(
        OUT / "mental_service_mix.csv", index=False
    )
    windows.to_csv(OUT / "topic_windows_by_patient.csv", index=False)
    topics = topic_summary(windows)
    topics.to_csv(OUT / "topic_occupation_by_window.csv", index=False)
    vocab = vocabulary_table(patients)
    vocab.to_csv(OUT / "vocabulary_by_cohort.csv", index=False)
    patients[
        [
            "cohort",
            "id",
            "n_sensor_days",
            "n_unique_codes",
            "top1_share",
            "top5_share",
            "entropy_bits",
            "n_effective",
            "observed_top_code",
            "gap_top_code",
            "gap_top_share",
        ]
    ].to_csv(OUT / "vocabulary_by_patient.csv", index=False)

    examples = select_examples(patients)
    example_dir = OUT / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    examples.to_csv(example_dir / "selection.csv", index=False)
    days = {**mental_result["days"], **hdo_result["days"], **val_result["days"]}
    for row in examples.dropna(subset=["id"]).itertuples(index=False):
        write_example(pd.Series(row._asdict()), days, windows, example_dir)

    make_figures(patients, features, topics, patients)
    archive = corpus_archive(set(hdo_ids), val_ids)
    archive["valset_clinical_n"] = len(val_ids)
    archive["valset_with_daily"] = [int(i) for i in val_present]
    archive["valset_without_daily"] = [int(i) for i in sorted(val_ids - set(val_present))]
    archive["valset_source"] = str(VALSET_CLINICAL.relative_to(ROOT))
    archive["valset_cutoff"] = "2026-09-03"
    archive["sensitivity_exclude"] = sorted(VALSET_SENSITIVITY_EXCLUDE)
    (OUT / "pretraining_corpus.json").write_text(json.dumps(archive, indent=2))
    val_patients = patients[patients["cohort"] == "VAL-SET"]
    sensitivity = pd.DataFrame(
        [
            {"subset": "all_with_daily", **{f"mae_{k}": v for k, v in iqr(val_patients["mae_scaled"]).items()}, **{f"quant_{k}": v for k, v in iqr(val_patients["quant_distance"]).items()}},
            {"subset": "excluding_21003_91021_31017", **{f"mae_{k}": v for k, v in iqr(val_patients.loc[~val_patients["sap_sensitivity_exclude"], "mae_scaled"]).items()}, **{f"quant_{k}": v for k, v in iqr(val_patients.loc[~val_patients["sap_sensitivity_exclude"], "quant_distance"]).items()}},
        ]
    )
    sensitivity.to_csv(OUT / "valset_sensitivity.csv", index=False)
    provenance = {
        "question": "How the frozen external model represents D178 and VAL-SET, against mental patients not used in the VQ-VAE fit. SAP 17.4 and the reviewer note are both applied.",
        "does_not_show": [
            "Absence of bias.",
            "Clinical relevance of a reconstruction or of a topic-distribution difference.",
            "That the cohorts should occupy different codes or topics for the representation to be usable.",
        ],
        "vqvae": "models/vq-vae/vqvae_a0.pt",
        "scaler": "models/vq-vae/preprocessing_scaler_vqvae.npz",
        "preprocessing": "Same clip bounds, log1p and frozen scaler as VQ-VAE inference. Weekend is filled from the calendar and is excluded from the error.",
        "observed": "A cell counts only when the value is present after the frozen clip. The patient MAE averages the eight continuous features and leaves practiced_sport out, because that binary flag is not on the same scale. Quantization distance is the L2 from the encoder state to the chosen code, on days with at least one of those nine values.",
        "coverage_definition": "The daily extracts do not store empty calendar days, so sensor-days/span is close to 1. The coverage that can be compared is the length of the record and the fraction of that record on which each variable is present (feature_presence.csv). This is not the study-calendar coverage used in the alarm audit.",
        "mental_reference": split_note,
        "mental_csv": str(MENTAL_CSV),
        "n_mental_held_out_with_daily": int((patients.cohort == "mental_held_out").sum()),
        "hdo_csv": str(HDO_CSV.relative_to(ROOT)),
        "n_hdo": int((patients.cohort == "HDO").sum()),
        "hdo_tumor": "Subjects_data.Cohort",
        "hdo_center": "Hospital on the stratified VAL-2 partition, short name; patients missing from that list take the modal hospital of the same id digit.",
        "hdo_age": "Subjects_data.Age. Age is not in the mental daily extract or in user_splits, so the age contrast is HDO-only.",
        "lda": "models/lda/LDA_model_vdec25.gensim and dictionary_LDA_vdec25.dict, not refit.",
        "windows": "Complete non-overlapping 30-day blocks from the first densified day, labeled T0–T5 in that order. Drop the incomplete tail. These positions are not the oncological F/U.",
        "lda_codes_absent_from_dictionary": missing_codes,
        "vocabulary": "Shannon entropy in bits on sensor-observed days only. Effective number is 2^entropy. Gap-day codes are reported separately so a missingness code is visible.",
        "examples": {
            "min_sensor_days": MIN_DAYS_EXAMPLE,
            "order": [
                "error_high: highest patient MAE",
                "error_low: lowest patient MAE",
                "diversity_low: lowest effective number among patients with at least 60 sensor days",
                "diversity_high: highest effective number",
                "coverage_low: coverage < 0.50, patient nearest the median MAE of that subset",
                "coverage_high: coverage >= 0.80, patient nearest the median MAE of that subset",
            ],
            "no_reuse_within_cohort": True,
            "day_panel": "VQ-VAE code per densified day",
            "window_panel": "LDA probabilities on T0–T5",
        },
        "populations": {
            "mental_held_out": "user_splits.vqvae == 0, oncology services removed. This is the reviewer external reference.",
            "HDO": "D178 digital population: Subjects_data ids present in DB1 daily_summary_checked (175 of 178).",
            "VAL-SET": "The 48 candidates in the DB3 clinical date table, cutoff 2026-09-03, with no id overlap with the 178. Error and topic metrics use the 42 who have a daily row.",
        },
        "sap_17_4": {
            "archive": "pretraining_corpus.json",
            "shifts": "feature levels, feature presence, reconstruction, codebook use, and topic posterior entropy, on the same clipped and scaled variables.",
            "codebook_vs_topics": "256 VQ-VAE codes are the vocabulary tables. Six LDA topics are the window tables.",
            "ablations_not_run": "A simple feature model and an oncology-only representation fitted inside each training fold are separate models in SAP 17.4. This run does not fit them.",
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
