#!/usr/bin/env python3
"""HDO-only k-means alternative to the external-vocabulary alarm.

One model, fixed before looking at its progression results. K-means with K=6
on the eight continuous daily variables, inside the SAP section 12 partition
already stored for the external-vocabulary alarm. Held-out patients and
VAL-SET are assigned to clusters that were fitted without them. They do not
enter the scaler, the 3/3 labels, or the Youden cut.

    python scripts/vqvae/run_hdo_only_kmeans.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/lda"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/alarm"))

from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from run_val2_anticipation import anticipation  # noqa: E402
from run_val2_blocks import block_spans  # noqa: E402
from run_val2_fu import (  # noqa: E402
    MIN_CLINICAL_DAYS,
    assign_labels,
    drop_outside_comparison,
    mark_comparison,
    rendimiento,
    topic_summary,
    youden_cut,
)
from run_val2_stratified import stratified_groups  # noqa: E402
from vqvae import constants as c  # noqa: E402
from vqvae.data import replace_out_of_bounds_with_nan  # noqa: E402

VARIABLES = [
    "sleep_start",
    "location_distance",
    "location_time_home",
    "sleep_duration",
    "activity_walking",
    "app_usage_total",
    "location_clusters_count",
    "steps_steps_total",
]
POSITIVE = [name for name in VARIABLES if name != "sleep_start"]
K = 6
N_OUTER = 5
N_INNER = 4
SAP_BASE = 20261031
N_REPEATS = 10

DAILY_CSV = ROOT / "data/daily_summaries/DB1/daily_summary_checked.csv"
WINDOWS_CSV = ROOT / "data/revision/user_embeddings_checked_DB1_a0_summary.csv"
AUDIT_XLSX = ROOT / "data/processed/DCABPs/DB1/178_audit_oct_2_2026_topics.xlsx"
VALSET_XLSX = ROOT / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026.xlsx"
SAP_ROOT = ROOT / "results/lda/val_2/anidada_estratificada_sap10"
OUT = ROOT / "results/vq-vae/training_provenance/hdo_only"

# Filled in the parent before the fork, then visible to each worker.
DAYS: dict | None = None
CLINICAL: pd.DataFrame | None = None
PARTITIONS: dict[int, pd.DataFrame] | None = None
SEEDS: pd.DataFrame | None = None
PAIRS: set[tuple[int, int]] | None = None


def build_days() -> dict:
    """Dense day matrix on the same calendar windows as the embedding sequences."""
    windows = pd.read_csv(WINDOWS_CSV)
    windows["user_id"] = windows["user_id"].astype(int)
    windows["date_start"] = pd.to_datetime(windows["date_start"])
    windows["date_end"] = pd.to_datetime(windows["date_end"])
    span = (windows["date_end"] - windows["date_start"]).dt.days + 1
    if not (span == windows["n_days"]).all():
        raise SystemExit("embedding window length does not match date_start to date_end")

    daily = pd.read_csv(DAILY_CSV, usecols=["user", "date_time", *VARIABLES])
    daily["user"] = daily["user"].astype(int)
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    daily = replace_out_of_bounds_with_nan(daily)
    if daily.duplicated(["user", "date_time"]).any():
        daily = daily.groupby(["user", "date_time"], as_index=False)[VARIABLES].mean()

    pieces = []
    owners = []
    absent = 0
    for row in windows.sort_values("user_id").itertuples(index=False):
        idx = pd.date_range(row.date_start, row.date_end, freq="D")
        sub = daily.loc[daily["user"] == row.user_id].drop_duplicates("date_time").set_index("date_time")
        absent += int((~idx.isin(sub.index)).sum())
        aligned = sub.reindex(idx)[VARIABLES].to_numpy(dtype=float)
        if len(aligned) != int(row.n_days):
            raise SystemExit(f"{row.user_id}: window length {len(aligned)} != {row.n_days}")
        pieces.append(aligned)
        owners.append(np.full(len(aligned), int(row.user_id), dtype=np.int64))

    values = np.vstack(pieces)
    user = np.concatenate(owners)
    observed = np.isfinite(values)
    for name in POSITIVE:
        column = VARIABLES.index(name)
        ok = observed[:, column]
        values[ok, column] = np.log1p(values[ok, column])

    uniq, start, counts = np.unique(user, return_index=True, return_counts=True)
    slices = {
        int(uid): (int(begin), int(begin + count))
        for uid, begin, count in zip(uniq, start, counts)
    }
    all_missing = ~observed.any(axis=1)
    coverage = {
        "patients": int(len(slices)),
        "days": int(len(values)),
        "days_with_no_daily_row": int(absent),
        "days_all_eight_missing_after_clip": int(all_missing.sum()),
        "fraction_all_eight_missing_after_clip": float(all_missing.mean()),
        "observed_fraction_by_variable": {
            name: float(observed[:, j].mean()) for j, name in enumerate(VARIABLES)
        },
    }
    return {"X": values, "user": user, "slices": slices, "all_missing": all_missing, "coverage": coverage}


def load_clinical() -> pd.DataFrame:
    audit = pd.read_excel(AUDIT_XLSX).dropna(subset=["id"]).copy()
    audit["id"] = audit["id"].astype(int)
    keep = ["id", "Evento_DB1", "Tiempo_obs_DB1", "Tiempo_obs_digital", "Fecha_Fin_DB1_Actualizado"]
    clinical = audit[keep].drop_duplicates("id").set_index("id")
    return clinical


def load_partitions() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    seeds = pd.read_csv(SAP_ROOT / "seeds.csv")
    partitions = {}
    for repeat in range(N_REPEATS):
        frame = pd.read_csv(SAP_ROOT / f"rep_{repeat:02d}" / "partitions.csv")
        frame["id"] = frame["id"].astype(int)
        frame["outer_fold"] = frame["outer_fold"].astype(int)
        frame["event"] = frame["event"].astype(int)
        frame["tumor_group"] = frame["tumor_group"].astype(int)
        partitions[repeat] = frame
    return seeds, partitions


def verify_partitions(seeds: pd.DataFrame, partitions: dict[int, pd.DataFrame]) -> None:
    """The stored SAP folds must be the folds these seeds still produce."""
    for repeat, frame in partitions.items():
        seed = seeds.loc[seeds["repeat"] == repeat].iloc[0]
        external = frame[frame["side"] == "external"][["id", "event", "tumor_group"]].drop_duplicates("id")
        cohort = external.rename(columns={"tumor_group": "tumor"}).copy()
        cohort["centre"] = cohort["id"].astype(str).str[0].astype(int)
        outer_groups = stratified_groups(cohort, N_OUTER, int(seed.outer_seed))
        saved_external = frame[frame["side"] == "external"]
        for fold, members in enumerate(outer_groups):
            saved = set(saved_external.loc[saved_external["outer_fold"] == fold, "id"].astype(int))
            if saved != set(int(i) for i in members):
                raise SystemExit(f"repeat {repeat} outer fold {fold} does not match seed {int(seed.outer_seed)}")
            train = cohort.loc[~cohort["id"].isin(members)].copy()
            inner_groups = stratified_groups(train, N_INNER, int(seed.inner_seed) + fold)
            saved_train = frame[(frame["side"] == "train") & (frame["outer_fold"] == fold)]
            for inner, held in enumerate(inner_groups):
                got = set(saved_train.loc[saved_train["inner_fold"] == inner, "id"].astype(int))
                if got != set(int(i) for i in held):
                    raise SystemExit(f"repeat {repeat} outer {fold} inner {inner} does not match its seed")


def sap_landmark_pairs() -> set[tuple[int, int]]:
    """The 1013 pairs frozen in the external-vocabulary alarm.

    decision_points.csv has since replaced six of those pairs. The comparison
    uses the pairs the vocabulary alarm was actually scored on.
    """
    frame = pd.read_csv(SAP_ROOT / "rep_00" / "alarma" / "decisiones.csv")
    external = frame[(frame["role"] == "external_test") & (frame["in_comparison"] == 1)].copy()
    external = external[external["outcome"].astype(str).isin(["0", "1"])]
    pairs = set(zip(external["id"].astype(int), external["block"].astype(int)))
    if len(pairs) != 1013:
        raise SystemExit(f"expected 1013 frozen landmarks, found {len(pairs)}")
    return pairs


def verify_valset(patient_ids: set[int]) -> dict:
    valset = pd.read_excel(VALSET_XLSX).dropna(subset=["S_RECORD_id"])
    val_ids = set(valset["S_RECORD_id"].astype(int))
    overlap = sorted(patient_ids & val_ids)
    if overlap:
        raise SystemExit(f"VAL-SET identifiers in the fitting cohort: {overlap}")
    return {"n_valset": int(len(val_ids)), "overlap_with_fitting_cohort": overlap}


def _scale(values: np.ndarray, row_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = values[row_mask]
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scaled = (values - center) / scale
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return scaled, center, scale


def _fit_kmeans(scaled: np.ndarray, row_mask: np.ndarray, seed: int) -> KMeans:
    model = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=int(seed),
        algorithm="lloyd",
    )
    model.fit(scaled[row_mask])
    return model


def _dominant(labels: np.ndarray) -> int:
    return int(np.bincount(labels, minlength=K).argmax())


def _patient_rows(fit_ids: np.ndarray, labels: np.ndarray, clinical: pd.DataFrame) -> pd.DataFrame:
    days = DAYS
    assert days is not None
    rows = []
    for pid in fit_ids:
        start, end = days["slices"][int(pid)]
        topic = _dominant(labels[start:end])
        rows.append({"id": int(pid), "topic": topic})
    frame = pd.DataFrame(rows).join(clinical, on="id")
    frame["t_pfs"] = frame["Tiempo_obs_DB1"].astype(float)
    frame["event"] = frame["Evento_DB1"].astype(int)
    return frame.loc[frame["t_pfs"] >= MIN_CLINICAL_DAYS].copy()


def _blocks_for(
    ids: np.ndarray,
    labels: np.ndarray,
    unfavorable: dict[int, str] | None,
    *,
    outer: int,
    inner: str,
    role: str,
    model_name: str,
) -> list[dict]:
    days = DAYS
    assert days is not None
    unfavorable_ids = [] if unfavorable is None else [topic for topic, label in unfavorable.items() if label == "U"]
    rows = []
    for pid in ids:
        start, end = days["slices"][int(pid)]
        sequence = labels[start:end]
        for block, (a, b) in enumerate(block_spans(len(sequence)), start=1):
            window = sequence[a:b]
            counts = np.bincount(window, minlength=K).astype(float)
            props = counts / counts.sum()
            if unfavorable is None:
                puf = np.nan
            else:
                puf = float(props[unfavorable_ids].sum()) if unfavorable_ids else 0.0
            row = {
                "id": int(pid),
                "outer_fold": int(outer),
                "inner_fold": inner,
                "role": role,
                "model": model_name,
                "block": int(block),
                "n_days": int(len(window)),
                "pUF": puf,
            }
            for topic in range(K):
                row[f"topic_{topic}"] = float(props[topic])
            rows.append(row)
    return rows


def _decisions(blocks: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = blocks.groupby(["id", "outer_fold", "inner_fold", "role", "model"], sort=True)
    for (pid, outer, inner, role, model_name), frame in grouped:
        frame = frame.sort_values("block")
        if len(frame) < 3:
            continue
        t_digital = float(clinical.loc[int(pid), "Tiempo_obs_digital"])
        event = int(clinical.loc[int(pid), "Evento_DB1"])
        fecha = clinical.loc[int(pid), "Fecha_Fin_DB1_Actualizado"]
        values = frame["pUF"].to_numpy(float)
        block_no = frame["block"].to_numpy(int)
        for i in range(2, len(frame)):
            window = values[i - 2 : i + 1]
            day = float(block_no[i]) * 30.0
            rows.append({
                "id": int(pid),
                "outer_fold": int(outer),
                "inner_fold": inner,
                "role": role,
                "model": model_name,
                "block": int(block_no[i]),
                "decision_day": day,
                "puf_1": float(window[0]),
                "puf_2": float(window[1]),
                "puf_3": float(window[2]),
                "score": float(np.sum(window)),
                "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
                "event": event,
                "t_digital": t_digital,
                "fecha_fin": str(fecha)[:10],
            })
    return pd.DataFrame(rows)


def _one_model(
    fit_ids: np.ndarray,
    score_ids: np.ndarray,
    seed: int,
    *,
    outer: int,
    inner: str,
    role: str,
    model_name: str,
) -> tuple[list[dict], list[dict], dict]:
    days = DAYS
    clinical = CLINICAL
    assert days is not None and clinical is not None
    row_mask = np.isin(days["user"], fit_ids)
    scaled, center, scale = _scale(days["X"], row_mask)
    model = _fit_kmeans(scaled, row_mask, seed)
    labels = model.predict(scaled)
    patients = _patient_rows(fit_ids, labels, clinical)
    summaries = [topic_summary(patients, topic) for topic in range(K)]
    mapping, rule = assign_labels(summaries)
    table = []
    for summary in summaries:
        table.append({
            "model": model_name,
            "outer_fold": int(outer),
            "inner_fold": inner,
            "n_train_patients_in_curve": int(len(patients)),
            "topic": summary["topic"],
            "n": summary["n"],
            "events": summary["events"],
            "pfs_365": summary["pfs_365"],
            "rmst_365": summary["rmst_365"],
            "supported_365": summary["supported_365"],
            "etiqueta": "" if mapping is None else mapping[summary["topic"]],
            "criterio": rule,
        })
    zero_cluster = int(model.predict(np.zeros((1, len(VARIABLES))))[0])
    train_labels = labels[row_mask]
    record = {
        "model": model_name,
        "outer_fold": int(outer),
        "inner_fold": inner,
        "role_scored": role,
        "n_fit_patients": int(len(fit_ids)),
        "n_fit_days": int(row_mask.sum()),
        "fraction_fit_days_all_missing": float(days["all_missing"][row_mask].mean()),
        "zero_cluster": zero_cluster,
        "fraction_fit_days_in_zero_cluster": float((train_labels == zero_cluster).mean()),
        "labelled": mapping is not None,
        "criterio": rule,
        "inertia": float(model.inertia_),
        "scaler_center": [float(v) for v in center],
        "scaler_scale": [float(v) for v in scale],
    }
    blocks = _blocks_for(
        score_ids, labels, mapping, outer=outer, inner=inner, role=role, model_name=model_name
    )
    return table, blocks, record


def run_repeat(repeat: int) -> str:
    partitions = PARTITIONS[repeat]
    seed = int(SEEDS.loc[SEEDS["repeat"] == repeat, "lda_seed"].iloc[0])
    clinical = CLINICAL
    assert clinical is not None and PAIRS is not None
    folder = OUT / f"rep_{repeat:02d}"
    alarm = folder / "alarma"
    fu = folder / "fu"
    alarm.mkdir(parents=True, exist_ok=True)
    fu.mkdir(parents=True, exist_ok=True)

    tables: list[dict] = []
    blocks: list[dict] = []
    records: list[dict] = []
    for outer in range(N_OUTER):
        train = partitions[(partitions["outer_fold"] == outer) & (partitions["side"] == "train")]
        test_ids = partitions.loc[
            (partitions["outer_fold"] == outer) & (partitions["side"] == "external"), "id"
        ].to_numpy(int)
        for inner in range(N_INNER):
            fit_ids = train.loc[train["inner_fold"] != inner, "id"].to_numpy(int)
            held_ids = train.loc[train["inner_fold"] == inner, "id"].to_numpy(int)
            name = f"outer_{outer}_inner_{inner}"
            table, block_rows, record = _one_model(
                fit_ids, held_ids, seed, outer=outer, inner=str(inner), role="held_out", model_name=name
            )
            tables.extend(table)
            blocks.extend(block_rows)
            records.append(record)
        fit_ids = train["id"].to_numpy(int)
        name = f"outer_{outer}_refit"
        table, block_rows, record = _one_model(
            fit_ids, test_ids, seed, outer=outer, inner="refit", role="external_test", model_name=name
        )
        tables.extend(table)
        blocks.extend(block_rows)
        records.append(record)
        print(f"repeat {repeat} outer {outer} fitted", flush=True)

    pd.DataFrame(tables).to_csv(fu / "km_por_cluster.csv", index=False)
    pd.DataFrame(records).to_csv(folder / "modelos.csv", index=False)
    reserved = (
        pd.DataFrame(records).loc[lambda frame: ~frame["labelled"], ["model", "outer_fold", "inner_fold", "criterio"]]
    )
    reserved.to_csv(fu / "modelos_reservados.csv", index=False)

    decision_df = drop_outside_comparison(mark_comparison(_decisions(pd.DataFrame(blocks), clinical), PAIRS))
    raw = decision_df[
        (decision_df["role"] == "external_test")
        & (decision_df["in_comparison"] == 1)
        & decision_df["outcome"].isin(["0", "1"])
    ]
    n_pos = int((raw["outcome"] == "1").sum())
    n_neg = int((raw["outcome"] == "0").sum())
    if len(raw) != 1013 or n_pos != 155 or n_neg != 858:
        raise RuntimeError(
            f"repeat {repeat}: external landmarks {len(raw)} ({n_pos} positive, {n_neg} negative), expected 1013 (155, 858)"
        )

    cuts = []
    threshold_of = {}
    held = decision_df[decision_df["role"] == "held_out"]
    for outer, frame in held.groupby("outer_fold"):
        usable = frame[
            frame["in_comparison"].eq(1)
            & frame["outcome"].isin(["0", "1"])
            & np.isfinite(frame["score"])
        ]
        found = None if usable.empty else youden_cut(
            usable["score"].to_numpy(float), usable["outcome"].astype(int).to_numpy()
        )
        if found is None:
            cuts.append({"outer_fold": int(outer), "threshold": np.nan, "reason": "sin positivos o sin negativos"})
            continue
        found["outer_fold"] = int(outer)
        found["tie_rule"] = "menor corte"
        found["time"] = "Tiempo_obs_digital"
        found["horizon_days"] = HORIZON_DAYS
        found["comparison"] = "1013 every 30 days"
        cuts.append(found)
        threshold_of[int(outer)] = found["threshold"]
    cut_df = pd.DataFrame(cuts)
    cut_df.to_csv(alarm / "cortes.csv", index=False)

    decision_df["threshold"] = decision_df["outer_fold"].map(threshold_of).astype(float)
    decision_df.loc[~np.isfinite(decision_df["threshold"]), "threshold"] = np.nan
    decision_df["alarm"] = decision_df["score"] >= decision_df["threshold"]
    decision_df.loc[decision_df["threshold"].isna() | ~np.isfinite(decision_df["score"]), "alarm"] = pd.NA
    decision_df.to_csv(alarm / "decisiones.csv", index=False)
    metrics = rendimiento(decision_df.dropna(subset=["threshold"]), ["outer_fold", "role"])
    metrics.to_csv(alarm / "rendimiento_por_vuelta.csv", index=False)

    alarm_bool = decision_df.copy()
    alarm_bool["alarm"] = alarm_bool["alarm"].map({True: True, False: False, "True": True, "False": False})
    by_patient, summary = anticipation(alarm_bool, "outer_fold")
    by_patient.to_csv(alarm / "anticipacion_por_paciente.csv", index=False)
    summary.to_csv(alarm / "anticipacion_resumen.csv", index=False)
    external = metrics[metrics["role"] == "external_test"]
    print(
        f"repeat {repeat} external folds {len(external)} "
        f"AUROC {external['AUROC'].mean():.3f} reserved {len(reserved)}",
        flush=True,
    )
    return f"rep_{repeat:02d} done"


def _repeat_metrics(folder: Path) -> dict:
    metrics = pd.read_csv(folder / "alarma" / "rendimiento_por_vuelta.csv")
    external = metrics[metrics["role"] == "external_test"]
    leads = pd.read_csv(folder / "alarma" / "anticipacion_por_paciente.csv")["first_tp_lead"].dropna()
    return {
        "n_external_folds": int(len(external)),
        "AUROC": float(external["AUROC"].mean()) if len(external) else np.nan,
        "sensitivity": float(external["sensitivity"].mean()) if len(external) else np.nan,
        "specificity": float(external["specificity"].mean()) if len(external) else np.nan,
        "PPV": float(external["PPV"].mean()) if len(external) else np.nan,
        "NPV": float(external["NPV"].mean()) if len(external) else np.nan,
        "detected": int(external["patients_detected"].sum()) if len(external) else 0,
        "false_alarm": int(external["patients_false_alarm"].sum()) if len(external) else 0,
        "n_event_patients": int(external["patients_with_event"].sum()) if len(external) else 0,
        "lead": float(np.median(leads)) if len(leads) else np.nan,
        "n_detected_leads": int(len(leads)),
    }


def external_reference() -> pd.DataFrame:
    rows = []
    for repeat in range(N_REPEATS):
        row = _repeat_metrics(SAP_ROOT / f"rep_{repeat:02d}")
        row["repeat"] = repeat
        row["arm"] = "external_vocabulary"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize() -> None:
    hdo_rows = []
    fold_frames = []
    for repeat in range(N_REPEATS):
        folder = OUT / f"rep_{repeat:02d}"
        row = _repeat_metrics(folder)
        row["repeat"] = repeat
        row["arm"] = "hdo_only_kmeans"
        hdo_rows.append(row)
        folds = pd.read_csv(folder / "alarma" / "rendimiento_por_vuelta.csv")
        folds.insert(0, "repeat", repeat)
        fold_frames.append(folds)
    hdo = pd.DataFrame(hdo_rows)
    reference = external_reference()
    per_repeat = hdo.merge(reference, on="repeat", suffixes=("_hdo", "_external"))
    for name in ("AUROC", "sensitivity", "specificity", "lead"):
        per_repeat[f"{name}_difference"] = per_repeat[f"{name}_hdo"] - per_repeat[f"{name}_external"]
    per_repeat.to_csv(OUT / "comparacion_por_repeticion.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUT / "rendimiento_por_vuelta.csv", index=False)
    hdo.to_csv(OUT / "rendimiento_medio_por_repeticion.csv", index=False)

    metric_names = ["AUROC", "sensitivity", "specificity", "lead"]
    summary = {
        "hdo_only_mean_of_repeat_means": {name: float(hdo[name].mean()) for name in metric_names},
        "hdo_only_sd_of_repeat_means": {name: float(hdo[name].std(ddof=1)) for name in metric_names},
        "external_vocabulary_mean_of_repeat_means": {
            name: float(reference[name].mean()) for name in metric_names
        },
        "external_vocabulary_sd_of_repeat_means": {
            name: float(reference[name].std(ddof=1)) for name in metric_names
        },
        "difference_hdo_minus_external": {
            name: float(hdo[name].mean() - reference[name].mean()) for name in metric_names
        },
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    global DAYS, CLINICAL, PARTITIONS, SEEDS, PAIRS
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building the day matrix", flush=True)
    DAYS = build_days()
    patient_ids = set(DAYS["slices"])
    seeds, partitions = load_partitions()
    if set().union(*[set(frame["id"]) for frame in partitions.values()]) != patient_ids:
        raise SystemExit("partition identifiers and embedding windows are not the same 175 patients")
    print("Checking that the stored folds are the SAP seeds", flush=True)
    verify_partitions(seeds, partitions)
    valset = verify_valset(patient_ids)
    CLINICAL = load_clinical()
    missing = sorted(patient_ids - set(CLINICAL.index))
    if missing:
        raise SystemExit(f"patients without clinical follow-up: {missing}")
    PARTITIONS = partitions
    SEEDS = seeds
    PAIRS = sap_landmark_pairs()
    manifest = {
        "status": "fitted",
        "algorithm": "k-means",
        "k": K,
        "init": "k-means++",
        "n_init": 1,
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": "lda_seed of the SAP repeat. One initialisation, not a search over starts.",
        "variables": VARIABLES,
        "clip": "scripts/vqvae/constants.py CLIP_INFO",
        "transform": (
            "log1p on the seven positive variables. Mean and standard deviation of the "
            "observed training-fold values. A missing value is 0 after that scaling, the training mean."
        ),
        "days": (
            "The calendar window of user_embeddings_checked_DB1_a0_summary.csv. "
            "Its length matches the embedding sequence used by the external-vocabulary alarm."
        ),
        "partition_source": str(SAP_ROOT.relative_to(ROOT)),
        "partition_check": "outer and inner folds recomputed from seeds.csv and matched to partitions.csv",
        "sap_base_seed": SAP_BASE,
        "outcome_rule": "3/3 Kaplan-Meier at 365 days on training patients with Tiempo_obs_DB1 >= 28",
        "youden": "one cut per outer fold, on inner held-out decisions among the 1013 landmarks, lowest cut if tied",
        "refit": (
            "Each outer test uses a k-means fitted on all 140 training patients of that fold. "
            "The 35 external patients are assigned only."
        ),
        "valset": valset,
        "accumulation_blocks": 3,
        "horizon_days": HORIZON_DAYS,
        "comparison_landmarks": (
            "1013 (id, block) pairs frozen in the external-vocabulary decisiones.csv. "
            "The current decision_points.csv differs by six pairs, so it is not used here. "
            "Each repeat has 155 positive and 858 negative external landmarks."
        ),
        "coverage": DAYS["coverage"],
        "tie_on_dominant_cluster": "lowest cluster index",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    seeds.to_csv(OUT / "seeds.csv", index=False)

    import multiprocessing as mp

    failures = []
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = {pool.submit(run_repeat, repeat): repeat for repeat in range(N_REPEATS)}
        for future in as_completed(futures):
            repeat = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((repeat, str(exc)))
                print(f"rep_{repeat:02d} FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} repeats failed: {failures}")
    summarize()


if __name__ == "__main__":
    main()
