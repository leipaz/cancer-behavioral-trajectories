#!/usr/bin/env python3
"""Oncology-only alarm: k-means on HDO days, then LDA, then the same alarm.

SAP 17.4 asks for an oncology-only representation fitted inside each development
training fold, to see whether the alarm depends on the external vocabulary.
The external model is the VQ-VAE. This script replaces that step with k-means
(K=6) on the eight HDO variables, then fits the same LDA (K=6) on the resulting
day-type sequences. Labels and the Youden cut stay inside the training fold.

Development is scored on the 1013 landmarks of the external-vocabulary alarm.
VAL-SET is scored afterwards with those already fixed models and cuts. It does
not enter the fit, the labels, or the choice of cut.

    python scripts/vqvae/run_hdo_only_kmeans_lda.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/lda"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/alarm"))

from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from run_hdo_only_kmeans import (  # noqa: E402
    AUDIT_XLSX,
    K,
    N_INNER,
    N_OUTER,
    N_REPEATS,
    SAP_BASE,
    SAP_ROOT,
    VARIABLES,
    _fit_kmeans,
    _scale,
    build_days,
    load_clinical,
    load_partitions,
    sap_landmark_pairs,
    verify_partitions,
    verify_valset,
)
from run_val2_anticipation import one_patient  # noqa: E402
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
from run_val2_nested import ITERATIONS, PASSES, fit_lda, topic_distribution  # noqa: E402
from vqvae.data import replace_out_of_bounds_with_nan  # noqa: E402

OUT = ROOT / "results" / "vq-vae" / "training_provenance" / "hdo_only_lda"
VALSET_XLSX = ROOT / "data/processed/DCABPs/DB3/DB3_oct_2_2026_topics.xlsx"
VALSET_DAILY = ROOT / "data/daily_summaries/DB3/daily_summary_checked.csv"
POSITIVE = [name for name in VARIABLES if name != "sleep_start"]

DAYS: dict | None = None
VALSET: dict | None = None
CLINICAL: pd.DataFrame | None = None
PARTITIONS: dict[int, pd.DataFrame] | None = None
SEEDS: pd.DataFrame | None = None
PAIRS: set[tuple[int, int]] | None = None


def build_valset(reference_columns: np.ndarray) -> dict:
    """Dense VAL-SET windows on the official DB3 clock. No fitting happens here."""
    table = pd.read_excel(VALSET_XLSX).dropna(subset=["S_RECORD_id"]).copy()
    table["id"] = table["S_RECORD_id"].astype(int)
    table["inicio"] = pd.to_datetime(table["Inicio_datos_DB3"])
    table["fin"] = pd.to_datetime(table["Fin_datos_DB3"])
    table["event"] = table["Estado"].eq("PD Segundo corte").astype(int)
    stored = pd.to_numeric(table["Tiempo_obs_digital"], errors="coerce")
    fallback = (table["fin"] - table["inicio"]).dt.days.astype(float)
    table["t_digital"] = stored.where(stored.notna(), fallback)

    daily = pd.read_csv(VALSET_DAILY, usecols=["user", "date_time", *VARIABLES])
    daily["user"] = daily["user"].astype(int)
    daily["date_time"] = pd.to_datetime(daily["date_time"]).dt.normalize()
    daily = replace_out_of_bounds_with_nan(daily)
    if daily.duplicated(["user", "date_time"]).any():
        daily = daily.groupby(["user", "date_time"], as_index=False)[VARIABLES].mean()

    pieces = []
    owners = []
    kept = []
    excluded = []
    for row in table.sort_values("id").itertuples(index=False):
        if pd.isna(row.inicio) or pd.isna(row.fin) or row.fin < row.inicio or not np.isfinite(row.t_digital):
            excluded.append({"id": int(row.id), "reason": "sin reloj"})
            continue
        idx = pd.date_range(row.inicio, row.fin, freq="D")
        sub = daily.loc[daily["user"] == row.id].drop_duplicates("date_time").set_index("date_time")
        aligned = sub.reindex(idx)[VARIABLES].to_numpy(dtype=float)
        if not np.isfinite(aligned).any():
            excluded.append({"id": int(row.id), "reason": "sin ningún día observado"})
            continue
        pieces.append(aligned)
        owners.append(np.full(len(aligned), int(row.id), dtype=np.int64))
        kept.append({"id": int(row.id), "event": int(row.event), "t_digital": float(row.t_digital), "n_days": int(len(aligned))})

    values = np.vstack(pieces)
    observed = np.isfinite(values)
    for name in POSITIVE:
        column = VARIABLES.index(name)
        ok = observed[:, column]
        values[ok, column] = np.log1p(values[ok, column])
    user = np.concatenate(owners)
    uniq, start, counts = np.unique(user, return_index=True, return_counts=True)
    slices = {int(uid): (int(begin), int(begin + count)) for uid, begin, count in zip(uniq, start, counts)}
    clinical = pd.DataFrame(kept).set_index("id")
    return {
        "X": values,
        "user": user,
        "slices": slices,
        "clinical": clinical,
        "excluded": excluded,
        "n_events": int(clinical["event"].sum()),
        "columns_match_development": list(VARIABLES) == list(reference_columns),
    }


def _tokens(labels: np.ndarray) -> list[str]:
    return [str(int(value)) for value in labels]


def _patient_topics(ids: np.ndarray, labels: np.ndarray, slices: dict, model, dictionary, clinical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid in ids:
        start, end = slices[int(pid)]
        dist = topic_distribution(model, dictionary.doc2bow(_tokens(labels[start:end])))
        rows.append({"id": int(pid), "topic": int(np.argmax(dist))})
    frame = pd.DataFrame(rows).join(clinical, on="id")
    frame["t_pfs"] = frame["Tiempo_obs_DB1"].astype(float)
    frame["event"] = frame["Evento_DB1"].astype(int)
    return frame.loc[frame["t_pfs"] >= MIN_CLINICAL_DAYS].copy()


def _blocks(
    ids: np.ndarray,
    labels: np.ndarray,
    slices: dict,
    model,
    dictionary,
    unfavorable: dict[int, str] | None,
    *,
    outer: int,
    inner: str,
    role: str,
    model_name: str,
) -> list[dict]:
    unfavorable_ids = [] if unfavorable is None else [topic for topic, label in unfavorable.items() if label == "U"]
    rows = []
    for pid in ids:
        start, end = slices[int(pid)]
        sequence = _tokens(labels[start:end])
        for block, (a, b) in enumerate(block_spans(len(sequence)), start=1):
            dist = topic_distribution(model, dictionary.doc2bow(sequence[a:b]))
            puf = np.nan if unfavorable is None else float(dist[unfavorable_ids].sum()) if unfavorable_ids else 0.0
            row = {
                "id": int(pid),
                "outer_fold": int(outer),
                "inner_fold": inner,
                "role": role,
                "model": model_name,
                "block": int(block),
                "n_days": int(b - a),
                "pUF": puf,
            }
            for topic in range(K):
                row[f"topic_{topic}"] = float(dist[topic])
            rows.append(row)
    return rows


def _fit_representation(fit_ids: np.ndarray, matrices: list[tuple[np.ndarray, np.ndarray, dict]], seed: int):
    """Fit scaler, k-means and LDA on development training days. Score every matrix."""
    values, users, slices = matrices[0]
    row_mask = np.isin(users, fit_ids)
    scaled, center, scale = _scale(values, row_mask)
    kmeans = _fit_kmeans(scaled, row_mask, seed)
    documents = []
    assigned = []
    for matrix_values, matrix_users, matrix_slices in matrices:
        matrix_scaled = (matrix_values - center) / scale
        matrix_scaled = np.where(np.isfinite(matrix_scaled), matrix_scaled, 0.0)
        labels = kmeans.predict(matrix_scaled)
        assigned.append(labels)
        if matrix_values is values:
            for pid in sorted(int(i) for i in fit_ids):
                start, end = matrix_slices[pid]
                documents.append(_tokens(labels[start:end]))
    dictionary, lda = fit_lda(documents, seed)
    return assigned, dictionary, lda


def _decisions(blocks: pd.DataFrame, clinical_of) -> pd.DataFrame:
    rows = []
    grouped = blocks.groupby(["id", "outer_fold", "inner_fold", "role", "model"], sort=True)
    for (pid, outer, inner, role, model_name), frame in grouped:
        frame = frame.sort_values("block")
        if len(frame) < 3:
            continue
        record = clinical_of(int(pid), role)
        if record is None:
            continue
        t_digital = float(record["t_digital"])
        event = int(record["event"])
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
                "score": float(np.sum(window)),
                "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
                "event": event,
                "t_digital": t_digital,
            })
    return pd.DataFrame(rows)


def _apply_cuts(decision_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cuts = []
    threshold_of = {}
    held = decision_df[decision_df["role"] == "held_out"]
    for outer, frame in held.groupby("outer_fold"):
        usable = frame[frame["in_comparison"].eq(1) & frame["outcome"].isin(["0", "1"]) & np.isfinite(frame["score"])]
        found = None if usable.empty else youden_cut(usable["score"].to_numpy(float), usable["outcome"].astype(int).to_numpy())
        if found is None:
            cuts.append({"outer_fold": int(outer), "threshold": np.nan, "reason": "sin positivos o sin negativos"})
            continue
        found["outer_fold"] = int(outer)
        found["tie_rule"] = "menor corte"
        cuts.append(found)
        threshold_of[int(outer)] = found["threshold"]
    decision_df["threshold"] = decision_df["outer_fold"].map(threshold_of).astype(float)
    decision_df.loc[~np.isfinite(decision_df["threshold"]), "threshold"] = np.nan
    alarm = pd.Series(pd.NA, index=decision_df.index, dtype="boolean")
    usable = decision_df["threshold"].notna() & np.isfinite(decision_df["score"])
    alarm.loc[usable] = decision_df.loc[usable, "score"] >= decision_df.loc[usable, "threshold"]
    decision_df["alarm"] = alarm
    return decision_df, pd.DataFrame(cuts)


def run_repeat(repeat: int) -> str:
    partitions = PARTITIONS[repeat]
    seed = int(SEEDS.loc[SEEDS["repeat"] == repeat, "lda_seed"].iloc[0])
    clinical = CLINICAL
    valset = VALSET
    days = DAYS
    assert clinical is not None and valset is not None and days is not None and PAIRS is not None
    folder = OUT / f"rep_{repeat:02d}"
    alarm = folder / "alarma"
    alarm.mkdir(parents=True, exist_ok=True)

    dev_matrix = (days["X"], days["user"], days["slices"])
    val_matrix = (valset["X"], valset["user"], valset["slices"])
    tables = []
    blocks = []
    reserved = []
    for outer in range(N_OUTER):
        train = partitions[(partitions["outer_fold"] == outer) & (partitions["side"] == "train")]
        test_ids = partitions.loc[(partitions["outer_fold"] == outer) & (partitions["side"] == "external"), "id"].to_numpy(int)
        jobs = []
        for inner in range(N_INNER):
            fit_ids = np.unique(train.loc[train["inner_fold"] != inner, "id"].to_numpy(int))
            held_ids = np.unique(train.loc[train["inner_fold"] == inner, "id"].to_numpy(int))
            jobs.append((fit_ids, held_ids, str(inner), "held_out", f"outer_{outer}_inner_{inner}", False))
        jobs.append((np.unique(train["id"].to_numpy(int)), test_ids, "refit", "external_test", f"outer_{outer}_refit", True))
        for fit_ids, score_ids, inner, role, name, score_valset in jobs:
            assigned, dictionary, lda = _fit_representation(fit_ids, [dev_matrix, val_matrix], seed)
            dev_labels, val_labels = assigned
            patients = _patient_topics(fit_ids, dev_labels, days["slices"], lda, dictionary, clinical)
            summaries = [topic_summary(patients, topic) for topic in range(K)]
            mapping, rule = assign_labels(summaries)
            if mapping is None:
                reserved.append({"model": name, "outer_fold": outer, "inner_fold": inner, "criterio": rule})
            for summary in summaries:
                tables.append({
                    "model": name,
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "topic": summary["topic"],
                    "n": summary["n"],
                    "events": summary["events"],
                    "pfs_365": summary["pfs_365"],
                    "etiqueta": "" if mapping is None else mapping[summary["topic"]],
                    "criterio": rule,
                })
            blocks.extend(_blocks(
                score_ids, dev_labels, days["slices"], lda, dictionary, mapping,
                outer=outer, inner=inner, role=role, model_name=name,
            ))
            if score_valset:
                blocks.extend(_blocks(
                    np.array(sorted(valset["slices"]), dtype=int), val_labels, valset["slices"], lda, dictionary, mapping,
                    outer=outer, inner="refit", role="valset", model_name=name,
                ))
        print(f"repeat {repeat} outer {outer} fitted", flush=True)

    pd.DataFrame(tables).to_csv(folder / "km_lda_por_topic.csv", index=False)
    pd.DataFrame(reserved).to_csv(folder / "modelos_reservados.csv", index=False)

    def clinical_of(pid: int, role: str):
        source = valset["clinical"] if role == "valset" else clinical
        if pid not in source.index:
            return None
        row = source.loc[pid]
        if role == "valset":
            return {"t_digital": row["t_digital"], "event": row["event"]}
        return {"t_digital": row["Tiempo_obs_digital"], "event": row["Evento_DB1"]}

    decision_df = _decisions(pd.DataFrame(blocks), clinical_of)
    development = decision_df[decision_df["role"] != "valset"].copy()
    valset_decisions = decision_df[decision_df["role"] == "valset"].copy()
    development = drop_outside_comparison(mark_comparison(development, PAIRS))
    raw = development[(development["role"] == "external_test") & (development["in_comparison"] == 1) & development["outcome"].isin(["0", "1"])]
    n_pos = int((raw["outcome"] == "1").sum())
    n_neg = int((raw["outcome"] == "0").sum())
    if len(raw) != 1013 or n_pos != 155 or n_neg != 858:
        raise RuntimeError(f"repeat {repeat}: external landmarks {len(raw)} ({n_pos} positive, {n_neg} negative)")

    development, cuts = _apply_cuts(development)
    cuts.to_csv(alarm / "cortes.csv", index=False)
    valset_decisions["threshold"] = valset_decisions["outer_fold"].map(dict(zip(cuts["outer_fold"], cuts["threshold"]))).astype(float)
    valset_decisions.loc[~np.isfinite(valset_decisions["threshold"]), "threshold"] = np.nan
    val_alarm = pd.Series(pd.NA, index=valset_decisions.index, dtype="boolean")
    usable = valset_decisions["threshold"].notna() & np.isfinite(valset_decisions["score"])
    val_alarm.loc[usable] = valset_decisions.loc[usable, "score"] >= valset_decisions.loc[usable, "threshold"]
    valset_decisions["alarm"] = val_alarm
    valset_decisions["in_comparison"] = valset_decisions["outcome"].isin(["0", "1"]).astype(int)
    both = pd.concat([development, valset_decisions], ignore_index=True)
    both.to_csv(alarm / "decisiones.csv", index=False)

    metrics = rendimiento(development.dropna(subset=["threshold"]), ["outer_fold", "role"])
    metrics.to_csv(alarm / "rendimiento_por_vuelta.csv", index=False)
    val_metrics = rendimiento(valset_decisions.dropna(subset=["threshold"]), ["outer_fold"])
    val_metrics.to_csv(alarm / "valset_por_vuelta.csv", index=False)

    flagged = development.copy()
    flagged["alarm"] = flagged["alarm"].map({True: True, False: False})
    external = flagged[(flagged["role"] == "external_test") & (flagged["event"] == 1) & (flagged["in_comparison"] == 1)]
    leads = []
    for (_, pid), group in external.groupby(["outer_fold", "id"], sort=True):
        leads.append(one_patient(group))
    pd.DataFrame(leads).to_csv(alarm / "anticipacion_por_paciente.csv", index=False)
    val_flagged = valset_decisions.copy()
    val_flagged["alarm"] = val_flagged["alarm"].map({True: True, False: False})
    val_events = val_flagged[(val_flagged["event"] == 1) & (val_flagged["in_comparison"] == 1)]
    val_leads = []
    for (fold, pid), group in val_events.groupby(["outer_fold", "id"], sort=True):
        row = one_patient(group)
        row["outer_fold"] = int(fold)
        val_leads.append(row)
    pd.DataFrame(val_leads).to_csv(alarm / "valset_anticipacion_por_paciente.csv", index=False)
    external_metrics = metrics[metrics["role"] == "external_test"]
    print(
        f"repeat {repeat} development folds {len(external_metrics)} "
        f"AUROC {external_metrics['AUROC'].mean():.3f} reserved {len(reserved)}",
        flush=True,
    )
    return f"rep_{repeat:02d} done"


def _mean_metrics(folder: Path, role: str | None) -> dict:
    if role == "valset":
        metrics = pd.read_csv(folder / "alarma" / "valset_por_vuelta.csv")
        leads = pd.read_csv(folder / "alarma" / "valset_anticipacion_por_paciente.csv")["first_tp_lead"].dropna()
    else:
        metrics = pd.read_csv(folder / "alarma" / "rendimiento_por_vuelta.csv")
        metrics = metrics[metrics["role"] == "external_test"]
        leads = pd.read_csv(folder / "alarma" / "anticipacion_por_paciente.csv")["first_tp_lead"].dropna()
    return {
        "n_folds": int(len(metrics)),
        "AUROC": float(metrics["AUROC"].mean()) if len(metrics) else np.nan,
        "sensitivity": float(metrics["sensitivity"].mean()) if len(metrics) else np.nan,
        "specificity": float(metrics["specificity"].mean()) if len(metrics) else np.nan,
        "detected": int(metrics["patients_detected"].sum()) if len(metrics) else 0,
        "n_event_patients": int(metrics["patients_with_event"].sum()) if len(metrics) else 0,
        "lead": float(np.median(leads)) if len(leads) else np.nan,
        "n_decisions": int(metrics["n_decisions"].sum()) if len(metrics) else 0,
    }


def external_reference() -> pd.DataFrame:
    rows = []
    for repeat in range(N_REPEATS):
        metrics = pd.read_csv(SAP_ROOT / f"rep_{repeat:02d}" / "alarma" / "rendimiento_por_vuelta.csv")
        external = metrics[metrics["role"] == "external_test"]
        leads = pd.read_csv(SAP_ROOT / f"rep_{repeat:02d}" / "alarma" / "anticipacion_por_paciente.csv")["first_tp_lead"].dropna()
        rows.append({
            "repeat": repeat,
            "AUROC": float(external["AUROC"].mean()),
            "sensitivity": float(external["sensitivity"].mean()),
            "specificity": float(external["specificity"].mean()),
            "lead": float(np.median(leads)),
        })
    return pd.DataFrame(rows)


def summarize() -> None:
    development_rows = []
    valset_rows = []
    for repeat in range(N_REPEATS):
        folder = OUT / f"rep_{repeat:02d}"
        development = _mean_metrics(folder, None)
        development["repeat"] = repeat
        development_rows.append(development)
        valset = _mean_metrics(folder, "valset")
        valset["repeat"] = repeat
        valset_rows.append(valset)
    development = pd.DataFrame(development_rows)
    valset = pd.DataFrame(valset_rows)
    reference = external_reference()
    merged = development.merge(reference, on="repeat", suffixes=("_hdo", "_external"))
    for name in ("AUROC", "sensitivity", "specificity", "lead"):
        merged[f"{name}_difference"] = merged[f"{name}_hdo"] - merged[f"{name}_external"]
    merged.to_csv(OUT / "comparacion_por_repeticion.csv", index=False)
    valset.to_csv(OUT / "valset_por_repeticion.csv", index=False)
    names = ["AUROC", "sensitivity", "specificity", "lead"]
    summary = {
        "development_1013_hdo_mean": {name: float(development[name].mean()) for name in names},
        "development_1013_hdo_sd": {name: float(development[name].std(ddof=1)) for name in names},
        "development_1013_external_vocabulary_mean": {name: float(reference[name].mean()) for name in names},
        "difference_hdo_minus_external": {
            name: float(development[name].mean() - reference[name].mean()) for name in names
        },
        "valset_mean": {name: float(valset[name].mean()) for name in names},
        "valset_sd": {name: float(valset[name].std(ddof=1)) for name in names},
        "development_folds_scored_per_repeat": development["n_folds"].astype(int).tolist(),
        "valset_folds_scored_per_repeat": valset["n_folds"].astype(int).tolist(),
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    global DAYS, VALSET, CLINICAL, PARTITIONS, SEEDS, PAIRS
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building development days", flush=True)
    DAYS = build_days()
    patient_ids = set(DAYS["slices"])
    seeds, partitions = load_partitions()
    verify_partitions(seeds, partitions)
    overlap = verify_valset(patient_ids)
    print("Building VAL-SET days", flush=True)
    VALSET = build_valset(np.array(VARIABLES))
    CLINICAL = load_clinical()
    PARTITIONS = partitions
    SEEDS = seeds
    PAIRS = sap_landmark_pairs()
    manifest = {
        "sap": "17.4 oncology-only representation inside each development training fold",
        "steps": [
            "k-means K=6 on the eight HDO variables of the training patients",
            "LDA K=6 on those cluster sequences, same passes and iterations as the nested SAP LDA",
            "3/3 labels and Youden cut inside the training fold",
            "development external folds scored on the frozen 1013 landmarks",
            "VAL-SET scored with the refit of each outer fold and that fold's already chosen cut",
        ],
        "lda_passes": PASSES,
        "lda_iterations": ITERATIONS,
        "seed": "the SAP lda_seed is the random state of both the single k-means start and the LDA",
        "sap_base_seed": SAP_BASE,
        "valset_used_for_fit_labels_or_threshold": False,
        "valset_overlap_with_development": overlap,
        "valset_patients_scored": int(len(VALSET["clinical"])),
        "valset_events": int(VALSET["n_events"]),
        "valset_excluded": VALSET["excluded"],
        "valset_event_rule": "Estado == PD Segundo corte. Time is Tiempo_obs_digital, or Fin_datos minus Inicio_datos when that time is blank.",
        "horizon_days": HORIZON_DAYS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    import multiprocessing as mp

    failures = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
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
