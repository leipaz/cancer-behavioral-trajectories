#!/usr/bin/env python3
"""Simple k-means baseline, one model, no LDA.

Standardize the eight daily variables with the mean and standard deviation of
every observed day. Fill what is still missing with that mean. Fit one
k-means (K=6) on every day of every sequence. A cluster is unfavorable by the
same 3/3 one-year rule, using the dominant cluster of each full sequence.

The prediction does not use LDA. At each decision, the fitted k-means is
applied to the days inside the same 90-day window as the usual alarm (three
blocks of 30). The score is the sum of the three block shares of unfavorable
days. The cut is Youden on those same points, because this baseline is fit
once on the whole cohort.

    python scripts/vqvae/run_hdo_kmeans_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/vqvae"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/lda"))
sys.path.insert(0, str(ROOT / "scripts/03_analysis/alarm"))

from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from rolling_burden_alarm import PAPER_THRESHOLD  # noqa: E402
from run_hdo_only_kmeans import (  # noqa: E402
    K,
    VARIABLES,
    _fit_kmeans,
    _scale,
    build_days,
    load_clinical,
    sap_landmark_pairs,
)
from run_val2_blocks import block_spans  # noqa: E402
from run_val2_fu import MIN_CLINICAL_DAYS, assign_labels, topic_summary, youden_cut  # noqa: E402

SEED = 20261031
N_BOOT = 2000
BOOT_SEED = 42
OUT = ROOT / "results/vq-vae/training_provenance/hdo_kmeans_baseline"
COLLAPSED = ROOT / "results/alarm/runs/Evento_DB1_clock/Bt_vs_mensual/decision_points_collapsed_teb2.csv"


def _metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["outcome"].astype(int).to_numpy()
    score = frame["score"].to_numpy(float)
    alarm = score >= threshold
    tp = int(((y == 1) & alarm).sum())
    fn = int(((y == 1) & ~alarm).sum())
    fp = int(((y == 0) & alarm).sum())
    tn = int(((y == 0) & ~alarm).sum())
    event_ids = frame.loc[frame["event"] == 1, "id"].unique()
    detected = 0
    leads = []
    for pid in event_ids:
        group = frame[frame["id"] == pid].sort_values("decision_day")
        hits = group[(group["outcome"].astype(str) == "1") & (group["score"] >= threshold)]
        if len(hits) == 0:
            continue
        detected += 1
        event_day = float(group["t_digital"].iloc[0]) + 1.0
        leads.append(event_day - float(hits["decision_day"].iloc[0]))
    return {
        "n_decisions": int(len(frame)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_patients": int(frame["id"].nunique()),
        "patients_with_event": int(len(event_ids)),
        "patients_detected": int(detected),
        "threshold": float(threshold),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "AUROC": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "median_lead_days": float(np.median(leads)) if leads else None,
        "n_leads": int(len(leads)),
    }


def _percentile(values: list[float]) -> tuple[float, float] | tuple[None, None]:
    if len(values) < 100:
        return None, None
    lo, hi = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return float(lo), float(hi)


def _operating(frame: pd.DataFrame, threshold: float) -> dict[int, dict]:
    """Per-patient counts at a fixed cut. The resample adds these counts up."""
    out = {}
    for pid, group in frame.groupby("id", sort=False):
        y = group["outcome"].astype(int).to_numpy()
        hit = group["score"].to_numpy(float) >= threshold
        event = int(group["event"].iloc[0])
        lead = None
        if event == 1:
            ordered = group.sort_values("decision_day")
            hits = ordered[(ordered["outcome"].astype(str) == "1") & (ordered["score"] >= threshold)]
            if len(hits):
                lead = float(ordered["t_digital"].iloc[0]) + 1.0 - float(hits["decision_day"].iloc[0])
        out[int(pid)] = {
            "tp": int(((y == 1) & hit).sum()),
            "fn": int(((y == 1) & ~hit).sum()),
            "fp": int(((y == 0) & hit).sum()),
            "tn": int(((y == 0) & ~hit).sum()),
            "lead": lead,
        }
    return out


def patient_bootstrap(
    baseline: pd.DataFrame,
    normal: pd.DataFrame,
    baseline_threshold: float,
    normal_threshold: float,
) -> dict:
    """Same patient bootstrap as the alarm figure.

    Two thousand resamples, seed 42, patients drawn with replacement, and the
    2.5 and 97.5 percentiles. One draw is shared by both scores. AUROC follows
    ``plot_analysis_figures._clustered_auc_ci``. Sensitivity, specificity and
    the median lead use that same draw with the cut held fixed.
    """
    baseline = baseline.sort_values(["id", "block"]).reset_index(drop=True)
    normal = normal.sort_values(["id", "block"]).reset_index(drop=True)
    ids = np.unique(baseline["id"].to_numpy())
    if not np.array_equal(ids, np.unique(normal["id"].to_numpy())):
        raise RuntimeError("the two scores do not cover the same patients")
    by_baseline = {int(pid): np.flatnonzero(baseline["id"].to_numpy() == pid) for pid in ids}
    by_normal = {int(pid): np.flatnonzero(normal["id"].to_numpy() == pid) for pid in ids}
    y_b = baseline["outcome"].astype(int).to_numpy()
    s_b = baseline["score"].to_numpy(float)
    y_n = normal["outcome"].astype(int).to_numpy()
    s_n = normal["score"].to_numpy(float)
    op_b = _operating(baseline, baseline_threshold)
    op_n = _operating(normal, normal_threshold)

    point_b = float(roc_auc_score(y_b, s_b))
    point_n = float(roc_auc_score(y_n, s_n))
    store = {name: [] for name in (
        "auc_baseline", "auc_normal", "auc_difference",
        "sens_baseline", "sens_normal",
        "spec_baseline", "spec_normal",
        "lead_baseline", "lead_normal",
    )}
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(N_BOOT):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx_b = np.concatenate([by_baseline[int(pid)] for pid in draw])
        idx_n = np.concatenate([by_normal[int(pid)] for pid in draw])
        yb, sb = y_b[idx_b], s_b[idx_b]
        yn, sn = y_n[idx_n], s_n[idx_n]
        auc_b = float(roc_auc_score(yb, sb)) if len(np.unique(yb)) > 1 else None
        auc_n = float(roc_auc_score(yn, sn)) if len(np.unique(yn)) > 1 else None
        if auc_b is not None:
            store["auc_baseline"].append(auc_b)
        if auc_n is not None:
            store["auc_normal"].append(auc_n)
        if auc_b is not None and auc_n is not None:
            store["auc_difference"].append(auc_b - auc_n)

        counts = {"b": {"tp": 0, "fn": 0, "fp": 0, "tn": 0}, "n": {"tp": 0, "fn": 0, "fp": 0, "tn": 0}}
        leads_b, leads_n = [], []
        for pid in draw:
            pid = int(pid)
            for key, op, leads in (("b", op_b, leads_b), ("n", op_n, leads_n)):
                row = op[pid]
                for part in ("tp", "fn", "fp", "tn"):
                    counts[key][part] += row[part]
                if row["lead"] is not None:
                    leads.append(row["lead"])
        for key, sens_name, spec_name in (
            ("b", "sens_baseline", "spec_baseline"),
            ("n", "sens_normal", "spec_normal"),
        ):
            pos = counts[key]["tp"] + counts[key]["fn"]
            neg = counts[key]["tn"] + counts[key]["fp"]
            if pos:
                store[sens_name].append(counts[key]["tp"] / pos)
            if neg:
                store[spec_name].append(counts[key]["tn"] / neg)
        if leads_b:
            store["lead_baseline"].append(float(np.median(leads_b)))
        if leads_n:
            store["lead_normal"].append(float(np.median(leads_n)))

    def pack(point: float, name: str) -> dict:
        lo, hi = _percentile(store[name])
        return {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_boot_used": int(len(store[name])),
        }

    base_m = _metrics(baseline, baseline_threshold)
    norm_m = _metrics(normal, normal_threshold)
    return {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "ci_method": "patient-clustered bootstrap percentile 95%",
        "same_function_as": "scripts/03_analysis/alarm/plot_analysis_figures.py _clustered_auc_ci",
        "cut_held_fixed": True,
        "n_patients": int(len(ids)),
        "n_decision_points": int(len(baseline)),
        "AUROC_baseline": pack(point_b, "auc_baseline"),
        "AUROC_normal": pack(point_n, "auc_normal"),
        "AUROC_baseline_minus_normal": pack(point_b - point_n, "auc_difference"),
        "sensitivity_baseline": pack(base_m["sensitivity"], "sens_baseline"),
        "sensitivity_normal": pack(norm_m["sensitivity"], "sens_normal"),
        "specificity_baseline": pack(base_m["specificity"], "spec_baseline"),
        "specificity_normal": pack(norm_m["specificity"], "spec_normal"),
        "median_lead_baseline": pack(base_m["median_lead_days"], "lead_baseline"),
        "median_lead_normal": pack(norm_m["median_lead_days"], "lead_normal"),
    }


def _decisions(labels: np.ndarray, unfavorable: dict[int, str], days: dict, clinical: pd.DataFrame) -> pd.DataFrame:
    unfavorable_ids = [topic for topic, label in unfavorable.items() if label == "U"]
    rows = []
    for pid, (start, end) in days["slices"].items():
        if pid not in clinical.index:
            continue
        sequence = labels[start:end]
        blocks = []
        for block, (a, b) in enumerate(block_spans(len(sequence)), start=1):
            window = sequence[a:b]
            share = float(np.isin(window, unfavorable_ids).mean()) if len(window) else np.nan
            blocks.append((block, share))
        if len(blocks) < 3:
            continue
        t_digital = float(clinical.loc[pid, "Tiempo_obs_digital"])
        event = int(clinical.loc[pid, "Evento_DB1"])
        for i in range(2, len(blocks)):
            trio = blocks[i - 2 : i + 1]
            day = float(trio[2][0]) * 30.0
            rows.append({
                "id": int(pid),
                "block": int(trio[2][0]),
                "decision_day": day,
                "puf_1": float(trio[0][1]),
                "puf_2": float(trio[1][1]),
                "puf_3": float(trio[2][1]),
                "score": float(sum(item[1] for item in trio)),
                "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
                "event": event,
                "t_digital": t_digital,
            })
    return pd.DataFrame(rows)


def _normal_on_same_points(pairs: set[tuple[int, int]], clinical: pd.DataFrame) -> pd.DataFrame:
    collapsed = pd.read_csv(COLLAPSED)
    collapsed["id"] = collapsed["id"].astype(int)
    collapsed["block"] = collapsed["month"].astype(int)
    collapsed = collapsed[collapsed.apply(lambda row: (int(row.id), int(row.block)) in pairs, axis=1)].copy()
    rows = []
    for row in collapsed.itertuples(index=False):
        pid = int(row.id)
        t_digital = float(clinical.loc[pid, "Tiempo_obs_digital"])
        event = int(clinical.loc[pid, "Evento_DB1"])
        day = float(row.decision_day)
        rows.append({
            "id": pid,
            "block": int(row.block),
            "decision_day": day,
            "score": float(row.AUC_W),
            "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
            "event": event,
            "t_digital": t_digital,
        })
    return pd.DataFrame(rows)


def main() -> None:
    days = build_days()
    clinical = load_clinical()
    pairs = sap_landmark_pairs()
    observed = np.isfinite(days["X"])
    row_mask = np.ones(len(days["X"]), dtype=bool)
    scaled, center, scale = _scale(days["X"], row_mask)
    model = _fit_kmeans(scaled, row_mask, SEED)
    labels = model.predict(scaled)

    patient_rows = []
    for pid, (start, end) in days["slices"].items():
        topic = int(np.bincount(labels[start:end], minlength=K).argmax())
        patient_rows.append({"id": int(pid), "topic": topic})
    patients = pd.DataFrame(patient_rows).join(clinical, on="id")
    patients["t_pfs"] = patients["Tiempo_obs_DB1"].astype(float)
    patients["event"] = patients["Evento_DB1"].astype(int)
    patients = patients.loc[patients["t_pfs"] >= MIN_CLINICAL_DAYS].copy()
    summaries = [topic_summary(patients, topic) for topic in range(K)]
    mapping, rule = assign_labels(summaries)
    if mapping is None:
        raise SystemExit(f"clusters could not be labelled: {rule}")

    decisions = _decisions(labels, mapping, days, clinical)
    kept = decisions[
        decisions.apply(lambda row: (int(row.id), int(row.block)) in pairs, axis=1)
        & decisions["outcome"].isin(["0", "1"])
    ].copy()
    n_pos = int((kept["outcome"] == "1").sum())
    n_neg = int((kept["outcome"] == "0").sum())
    if len(kept) != 1013 or n_pos != 155 or n_neg != 858:
        raise SystemExit(
            f"baseline landmarks {len(kept)} ({n_pos} positive, {n_neg} negative), expected 1013 (155, 858)"
        )

    cut = youden_cut(kept["score"].to_numpy(float), kept["outcome"].astype(int).to_numpy())
    baseline = _metrics(kept, cut["threshold"])
    normal_frame = _normal_on_same_points(pairs, clinical)
    normal_kept = normal_frame[normal_frame["outcome"].isin(["0", "1"])].copy()
    normal = _metrics(normal_kept, PAPER_THRESHOLD)
    intervals = patient_bootstrap(kept, normal_kept, cut["threshold"], PAPER_THRESHOLD)

    cluster_table = []
    for summary in summaries:
        cluster_table.append({
            "topic": summary["topic"],
            "n": summary["n"],
            "events": summary["events"],
            "pfs_365": summary["pfs_365"],
            "rmst_365": summary["rmst_365"],
            "etiqueta": mapping[summary["topic"]],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    kept.to_csv(OUT / "decisiones.csv", index=False)
    pd.DataFrame(cluster_table).to_csv(OUT / "clusters.csv", index=False)
    comparison = {
        "baseline": "k-means on every standardized day, mean imputation, no LDA",
        "fit": "all 175 sequences, one model",
        "seed": SEED,
        "k": K,
        "n_init": 1,
        "standardize": "mean and standard deviation of every observed day, then missing filled with the mean",
        "prediction": "fitted k-means applied to the days of the 90-day window; score is the sum of the three unfavorable shares",
        "labels": rule,
        "unfavorable_clusters": [topic for topic, label in mapping.items() if label == "U"],
        "youden_on_the_same_points": True,
        "n_fit_days": int(row_mask.sum()),
        "n_imputed_cells": int((~observed).sum()),
        "inertia": float(model.inertia_),
        "center": [float(v) for v in center],
        "scale": [float(v) for v in scale],
        "baseline_metrics": baseline,
        "normal_pipeline_same_points": normal,
        "normal_pipeline_threshold": PAPER_THRESHOLD,
        "normal_pipeline_source": str(COLLAPSED.relative_to(ROOT)),
        "same_pairs": int(len(set(zip(kept["id"], kept["block"])) & set(zip(normal_kept["id"], normal_kept["block"])))),
        "bootstrap": intervals,
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"measure": key, **value}
        for key, value in intervals.items()
        if isinstance(value, dict) and "point" in value
    ]).to_csv(OUT / "intervalos_bootstrap.csv", index=False)
    print(json.dumps({"baseline": baseline, "normal": normal, "rule": rule, "bootstrap": intervals}, indent=2))


if __name__ == "__main__":
    main()
