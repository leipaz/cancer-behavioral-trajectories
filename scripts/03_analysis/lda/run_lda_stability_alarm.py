#!/usr/bin/env python3
"""Monthly-block and alarm stability of the K=6 LDA replicates on audited DB1.

The published model is not retrained. Each replicate is a fresh K=6 fit
(100 passes, 50 iterations) on the audited day-type documents, Hungarian-aligned
to the published topic profiles by cosine. Alignment does not use outcomes.

After alignment, F = topics 0–2 and U = topics 3–5. Monthly blocks are the same
collapsed 30-day windows used for the teb2 alarm (the last block may be the
final 30 profile days if follow-up ends before the nominal month end). The
three-block load is

    AUC3(T) = pUF(T-60) + pUF(T-30) + pUF(T)

and the alarm is AUC3 >= 1.0596384140662845. The cut is not refit.

Leave-20%-out replicates are inferred on the patients left out of that fit as
well as on the patients used to fit. Those two sets are summarised separately.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UNFAVOURABLE = (3, 4, 5)
THETA = 1.0596384140662845
BLOCK = 30
W = 3
H_DAYS = 120


def topic_matrix(lda: LdaModel) -> np.ndarray:
    return np.asarray(lda.get_topics(), dtype=float)


def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    return float(np.dot(p, q) / den) if den > 0 else 0.0


def align_topics_cosine(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    k = ref.shape[0]
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -cosine_sim(ref[i], other[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(k, dtype=int)
    perm[row_ind] = col_ind
    return perm


def doc_topics(lda: LdaModel, bow) -> np.ndarray:
    probs = np.zeros(lda.num_topics, dtype=float)
    if not bow:
        return probs
    for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
        probs[int(tid)] = float(p)
    return probs


def train_lda(corpus_bow, dictionary, seed: int, passes: int) -> LdaModel:
    return LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=6,
        random_state=seed,
        passes=passes,
        chunksize=100,
        alpha="symmetric",
        eta="symmetric",
        iterations=50,
        eval_every=None,
    )


def month_windows(ids: list) -> list[tuple[int, list, bool]]:
    """Collapsed 30-day blocks. Last block is the final 30 days if the profile ends early."""
    n = len(ids)
    if n < 1:
        return []
    if n < BLOCK:
        return [(1, list(ids), True)]
    out: list[tuple[int, list, bool]] = []
    period = 0
    for start in range(0, n, BLOCK):
        end = start + BLOCK
        period += 1
        if end > n:
            out.append((period, list(ids[-BLOCK:]), True))
            break
        out.append((period, list(ids[start:end]), False))
    return out


def load_inputs(root: Path) -> tuple[pd.DataFrame, list, list[str]]:
    emb = pd.read_csv(root / "data/processed/lda/user_embeddings_from_pkl.csv")
    clin = pd.read_csv(root / "data/processed/clinical/obs_vs_teb2_start_dates.csv")
    rang = pd.read_csv(root / "results/alarm/rangos_meses_t_eb2.csv")
    dp = pd.read_csv(root / "results/alarm/puntos_decision_teb2.csv")
    clk = dp.groupby("id", as_index=False).agg(event_dp=("event", "first"))
    meta = emb.merge(clin[["id", "PD_event"]], left_on="user_id", right_on="id", how="left")
    meta = meta.merge(rang[["id", "t_evento_eb2"]], on="id", how="left")
    meta = meta.merge(clk, on="id", how="left")
    meta["event"] = meta["event_dp"].fillna(meta["PD_event"]).astype(int)
    ids = [int(x) for x in meta["user_id"].tolist()]
    docs = []
    for raw in meta["embedding_ids"]:
        docs.append([str(t) for t in ast.literal_eval(raw)])
    return meta, docs, [str(i) for i in ids]


def build_months_and_landmarks(meta: pd.DataFrame, docs: list, dictionary) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    month_rows = []
    bows = []
    for rec, ids in zip(meta.to_dict("records"), docs):
        pid = int(rec["user_id"])
        n = len(ids)
        te = float(rec["t_evento_eb2"])
        ev = int(rec["event"])
        for period, window, past in month_windows(ids):
            bow = dictionary.doc2bow(window)
            bows.append(bow)
            month_rows.append(
                {
                    "id": pid,
                    "month": period,
                    "study_day": BLOCK * period,
                    "n_days": n,
                    "n_tokens": len(window),
                    "past_profile": int(past or (BLOCK * period > n)),
                    "empty_bow": int(len(bow) == 0),
                    "event": ev,
                    "t_evento_eb2": te,
                }
            )
    months = pd.DataFrame(month_rows)
    # landmarks: three preceding blocks exist
    by_id = {pid: g.sort_values("month") for pid, g in months.groupby("id")}
    land_rows = []
    for pid, g in by_id.items():
        puf_index = {int(r.month): i for i, r in enumerate(g.itertuples(index=False))}
        # global row positions filled later; store month keys
        for r in g.itertuples(index=False):
            if int(r.month) < W:
                continue
            T = int(r.study_day)
            study = float(r.t_evento_eb2) + 1.0
            if not (T < study):
                y = np.nan
                status = "after_event"
            elif int(r.event) == 1 and study <= T + H_DAYS:
                y = 1.0
                status = "positive"
            elif study >= T + H_DAYS:
                y = 0.0
                status = "negative"
            else:
                y = np.nan
                status = "unresolved"
            land_rows.append(
                {
                    "id": pid,
                    "month": int(r.month),
                    "study_day": T,
                    "y": y,
                    "status": status,
                    "past_profile": int(r.past_profile),
                    "event": int(r.event),
                    "t_evento_eb2": float(r.t_evento_eb2),
                    "n_days": int(r.n_days),
                    "month_row_m2": int(g.index[puf_index[int(r.month) - 2]]),
                    "month_row_m1": int(g.index[puf_index[int(r.month) - 1]]),
                    "month_row_m": int(g.index[puf_index[int(r.month)]]),
                }
            )
    landmarks = pd.DataFrame(land_rows)
    # groupby index is the original months index if we didn't reset; g.index is months index. Good.
    return months, landmarks, bows


def puf_from_topics(probs: np.ndarray) -> float:
    return float(probs[list(UNFAVOURABLE)].sum())


def score_model(lda: LdaModel, bows: list, perm: np.ndarray | None) -> np.ndarray:
    out = np.zeros(len(bows), dtype=float)
    for i, bow in enumerate(bows):
        probs = doc_topics(lda, bow)
        if perm is not None:
            probs = probs[perm]
        out[i] = puf_from_topics(probs)
    return out


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = y.astype(int)
    pred = pred.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
    }


def agreement_block(ref: np.ndarray, rep: np.ndarray, y: np.ndarray, alarm_ref: np.ndarray) -> dict:
    """ref/rep are AUC3; missing rep scores are NaN and stay in the denominator."""
    n = int(len(ref))
    scored = np.isfinite(rep)
    n_scored = int(scored.sum())
    delta = rep[scored] - ref[scored]
    alarm_rep = np.full(n, np.nan)
    alarm_rep[scored] = (rep[scored] >= THETA).astype(float)
    both = scored
    changed = both & (alarm_rep != alarm_ref)
    ref_neg = both & (alarm_ref == 0)
    ref_pos = both & (alarm_ref == 1)
    neg_to_pos = int((ref_neg & (alarm_rep == 1)).sum())
    pos_to_neg = int((ref_pos & (alarm_rep == 0)).sum())
    y_scored = y[scored]
    s_scored = rep[scored]
    perf = binary_metrics(y_scored, (s_scored >= THETA).astype(int)) if n_scored else {
        "tp": 0, "fn": 0, "fp": 0, "tn": 0,
        "sensitivity": float("nan"), "specificity": float("nan"),
        "ppv": float("nan"), "npv": float("nan"),
    }
    if n_scored and len(np.unique(y_scored)) == 2:
        auroc = float(roc_auc_score(y_scored, s_scored))
    else:
        auroc = float("nan")
    return {
        "n_landmarks": n,
        "n_scored": n_scored,
        "n_missing": n - n_scored,
        "auc3_pearson": _safe_pearson(ref[scored], rep[scored]),
        "auc3_bias": float(np.mean(delta)) if n_scored else float("nan"),
        "auc3_mae": float(np.mean(np.abs(delta))) if n_scored else float("nan"),
        "n_decisions_changed": int(changed.sum()),
        "pct_decisions_changed": float(changed.sum() / n) if n else float("nan"),
        "n_refneg_reppos": neg_to_pos,
        "n_refpos_repneg": pos_to_neg,
        "pct_refneg_to_reppos": float(neg_to_pos / ref_neg.sum()) if ref_neg.sum() else float("nan"),
        "pct_refpos_to_repneg": float(pos_to_neg / ref_pos.sum()) if ref_pos.sum() else float("nan"),
        "n_ref_negative": int((alarm_ref == 0).sum()),
        "n_ref_positive": int((alarm_ref == 1).sum()),
        "auroc": auroc,
        **perf,
        "n_positive_labels": int((y == 1).sum()),
        "n_negative_labels": int((y == 0).sum()),
        "n_patients": None,
    }


_WORKER: dict = {}


def _init_worker(dictionary_path: str, full_bows, month_bows, ref_topics, passes: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER["dictionary"] = corpora.Dictionary.load(dictionary_path)
    _WORKER["full_bows"] = full_bows
    _WORKER["month_bows"] = month_bows
    _WORKER["ref_topics"] = ref_topics
    _WORKER["passes"] = passes


def fit_one(spec: dict) -> dict:
    dictionary = _WORKER["dictionary"]
    train_bows = [_WORKER["full_bows"][i] for i in spec["train_idx"]]
    model = train_lda(train_bows, dictionary, spec["seed"], _WORKER["passes"])
    perm = align_topics_cosine(_WORKER["ref_topics"], topic_matrix(model))
    puf = score_model(model, _WORKER["month_bows"], perm)
    return {
        "rep": spec["rep"],
        "seed": spec["seed"],
        "mode": spec["mode"],
        "n_train": int(len(spec["train_idx"])),
        "alignment_perm": ",".join(str(int(x)) for x in perm),
        "held_out_ids": spec["held_out_ids"],
        "puf": puf,
    }


def summarise_distribution(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"n_reps": int(len(df))}
    for c in cols:
        s = df[c].astype(float)
        out[c] = {
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def plot_level(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    panels = [
        (axes[0, 0], "auc3_pearson", "Pearson r, three-block load"),
        (axes[0, 1], "auc3_mae", "MAE of three-block load"),
        (axes[1, 0], "pct_decisions_changed", "Fraction of alarm decisions that change"),
        (axes[1, 1], "auroc", "AUROC at the fixed cut"),
    ]
    for ax, col, label in panels:
        vals = df[col].to_numpy(dtype=float)
        ax.hist(vals, bins=12, color="#4c72b0", edgecolor="white")
        med = float(np.nanmedian(vals))
        ax.axvline(med, color="crimson", lw=1.6, label=f"median {med:.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Replicates")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("--leave-out", type=float, default=0.2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results/lda/stability/revision/alarm",
    )
    args = parser.parse_args()
    root = PROJECT_ROOT
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpus and reference model…", flush=True)
    meta, docs, _ = load_inputs(root)
    dictionary = corpora.Dictionary.load(str(root / "models/lda/dictionary_LDA_vdec25.dict"))
    ref_model = LdaModel.load(str(root / "models/lda/LDA_model_vdec25.gensim"))
    ref_topics = topic_matrix(ref_model)
    full_bows = [dictionary.doc2bow(doc) for doc in docs]
    patient_ids = [int(x) for x in meta["user_id"].tolist()]

    months, landmarks, month_bows = build_months_and_landmarks(meta, docs, dictionary)
    print(f"Monthly blocks {len(months)}; landmark rows {len(landmarks)}", flush=True)
    ref_puf = score_model(ref_model, month_bows, perm=None)
    months = months.copy()
    months["pUF_ref"] = ref_puf
    # map month-table position -> pUF. build_months used months.index before any filter.
    # landmarks store months.index values from the groupby frame, which match months.index.
    puf_by_row = months["pUF_ref"]
    landmarks = landmarks.copy()
    landmarks["pUF_m2"] = landmarks["month_row_m2"].map(puf_by_row)
    landmarks["pUF_m1"] = landmarks["month_row_m1"].map(puf_by_row)
    landmarks["pUF_m"] = landmarks["month_row_m"].map(puf_by_row)
    landmarks["AUC3_ref"] = landmarks["pUF_m2"] + landmarks["pUF_m1"] + landmarks["pUF_m"]
    landmarks["alarm_ref"] = (landmarks["AUC3_ref"] >= THETA).astype(int)

    # Cotejo every-30 scores, for the link to the closed reference table.
    dp = pd.read_csv(root / "results/alarm/puntos_decision_teb2.csv")
    e30 = dp[(dp.cadence == "Every 30 days") & (dp.included_in_metrics == 1)][
        ["id", "study_day", "score_B", "label"]
    ].copy()
    e30["AUC3_cotejo"] = e30["score_B"] * 3.0
    landmarks = landmarks.merge(e30[["id", "study_day", "AUC3_cotejo", "label"]], on=["id", "study_day"], how="left")
    # All labeled collapsed-monthly landmarks, including the six end-of-profile
    # blocks. This is the 1,019-point set, not the 1,013 of section 2.
    evaluable = landmarks[landmarks["status"].isin(["positive", "negative"])].copy()
    evaluable = evaluable.reset_index(drop=True)
    print(
        f"Evaluable landmarks {len(evaluable)} "
        f"(profile reaches T: {(evaluable.past_profile == 0).sum()}; "
        f"end-of-profile block: {(evaluable.past_profile == 1).sum()})",
        flush=True,
    )

    # Reproduce leave-out masks with the same Generator sequence as run_lda_stability.py.
    n = len(full_bows)
    n_out = max(1, int(round(args.leave_out * n)))
    rng = np.random.default_rng(args.seed_start)
    masks = []
    for r in range(args.n_reps):
        out_idx = rng.choice(n, size=n_out, replace=False)
        held = sorted(int(patient_ids[i]) for i in out_idx)
        train_idx = np.setdiff1d(np.arange(n), out_idx)
        masks.append((train_idx.tolist(), held))

    specs = []
    for r in range(args.n_reps):
        specs.append(
            {
                "rep": r,
                "seed": args.seed_start + r,
                "mode": "seed",
                "train_idx": list(range(n)),
                "held_out_ids": [],
            }
        )
    for r, (train_idx, held) in enumerate(masks):
        specs.append(
            {
                "rep": r,
                "seed": args.seed_start + r,
                "mode": "resample",
                "train_idx": train_idx,
                "held_out_ids": held,
            }
        )

    print(f"Fitting {len(specs)} replicates on {args.workers} workers…", flush=True)
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            str(root / "models/lda/dictionary_LDA_vdec25.dict"),
            full_bows,
            month_bows,
            ref_topics,
            args.passes,
        ),
    ) as pool:
        futures = [pool.submit(fit_one, spec) for spec in specs]
        for k, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if k % 10 == 0 or k == len(futures):
                print(f"  finished {k}/{len(futures)}", flush=True)

    month_ids = months["id"].to_numpy()
    eval_ids = evaluable["id"].to_numpy()
    ref_auc = evaluable["AUC3_ref"].to_numpy(dtype=float)
    alarm_ref = evaluable["alarm_ref"].to_numpy(dtype=float)
    y = evaluable["y"].to_numpy(dtype=float)
    # Monthly pUF is compared only on the blocks that enter these landmarks.
    used_months = np.unique(
        np.concatenate(
            [
                evaluable["month_row_m2"].to_numpy(),
                evaluable["month_row_m1"].to_numpy(),
                evaluable["month_row_m"].to_numpy(),
            ]
        )
    )
    ref_month_all = months["pUF_ref"].to_numpy(dtype=float)
    ref_month = ref_month_all[used_months]

    def rows_for(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        per = []
        split_rows = []
        subset = [r for r in results if r["mode"] == mode]
        subset.sort(key=lambda r: r["rep"])
        for r in subset:
            puf = r["puf"]
            puf_used = puf[used_months]
            auc = (
                puf[evaluable["month_row_m2"].to_numpy()]
                + puf[evaluable["month_row_m1"].to_numpy()]
                + puf[evaluable["month_row_m"].to_numpy()]
            )
            block = agreement_block(ref_auc, auc, y, alarm_ref)
            month_delta = puf_used - ref_month
            block.update(
                {
                    "rep": r["rep"],
                    "seed": r["seed"],
                    "mode": mode,
                    "n_train": r["n_train"],
                    "alignment_perm": r["alignment_perm"],
                    "n_patients": int(evaluable["id"].nunique()),
                    "puf_pearson": _safe_pearson(ref_month, puf_used),
                    "puf_bias": float(np.mean(month_delta)),
                    "puf_mae": float(np.mean(np.abs(month_delta))),
                    "n_months": int(len(puf_used)),
                }
            )
            per.append(block)
            if mode == "resample":
                held = set(r["held_out_ids"])
                for name, mask in (
                    ("included_in_fit", ~np.isin(eval_ids, list(held))),
                    ("held_out_of_fit", np.isin(eval_ids, list(held))),
                ):
                    sub = agreement_block(ref_auc[mask], auc[mask], y[mask], alarm_ref[mask])
                    # monthly pUF on the same patients
                    in_landmarks = np.zeros(len(puf), dtype=bool)
                    in_landmarks[used_months] = True
                    if name == "included_in_fit":
                        msk_m = in_landmarks & ~np.isin(month_ids, list(held))
                    else:
                        msk_m = in_landmarks & np.isin(month_ids, list(held))
                    sub.update(
                        {
                            "rep": r["rep"],
                            "seed": r["seed"],
                            "subset": name,
                            "n_patients_subset": int(len(held) if name == "held_out_of_fit" else n - len(held)),
                            "n_patients_with_landmarks": int(pd.unique(eval_ids[mask]).size),
                            "puf_pearson": _safe_pearson(ref_month_all[msk_m], puf[msk_m]),
                            "puf_bias": float(np.mean(puf[msk_m] - ref_month_all[msk_m])) if msk_m.any() else float("nan"),
                            "puf_mae": float(np.mean(np.abs(puf[msk_m] - ref_month_all[msk_m]))) if msk_m.any() else float("nan"),
                            "n_months": int(msk_m.sum()),
                        }
                    )
                    split_rows.append(sub)
        return pd.DataFrame(per), pd.DataFrame(split_rows)

    level1, _ = rows_for("seed")
    level2, level2_split = rows_for("resample")

    # Reference operating point on the same landmarks.
    ref_perf = binary_metrics(y, alarm_ref)
    if len(np.unique(y)) == 2:
        ref_perf["auroc"] = float(roc_auc_score(y, ref_auc))
    else:
        ref_perf["auroc"] = float("nan")
    joined = evaluable["AUC3_cotejo"].notna()
    cotejo_delta = (evaluable.loc[joined, "AUC3_ref"] - evaluable.loc[joined, "AUC3_cotejo"]).abs()

    metric_cols = [
        "puf_pearson", "puf_bias", "puf_mae",
        "auc3_pearson", "auc3_bias", "auc3_mae",
        "pct_decisions_changed", "pct_refneg_to_reppos", "pct_refpos_to_repneg",
        "auroc", "sensitivity", "specificity", "ppv", "npv",
    ]
    summary = {
        "theta_AUC3": THETA,
        "passes": args.passes,
        "iterations": 50,
        "alignment": "Hungarian cosine of P(day-type | topic); outcomes not used",
        "hypertopic": "F=T0-2, U=T3-5, pUF=P3+P4+P5 after alignment",
        "blocks": "non-overlapping 30-day blocks; if the profile ends before T=30m the last block is the final 30 days and the decision stays at T",
        "load": "AUC3 = sum of the last 3 monthly pUF; alarm if AUC3 >= theta",
        "horizon_days": H_DAYS,
        "n_patients_documents": n,
        "n_monthly_blocks": int(len(months)),
        "n_evaluable_landmarks": int(len(evaluable)),
        "n_profile_reaches_T": int((evaluable.past_profile == 0).sum()),
        "n_end_of_profile_block": int((evaluable.past_profile == 1).sum()),
        "n_positive_labels": int((evaluable.y == 1).sum()),
        "n_negative_labels": int((evaluable.y == 0).sum()),
        "n_patients_evaluable": int(evaluable.id.nunique()),
        "n_events_with_a_positive_landmark": int(evaluable.loc[evaluable.y == 1, "id"].nunique()),
        "cotejo_every30_joined": int(joined.sum()),
        "cotejo_abs_AUC3_median": float(cotejo_delta.median()) if joined.any() else None,
        "cotejo_abs_AUC3_max": float(cotejo_delta.max()) if joined.any() else None,
        "reference_performance": ref_perf,
        "level1_seed": summarise_distribution(level1, metric_cols),
        "level2_resample_all_landmarks": summarise_distribution(level2, metric_cols),
    }
    if len(level2_split):
        for subset, g in level2_split.groupby("subset"):
            summary[f"level2_{subset}"] = summarise_distribution(g, metric_cols)
            summary[f"level2_{subset}"]["n_landmarks_median"] = float(g["n_landmarks"].median())
            summary[f"level2_{subset}"]["n_patients_with_landmarks_median"] = float(g["n_patients_with_landmarks"].median())

    # Confirm the first seed replicate matches the saved stability permutation.
    expected = pd.read_csv(root / "results/lda/stability/revision/csv/level1_seed_per_rep.csv")
    got = level1.loc[level1.rep == 0, "alignment_perm"].iloc[0]
    summary["seed0_alignment_matches_stability_csv"] = bool(got == expected.loc[expected.rep == 0, "alignment_perm"].iloc[0])

    evaluable.drop(columns=["month_row_m2", "month_row_m1", "month_row_m"]).to_csv(
        out / "reference_landmarks.csv", index=False
    )
    months.drop(columns=[], errors="ignore").to_csv(out / "reference_monthly_puf.csv", index=False)
    level1.to_csv(out / "level1_alarm_per_rep.csv", index=False)
    level2.to_csv(out / "level2_alarm_per_rep.csv", index=False)
    level2_split.to_csv(out / "level2_alarm_by_split.csv", index=False)
    held_rows = []
    for r in results:
        if r["mode"] != "resample":
            continue
        for pid in r["held_out_ids"]:
            held_rows.append({"rep": r["rep"], "seed": r["seed"], "held_out_id": pid})
    pd.DataFrame(held_rows).to_csv(out / "level2_held_out_ids.csv", index=False)
    (out / "alarm_summary.json").write_text(json.dumps(summary, indent=2))

    plot_level(
        level1,
        out / "figures" / "level1_alarm_stability.png",
        "Level 1 — seed replicates, monthly load and alarm",
    )
    plot_level(
        level2,
        out / "figures" / "level2_alarm_stability.png",
        "Level 2 — leave-20%-out replicates, all evaluable landmarks",
    )
    print(json.dumps(summary["reference_performance"], indent=2))
    print("seed0 match", summary["seed0_alignment_matches_stability_csv"])
    print("level1 median auc3 r", summary["level1_seed"]["auc3_pearson"]["median"])
    print("level2 median auc3 r", summary["level2_resample_all_landmarks"]["auc3_pearson"]["median"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
