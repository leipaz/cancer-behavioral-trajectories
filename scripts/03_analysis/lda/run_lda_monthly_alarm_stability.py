#!/usr/bin/env python3
"""Extend LDA stability to monthly collapsed alarm landmarks (digital observation time).

Starting point (locked to ``results/alarm/base_analysis/granularity_report.docx``):
**1019** evaluable monthly-collapsed landmarks from
``results/alarm/base_analysis/decision_points_collapsed.csv``,
W=3 months, H=4 months (120 d), clock = Tiempo_obs digital / t_evento_eb2,
fixed cut ``S_m = AUC_W >= 1.0596384140662845`` (Youden on the prior 1013 set;
θ/3 ≈ 0.3532; not re-tuned per replica). Same denominators as §2 of that report
(159 positive / 860 negative; sens/spec ≈ 0.884 / 0.451 at this θ).

For each aligned LDA replicate (Hungarian cosine; F=T0–2, U=T3–5; no outcomes):
  1) monthly pUF and 3-block load vs reference (corr, bias, MAE)
  2) alarm decision flips on the 1019 base (ref−→rep+, ref+→rep−); missing scores kept
  3) AUROC / sens / spec / PPV / NPV on the same evaluable points

Leave-20%-out: infer on held-out patients too; report in-fit vs held-out separately.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import LdaModel
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_NUM_TOPICS,
    DICTIONARY_PATH,
    LDA_MODEL_PATH,
    PROJECT_ROOT,
    USER_EMBEDDINGS_CSV,
)
from lda_utils import load_lda_artifacts  # noqa: E402

UNFAV = (3, 4, 5)
FAV = (0, 1, 2)
# Youden cut from base_analysis/granularity_report.docx §2 (sum scale S_m = 3 B_m).
THR_SUM = 1.0596384140662845
THR_MEAN = THR_SUM / 3.0  # ≈ 0.3532128046887615
H = 120
WINDOW = 30
N_BASE = 1019
COLLAPSED_BASE = PROJECT_ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv"
GRANULARITY_REPORT = PROJECT_ROOT / "results/alarm/base_analysis/granularity_report.docx"
DEFAULT_OUT = PROJECT_ROOT / "results/lda/stability/monthly_alarm"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "results/lda/stability/replica_models"
SUBSET_ALL = "all_1019"


def parse_ids(v):
    return list(ast.literal_eval(v)) if isinstance(v, str) else list(v)


def topic_matrix(lda: LdaModel) -> np.ndarray:
    return np.asarray(lda.get_topics(), dtype=float)


def cosine_sim(p, q) -> float:
    num = float(np.dot(p, q))
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    return num / den if den > 0 else 0.0


def align_topics_cosine(ref: np.ndarray, other: np.ndarray):
    k = ref.shape[0]
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = -cosine_sim(ref[i], other[j])
    row, col = linear_sum_assignment(cost)
    perm = np.empty(k, dtype=int)
    perm[row] = col
    return perm


def train_lda(corpus_bow, dictionary, num_topics: int, passes: int, seed: int) -> LdaModel:
    return LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=passes,
        chunksize=100,
        alpha="symmetric",
        eta="symmetric",
        iterations=50,
        eval_every=None,
    )


def puf_series_for_patient(emb_ids: list, lda: LdaModel, dictionary, perm: np.ndarray) -> dict[int, float]:
    """study_day -> pUF after aligning replica topics to reference indices."""
    n = len(emb_ids)
    out = {}
    if n < 15:
        return out
    if n < WINDOW:
        bow = dictionary.doc2bow([str(t) for t in emb_ids])
        probs = np.zeros(lda.num_topics)
        for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
            probs[int(tid)] = float(p)
        aligned = probs[perm]
        out[n] = float(aligned[list(UNFAV)].sum())
        return out
    for end in range(WINDOW, n + 1):
        bow = dictionary.doc2bow([str(t) for t in emb_ids[end - WINDOW : end]])
        probs = np.zeros(lda.num_topics)
        for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
            probs[int(tid)] = float(p)
        aligned = probs[perm]
        out[end] = float(aligned[list(UNFAV)].sum())
    return out


def clasif(y, score, thr):
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    pred = s >= thr
    tp = float(((pred) & (y == 1)).sum())
    fn = float(((~pred) & (y == 1)).sum())
    fp = float(((pred) & (y == 0)).sum())
    tn = float(((~pred) & (y == 0)).sum())

    def r(a, b):
        return a / b if b else np.nan

    try:
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    return dict(
        AUROC=auc,
        sensitivity=r(tp, tp + fn),
        specificity=r(tn, tn + fp),
        PPV=r(tp, tp + fp),
        NPV=r(tn, tn + fn),
        TP=tp,
        FN=fn,
        FP=fp,
        TN=tn,
        n=int(len(y)),
    )


def load_base_collapsed() -> pd.DataFrame:
    """Monthly-collapsed landmarks with digital-observation labels (base_analysis)."""
    pts = pd.read_csv(COLLAPSED_BASE)
    base = pts.copy()
    base["id"] = base["id"].astype(int)
    base["study_day"] = base["decision_day"].astype(int)
    base["month"] = base["month"].astype(int)
    base["y"] = base["target"].astype(int)
    base["score_sum_ref"] = base["AUC_W"].astype(float)
    base["alarm_ref"] = (base["score_sum_ref"] >= THR_SUM).astype(int)
    assert len(base) == N_BASE, f"expected {N_BASE} evaluable points, got {len(base)}"
    return base.reset_index(drop=True)


def fixed_leave_masks(n: int, n_reps: int, leave_out: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_out = max(1, int(round(leave_out * n)))
    return [np.sort(rng.choice(n, size=n_out, replace=False)) for _ in range(n_reps)]


def score_at_points(
    qmap: dict[int, dict[int, float]],
    base: pd.DataFrame,
    q_ref: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for r in base.itertuples(index=False):
        pid = int(r.id)
        q = qmap.get(pid, {})
        qr = (q_ref or {}).get(pid, {})
        t = int(r.study_day)
        ok = (t - 60) in q and (t - 30) in q and t in q
        if ok:
            s = float(q[t - 60] + q[t - 30] + q[t])
            pufs = (q[t - 60], q[t - 30], q[t])
        else:
            s = np.nan
            pufs = (np.nan, np.nan, np.nan)
        ok_r = (t - 60) in qr and (t - 30) in qr and t in qr
        pufs_r = (qr.get(t - 60, np.nan), qr.get(t - 30, np.nan), qr.get(t, np.nan)) if ok_r else (np.nan,) * 3
        rows.append(
            {
                "id": pid,
                "study_day": t,
                "y": int(r.y),
                "score_sum_ref": float(r.score_sum_ref),
                "alarm_ref": int(r.alarm_ref),
                "pUF_m2_ref": pufs_r[0],
                "pUF_m1_ref": pufs_r[1],
                "pUF_m0_ref": pufs_r[2],
                "pUF_m2": pufs[0],
                "pUF_m1": pufs[1],
                "pUF_m0": pufs[2],
                "score_sum_rep": s,
                "score_missing": int(not ok),
            }
        )
    return pd.DataFrame(rows)


def summarize_replica(cmp: pd.DataFrame, subset_name: str, rep_meta: dict) -> dict:
    have = cmp[cmp.score_missing == 0]
    miss = int(cmp.score_missing.sum())
    denom = int(len(cmp))
    out = {
        **rep_meta,
        "subset": subset_name,
        "denom_base_points": denom,
        "n_scored": int(len(have)),
        "n_score_missing": miss,
        "frac_score_missing": miss / denom if denom else np.nan,
    }
    if len(have) < 2:
        out.update({k: np.nan for k in [
            "puf_block_pearson", "puf_block_bias", "puf_block_mae",
            "load_pearson", "load_bias", "load_mae",
            "flip_neg_to_pos", "flip_pos_to_neg", "n_decision_change", "frac_decision_change",
            "AUROC", "sensitivity", "specificity", "PPV", "NPV",
        ]})
        return out

    r_load, _ = pearsonr(have.score_sum_ref, have.score_sum_rep)
    bias = float((have.score_sum_rep - have.score_sum_ref).mean())
    mae = float((have.score_sum_rep - have.score_sum_ref).abs().mean())

    # pooled monthly block pUF (three blocks × points)
    xref = np.concatenate([have.pUF_m2_ref, have.pUF_m1_ref, have.pUF_m0_ref])
    yrep = np.concatenate([have.pUF_m2, have.pUF_m1, have.pUF_m0])
    m = np.isfinite(xref) & np.isfinite(yrep)
    if m.sum() >= 2:
        r_puf, _ = pearsonr(xref[m], yrep[m])
        puf_bias = float((yrep[m] - xref[m]).mean())
        puf_mae = float(np.abs(yrep[m] - xref[m]).mean())
    else:
        r_puf, puf_bias, puf_mae = np.nan, np.nan, np.nan

    alarm_rep = (have.score_sum_rep >= THR_SUM).astype(int)
    flip_np = int(((have.alarm_ref == 0) & (alarm_rep == 1)).sum())
    flip_pn = int(((have.alarm_ref == 1) & (alarm_rep == 0)).sum())
    changed = flip_np + flip_pn
    perf = clasif(have.y.to_numpy(), have.score_sum_rep.to_numpy(), THR_SUM)
    out.update(
        dict(
            puf_block_pearson=float(r_puf) if np.isfinite(r_puf) else np.nan,
            puf_block_bias=puf_bias,
            puf_block_mae=puf_mae,
            load_pearson=float(r_load) if np.isfinite(r_load) else np.nan,
            load_bias=bias,
            load_mae=mae,
            flip_neg_to_pos=flip_np,
            flip_pos_to_neg=flip_pn,
            n_decision_change=changed,
            frac_decision_change=changed / len(have),
            **perf,
        )
    )
    return out


def monthly_puf_corr_vs_ref(
    q_rep: dict[int, dict[int, float]],
    q_ref: dict[int, dict[int, float]],
    ids: list[int] | None = None,
) -> dict:
    xs, ys = [], []
    for pid, q in q_rep.items():
        if ids is not None and pid not in ids:
            continue
        qr = q_ref.get(pid, {})
        for day, p in q.items():
            if day in qr:
                xs.append(qr[day])
                ys.append(p)
    if len(xs) < 2:
        return dict(puf_day_pearson=np.nan, puf_day_bias=np.nan, puf_day_mae=np.nan, n_day_pairs=0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    r, _ = pearsonr(xs, ys)
    return dict(
        puf_day_pearson=float(r) if np.isfinite(r) else np.nan,
        puf_day_bias=float((ys - xs).mean()),
        puf_day_mae=float(np.abs(ys - xs).mean()),
        n_day_pairs=int(len(xs)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--reference-model", type=Path, default=LDA_MODEL_PATH)
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("--n-seed", type=int, default=5, help="Level-1 seeds (pilot default 5; use 50 for full)")
    parser.add_argument("--n-resample", type=int, default=5)
    parser.add_argument("--leave-out", type=float, default=0.2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--resample-seed", type=int, default=0)
    parser.add_argument("--mode", choices=("seed", "resample", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save/load replica LDA models (avoid re-training).",
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Load replica models from --models-dir when present instead of training.",
    )
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "seed").mkdir(parents=True, exist_ok=True)
    (models_dir / "resample").mkdir(parents=True, exist_ok=True)

    print(f"Loading reference LDA + embeddings + collapsed base ({N_BASE}, θ={THR_SUM})…")
    dictionary, ref_model, df_profiles, documents, corpus_bow = load_lda_artifacts(
        args.dictionary, args.reference_model, args.profiles_csv
    )
    ref_topics = topic_matrix(ref_model)
    base = load_base_collapsed()
    emb_map = {
        int(r.user_id): parse_ids(r.embedding_ids)
        for r in df_profiles.itertuples(index=False)
    }
    # identity perm for reference
    ref_perm = np.arange(ref_model.num_topics)
    q_ref = {pid: puf_series_for_patient(ids, ref_model, dictionary, ref_perm) for pid, ids in emb_map.items()}

    # sanity: rebuild ref scores on base and compare to stored AUC_W
    chk = score_at_points(q_ref, base, q_ref)
    ok = chk.score_missing == 0
    if ok.any():
        mae = float((chk.loc[ok, "score_sum_rep"] - chk.loc[ok, "score_sum_ref"]).abs().mean())
        print(f"  ref rebuild vs stored AUC_W MAE={mae:.6f} (near 0 if monthly pUF = 30d sliding end)")

    n_docs = len(corpus_bow)
    user_ids = df_profiles["user_id"].astype(int).to_numpy()
    leave_masks = fixed_leave_masks(n_docs, args.n_resample, args.leave_out, args.resample_seed)

    design = {
        "starting_point": {
            "alarm_run": "base_analysis monthly collapsed (§2 granularity_report.docx)",
            "report": str(GRANULARITY_REPORT.relative_to(PROJECT_ROOT)),
            "clock": "Tiempo_obs digital / t_evento_eb2 (gap-corrected; labels in CSV target)",
            "landmarks_csv": str(COLLAPSED_BASE.relative_to(PROJECT_ROOT)),
            "n_evaluable_landmarks": N_BASE,
            "n_positive_negative": "159 / 860",
            "cadence": "monthly collapsed (decision_day = month × 30)",
            "W_blocks": 3,
            "H_days": H,
            "threshold_sum_S_m": THR_SUM,
            "threshold_mean_B_m": THR_MEAN,
            "threshold_not_retuned_per_replica": True,
            "note": "Youden from prior 1013 set, applied to 1019 collapsed landmarks as in the report table",
        },
        "lda": {
            "K": 6,
            "passes": args.passes,
            "iterations": 50,
            "hungarian": "cosine P(day-type|topic), no outcomes",
            "F": list(FAV),
            "U": list(UNFAV),
        },
        "seeds": list(range(args.seed_start, args.seed_start + args.n_seed)),
        "leave_out": {
            "fraction": args.leave_out,
            "rng_seed": args.resample_seed,
            "masks": [m.tolist() for m in leave_masks],
        },
        "models_dir": str(models_dir),
        "reuse_models": bool(args.reuse_models),
    }
    (out / "design_meta.json").write_text(json.dumps(design, indent=2))
    (models_dir / "design_meta.json").write_text(json.dumps(design, indent=2))
    base.to_csv(out / "base_landmarks_1019_collapsed.csv", index=False)

    summary_rows = []
    point_rows = []

    def replica_stem(mode: str, key: int) -> Path:
        return models_dir / mode / f"rep_{key:03d}"

    def save_replica(stem: Path, model: LdaModel, perm: np.ndarray, train_idx: np.ndarray, seed: int, mode: str, key: int):
        stem.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(stem.with_suffix(".gensim")))
        meta = {
            "mode": mode,
            "key": int(key),
            "seed": int(seed),
            "passes": int(args.passes),
            "n_train": int(len(train_idx)),
            "train_idx": train_idx.astype(int).tolist(),
            "alignment_perm": [int(x) for x in perm.tolist()],
            "F": list(FAV),
            "U": list(UNFAV),
        }
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    def load_or_train(mode: str, key: int, train_idx: np.ndarray, seed: int) -> tuple[LdaModel, np.ndarray]:
        stem = replica_stem(mode, key)
        gensim_path = stem.with_suffix(".gensim")
        meta_path = stem.with_suffix(".json")
        if args.reuse_models and gensim_path.exists() and meta_path.exists():
            print(f"  [{mode} key={key}] LOAD {gensim_path.name}", flush=True)
            model = LdaModel.load(str(gensim_path))
            meta = json.loads(meta_path.read_text())
            perm = np.asarray(meta["alignment_perm"], dtype=int)
            return model, perm
        bow_train = [corpus_bow[i] for i in train_idx]
        print(f"  [{mode} key={key}] TRAIN n={len(train_idx)} seed={seed} → save {stem.name}", flush=True)
        model = train_lda(bow_train, dictionary, DEFAULT_NUM_TOPICS, args.passes, seed)
        perm = align_topics_cosine(ref_topics, topic_matrix(model))
        save_replica(stem, model, perm, train_idx, seed, mode, key)
        return model, perm

    def run_one(mode: str, key: int, train_idx: np.ndarray, seed: int, held_out_ids: set[int] | None):
        nonlocal summary_rows, point_rows
        model, perm = load_or_train(mode, key, train_idx, seed)
        q_rep = {pid: puf_series_for_patient(ids, model, dictionary, perm) for pid, ids in emb_map.items()}
        day_corr = monthly_puf_corr_vs_ref(q_rep, q_ref)
        cmp_all = score_at_points(q_rep, base, q_ref)
        cmp_all["mode"] = mode
        cmp_all["key"] = key
        cmp_all["seed"] = seed
        meta = dict(mode=mode, key=key, seed=seed, n_train=int(len(train_idx)), passes=args.passes)
        meta.update(day_corr)

        if mode == "seed":
            summary_rows.append(summarize_replica(cmp_all, SUBSET_ALL, meta))
            point_rows.append(cmp_all)
        else:
            assert held_out_ids is not None
            in_ids = set(user_ids[train_idx].tolist())
            out_ids = held_out_ids
            for name, idset in (("in_fit", in_ids), ("held_out", out_ids), (SUBSET_ALL, None)):
                sub = cmp_all if idset is None else cmp_all[cmp_all.id.isin(idset)]
                summary_rows.append(summarize_replica(sub, name, meta))
            point_rows.append(cmp_all)

    modes = ["seed", "resample"] if args.mode == "both" else [args.mode]
    if "seed" in modes:
        print("\n=== Level 1 seeds → monthly / alarm ===")
        for k, seed in enumerate(range(args.seed_start, args.seed_start + args.n_seed)):
            run_one("seed", k, np.arange(n_docs), seed, None)
    if "resample" in modes:
        print("\n=== Level 2 leave-20% → monthly / alarm (in-fit + held-out) ===")
        for k, out_idx in enumerate(leave_masks):
            train_idx = np.setdiff1d(np.arange(n_docs), out_idx)
            held = set(user_ids[out_idx].tolist())
            run_one("resample", k, train_idx, args.resample_seed + k, held)

    summ = pd.DataFrame(summary_rows)
    pts = pd.concat(point_rows, ignore_index=True)
    summ.to_csv(out / "per_replica_summary.csv", index=False)
    pts.to_csv(out / "per_point_scores.csv", index=False)

    # short headline JSON
    headline = {}
    for mode in summ["mode"].unique():
        for subset in summ.loc[summ["mode"] == mode, "subset"].unique():
            sub = summ[(summ["mode"] == mode) & (summ["subset"] == subset)]
            headline[f"{mode}:{subset}"] = {
                "n_reps": int(len(sub)),
                "puf_block_pearson_median": float(sub.puf_block_pearson.median()),
                "puf_block_mae_median": float(sub.puf_block_mae.median()),
                "load_pearson_median": float(sub.load_pearson.median()),
                "load_mae_median": float(sub.load_mae.median()),
                "load_bias_median": float(sub.load_bias.median()),
                "frac_decision_change_median": float(sub.frac_decision_change.median()),
                "flip_neg_to_pos_median": float(sub.flip_neg_to_pos.median()),
                "flip_pos_to_neg_median": float(sub.flip_pos_to_neg.median()),
                "AUROC_median": float(sub.AUROC.median()),
                "sensitivity_median": float(sub.sensitivity.median()),
                "specificity_median": float(sub.specificity.median()),
                "PPV_median": float(sub.PPV.median()),
                "NPV_median": float(sub.NPV.median()),
                "n_score_missing_median": float(sub.n_score_missing.median()),
            }
    (out / "headline_summary.json").write_text(json.dumps(headline, indent=2))

    readme = f"""# Monthly-block + alarm stability (collapsed, digital observation time)

## Starting point (locked)
- Report: `results/alarm/base_analysis/granularity_report.docx` (§2 Collapsed monthly)
- Landmarks: `results/alarm/base_analysis/decision_points_collapsed.csv`
- **{N_BASE}** evaluable monthly-collapsed landmarks (159 pos / 860 neg; 116 patients)
- Clock: **Tiempo_obs digital** / t_evento_eb2 (labels in `target`)
- W = 3 months, H = 4 months (120 days)
- Fixed cut: **S_m = AUC_W ≥ {THR_SUM}** (θ/3 = {THR_MEAN}; Youden on prior 1013; not re-tuned per replica)

## Method
- Freeze VQ-VAE day-types + dictionary; re-fit LDA replicates (K=6, {args.passes} passes)
- Hungarian align topics by cosine of P(day-type|topic); **no outcomes**
- After alignment: F=T0–2, U=T3–5, pUF = P(U)
- 30-day block pUF at month ends → 3-block load (`AUC_W` / `S_m`) at each of the {N_BASE} points
- Leave-20%-out: inference on held-out patients allowed; metrics split **in_fit** vs **held_out**

## Outputs
- `design_meta.json`, `base_landmarks_1019_collapsed.csv`
- `per_replica_summary.csv`, `per_point_scores.csv`, `headline_summary.json`
- Replica LDA models (reusable): `{models_dir}/seed/rep_XXX.gensim` (+ `.json` with alignment perm / train_idx)
  and `{models_dir}/resample/rep_XXX.gensim`. Re-run with `--reuse-models` to skip training.
"""
    (out / "README.md").write_text(readme)
    (models_dir / "README.md").write_text(
        "# Saved LDA stability replicas\n\n"
        "Each `rep_XXX.gensim` is a K=6 model trained with 100 passes "
        "(same hyper-parameters as the stability battery). "
        "Matching `rep_XXX.json` stores seed, train_idx, and Hungarian "
        "`alignment_perm` onto the published reference topics.\n\n"
        "Reuse with:\n"
        "`python scripts/03_analysis/lda/run_lda_monthly_alarm_stability.py "
        "--reuse-models --models-dir results/lda/stability/replica_models ...`\n"
    )
    print(json.dumps(headline, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
