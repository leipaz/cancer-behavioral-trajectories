#!/usr/bin/env python3
"""Passes-budget convergence check for LDA stability replicates.

Compares the current replicate budget (100 gensim ``passes``) with larger
budgets on a **small, a-priori fixed** set of seeds and leave-out masks.
Same DB1-reconciled documents and the same init hyper-parameters as
``run_lda_stability.py``. Does **not** optimise clinical metrics.

Outputs under ``results/lda/stability/convergence_passes/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_NUM_TOPICS,
    DICTIONARY_PATH,
    LDA_MODEL_PATH,
    USER_EMBEDDINGS_CSV,
)
from lda_utils import load_lda_artifacts  # noqa: E402

UNFAVOURABLE_REF = (3, 4, 5)
FAVOURABLE_REF = (0, 1, 2)

# Fixed a priori (do not change after first delivery without noting it)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BUDGETS = (100, 1000, 5000)
DEFAULT_LEAVE_OUT = 0.2
DEFAULT_RESAMPLE_SEED = 0
DEFAULT_N_RESAMPLE = 5
DEFAULT_CURVE_SEED = 0
DEFAULT_CURVE_MAX = 2000
DEFAULT_CURVE_EVERY = 25


def topic_matrix(lda: LdaModel) -> np.ndarray:
    return np.asarray(lda.get_topics(), dtype=float)


def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    num = float(np.dot(p, q))
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    return num / den if den > 0 else 0.0


def mix_topics(topic_mat: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    m = topic_mat[list(indices)].mean(axis=0)
    s = float(m.sum())
    return m / s if s > 0 else m


def align_topics_cosine(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = ref.shape[0]
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -cosine_sim(ref[i], other[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(k, dtype=int)
    perm[row_ind] = col_ind
    cos = np.array([cosine_sim(ref[i], other[perm[i]]) for i in range(k)])
    return perm, cos


def doc_topic_matrix(lda: LdaModel, corpus_bow) -> np.ndarray:
    k = lda.num_topics
    rows = []
    for bow in corpus_bow:
        probs = np.zeros(k, dtype=float)
        for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
            probs[int(tid)] = float(p)
        rows.append(probs)
    return np.asarray(rows, dtype=float)


def train_lda(corpus_bow, dictionary, num_topics: int, passes: int, seed: int) -> LdaModel:
    """Same hyper-parameters as stability replicates (except passes)."""
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


def fixed_leave_out_masks(n: int, n_reps: int, leave_out: float, seed: int) -> list[np.ndarray]:
    """Pre-declare leave-out index sets (train = complement)."""
    rng = np.random.default_rng(seed)
    n_out = max(1, int(round(leave_out * n)))
    masks = []
    for _ in range(n_reps):
        out_idx = np.sort(rng.choice(n, size=n_out, replace=False))
        masks.append(out_idx)
    return masks


def metrics_vs_ref(
    model: LdaModel,
    ref_topics: np.ndarray,
    ref_doc: np.ndarray,
    corpus_bow,
    eval_idx: np.ndarray,
) -> dict:
    other = topic_matrix(model)
    perm, cos = align_topics_cosine(ref_topics, other)
    aligned = other[perm]
    bow_eval = [corpus_bow[i] for i in eval_idx]
    doc = doc_topic_matrix(model, bow_eval)[:, perm]
    ref_sub = ref_doc[eval_idx]
    puf = doc[:, list(UNFAVOURABLE_REF)].sum(axis=1)
    ref_puf = ref_sub[:, list(UNFAVOURABLE_REF)].sum(axis=1)
    puf_r, _ = pearsonr(ref_puf, puf)
    fav_c = cosine_sim(mix_topics(ref_topics, FAVOURABLE_REF), mix_topics(aligned, FAVOURABLE_REF))
    unfav_c = cosine_sim(mix_topics(ref_topics, UNFAVOURABLE_REF), mix_topics(aligned, UNFAVOURABLE_REF))
    bound = float(model.bound(bow_eval))
    return {
        "mean_cosine_vs_ref": float(np.mean(cos)),
        "min_cosine_vs_ref": float(np.min(cos)),
        "fav_cosine_vs_ref": float(fav_c),
        "unfav_cosine_vs_ref": float(unfav_c),
        "puf_pearson_vs_ref": float(puf_r) if np.isfinite(puf_r) else np.nan,
        "bound_eval": bound,
        "alignment_perm": ",".join(map(str, map(int, perm))),
        "topic_mat": aligned,
        "doc_topics": doc,
    }


def pair_budget_metrics(a: dict, b: dict) -> dict:
    """Compare two budgets.

    Topic mats / doc topics in ``a`` and ``b`` are each Hungarian-aligned to the
    published reference. For between-budget pUF we further align ``b`` to ``a``
    so a topic-index swap vs the reference does not look like a pUF change.
    """
    cos = np.array(
        [cosine_sim(a["topic_mat"][t], b["topic_mat"][t]) for t in range(a["topic_mat"].shape[0])]
    )
    perm, cos_h = align_topics_cosine(a["topic_mat"], b["topic_mat"])
    puf_a = a["doc_topics"][:, list(UNFAVOURABLE_REF)].sum(axis=1)
    doc_b = b["doc_topics"][:, perm]
    puf_b = doc_b[:, list(UNFAVOURABLE_REF)].sum(axis=1)
    n = min(len(puf_a), len(puf_b))
    r, _ = pearsonr(puf_a[:n], puf_b[:n])
    return {
        "mean_cosine_refaligned": float(np.mean(cos)),
        "mean_cosine_hungarian": float(np.mean(cos_h)),
        "min_cosine_hungarian": float(np.min(cos_h)),
        "puf_pearson_between_budgets": float(r) if np.isfinite(r) else np.nan,
        "delta_bound": float(b["bound_eval"] - a["bound_eval"]),
    }


def run_budget_grid(
    *,
    mode: str,
    corpus_bow,
    dictionary,
    ref_topics: np.ndarray,
    ref_doc: np.ndarray,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...] | None,
    leave_masks: list[np.ndarray] | None,
    num_topics: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n = len(corpus_bow)
    snapshot_rows = []
    pair_rows = []
    store: dict[tuple, dict] = {}

    if mode == "seed":
        assert seeds is not None
        jobs = [("seed", s, np.arange(n), s) for s in seeds]
    else:
        assert leave_masks is not None
        jobs = []
        for r, out_idx in enumerate(leave_masks):
            train_idx = np.setdiff1d(np.arange(n), out_idx)
            jobs.append(("resample", r, train_idx, DEFAULT_RESAMPLE_SEED + r))

    for mode_name, key, train_idx, seed in jobs:
        bow_train = [corpus_bow[i] for i in train_idx]
        eval_idx = train_idx
        for passes in budgets:
            print(
                f"  [{mode_name} key={key}] passes={passes} n_train={len(train_idx)} seed={seed}",
                flush=True,
            )
            model = train_lda(bow_train, dictionary, num_topics, passes, seed)
            m = metrics_vs_ref(model, ref_topics, ref_doc, corpus_bow, eval_idx)
            store[(mode_name, key, passes)] = m
            snapshot_rows.append(
                {
                    "mode": mode_name,
                    "key": int(key),
                    "seed": int(seed),
                    "n_train": int(len(train_idx)),
                    "passes": int(passes),
                    "mean_cosine_vs_ref": m["mean_cosine_vs_ref"],
                    "min_cosine_vs_ref": m["min_cosine_vs_ref"],
                    "fav_cosine_vs_ref": m["fav_cosine_vs_ref"],
                    "unfav_cosine_vs_ref": m["unfav_cosine_vs_ref"],
                    "puf_pearson_vs_ref": m["puf_pearson_vs_ref"],
                    "bound_eval": m["bound_eval"],
                }
            )

        base = budgets[0]
        for other in budgets[1:]:
            cmp = pair_budget_metrics(store[(mode_name, key, base)], store[(mode_name, key, other)])
            pair_rows.append(
                {
                    "mode": mode_name,
                    "key": int(key),
                    "seed": int(seed),
                    "passes_a": int(base),
                    "passes_b": int(other),
                    **{k: v for k, v in cmp.items()},
                }
            )

    return pd.DataFrame(snapshot_rows), pd.DataFrame(pair_rows), store


def run_bound_curve(
    corpus_bow,
    dictionary,
    num_topics: int,
    seed: int,
    max_passes: int,
    every: int,
) -> pd.DataFrame:
    """Incremental updates; log variational bound every ``every`` passes."""
    print(f"  [bound curve] seed={seed} max_passes={max_passes} every={every}", flush=True)
    model = LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=1,
        chunksize=100,
        alpha="symmetric",
        eta="symmetric",
        iterations=50,
        eval_every=None,
    )
    rows = []
    for p in range(1, max_passes + 1):
        if p > 1:
            model.update(corpus_bow, passes=1)
        if p == 1 or p % every == 0 or p in (100, 1000) or p == max_passes:
            bound = float(model.bound(corpus_bow))
            rows.append({"pass": p, "bound": bound, "seed": seed})
            print(f"    pass={p} bound={bound:.4f}", flush=True)
    return pd.DataFrame(rows)


def plot_bound_curve(df: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["pass"], df["bound"], "-o", ms=3, lw=1.5, color="#1f77b4")
    for mark in (100, 1000):
        if (df["pass"] == mark).any():
            ax.axvline(mark, color="#2ca02c" if mark == 100 else "#ff7f0e", ls="--", lw=1.2, label=f"{mark} passes")
    ax.set_xlabel("Training passes")
    ax.set_ylabel("Variational bound (higher is better)")
    ax.set_title("LDA training objective vs passes (fixed seed, full cohort)")
    # Absolute scale from below the minimum up to 0 (bound is negative)
    ymin = float(df["bound"].min())
    ax.set_ylim(ymin - 0.02 * abs(ymin), 0)
    ax.axhline(0, color="0.5", lw=0.8, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(Path(str(out_stem) + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_budget_pairs(pairs: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, title in zip(
        axes,
        ["mean_cosine_hungarian", "puf_pearson_between_budgets"],
        ["Topic-profile cosine (100 vs higher)", "Patient pUF Pearson (100 vs higher)"],
    ):
        for mode in ("seed", "resample"):
            sub = pairs[pairs["mode"] == mode]
            for passes_b in sorted(sub["passes_b"].unique()):
                ss = sub[sub["passes_b"] == passes_b]
                ax.scatter(
                    np.full(len(ss), float(passes_b)) + (-40 if mode == "seed" else 40),
                    ss[metric],
                    label=f"{mode}, vs {int(passes_b)}",
                    alpha=0.75,
                    s=40,
                )
        ax.axhline(0.95 if "cosine" in metric else 0.9, color="green", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("Higher pass budget")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    # unique labels
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    axes[0].legend(h2, l2, fontsize=7, loc="best")
    fig.tight_layout()
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(Path(str(out_stem) + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--reference-model", type=Path, default=LDA_MODEL_PATH)
    parser.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS)
    parser.add_argument(
        "--budgets",
        type=str,
        default=",".join(map(str, DEFAULT_BUDGETS)),
        help="Comma-separated pass budgets (first = current replicate budget)",
    )
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n-resample", type=int, default=DEFAULT_N_RESAMPLE)
    parser.add_argument("--leave-out", type=float, default=DEFAULT_LEAVE_OUT)
    parser.add_argument("--resample-seed", type=int, default=DEFAULT_RESAMPLE_SEED)
    parser.add_argument("--curve-seed", type=int, default=DEFAULT_CURVE_SEED)
    parser.add_argument("--curve-max", type=int, default=DEFAULT_CURVE_MAX)
    parser.add_argument("--curve-every", type=int, default=DEFAULT_CURVE_EVERY)
    parser.add_argument("--skip-curve", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/lda/stability/convergence_passes"),
    )
    args = parser.parse_args()

    budgets = tuple(int(x) for x in args.budgets.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts…")
    dictionary, ref_model, _df, _docs, corpus_bow = load_lda_artifacts(
        args.dictionary, args.reference_model, args.profiles_csv
    )
    ref_topics = topic_matrix(ref_model)
    ref_doc = doc_topic_matrix(ref_model, corpus_bow)
    n = len(corpus_bow)

    leave_masks = fixed_leave_out_masks(n, args.n_resample, args.leave_out, args.resample_seed)

    design = {
        "purpose": "Check whether stability replicates at 100 passes are under-trained vs higher pass budgets",
        "not_for": "Clinical metric tuning or selecting a new operating point",
        "reference_model": {
            "passes": int(getattr(ref_model, "passes", -1)),
            "iterations": int(getattr(ref_model, "iterations", -1)),
            "note": "Published reference; not re-trained here",
        },
        "replicate_match": {
            "chunksize": 100,
            "alpha": "symmetric",
            "eta": "symmetric",
            "iterations": 50,
            "current_stability_passes": 100,
        },
        "fixed_seeds_level1": list(seeds),
        "fixed_resample": {
            "n_reps": args.n_resample,
            "leave_out_fraction": args.leave_out,
            "rng_seed": args.resample_seed,
            "leave_out_indices": [m.tolist() for m in leave_masks],
        },
        "budgets": list(budgets),
        "bound_curve": {
            "seed": args.curve_seed,
            "max_passes": args.curve_max,
            "log_every": args.curve_every,
            "skipped": bool(args.skip_curve),
        },
        "corpus": "DB1-reconciled day-type documents (same as stability folder)",
        "n_documents": n,
    }
    (out_dir / "design_meta.json").write_text(json.dumps(design, indent=2))

    all_snap, all_pairs = [], []
    for mode in ("seed", "resample"):
        print(f"\n=== Budget grid: {mode} ===")
        snap, pairs, _ = run_budget_grid(
            mode=mode,
            corpus_bow=corpus_bow,
            dictionary=dictionary,
            ref_topics=ref_topics,
            ref_doc=ref_doc,
            budgets=budgets,
            seeds=seeds if mode == "seed" else None,
            leave_masks=leave_masks if mode == "resample" else None,
            num_topics=args.num_topics,
        )
        all_snap.append(snap)
        all_pairs.append(pairs)

    snapshots = pd.concat(all_snap, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    snapshots.to_csv(out_dir / "snapshots_by_budget.csv", index=False)
    pairs.to_csv(out_dir / "pairwise_100_vs_higher.csv", index=False)

    # Summary: is 100 close to higher budgets?
    summary = {
        "budgets": list(budgets),
        "by_comparison": {},
    }
    base = budgets[0]
    for passes_b in budgets[1:]:
        sub = pairs[pairs["passes_b"] == passes_b]
        summary["by_comparison"][f"{base}_vs_{passes_b}"] = {
            "n": int(len(sub)),
            "mean_cosine_hungarian_median": float(sub["mean_cosine_hungarian"].median()),
            "mean_cosine_hungarian_min": float(sub["mean_cosine_hungarian"].min()),
            "puf_pearson_between_budgets_median": float(sub["puf_pearson_between_budgets"].median()),
            "puf_pearson_between_budgets_min": float(sub["puf_pearson_between_budgets"].min()),
            "delta_bound_median": float(sub["delta_bound"].median()),
        }
    # Heuristic sufficiency note (not a clinical threshold)
    mid = next((b for b in budgets[1:] if b >= 1000), budgets[1] if len(budgets) > 1 else None)
    cmp_mid = summary["by_comparison"].get(f"{base}_vs_{mid}", {}) if mid is not None else {}
    sufficient = (
        cmp_mid.get("mean_cosine_hungarian_median", 0) >= 0.95
        and cmp_mid.get("puf_pearson_between_budgets_median", 0) >= 0.95
    )
    summary["conclusion"] = (
        f"{base} passes already close to {mid} on this fixed design (topic cosine and pUF); "
        f"keep replicate budget at {base} unless a higher budget shows a material gap."
        if sufficient
        else f"{base} passes still move relative to {mid} on this design; document the gap and "
        "fix a higher budget before refreshing the full replicate set."
    )
    summary["sufficient_heuristic"] = bool(sufficient)
    summary["compared_to"] = mid
    (out_dir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))

    plot_budget_pairs(pairs, fig_dir / "passes_budget_agreement")

    if not args.skip_curve:
        print("\n=== Bound learning curve ===")
        curve = run_bound_curve(
            corpus_bow,
            dictionary,
            args.num_topics,
            args.curve_seed,
            args.curve_max,
            args.curve_every,
        )
        curve.to_csv(out_dir / "bound_curve.csv", index=False)
        plot_bound_curve(curve, fig_dir / "bound_vs_passes")

    readme = f"""# Passes convergence check (LDA stability)

## Question
Are the stability replicates at **100 gensim passes** under-trained relative to a larger pass budget?

## Design (fixed a priori)
- Same audited DB1 day-type documents as the stability analysis
- Same LDA hyper-parameters as replicates (`iterations=50`, symmetric α/η, chunksize=100)
- Level-1 style: seeds `{list(seeds)}` on the full cohort
- Level-2 style: `{args.n_resample}` leave-{int(100*args.leave_out)}%-out masks from RNG seed `{args.resample_seed}` (indices in `design_meta.json`)
- Pass budgets: `{list(budgets)}` (first = current replicate budget)
- Bound curve: seed `{args.curve_seed}`, up to `{args.curve_max}` passes

## Reference (not re-trained)
Published model: **passes = {design['reference_model']['passes']}**, **iterations = {design['reference_model']['iterations']}**.

## What this is not
Not a search for better clinical concordance; not a reason to change K or alarm thresholds by itself.

## Outputs
- `design_meta.json` — locked seeds / masks / budgets
- `snapshots_by_budget.csv` — metrics at each budget
- `pairwise_100_vs_higher.csv` — 100 vs each higher budget
- `bound_curve.csv` / figures — training objective vs passes
- `convergence_summary.json` — short quantitative wrap-up
"""
    (out_dir / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
