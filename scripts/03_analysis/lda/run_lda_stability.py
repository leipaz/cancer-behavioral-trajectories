#!/usr/bin/env python3
"""LDA stability at fixed K=6 (cosine similarity only).

Level 1 — seed stability: re-fit on all patients with different random seeds.
Level 2 — data stability: leave-out a fraction of patients and re-fit (resampling).

Topics are aligned to the published reference model with the Hungarian algorithm
on cosine similarity of ``P(profile | topic)``. Reports cosine distributions and
patient-level pUF concordance (topics 3+4+5 after alignment).

Examples
--------
.venv/bin/python scripts/03_analysis/lda/run_lda_stability.py --mode seed --n-reps 50
.venv/bin/python scripts/03_analysis/lda/run_lda_stability.py --mode resample --n-reps 50 --leave-out 0.2
.venv/bin/python scripts/03_analysis/lda/run_lda_stability.py --mode both --n-reps 50
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
    RESULTS_LDA_FIGURES,
    USER_EMBEDDINGS_CSV,
)
from lda_utils import load_lda_artifacts  # noqa: E402

UNFAVOURABLE_REF = (3, 4, 5)
FAVOURABLE_REF = (0, 1, 2)
DPI = 300


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
    """Hungarian match maximizing cosine similarity. Returns perm, cos_per_topic."""
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


def plot_level(
    per_rep: pd.DataFrame,
    by_topic: pd.DataFrame,
    out_stem: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax = axes[0, 0]
    data = [by_topic.loc[by_topic.ref_topic == t, "cosine_similarity"].to_numpy() for t in range(6)]
    ax.boxplot(data, tick_labels=[f"T{t}" for t in range(6)], showfliers=False)
    for t, ys in enumerate(data):
        color = "#c44e52" if t >= 3 else "#4c72b0"
        ax.scatter(np.full(len(ys), t + 1), ys, alpha=0.35, s=12, color=color)
    ax.axhline(0.9, color="green", ls="--", lw=1.2, label="0.9")
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Cosine similarity to reference")
    ax.set_title("A. Six topics (aligned profile distributions)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    hyp = [
        per_rep["fav_cosine"].to_numpy(),
        per_rep["unfav_cosine"].to_numpy(),
    ]
    ax.boxplot(hyp, tick_labels=["Favourable\n(T0–2)", "Unfavourable\n(T3–5)"], showfliers=False)
    ax.scatter(np.full(len(hyp[0]), 1), hyp[0], alpha=0.4, s=14, color="#4c72b0")
    ax.scatter(np.full(len(hyp[1]), 2), hyp[1], alpha=0.4, s=14, color="#c44e52")
    ax.axhline(0.9, color="green", ls="--", lw=1.2)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Cosine similarity to reference")
    ax.set_title("B. Two hypertopics (mixed profiles)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.hist(per_rep["mean_cosine"], bins=12, color="#4c72b0", edgecolor="white")
    med = float(per_rep["mean_cosine"].median())
    q1, q3 = float(per_rep["mean_cosine"].quantile(0.25)), float(per_rep["mean_cosine"].quantile(0.75))
    ax.axvline(med, color="crimson", lw=1.8, label=f"median={med:.2f}")
    ax.axvspan(q1, q3, color="crimson", alpha=0.15, label=f"IQR=[{q1:.2f},{q3:.2f}]")
    ax.axvline(0.9, color="green", ls="--", lw=1.2, label="0.9")
    ax.set_xlabel("Mean cosine over 6 topics")
    ax.set_ylabel("Count")
    ax.set_title("C. Distribution across replicates")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.hist(per_rep["puf_pearson"], bins=12, color="#c44e52", edgecolor="white")
    mp = float(per_rep["puf_pearson"].median())
    ax.axvline(mp, color="black", ls="--", lw=1.5, label=f"median r={mp:.2f}")
    ax.set_xlabel("Pearson r (pUF vs reference)")
    ax.set_ylabel("Count")
    ax.set_title("D. Alarm hypertopic (pUF = T3+T4+T5)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_profile_heatmap(
    topic_stack: np.ndarray,
    vocab: list[str],
    out_stem: Path,
    title: str,
    top_n: int = 15,
) -> None:
    """Heatmap of mean±SD top profiles per reference topic across replicates."""
    # topic_stack: (n_reps, K, V) already aligned
    mean = topic_stack.mean(axis=0)
    std = topic_stack.std(axis=0)
    k, v = mean.shape
    # pick union of top_n profiles per topic by mean mass
    keep = set()
    for t in range(k):
        keep.update(np.argsort(mean[t])[::-1][:top_n].tolist())
    keep_idx = sorted(keep, key=lambda i: -mean[:, i].max())
    labels = [vocab[i] if i < len(vocab) else str(i) for i in keep_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    mshow = mean[:, keep_idx]
    sshow = std[:, keep_idx]
    im0 = axes[0].imshow(mshow, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.05, float(mshow.max())))
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([f"T{t}" + (" (unfav)" if t >= 3 else " (fav)") for t in range(k)])
    axes[0].set_xticks(range(len(keep_idx)))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_title("Mean P(profile | topic)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sshow, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.02, float(sshow.max())))
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([f"T{t}" for t in range(k)])
    axes[1].set_xticks(range(len(keep_idx)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_title("SD across replicates")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def summarize(per_rep: pd.DataFrame, by_topic: pd.DataFrame, tag: str) -> dict:
    return {
        "level": tag,
        "n_reps": int(len(per_rep)),
        "cosine_all_pairs_median": float(by_topic["cosine_similarity"].median()),
        "cosine_all_pairs_q25": float(by_topic["cosine_similarity"].quantile(0.25)),
        "cosine_all_pairs_q75": float(by_topic["cosine_similarity"].quantile(0.75)),
        "mean_cosine_per_rep_median": float(per_rep["mean_cosine"].median()),
        "mean_cosine_per_rep_q25": float(per_rep["mean_cosine"].quantile(0.25)),
        "mean_cosine_per_rep_q75": float(per_rep["mean_cosine"].quantile(0.75)),
        "fav_cosine_median": float(per_rep["fav_cosine"].median()),
        "unfav_cosine_median": float(per_rep["unfav_cosine"].median()),
        "puf_pearson_median": float(per_rep["puf_pearson"].median()),
        "puf_pearson_q25": float(per_rep["puf_pearson"].quantile(0.25)),
        "puf_pearson_q75": float(per_rep["puf_pearson"].quantile(0.75)),
        "cosine_by_topic_median": {
            str(t): float(by_topic.loc[by_topic.ref_topic == t, "cosine_similarity"].median())
            for t in range(6)
        },
    }


def run_replicates(
    *,
    mode: str,
    corpus_bow,
    dictionary,
    ref_topics: np.ndarray,
    ref_doc: np.ndarray,
    documents: list,
    n_reps: int,
    passes: int,
    leave_out: float,
    seed0: int,
    num_topics: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = len(corpus_bow)
    ref_fav_mix = mix_topics(ref_topics, FAVOURABLE_REF)
    ref_unfav_mix = mix_topics(ref_topics, UNFAVOURABLE_REF)
    ref_puf = ref_doc[:, list(UNFAVOURABLE_REF)].sum(axis=1)

    per_rows, topic_rows = [], []
    stacks = []
    rng = np.random.default_rng(seed0)

    for r in range(n_reps):
        seed = seed0 + r
        if mode == "seed":
            train_idx = np.arange(n)
            eval_idx = train_idx
            label = f"seed={seed}"
        else:
            n_out = max(1, int(round(leave_out * n)))
            out_idx = rng.choice(n, size=n_out, replace=False)
            train_idx = np.setdiff1d(np.arange(n), out_idx)
            eval_idx = train_idx  # score concordance on patients used to fit
            label = f"rep={r}, leave_out={n_out}"

        print(f"  [{r+1}/{n_reps}] {mode}: {label} (n_train={len(train_idx)})", flush=True)
        bow_train = [corpus_bow[i] for i in train_idx]
        model = train_lda(bow_train, dictionary, num_topics, passes, seed)
        other = topic_matrix(model)
        perm, cos = align_topics_cosine(ref_topics, other)
        aligned = other[perm]
        stacks.append(aligned)

        bow_eval = [corpus_bow[i] for i in eval_idx]
        doc = doc_topic_matrix(model, bow_eval)[:, perm]
        # reference doc rows for same patients
        ref_sub = ref_doc[eval_idx]
        puf = doc[:, list(UNFAVOURABLE_REF)].sum(axis=1)
        ref_puf_sub = ref_sub[:, list(UNFAVOURABLE_REF)].sum(axis=1)
        puf_r, _ = pearsonr(ref_puf_sub, puf)

        fav_c = cosine_sim(ref_fav_mix, mix_topics(aligned, FAVOURABLE_REF))
        unfav_c = cosine_sim(ref_unfav_mix, mix_topics(aligned, UNFAVOURABLE_REF))

        per_rows.append(
            {
                "rep": r,
                "seed": seed,
                "n_train": int(len(train_idx)),
                "mean_cosine": float(np.mean(cos)),
                "min_cosine": float(np.min(cos)),
                "fav_cosine": float(fav_c),
                "unfav_cosine": float(unfav_c),
                "puf_pearson": float(puf_r) if np.isfinite(puf_r) else np.nan,
                "alignment_perm": ",".join(map(str, map(int, perm))),
                "passes": passes,
            }
        )
        for t in range(num_topics):
            topic_rows.append(
                {
                    "rep": r,
                    "ref_topic": t,
                    "matched_topic": int(perm[t]),
                    "cosine_similarity": float(cos[t]),
                    "hypertopic": "unfavourable" if t in UNFAVOURABLE_REF else "favourable",
                }
            )

    return pd.DataFrame(per_rows), pd.DataFrame(topic_rows), np.stack(stacks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("seed", "resample", "both"), default="both")
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("--leave-out", type=float, default=0.2, help="Fraction of patients left out (resample)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS)
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--reference-model", type=Path, default=LDA_MODEL_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/lda/stability"),
        help="Directory for CSV/JSON summaries (default: results/lda/stability)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Figure directory (default: results/lda/figures/stability)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.fig_dir is not None:
        fig_dir = args.fig_dir
    elif out_dir != Path("results/lda/stability"):
        fig_dir = out_dir / "figures"
    else:
        fig_dir = RESULTS_LDA_FIGURES / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reference…")
    dictionary, ref_model, df_profiles, documents, corpus_bow = load_lda_artifacts(
        args.dictionary, args.reference_model, args.profiles_csv
    )
    ref_topics = topic_matrix(ref_model)
    ref_doc = doc_topic_matrix(ref_model, corpus_bow)
    vocab = [dictionary[i] for i in range(len(dictionary))]

    modes = ["seed", "resample"] if args.mode == "both" else [args.mode]
    summaries = {}

    for mode in modes:
        print(f"\n=== Level {'1 (seed)' if mode=='seed' else '2 (resample)'} ===")
        per, by_topic, stack = run_replicates(
            mode=mode,
            corpus_bow=corpus_bow,
            dictionary=dictionary,
            ref_topics=ref_topics,
            ref_doc=ref_doc,
            documents=documents,
            n_reps=args.n_reps,
            passes=args.passes,
            leave_out=args.leave_out,
            seed0=args.seed_start,
            num_topics=args.num_topics,
        )
        tag = "level1_seed" if mode == "seed" else "level2_resample"
        per.to_csv(out_dir / f"{tag}_per_rep.csv", index=False)
        by_topic.to_csv(out_dir / f"{tag}_by_topic.csv", index=False)
        # also keep legacy names for level1
        if mode == "seed":
            per.to_csv(out_dir / "stability_per_seed.csv", index=False)
            by_topic.to_csv(out_dir / "stability_by_topic_cosine.csv", index=False)

        summ = summarize(per, by_topic, tag)
        if mode == "resample":
            summ["leave_out_fraction"] = args.leave_out
        summaries[tag] = summ
        (out_dir / f"{tag}_summary.json").write_text(json.dumps(summ, indent=2))

        title = (
            "Level 1 — seed stability at K=6 (cosine, Hungarian-aligned)"
            if mode == "seed"
            else f"Level 2 — leave-{int(100*args.leave_out)}%-out stability at K=6 (cosine)"
        )
        stem = (
            fig_dir / "lda_stability_level1_cosine"
            if mode == "seed"
            else fig_dir / "lda_stability_level2_resample_cosine"
        )
        plot_level(per, by_topic, stem, title)
        plot_profile_heatmap(
            stack,
            vocab,
            fig_dir / ("lda_stability_level1_profile_heatmap" if mode == "seed" else "lda_stability_level2_profile_heatmap"),
            title.replace("cosine, Hungarian-aligned", "mean ± SD profile weights").replace("(cosine)", "profile weights"),
        )
        print(json.dumps(summ, indent=2))

    (out_dir / "stability_cosine_summary.json").write_text(json.dumps(summaries, indent=2))
    # remove JS-centric legacy figure names from being the main ones; leave files if present
    print(f"\nWrote cosine-only outputs under {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
