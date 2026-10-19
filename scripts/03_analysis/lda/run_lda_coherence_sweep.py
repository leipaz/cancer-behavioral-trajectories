#!/usr/bin/env python3
"""Fine-grid LDA coherence sweep (for K-selection figure / report)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import DICTIONARY_PATH, LDA_MODEL_PATH, RESULTS_LDA_FIGURES, USER_EMBEDDINGS_CSV  # noqa: E402
from lda_utils import load_lda_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument(
        "--k-list",
        type=str,
        default=None,
        help="Comma-separated candidate K values (e.g. 2,6,10,14,18). Overrides k-min/k-max.",
    )
    parser.add_argument("--n-inits", type=int, default=5)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/lda/stability"),
        help="CSV/JSON output directory (default: results/lda/stability)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Figure directory (default: results/lda/figures)",
    )
    args = parser.parse_args()

    if args.k_list:
        k_values = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    else:
        k_values = list(range(args.k_min, args.k_max + 1))
    dictionary, _ref, _df, documents, corpus = load_lda_artifacts(
        DICTIONARY_PATH, LDA_MODEL_PATH, args.profiles_csv
    )
    idx = np.arange(len(documents))
    tr, va = train_test_split(idx, test_size=0.3, random_state=args.seed)
    texts_tr = [documents[i] for i in tr]
    texts_va = [documents[i] for i in va]
    bow_tr = [corpus[i] for i in tr]

    rows = []
    for k in k_values:
        tr_s, va_s = [], []
        for i in range(args.n_inits):
            print(f"K={k} init={i+1}/{args.n_inits}", flush=True)
            model = LdaModel(
                corpus=bow_tr,
                id2word=dictionary,
                num_topics=k,
                random_state=args.seed + i,
                chunksize=100,
                passes=args.passes,
                alpha="auto",
                eta="auto",
            )
            cm_tr = CoherenceModel(
                model=model,
                texts=texts_tr,
                dictionary=dictionary,
                coherence="c_v",
                processes=1,
            )
            cm_va = CoherenceModel(
                model=model,
                texts=texts_va,
                dictionary=dictionary,
                coherence="c_v",
                processes=1,
            )
            tr_s.append(cm_tr.get_coherence())
            va_s.append(cm_va.get_coherence())
        rows.append(
            {
                "K": k,
                "train_mean": float(np.mean(tr_s)),
                "train_sd": float(np.std(tr_s)),
                "val_mean": float(np.mean(va_s)),
                "val_sd": float(np.std(va_s)),
            }
        )
        print(
            f"  -> train={rows[-1]['train_mean']:.3f} val={rows[-1]['val_mean']:.3f}",
            flush=True,
        )

    out_dir = args.out_dir
    if args.fig_dir is not None:
        fig_dir = args.fig_dir
    elif out_dir != Path("results/lda/stability"):
        fig_dir = out_dir / "figures"
    else:
        fig_dir = RESULTS_LDA_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    grid_name = "coherence_coarse_grid.csv" if args.k_list else "coherence_fine_grid.csv"
    df.to_csv(out_dir / grid_name, index=False)
    # always also write fine_grid name for report compatibility when using k-list
    if args.k_list:
        df.to_csv(out_dir / "coherence_fine_grid.csv", index=False)

    best_k = int(df.loc[df["val_mean"].idxmax(), "K"])
    # Parsimony: among K within 98% of peak val coherence, pick smallest K
    peak = float(df["val_mean"].max())
    near = df[df["val_mean"] >= 0.98 * peak]
    parsimonious_k = int(near["K"].min())

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(df["K"], df["train_mean"], "o-", color="#1f77b4", lw=2, label="Train (mean over inits)")
    ax.fill_between(
        df["K"],
        df["train_mean"] - df["train_sd"],
        df["train_mean"] + df["train_sd"],
        color="#1f77b4",
        alpha=0.15,
    )
    ax.plot(df["K"], df["val_mean"], "o-", color="#ff7f0e", lw=2, label="Validation (mean over inits)")
    ax.fill_between(
        df["K"],
        df["val_mean"] - df["val_sd"],
        df["val_mean"] + df["val_sd"],
        color="#ff7f0e",
        alpha=0.15,
    )
    ax.axvline(6, color="#2ca02c", ls="--", lw=2, label="Selected K = 6")
    if best_k != 6:
        ax.axvline(best_k, color="#d62728", ls=":", lw=1.5, label=f"Val. peak K = {best_k}")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Coherence ($c_v$)")
    ax.set_title("Unsupervised topic-number selection (average $c_v$ over initializations)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(fig_dir / f"lda_coherence_k_selection{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "best_val_K": best_k,
        "parsimonious_K_within_98pct_peak": parsimonious_k,
        "selected_K": 6,
        "peak_val_coherence": peak,
        "val_at_K6": float(df.loc[df.K == 6, "val_mean"].iloc[0]) if (df.K == 6).any() else None,
    }
    pd.Series(meta).to_json(out_dir / "coherence_selection_meta.json")
    print(meta)
    print(f"Wrote {fig_dir / 'lda_coherence_k_selection.png'}")


if __name__ == "__main__":
    main()
