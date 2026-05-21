#!/usr/bin/env python3
"""Assign LDA topics per user, export top terms table and barplot grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DICTIONARY_PATH,
    FIGURE_TOP_TERMS_GRID,
    LDA_MODEL_PATH,
    TABLE_TOP_TERMS,
    USER_EMBEDDINGS_CSV,
    USER_TOPIC_CLUSTER_CSV,
    DEFAULT_TOP_N_TERMS,
)
from lda_utils import (  # noqa: E402
    assign_topics_to_users,
    get_top_terms_positive_probability,
    load_lda_artifacts,
    plot_top_terms_by_topic_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign LDA topics to users.")
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--model", type=Path, default=LDA_MODEL_PATH)
    parser.add_argument("--topics-out", type=Path, default=USER_TOPIC_CLUSTER_CSV)
    parser.add_argument("--top-terms-out", type=Path, default=TABLE_TOP_TERMS)
    parser.add_argument("--figure-out", type=Path, default=FIGURE_TOP_TERMS_GRID)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_TERMS)
    args = parser.parse_args()

    dictionary, lda_model, df_profiles, _documents, corpus_bow = load_lda_artifacts(
        args.dictionary, args.model, args.profiles_csv
    )

    result_df = assign_topics_to_users(lda_model, corpus_bow, df_profiles)
    args.topics_out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.topics_out, index=False)
    print(f"Saved user topics: {args.topics_out} ({len(result_df)} users)")

    terms_df = get_top_terms_positive_probability(
        lda_model, corpus_bow, dictionary, top_n=args.top_n
    )
    terms_export = terms_df[["topic", "profile"]].copy()
    args.top_terms_out.parent.mkdir(parents=True, exist_ok=True)
    terms_export.to_csv(args.top_terms_out, index=False)
    print(f"Saved top terms: {args.top_terms_out}")

    plot_top_terms_by_topic_grid(terms_df, output_path=args.figure_out, top_n=args.top_n)


if __name__ == "__main__":
    main()
