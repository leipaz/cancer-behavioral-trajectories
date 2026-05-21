#!/usr/bin/env python3
"""
Build monthly predominant LDA topic table per patient.

Extracted from ``notebooks/6_1_entropy_variability_cleaned_dec25.ipynb`` (cells 13–14).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    FALLBACK_MONTH_PREDOM_CSV,
    MIN_EMBEDDINGS_FOR_MONTHLY,
    PROCESSED_LDA_DIR,
    resolve_dictionary_for_monthly,
    resolve_lda_model_for_monthly,
    resolve_topics_day_csv,
)
from monthly_topics_utils import build_monthly_predominant_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build monthly predominant topic table from per-day embeddings."
    )
    parser.add_argument(
        "--topics-day-csv",
        type=Path,
        default=None,
        help="Input CSV with id, embedding_ids, Evento PD, Evento PD diff",
    )
    parser.add_argument(
        "--lda-model",
        type=Path,
        default=None,
        help="Gensim LDA model path",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Gensim dictionary path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=FALLBACK_MONTH_PREDOM_CSV,
        help="Output monthly predominant topics CSV",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--min-embeddings",
        type=int,
        default=MIN_EMBEDDINGS_FOR_MONTHLY,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    topics_csv = args.topics_day_csv or resolve_topics_day_csv()
    lda_path = args.lda_model or resolve_lda_model_for_monthly()
    dict_path = args.dictionary or resolve_dictionary_for_monthly()

    print(f"Loading topics day data: {topics_csv}")
    df_day = pd.read_csv(topics_csv)
    print(f"Loading LDA model: {lda_path}")
    lda_model = LdaModel.load(str(lda_path))
    print(f"Loading dictionary: {dict_path}")
    dictionary = corpora.Dictionary.load(str(dict_path))

    ordered_df = build_monthly_predominant_table(
        df_day,
        lda_model,
        dictionary,
        window_size=args.window_size,
        min_embeddings=args.min_embeddings,
        verbose=args.verbose,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(ordered_df)} patients → {args.output_csv}")


if __name__ == "__main__":
    main()
