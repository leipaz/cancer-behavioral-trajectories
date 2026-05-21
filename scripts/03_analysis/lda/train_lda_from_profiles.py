#!/usr/bin/env python3
"""
Build user embedding documents from the VQ-VAE profiles PKL and train an LDA model.

Pipeline (from ``notebooks/4_01_LDA_dec25.ipynb``, up to LDA training):
  1. Load ``profiles_per_sample_oncology_28_12_2025.pkl``
  2. Extract per-user ``embedding_ids`` for ``model_type`` / ``n``
  3. Build gensim dictionary and bag-of-words corpus
  4. Train ``LdaMulticore`` on BOW (not TF-IDF, as in the notebook)
  5. Save dictionary, model, and profiles CSV under ``data/processed/lda/``
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_MODEL_TYPE,
    DEFAULT_N,
    DEFAULT_NUM_TOPICS,
    DEFAULT_PASSES,
    DEFAULT_PROFILES_PKL,
    DICTIONARY_PATH,
    LDA_MODEL_PATH,
    PROCESSED_LDA_DIR,
    USER_EMBEDDINGS_CSV,
)


def load_profiles_pkl(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def profiles_to_dataframe(
    embedding_counts: dict,
    model_type: str,
    n: int,
) -> pd.DataFrame:
    """One row per user: user_id + embedding_ids list."""
    users = embedding_counts[model_type][n]
    rows = []
    for user_id, payload in users.items():
        user_embed = payload[0]
        embedding_ids = user_embed[1]
        rows.append(
            {
                "user_id": user_id,
                "embedding_ids": embedding_ids.tolist(),
            }
        )
    df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    return df


def parse_embedding_ids(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def build_documents(df_profiles: pd.DataFrame) -> list[list[str]]:
    ids = df_profiles["embedding_ids"].apply(parse_embedding_ids)
    return [[str(token) for token in doc] for doc in ids]


def train_lda(
    documents: list[list[str]],
    num_topics: int,
    passes: int,
    workers: int,
) -> tuple[corpora.Dictionary, list, LdaMulticore]:
    dictionary = corpora.Dictionary(documents)
    corpus_bow = [dictionary.doc2bow(doc) for doc in documents]

    print(f"Documents: {len(documents)}")
    print(f"Dictionary size: {len(dictionary)}")
    print(f"Training LDA: topics={num_topics}, passes={passes}, workers={workers}")

    lda_model = LdaMulticore(
        corpus_bow,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        workers=workers,
    )
    return dictionary, corpus_bow, lda_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LDA from VQ-VAE profiles PKL (embedding token sequences)."
    )
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PROFILES_PKL)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Profile length key (e.g. 30)")
    parser.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = all CPUs")
    parser.add_argument("--profiles-csv", type=Path, default=USER_EMBEDDINGS_CSV)
    parser.add_argument("--dictionary-out", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--model-out", type=Path, default=LDA_MODEL_PATH)
    args = parser.parse_args()

    if not args.pkl.is_file():
        raise FileNotFoundError(f"PKL not found: {args.pkl}")

    if args.model_type not in ("a0", "a1", "a2"):
        raise ValueError(f"Unknown model_type: {args.model_type}")

    embedding_counts = load_profiles_pkl(args.pkl)
    if args.n not in embedding_counts[args.model_type]:
        raise KeyError(
            f"n={args.n} not in PKL for model_type={args.model_type}. "
            f"Available: {sorted(embedding_counts[args.model_type].keys())}"
        )

    df_profiles = profiles_to_dataframe(embedding_counts, args.model_type, args.n)
    args.profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    df_profiles.to_csv(args.profiles_csv, index=False)
    print(f"Saved profiles CSV: {args.profiles_csv} ({len(df_profiles)} users)")

    documents = build_documents(df_profiles)
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    dictionary, corpus_bow, lda_model = train_lda(
        documents,
        num_topics=args.num_topics,
        passes=args.passes,
        workers=workers,
    )

    args.dictionary_out.parent.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(args.dictionary_out))
    lda_model.save(str(args.model_out))

    print(f"Saved dictionary: {args.dictionary_out}")
    print(f"Saved LDA model:  {args.model_out}")
    print(f"Corpus size: {len(corpus_bow)} docs, example bow length: {len(corpus_bow[0])}")


if __name__ == "__main__":
    main()
