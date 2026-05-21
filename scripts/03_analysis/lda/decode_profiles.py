#!/usr/bin/env python3
"""
Map LDA top-10 profile tokens to decoded VQ-VAE behavioral feature vectors.

Extracted from ``notebooks/5_decodificar_perfiles.ipynb``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    BEHAVIORAL_FEATURE_COLS,
    DECODED_PROFILES_CSV,
    DECODED_PROFILES_SCALED_CSV,
    PROFILE_FEATURE_NAMES,
    TABLE_TOP_TERMS,
    resolve_decoded_embeddings_pkl,
)


def load_embedding_counts(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_decoded_profiles(
    terms_topic: pd.DataFrame,
    embedding_counts: dict,
) -> pd.DataFrame:
    """Join top terms with decoded vectors and expand to named feature columns."""
    if "profile" not in terms_topic.columns or "topic" not in terms_topic.columns:
        raise ValueError("terms_topic must contain 'topic' and 'profile' columns")

    sample_key = next(iter(embedding_counts))
    vector_len = len(embedding_counts[sample_key])
    if vector_len != len(PROFILE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(PROFILE_FEATURE_NAMES)} features per profile, "
            f"got {vector_len}"
        )

    df = terms_topic.copy()
    df["embedding_vector"] = df["profile"].map(embedding_counts)
    missing = df["embedding_vector"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} profile(s) missing from decoded embeddings dictionary"
        )

    embedding_df = pd.DataFrame(
        df["embedding_vector"].tolist(),
        columns=[f"feat_{i}" for i in range(vector_len)],
    )
    df_final = pd.concat([df.drop(columns=["embedding_vector"]), embedding_df], axis=1)
    rename_map = {
        f"feat_{i}": PROFILE_FEATURE_NAMES[i] for i in range(vector_len)
    }
    df_final = df_final.rename(columns=rename_map)
    if "weekend" in df_final.columns:
        df_final = df_final.drop(columns=["weekend"])
    return df_final


def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score behavioral columns (notebook StandardScaler step)."""
    out = df.copy()
    values = out[feature_cols]
    out[feature_cols] = (values - values.mean()) / values.std(ddof=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode LDA top-10 profile tokens to behavioral features."
    )
    parser.add_argument(
        "--terms-csv",
        type=Path,
        default=TABLE_TOP_TERMS,
        help="LDA top-10 terms per topic (topic, profile)",
    )
    parser.add_argument(
        "--embeddings-pkl",
        type=Path,
        default=None,
        help="Decoded embedding vectors pickle (profile id → vector)",
    )
    parser.add_argument(
        "--model-type",
        choices=("a0", "finetune"),
        default="a0",
        help="Which decoded embeddings file to use if --embeddings-pkl omitted",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DECODED_PROFILES_CSV,
        help="Raw decoded features CSV",
    )
    parser.add_argument(
        "--output-scaled-csv",
        type=Path,
        default=DECODED_PROFILES_SCALED_CSV,
        help="Z-scored features CSV for boxplots",
    )
    args = parser.parse_args()

    if not args.terms_csv.is_file():
        raise FileNotFoundError(
            f"Top terms table not found: {args.terms_csv}. "
            "Run assign_lda_topics.py or run_lda_pipeline.py first."
        )

    pkl_path = args.embeddings_pkl or resolve_decoded_embeddings_pkl(args.model_type)
    print(f"Loading embeddings: {pkl_path}")
    embedding_counts = load_embedding_counts(pkl_path)

    terms_topic = pd.read_csv(args.terms_csv)
    df_final = build_decoded_profiles(terms_topic, embedding_counts)
    df_scaled = scale_features(df_final, BEHAVIORAL_FEATURE_COLS)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_scaled_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(args.output_csv, index=False)
    df_scaled.to_csv(args.output_scaled_csv, index=False)

    print(f"Saved decoded profiles: {args.output_csv} ({len(df_final)} rows)")
    print(f"Saved scaled profiles: {args.output_scaled_csv}")


if __name__ == "__main__":
    main()
