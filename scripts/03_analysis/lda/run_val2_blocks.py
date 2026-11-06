#!/usr/bin/env python3
"""30-day block topic probabilities for the VAL-2 alarm.

Uses the saved LDA of each held-out and external-test row. Training rows are
omitted: the alarm score of a patient comes from the model she did not train.
Blocks follow the usual monthly rule: non-overlapping 30 embedding days from
the first day, and a short final block uses the last 30 days.

    python scripts/03_analysis/lda/run_val2_blocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import USER_EMBEDDINGS_CSV  # noqa: E402
from monthly_topics_utils import parse_embedding_ids  # noqa: E402
from run_val2_nested import (  # noqa: E402
    LDA_SEED,
    N_TOPICS,
    OUT_DIR,
    load_lda,
    topic_distribution,
)

WINDOW = 30
MIN_DAYS = 15
SCORE_ROLES = ("held_out", "external_test")


def block_spans(n_days: int) -> list[tuple[int, int]]:
    """Return half-open [start, end) spans, 0-based, matching the monthly rule."""
    if n_days < MIN_DAYS:
        return []
    if n_days < WINDOW:
        return [(0, n_days)]
    spans = []
    for start in range(0, n_days, WINDOW):
        end = start + WINDOW
        if end > n_days:
            start = n_days - WINDOW
            end = n_days
        spans.append((start, end))
        if end >= n_days:
            break
    return spans


def model_dir(outer_fold: int, inner_fold: str) -> Path:
    if inner_fold == "external":
        name = f"outer_{outer_fold}_refit"
    else:
        name = f"outer_{outer_fold}_inner_{int(inner_fold)}"
    return OUT_DIR / "models" / name


def main() -> None:
    global OUT_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is not None:
        OUT_DIR = args.out
    assigned = pd.read_csv(OUT_DIR / "topic_probabilities.csv")
    assigned = assigned[assigned["role"].isin(SCORE_ROLES)].copy()
    assigned["inner_fold"] = assigned["inner_fold"].astype(str)
    embeddings = pd.read_csv(USER_EMBEDDINGS_CSV)
    embeddings["id"] = embeddings["user_id"].astype(int)
    sequences = {
        int(row.id): [str(token) for token in parse_embedding_ids(row.embedding_ids)]
        for row in embeddings.itertuples(index=False)
    }

    rows: list[dict] = []
    skipped = []
    for (outer_fold, inner_fold), group in assigned.groupby(["outer_fold", "inner_fold"], sort=True):
        directory = model_dir(int(outer_fold), str(inner_fold))
        dictionary, model, _train_ids = load_lda(directory)
        model_name = directory.name
        for patient in group.sort_values("id").itertuples(index=False):
            tokens = sequences[int(patient.id)]
            spans = block_spans(len(tokens))
            if not spans:
                skipped.append((int(patient.id), int(outer_fold), str(inner_fold), patient.role, len(tokens)))
                continue
            for block, (start, end) in enumerate(spans, start=1):
                dist = topic_distribution(model, dictionary.doc2bow(tokens[start:end]))
                topic = int(dist.argmax())
                row = {
                    "id": int(patient.id),
                    "hospital": patient.hospital,
                    "outer_fold": int(outer_fold),
                    "inner_fold": str(inner_fold),
                    "role": patient.role,
                    "model": model_name,
                    "block": block,
                    "day_start": start + 1,
                    "day_end": end,
                    "n_days": end - start,
                    "n_train": int(patient.n_train),
                    "lda_seed": LDA_SEED,
                    "topic": topic,
                    "topic_probability": float(dist[topic]),
                }
                for t in range(N_TOPICS):
                    row[f"topic_{t}"] = float(dist[t])
                rows.append(row)
        print(f"{model_name}: {len(group)} patients", flush=True)

    out = pd.DataFrame(rows)
    dest = OUT_DIR / "block_probabilities.csv"
    out.to_csv(dest, index=False)
    print(f"{len(out)} rows, {out['id'].nunique()} patients -> {dest}", flush=True)
    print(out.groupby("role").size().to_string(), flush=True)
    if skipped:
        print(f"skipped {len(skipped)} patient-model rows with fewer than {MIN_DAYS} days", flush=True)
        print(pd.DataFrame(skipped, columns=["id", "outer_fold", "inner_fold", "role", "n_days"]).drop_duplicates("id").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
