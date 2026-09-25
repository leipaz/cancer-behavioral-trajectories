#!/usr/bin/env python3
"""Build daily sliding-window ``pUF(d)`` (LDA on days [d-29, d]).

This is the expensive step. Prefer ``--skip-lda`` in ``run_granularity_sweep.py``
when ``results/alarm/tables/daily_sliding_puf.csv`` already exists.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

_SCRIPT_DIR = Path(__file__).resolve().parent
_LDA_DIR = _SCRIPT_DIR.parent / "lda"
for p in (_SCRIPT_DIR, _LDA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import (  # noqa: E402
    DICTIONARY_PATH,
    FALLBACK_TOPICS_DAY_CSV,
    LDA_MODEL_PATH,
    PROJECT_ROOT,
    resolve_dictionary_for_monthly,
    resolve_lda_model_for_monthly,
    resolve_topics_day_csv,
)
from rolling_burden_alarm import UNFAVOURABLE_TOPICS, default_clinical_path  # noqa: E402


def _parse_embedding_ids(value) -> list:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return list(value)


def daily_sliding_puf_for_patient(
    embedding_ids: list,
    lda_model: LdaModel,
    dictionary,
    window_size: int = 30,
    unfavourable_topics: tuple[int, ...] = UNFAVOURABLE_TOPICS,
    min_embeddings: int = 15,
) -> list[dict]:
    n = len(embedding_ids)
    if n < min_embeddings:
        return []
    short = int(n < window_size)
    rows: list[dict] = []
    if short:
        bow = dictionary.doc2bow([str(tok) for tok in embedding_ids])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        puf = float(sum(probs[t] for t in unfavourable_topics if t < len(probs)))
        rows.append(
            {
                "study_day": n,
                "pUF": puf,
                "short_series": 1,
                **{f"topic_{k}": float(probs[k]) if k < len(probs) else np.nan for k in range(6)},
            }
        )
        return rows

    for end in range(window_size, n + 1):
        window = embedding_ids[end - window_size : end]
        bow = dictionary.doc2bow([str(tok) for tok in window])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        puf = float(sum(probs[t] for t in unfavourable_topics if t < len(probs)))
        rows.append(
            {
                "study_day": end,
                "pUF": puf,
                "short_series": 0,
                **{f"topic_{k}": float(probs[k]) if k < len(probs) else np.nan for k in range(6)},
            }
        )
    return rows


def build_daily_sliding_puf_table(
    topics_day: pd.DataFrame,
    lda_model: LdaModel,
    dictionary,
    window_size: int = 30,
) -> pd.DataFrame:
    clinical = None
    if default_clinical_path().exists():
        clinical = pd.read_excel(default_clinical_path())
        clinical.columns = [str(c).strip() for c in clinical.columns]

    id_map = {}
    if clinical is not None:
        id_col = "S_RECORD_id" if "S_RECORD_id" in clinical.columns else "id"
        for _, r in clinical.iterrows():
            pid = int(r[id_col])
            event = int(r["PD_event"])
            fu = float(r["Obs_time"])
            t_ev = fu if event == 0 else float(r.get("pd_event_diff", fu) if False else fu)
            # Obs_time is censor/event time in Subjects_data; for events PD time ≈ Obs_time
            id_map[pid] = (event, fu, fu)

    # Prefer per-patient times on topics_day when present
    rows: list[dict] = []
    for _, row in topics_day.iterrows():
        pid = int(row["id"])
        emb = _parse_embedding_ids(row["embedding_ids"])
        block_rows = daily_sliding_puf_for_patient(emb, lda_model, dictionary, window_size)
        if not block_rows:
            continue
        if "pd_event" in topics_day.columns:
            event = int(row["pd_event"])
            fu = float(row["obs_time_final"])
            t_raw = row.get("pd_event_diff", np.nan)
            t_ev = float(fu if event == 0 else (t_raw if pd.notna(t_raw) else fu))
        elif pid in id_map:
            event, t_ev, fu = id_map[pid]
        else:
            continue
        for br in block_rows:
            rows.append(
                {
                    "id": pid,
                    "Evento PD": event,
                    "t_evento_eb2": t_ev,
                    "follow_up_days": fu,
                    **br,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--topics-day-csv", type=Path, default=None)
    parser.add_argument("--lda-model", type=Path, default=None)
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "results/alarm/tables/daily_sliding_puf.csv",
    )
    args = parser.parse_args()

    topics_csv = args.topics_day_csv or resolve_topics_day_csv()
    if not Path(topics_csv).exists():
        topics_csv = FALLBACK_TOPICS_DAY_CSV
    lda_path = args.lda_model or resolve_lda_model_for_monthly()
    if not Path(lda_path).exists():
        lda_path = LDA_MODEL_PATH
    dict_path = args.dictionary or resolve_dictionary_for_monthly()
    if not Path(dict_path).exists():
        dict_path = DICTIONARY_PATH

    print(f"Loading {topics_csv}")
    topics_day = pd.read_csv(topics_csv)
    print(f"Loading LDA {lda_path}")
    lda_model = LdaModel.load(str(lda_path))
    dictionary = corpora.Dictionary.load(str(dict_path))

    table = build_daily_sliding_puf_table(topics_day, lda_model, dictionary, args.window_size)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(table)} rows, {table['id'].nunique()} patients)")


if __name__ == "__main__":
    main()
