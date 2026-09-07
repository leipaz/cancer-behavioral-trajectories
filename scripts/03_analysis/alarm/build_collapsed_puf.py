#!/usr/bin/env python3
"""Build collapsed-block ``pUF`` tables (weekly / biweekly / monthly) via LDA."""

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
if str(_LDA_DIR) not in sys.path:
    sys.path.insert(0, str(_LDA_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

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


def collapsed_puf_for_patient(
    embedding_ids: list,
    lda_model: LdaModel,
    dictionary,
    block_size: int,
    unfavourable_topics: tuple[int, ...] = UNFAVOURABLE_TOPICS,
    min_embeddings: int = 15,
) -> list[dict]:
    """Non-overlapping blocks; last incomplete block uses the final ``block_size`` days."""
    n = len(embedding_ids)
    if n < min_embeddings:
        return []
    rows: list[dict] = []
    if min_embeddings <= n < block_size:
        windows = [embedding_ids]
    else:
        windows = []
        for start in range(0, n, block_size):
            end = start + block_size
            if end > n:
                windows.append(embedding_ids[-block_size:])
                break
            windows.append(embedding_ids[start:end])

    for period, window in enumerate(windows, start=1):
        bow = dictionary.doc2bow([str(tok) for tok in window])
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        puf = float(sum(probs[t] for t in unfavourable_topics if t < len(probs)))
        rows.append(
            {
                "period": period,
                "decision_day": period * block_size,
                "pUF": puf,
                **{f"topic_{k}": float(probs[k]) if k < len(probs) else np.nan for k in range(6)},
            }
        )
    return rows


def build_collapsed_puf_table(
    topics_day: pd.DataFrame,
    lda_model: LdaModel,
    dictionary,
    block_size: int,
    clinical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if clinical is None and default_clinical_path().exists():
        clinical = pd.read_excel(default_clinical_path())

    id_map = {}
    if clinical is not None:
        clin = clinical.copy()
        clin.columns = [str(c).strip() for c in clin.columns]
        id_col = "S_RECORD_id" if "S_RECORD_id" in clin.columns else "id"
        for _, r in clin.iterrows():
            pid = int(r[id_col])
            event = int(r["PD_event"]) if "PD_event" in clin.columns else int(r.get("Evento PD", 0))
            fu = float(r["Obs_time"]) if "Obs_time" in clin.columns else float(r.get("obs_time_final", np.nan))
            t_ev = float(r["Obs_time"]) if event == 0 else float(
                r["pd_event_diff"] if "pd_event_diff" in clin.columns else fu
            )
            # prefer columns already on topics_day when present
            id_map[pid] = (event, t_ev, fu)

    rows: list[dict] = []
    for _, row in topics_day.iterrows():
        pid = int(row["id"])
        emb = _parse_embedding_ids(row["embedding_ids"])
        block_rows = collapsed_puf_for_patient(emb, lda_model, dictionary, block_size)
        if not block_rows:
            continue
        if pid in id_map:
            event, t_ev, fu = id_map[pid]
        else:
            event = int(row.get("pd_event", row.get("Evento PD", 0)))
            fu = float(row.get("obs_time_final", row.get("follow_up_days", np.nan)))
            t_raw = row.get("pd_event_diff", row.get("t_evento_eb2", fu))
            t_ev = float(fu if event == 0 else (t_raw if pd.notna(t_raw) else fu))
        for br in block_rows:
            rows.append(
                {
                    "id": pid,
                    "month": np.nan,
                    "block_size_days": block_size,
                    "Evento PD": event,
                    "t_evento_eb2": t_ev,
                    "follow_up_days": fu,
                    **br,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, nargs="+", default=[7, 14], help="Block sizes in days")
    parser.add_argument("--topics-day-csv", type=Path, default=None)
    parser.add_argument("--lda-model", type=Path, default=None)
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results/alarm/tables",
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = {7: "weekly", 14: "biweekly", 30: "monthly"}
    for b in args.block_size:
        table = build_collapsed_puf_table(topics_day, lda_model, dictionary, b)
        out = args.out_dir / f"collapsed_puf_{name.get(b, str(b)+'d')}.csv"
        table.to_csv(out, index=False)
        print(f"Wrote {out} ({len(table)} rows, {table['id'].nunique()} patients)")


if __name__ == "__main__":
    main()
