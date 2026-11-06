#!/usr/bin/env python3
"""VAL-2 nested topic assignment on DB1.

Five outer patient partitions and four inner partitions. Every month of a
patient stays in the same side. The VQ-VAE embeddings stay frozen. Each LDA
(K=6) is fit only on the patients of that training split. Topic numbers are
not aligned to the published favourable/unfavourable grouping.

The inner held-out topic is the one Miguel uses to rank one-year disease
control. The outer-test topic is the refit applied to the patients who were
left out of that round. Their outcomes are not used to fit the model.

    python scripts/03_analysis/lda/run_val2_nested.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import PROJECT_ROOT, USER_EMBEDDINGS_CSV  # noqa: E402

AUDIT = (
    PROJECT_ROOT
    / "data/processed/DCABPs/DB1"
    / "2 178_audit_oct_2_2026_con grupos de tratamiento  copia.xlsx"
)
OUT_DIR = PROJECT_ROOT / "results/lda/val_2/anidada"

OUTER_SEED = 20261103
INNER_SEED = 20261104
LDA_SEED = 20261103
N_OUTER = 5
N_INNER = 4
N_TOPICS = 6
PASSES = 500
ITERATIONS = 50


def parse_ids(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def split_ids(ids: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """Shuffle with ``seed`` and cut into ``n_splits`` groups, as equal as possible."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(ids))
    return [np.array(part, dtype=int) for part in np.array_split(order, n_splits)]


def topic_distribution(model: LdaModel, bow) -> np.ndarray:
    probs = np.zeros(model.num_topics, dtype=float)
    for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
        probs[int(tid)] = float(prob)
    return probs


def load_lda(directory: Path) -> tuple[corpora.Dictionary, LdaModel, np.ndarray]:
    dictionary = corpora.Dictionary.load(str(directory / "dictionary.dict"))
    model = LdaModel.load(str(directory / "lda.gensim"))
    train_ids = np.array(json.loads((directory / "train_ids.json").read_text()), dtype=int)
    return dictionary, model, train_ids


def save_lda(directory: Path, dictionary: corpora.Dictionary, model: LdaModel, train_ids: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(directory / "dictionary.dict"))
    model.save(str(directory / "lda.gensim"))
    (directory / "train_ids.json").write_text(json.dumps([int(i) for i in train_ids]))


def fit_lda(documents: list[list[str]], seed: int) -> tuple[corpora.Dictionary, LdaModel]:
    dictionary = corpora.Dictionary(documents)
    bow = [dictionary.doc2bow(doc) for doc in documents]
    model = LdaModel(
        corpus=bow,
        id2word=dictionary,
        num_topics=N_TOPICS,
        random_state=seed,
        passes=PASSES,
        chunksize=100,
        alpha="symmetric",
        eta="symmetric",
        iterations=ITERATIONS,
        eval_every=None,
    )
    return dictionary, model


def predict(model: LdaModel, dictionary: corpora.Dictionary, documents: list[list[str]]) -> np.ndarray:
    rows = []
    for doc in documents:
        rows.append(topic_distribution(model, dictionary.doc2bow(doc)))
    return np.vstack(rows)


def assignment_rows(
    ids: np.ndarray,
    probs: np.ndarray,
    *,
    outer_fold: int,
    inner_fold: int | str,
    role: str,
    n_train: int,
    seed: int,
) -> list[dict]:
    rows = []
    for i, pid in enumerate(ids):
        dist = probs[i]
        topic = int(np.argmax(dist))
        row = {
            "id": int(pid),
            "outer_fold": int(outer_fold),
            "inner_fold": str(inner_fold),
            "role": role,
            "topic": topic,
            "topic_probability": float(dist[topic]),
            "n_train": int(n_train),
            "lda_seed": int(seed),
        }
        for t in range(N_TOPICS):
            row[f"topic_{t}"] = float(dist[t])
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_excel(AUDIT).dropna(subset=["id"])
    audit["id"] = audit["id"].astype(int)
    hospital = audit.set_index("id")["Hospital"]

    embeddings = pd.read_csv(USER_EMBEDDINGS_CSV)
    embeddings["id"] = embeddings["user_id"].astype(int)
    documents = {
        int(row.id): [str(token) for token in parse_ids(row.embedding_ids)]
        for row in embeddings.itertuples(index=False)
    }
    cohort = np.array(sorted(set(audit["id"]) & set(documents)), dtype=int)
    missing = sorted(set(audit["id"]) - set(documents))

    outer_groups = split_ids(cohort, N_OUTER, OUTER_SEED)
    fold_rows = []
    for fold, members in enumerate(outer_groups):
        for pid in members:
            fold_rows.append({"id": int(pid), "outer_fold": fold, "hospital": hospital.get(pid)})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT_DIR / "outer_folds.csv", index=False)

    manifest = {
        "cohort": "DB1",
        "audit": str(AUDIT.relative_to(PROJECT_ROOT)),
        "embeddings": str(USER_EMBEDDINGS_CSV.relative_to(PROJECT_ROOT)),
        "n_audit": int(audit["id"].nunique()),
        "n_with_profile": int(len(cohort)),
        "excluded_no_profile": missing,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "lda_seed": LDA_SEED,
        "num_topics": N_TOPICS,
        "passes": PASSES,
        "iterations": ITERATIONS,
        "vqvae": "frozen; one document per patient from embedding_ids",
        "topic_alignment": "none",
        "window_months": 3,
        "horizon_days": 120,
        "window_and_horizon": "fixed for the later alarm score; not used in this topic assignment",
        "outer_fold_sizes": [int(len(g)) for g in outer_groups],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    model_dir = OUT_DIR / "models"
    assigned: list[dict] = []
    for outer, test_ids in enumerate(outer_groups):
        train_ids = np.concatenate([g for i, g in enumerate(outer_groups) if i != outer])
        inner_groups = split_ids(train_ids, N_INNER, INNER_SEED + outer)
        print(
            f"outer {outer}: test {len(test_ids)}, train {len(train_ids)}",
            flush=True,
        )
        for inner, held in enumerate(inner_groups):
            fit_ids = np.concatenate([g for i, g in enumerate(inner_groups) if i != inner])
            saved = model_dir / f"outer_{outer}_inner_{inner}"
            if (saved / "lda.gensim").exists():
                dictionary, model, saved_ids = load_lda(saved)
                if not np.array_equal(saved_ids, fit_ids):
                    raise SystemExit(f"{saved.name} was trained on a different patient list")
            else:
                dictionary, model = fit_lda([documents[i] for i in fit_ids], LDA_SEED)
                save_lda(saved, dictionary, model, fit_ids)
            for ids, role in ((fit_ids, "train"), (held, "held_out")):
                probs = predict(model, dictionary, [documents[i] for i in ids])
                assigned.extend(
                    assignment_rows(
                        ids,
                        probs,
                        outer_fold=outer,
                        inner_fold=inner,
                        role=role,
                        n_train=len(fit_ids),
                        seed=LDA_SEED,
                    )
                )
            print(f"  inner {inner}: train {len(fit_ids)}, held out {len(held)}", flush=True)

        saved = model_dir / f"outer_{outer}_refit"
        if (saved / "lda.gensim").exists():
            dictionary, model, saved_ids = load_lda(saved)
            if not np.array_equal(saved_ids, train_ids):
                raise SystemExit(f"{saved.name} was trained on a different patient list")
        else:
            dictionary, model = fit_lda([documents[i] for i in train_ids], LDA_SEED)
            save_lda(saved, dictionary, model, train_ids)
        for ids, fold_label, role in (
            (train_ids, "refit", "refit"),
            (test_ids, "external", "external_test"),
        ):
            probs = predict(model, dictionary, [documents[i] for i in ids])
            assigned.extend(
                assignment_rows(
                    ids,
                    probs,
                    outer_fold=outer,
                    inner_fold=fold_label,
                    role=role,
                    n_train=len(train_ids),
                    seed=LDA_SEED,
                )
            )

    out = pd.DataFrame(assigned).merge(folds[["id", "hospital"]], on="id", how="left")
    front = [
        "id",
        "hospital",
        "outer_fold",
        "inner_fold",
        "role",
        "topic",
        "topic_probability",
        "n_train",
        "lda_seed",
    ]
    out = out[front + [f"topic_{t}" for t in range(N_TOPICS)]]
    out = out.sort_values(
        ["outer_fold", "inner_fold", "role", "id"], na_position="last"
    ).reset_index(drop=True)
    dest = OUT_DIR / "topic_probabilities.csv"
    if dest.exists():
        previous = pd.read_csv(dest)
        previous["inner_fold"] = previous["inner_fold"].map(
            lambda value: str(int(float(value))) if str(value).replace(".", "", 1).isdigit() else str(value)
        )
        keys = ["id", "outer_fold", "inner_fold", "role"]
        topic_cols = [f"topic_{t}" for t in range(N_TOPICS)]
        fresh = out.copy()
        fresh["inner_fold"] = fresh["inner_fold"].astype(str)
        merged = previous[keys + topic_cols].merge(
            fresh[keys + topic_cols], on=keys, suffixes=("_prev", "_new")
        )
        if len(merged) != len(previous) or len(merged) != len(fresh):
            raise SystemExit(
                f"topic rows do not match the saved models: {len(previous)} vs {len(fresh)}, joined {len(merged)}"
            )
        gap = max(
            (merged[f"{col}_prev"] - merged[f"{col}_new"]).abs().max()
            for col in topic_cols
        )
        gap = float(gap)
        topic_gap = max(
            (merged[f"topic_{t}_prev"] - merged[f"topic_{t}_new"]).abs().max()
            for t in range(N_TOPICS)
        )
        same_topic = int((
            merged[[f"topic_{t}_prev" for t in range(N_TOPICS)]].to_numpy().argmax(axis=1)
            == merged[[f"topic_{t}_new" for t in range(N_TOPICS)]].to_numpy().argmax(axis=1)
        ).sum())
        print(
            f"loaded models vs previous {dest.name}: max abs {topic_gap}, same predominant topic {same_topic}/{len(merged)}",
            flush=True,
        )
        del gap
    out.to_csv(dest, index=False)
    held = out[out["role"].isin(["held_out", "external_test"])]
    assignment = OUT_DIR / "topic_assignment.csv"
    held.to_csv(assignment, index=False)
    print(f"{len(out)} rows -> {dest}", flush=True)
    print(f"{len(held)} rows -> {assignment}", flush=True)


if __name__ == "__main__":
    main()
