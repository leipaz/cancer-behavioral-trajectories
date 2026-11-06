#!/usr/bin/env python3
"""Stratified repeat of the nested VAL-2 split.

Same seeds, same LDA and the same five-by-four design as the unstratified
run. The only change is that both the outer groups and the inner groups
spread each event × tumor group × hospital combination across folds. The
hospital is the first digit of the id. Results go to
results/lda/val_2/anidada_estratificada and do not replace anidada/.

    python scripts/03_analysis/lda/run_val2_stratified.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_val2_nested import (  # noqa: E402
    AUDIT,
    INNER_SEED,
    LDA_SEED,
    N_INNER,
    N_OUTER,
    N_TOPICS,
    OUTER_SEED,
    USER_EMBEDDINGS_CSV,
    assignment_rows,
    fit_lda,
    load_lda,
    parse_ids,
    predict,
    save_lda,
)
from config import PROJECT_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results/lda/val_2/anidada_estratificada"


def stratified_groups(frame: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    """Fill equal groups, balancing hospital, then event, then tumor group.

    The seed only shuffles patients inside each hospital × event × tumor cell.
    """
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    meta = frame.set_index("id")
    pieces = []
    for _, sub in frame.groupby(["centre", "event", "tumor"], dropna=False, sort=True):
        ids = np.array(sorted(int(i) for i in sub["id"]), dtype=int)
        rng.shuffle(ids)
        pieces.append(ids)
    for ids in pieces:
        for pid in ids:
            row = meta.loc[int(pid)]

            def cost(index: int, row=row) -> tuple[int, int, int, int, int]:
                members = buckets[index]
                same_centre = sum(int(meta.loc[i, "centre"]) == int(row.centre) for i in members)
                same_event = sum(int(meta.loc[i, "event"]) == int(row.event) for i in members)
                same_tumor = sum(int(meta.loc[i, "tumor"]) == int(row.tumor) for i in members)
                return (len(members), same_centre, same_event, same_tumor, index)

            buckets[min(range(n_splits), key=cost)].append(int(pid))
    return [np.array(bucket, dtype=int) for bucket in buckets]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_excel(AUDIT).dropna(subset=["id"]).copy()
    audit["id"] = audit["id"].astype(int)
    hospital = audit.set_index("id")["Hospital"]
    event = audit.set_index("id")["Evento_DB1"]
    tumor = audit.set_index("id")["Group tumor"]

    embeddings = pd.read_csv(USER_EMBEDDINGS_CSV)
    documents = {
        int(row.user_id): [str(token) for token in parse_ids(row.embedding_ids)]
        for row in embeddings.itertuples(index=False)
    }
    cohort_ids = np.array(sorted(set(audit["id"]) & set(documents)), dtype=int)
    missing = sorted(set(audit["id"]) - set(documents))
    cohort = pd.DataFrame({"id": cohort_ids})
    cohort["event"] = cohort["id"].map(event).astype(int)
    cohort["tumor"] = cohort["id"].map(tumor).astype(int)
    cohort["centre"] = cohort["id"].astype(str).str[0].astype(int)

    outer_groups = stratified_groups(cohort, N_OUTER, OUTER_SEED)
    fold_rows = []
    for fold, members in enumerate(outer_groups):
        for pid in members:
            fold_rows.append({
                "id": int(pid),
                "outer_fold": fold,
                "hospital": hospital.get(int(pid)),
                "event": int(event.loc[int(pid)]),
                "tumor_group": int(tumor.loc[int(pid)]),
                "side": "external",
            })
    print("outer sizes", [len(g) for g in outer_groups], flush=True)
    print(
        "outer events",
        [int(cohort.loc[cohort.id.isin(g), "event"].sum()) for g in outer_groups],
        flush=True,
    )
    for fold, members in enumerate(outer_groups):
        centres = cohort.loc[cohort.id.isin(members), "centre"].value_counts().sort_index()
        print(f"outer {fold} hospitals", centres.to_dict(), flush=True)

    model_dir = OUT_DIR / "models"
    assigned: list[dict] = []
    partitions = list(fold_rows)
    for outer, test_ids in enumerate(outer_groups):
        train = cohort.loc[~cohort["id"].isin(test_ids)].copy()
        inner_groups = stratified_groups(train, N_INNER, INNER_SEED + outer)
        print(
            f"outer {outer}: test {len(test_ids)}, train {len(train)}, "
            f"inner {[len(g) for g in inner_groups]}, "
            f"inner events {[int(cohort.loc[cohort.id.isin(g), 'event'].sum()) for g in inner_groups]}",
            flush=True,
        )
        for inner, members in enumerate(inner_groups):
            for pid in members:
                partitions.append({
                    "id": int(pid),
                    "outer_fold": outer,
                    "hospital": hospital.get(int(pid)),
                    "event": int(event.loc[int(pid)]),
                    "tumor_group": int(tumor.loc[int(pid)]),
                    "side": "train",
                    "inner_fold": inner,
                })
        for inner, held in enumerate(inner_groups):
            fit_ids = np.concatenate([g for i, g in enumerate(inner_groups) if i != inner])
            saved = model_dir / f"outer_{outer}_inner_{inner}"
            if (saved / "lda.gensim").exists():
                dictionary, model, saved_ids = load_lda(saved)
                if not np.array_equal(saved_ids, fit_ids):
                    raise SystemExit(f"{saved.name} was trained on a different patient list")
            else:
                dictionary, model = fit_lda([documents[int(i)] for i in fit_ids], LDA_SEED)
                save_lda(saved, dictionary, model, fit_ids)
            for ids, role in ((fit_ids, "train"), (held, "held_out")):
                probs = predict(model, dictionary, [documents[int(i)] for i in ids])
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

        train_ids = np.concatenate(inner_groups)
        saved = model_dir / f"outer_{outer}_refit"
        if (saved / "lda.gensim").exists():
            dictionary, model, saved_ids = load_lda(saved)
            if not np.array_equal(saved_ids, train_ids):
                raise SystemExit(f"{saved.name} was trained on a different patient list")
        else:
            dictionary, model = fit_lda([documents[int(i)] for i in train_ids], LDA_SEED)
            save_lda(saved, dictionary, model, train_ids)
        for ids, fold_label, role in (
            (train_ids, "refit", "refit"),
            (test_ids, "external", "external_test"),
        ):
            probs = predict(model, dictionary, [documents[int(i)] for i in ids])
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

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT_DIR / "outer_folds.csv", index=False)
    pd.DataFrame(partitions).to_csv(OUT_DIR / "partitions.csv", index=False)

    manifest = {
        "cohort": "DB1",
        "design": "nested CV; equal groups balancing hospital, then event, then tumor group",
        "compared_with": "results/lda/val_2/anidada, which shuffles without stratification",
        "n_audit": int(audit["id"].nunique()),
        "n_with_profile": int(len(cohort)),
        "excluded_no_profile": missing,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "outer_seed": OUTER_SEED,
        "inner_seed": f"{INNER_SEED} + outer fold",
        "lda_seed": LDA_SEED,
        "num_topics": N_TOPICS,
        "outer_fold_sizes": [int(len(g)) for g in outer_groups],
        "outer_events": [int(cohort.loc[cohort.id.isin(g), "event"].sum()) for g in outer_groups],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    out = pd.DataFrame(assigned).merge(folds[["id", "hospital"]], on="id", how="left")
    front = [
        "id", "hospital", "outer_fold", "inner_fold", "role",
        "topic", "topic_probability", "n_train", "lda_seed",
    ]
    out = out[front + [f"topic_{t}" for t in range(N_TOPICS)]]
    out = out.sort_values(["outer_fold", "inner_fold", "role", "id"]).reset_index(drop=True)
    dest = OUT_DIR / "topic_probabilities.csv"
    out.to_csv(dest, index=False)
    held = out[out["role"].isin(["held_out", "external_test"])]
    held.to_csv(OUT_DIR / "topic_assignment.csv", index=False)
    print(f"{len(out)} rows -> {dest}", flush=True)


if __name__ == "__main__":
    main()
