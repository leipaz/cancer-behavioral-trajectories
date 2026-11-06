#!/usr/bin/env python3
"""Leave-one-centre-out topic probabilities for VAL-2.

Each centre, taken from the first digit of the id, is left out once. The LDA
is fit only on the other centres. Their inner split is stratified by event and
tumor group with a fixed seed. The left-out centre is used only as external.

F/U, pUF and the Youden cut are not computed here.

    python scripts/03_analysis/lda/run_val2_loco.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import USER_EMBEDDINGS_CSV  # noqa: E402
from run_val2_blocks import block_spans  # noqa: E402
from run_val2_nested import (  # noqa: E402
    AUDIT,
    LDA_SEED,
    N_INNER,
    N_TOPICS,
    OUT_DIR,
    fit_lda,
    load_lda,
    parse_ids,
    predict,
    save_lda,
)

INNER_SEED = 20261104
LOCO_DIR = OUT_DIR.parent / "loco"
RANGOS = Path(__file__).resolve().parents[3] / "results/by_db/DB1/rangos_meses_t_eb2.csv"


def as_date(value):
    if pd.isna(value) or str(value).strip() in {"", "NaT", "nan"}:
        return None
    return datetime.fromisoformat(str(value)[:10]).date()


def stratified_groups(frame: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    """Spread each event × tumor stratum across groups. The seed is fixed."""
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    offset = 0
    for _, sub in frame.groupby(["event", "tumor"], dropna=False, sort=True):
        ids = np.array(sorted(int(i) for i in sub["id"]), dtype=int)
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            buckets[(offset + i) % n_splits].append(int(pid))
        offset = (offset + len(ids)) % n_splits
    return [np.array(bucket, dtype=int) for bucket in buckets]


def block_rows_for(ids, tokens, starts, meta, model, dictionary) -> list[dict]:
    rows = []
    for pid in ids:
        sequence = tokens[int(pid)]
        spans = block_spans(len(sequence))
        profile_start = starts.get(int(pid))
        for block, (start, end) in enumerate(spans, start=1):
            dist = predict(model, dictionary, [sequence[start:end]])[0]
            topic = int(np.argmax(dist))
            row = dict(meta)
            row.update({
                "id": int(pid),
                "block": block,
                "day_start": start + 1,
                "day_end": end,
                "n_days": end - start,
                "topic": topic,
                "topic_probability": float(dist[topic]),
            })
            if profile_start is not None:
                row["block_start"] = str(profile_start + timedelta(days=start))
                row["block_end"] = str(profile_start + timedelta(days=end - 1))
            else:
                row["block_start"] = ""
                row["block_end"] = ""
            for t in range(N_TOPICS):
                row[f"topic_{t}"] = float(dist[t])
            rows.append(row)
    return rows


def topic_rows(ids, probs, clinical, common) -> list[dict]:
    rows = []
    for i, pid in enumerate(ids):
        dist = probs[i]
        topic = int(np.argmax(dist))
        row = dict(common)
        row.update(clinical[int(pid)])
        row.update({
            "id": int(pid),
            "topic": topic,
            "topic_probability": float(dist[topic]),
            "n_train": common["n_train"],
            "lda_seed": LDA_SEED,
        })
        for t in range(N_TOPICS):
            row[f"topic_{t}"] = float(dist[t])
        rows.append(row)
    return rows


def model_for(directory: Path, fit_ids: np.ndarray, documents: dict):
    if (directory / "lda.gensim").exists():
        dictionary, model, saved_ids = load_lda(directory)
        if not np.array_equal(saved_ids, fit_ids):
            raise SystemExit(f"{directory.name} was trained on a different patient list")
        return dictionary, model
    dictionary, model = fit_lda([documents[i] for i in fit_ids], LDA_SEED)
    save_lda(directory, dictionary, model, fit_ids)
    return dictionary, model


def main() -> None:
    LOCO_DIR.mkdir(parents=True, exist_ok=True)
    (LOCO_DIR / "models").mkdir(parents=True, exist_ok=True)
    audit = pd.read_excel(AUDIT).dropna(subset=["id"]).copy()
    audit["id"] = audit["id"].astype(int)
    audit["centre"] = audit["id"].astype(str).str[0].astype(int)
    embeddings = pd.read_csv(USER_EMBEDDINGS_CSV)
    documents = {
        int(row.user_id): [str(token) for token in parse_ids(row.embedding_ids)]
        for row in embeddings.itertuples(index=False)
    }
    audit["profile"] = audit["id"].isin(documents)
    rangos = pd.read_csv(RANGOS)
    profile_start = {
        int(row.id): as_date(row.fecha_inicio_daily)
        for row in rangos.itertuples(index=False)
    }

    clinical = {}
    for row in audit.to_dict(orient="records"):
        clinical[int(row["id"])] = {
            "hospital": row["Hospital"],
            "centre": int(row["centre"]),
            "event": "" if pd.isna(row["Evento_DB1"]) else int(row["Evento_DB1"]),
            "tumor_group": "" if pd.isna(row["Group tumor"]) else int(row["Group tumor"]),
            "fecha_inicio": "" if as_date(row["Fecha_inicio_actualizada"]) is None else str(as_date(row["Fecha_inicio_actualizada"])),
            "fecha_fin": "" if as_date(row["Fecha_Fin_DB1_Actualizado"]) is None else str(as_date(row["Fecha_Fin_DB1_Actualizado"])),
            "tiempo_obs": "" if pd.isna(row["Tiempo_obs_DB1"]) else int(row["Tiempo_obs_DB1"]),
            "tiempo_obs_digital": "" if pd.isna(row["Tiempo_obs_digital"]) else int(row["Tiempo_obs_digital"]),
            "n_embedding_days": len(documents.get(int(row["id"]), [])),
        }

    exclusions = []
    for pid, info in clinical.items():
        if info["n_embedding_days"] == 0:
            exclusions.append({
                "id": pid,
                "centre": info["centre"],
                "hospital": info["hospital"],
                "stage": "lda",
                "reason": "sin documento digital",
            })
        elif info["n_embedding_days"] < 15:
            exclusions.append({
                "id": pid,
                "centre": info["centre"],
                "hospital": info["hospital"],
                "stage": "bloques",
                "reason": f"{info['n_embedding_days']} dias de perfil, por debajo de 15; entra en el LDA y no tiene mes",
            })

    cohort = audit.loc[audit["profile"], ["id", "centre", "Evento_DB1", "Group tumor"]].copy()
    cohort["event"] = cohort["Evento_DB1"].astype(int)
    cohort["tumor"] = cohort["Group tumor"].astype(int)

    assigned: list[dict] = []
    blocks: list[dict] = []
    partitions: list[dict] = []

    for centre in range(1, 10):
        test_ids = cohort.loc[cohort["centre"] == centre, "id"].to_numpy(dtype=int)
        train = cohort.loc[cohort["centre"] != centre].copy()
        seed = INNER_SEED + centre
        groups = stratified_groups(train, N_INNER, seed)
        print(
            f"centre {centre}: external {len(test_ids)}, train {len(train)}, "
            f"inner sizes {[len(g) for g in groups]}, seed {seed}",
            flush=True,
        )
        for pid in test_ids:
            info = clinical[int(pid)]
            partitions.append({
                "id": int(pid),
                "hospital": info["hospital"],
                "centre": centre,
                "left_out_centre": centre,
                "side": "left_out",
                "inner_fold": "external",
                "event": info["event"],
                "tumor_group": info["tumor_group"],
                "inner_seed": seed,
            })
        for inner, members in enumerate(groups):
            for pid in members:
                info = clinical[int(pid)]
                partitions.append({
                    "id": int(pid),
                    "hospital": info["hospital"],
                    "centre": info["centre"],
                    "left_out_centre": centre,
                    "side": "train_centres",
                    "inner_fold": inner,
                    "event": info["event"],
                    "tumor_group": info["tumor_group"],
                    "inner_seed": seed,
                })

        for inner, held in enumerate(groups):
            fit_ids = np.concatenate([g for i, g in enumerate(groups) if i != inner])
            directory = LOCO_DIR / "models" / f"left_{centre}_inner_{inner}"
            dictionary, model = model_for(directory, fit_ids, documents)
            for ids, role in ((fit_ids, "train"), (held, "held_out")):
                probs = predict(model, dictionary, [documents[i] for i in ids])
                common = {
                    "left_out_centre": centre,
                    "inner_fold": str(inner),
                    "role": role,
                    "model": directory.name,
                    "n_train": int(len(fit_ids)),
                }
                assigned.extend(topic_rows(ids, probs, clinical, common))
                if role == "held_out":
                    blocks.extend(block_rows_for(ids, documents, profile_start, common, model, dictionary))
            print(f"  inner {inner}: train {len(fit_ids)}, held out {len(held)}", flush=True)

        directory = LOCO_DIR / "models" / f"left_{centre}_refit"
        train_ids = np.concatenate(groups)
        dictionary, model = model_for(directory, train_ids, documents)
        for ids, fold_label, role in (
            (train_ids, "refit", "refit"),
            (test_ids, "external", "external_test"),
        ):
            probs = predict(model, dictionary, [documents[i] for i in ids])
            common = {
                "left_out_centre": centre,
                "inner_fold": fold_label,
                "role": role,
                "model": directory.name,
                "n_train": int(len(train_ids)),
            }
            assigned.extend(topic_rows(ids, probs, clinical, common))
            if role == "external_test":
                blocks.extend(block_rows_for(ids, documents, profile_start, common, model, dictionary))

    front = [
        "id", "hospital", "centre", "left_out_centre", "inner_fold", "role", "model",
        "event", "tumor_group", "fecha_inicio", "fecha_fin", "tiempo_obs", "tiempo_obs_digital",
        "n_embedding_days", "topic", "topic_probability", "n_train", "lda_seed",
    ]
    topics = pd.DataFrame(assigned)
    topics["lda_seed"] = LDA_SEED
    topics = topics[front + [f"topic_{t}" for t in range(N_TOPICS)]]
    topics = topics.sort_values(
        ["left_out_centre", "inner_fold", "role", "id"]
    ).reset_index(drop=True)
    topics_path = LOCO_DIR / "topic_probabilities.csv"
    topics.to_csv(topics_path, index=False)

    block_front = [
        "id", "hospital", "centre", "left_out_centre", "inner_fold", "role", "model",
        "block", "day_start", "day_end", "n_days", "block_start", "block_end",
        "event", "tumor_group", "fecha_inicio", "fecha_fin", "tiempo_obs", "tiempo_obs_digital",
        "topic", "topic_probability", "n_train", "lda_seed",
    ]
    block_df = pd.DataFrame(blocks)
    block_df["lda_seed"] = LDA_SEED
    # clinical fields are inside topic_rows but block rows only have meta + id
    # Merge clinical onto blocks.
    clin_df = pd.DataFrame([{"id": pid, **info} for pid, info in clinical.items()])
    keep_clin = ["id", "hospital", "centre", "event", "tumor_group", "fecha_inicio", "fecha_fin", "tiempo_obs", "tiempo_obs_digital"]
    block_df = block_df.drop(columns=[c for c in keep_clin if c in block_df.columns and c != "id"], errors="ignore")
    block_df = block_df.merge(clin_df[keep_clin], on="id", how="left")
    block_df = block_df[block_front + [f"topic_{t}" for t in range(N_TOPICS)]]
    block_df = block_df.sort_values(
        ["left_out_centre", "inner_fold", "role", "id", "block"]
    ).reset_index(drop=True)
    blocks_path = LOCO_DIR / "block_probabilities.csv"
    block_df.to_csv(blocks_path, index=False)

    part_df = pd.DataFrame(partitions).sort_values(
        ["left_out_centre", "side", "inner_fold", "id"]
    )
    part_df.to_csv(LOCO_DIR / "partitions.csv", index=False)
    pd.DataFrame(exclusions).to_csv(LOCO_DIR / "exclusions.csv", index=False)

    manifest = {
        "design": "leave one centre out; centre is the first digit of id",
        "n_audit": int(audit["id"].nunique()),
        "n_with_profile": int(cohort["id"].nunique()),
        "excluded_no_profile": sorted(int(i) for i, info in clinical.items() if info["n_embedding_days"] == 0),
        "excluded_no_month": sorted(int(i) for i, info in clinical.items() if 0 < info["n_embedding_days"] < 15),
        "inner_splits": N_INNER,
        "inner_stratification": "Evento_DB1 x Group tumor",
        "inner_seed": f"{INNER_SEED} + left-out centre",
        "lda_seed": LDA_SEED,
        "num_topics": N_TOPICS,
        "held_out_used_for": "internal Youden cut, after F/U is defined from train rows of the same model",
        "external_used_for": "evaluation of the left-out centre with the refit; not used to choose the cut",
        "fu_and_cut": "not computed",
    }
    (LOCO_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(topics)} topic rows -> {topics_path}", flush=True)
    print(f"{len(block_df)} block rows -> {blocks_path}", flush=True)


if __name__ == "__main__":
    main()
