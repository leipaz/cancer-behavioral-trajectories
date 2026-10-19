#!/usr/bin/env python3
"""Revision end-to-end: frozen VQ-VAE → embeddings → frozen LDA → topics_probs → alarms.

Optimized defaults: CUDA, mode a0 only, larger batch. Does NOT retrain models.
"""

from __future__ import annotations

import argparse
import ast
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/03_analysis/lda"))

from monthly_topics_utils import (  # noqa: E402
    get_topic_distributions_and_predominant_topics_relaxedwindow,
    parse_embedding_ids,
)

REV = PROJECT_ROOT / "data/revision"
MODELS_VQ = PROJECT_ROOT / "models/vq-vae"
MODELS_LDA = PROJECT_ROOT / "models/lda"
PY = sys.executable


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def step_vqvae(args) -> Path:
    out_pkl = REV / "profiles_per_sample_oncology_revision.pkl"
    cmd = [
        PY,
        str(PROJECT_ROOT / "scripts/03_analysis/vq-vae/generate_profiles.py"),
        "--data-csv",
        str(args.data_csv),
        "--models-dir",
        str(MODELS_VQ),
        "--scaler",
        str(MODELS_VQ / "preprocessing_scaler_vqvae.npz"),
        "--output",
        str(out_pkl),
        "--modes",
        *args.modes,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if out_pkl.exists() and args.skip_vqvae:
        print(f"[skip] VQ-VAE — using existing {out_pkl}")
        return out_pkl
    _run(cmd, "1/5 VQ-VAE inference (frozen)")
    return out_pkl


def step_embeddings(pkl_path: Path, model_type: str = "a0", n: int = 30) -> Path:
    print(f"\n{'=' * 70}\n2/5 Extract embedding_ids from PKL\n{'=' * 70}", flush=True)
    with pkl_path.open("rb") as f:
        embedding_counts = pickle.load(f)

    users = embedding_counts[model_type][n]
    rows = []
    for user_id, payload in users.items():
        length, seq, _mapped, dates, some_obs, *_rest = payload[0]
        # Keep full calendar sequence (same as paper PD_cutoff tables).
        emb = np.asarray(seq).tolist()
        dates = pd.to_datetime(dates)
        if hasattr(dates, "tolist"):
            date_list = [pd.Timestamp(d).date().isoformat() for d in dates.tolist()]
        else:
            date_list = [pd.Timestamp(d).date().isoformat() for d in dates]
        some = np.asarray(some_obs).astype(int).tolist()
        rows.append(
            {
                "id": int(user_id),
                "user_id": int(user_id),
                "embedding_ids": emb,
                "dates": date_list,
                "some_observed": some,
                "n_days": int(length),
                "n_observed": int(np.sum(some)),
            }
        )

    emb_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    emb_csv = REV / "user_embeddings_revision_a0.csv"
    emb_df[["user_id", "embedding_ids"]].to_csv(emb_csv, index=False)

    # Day table for monthly + alarm builders (clinical merge)
    clinical_path = PROJECT_ROOT / "data/processed/clinical/Subjects_data.xlsx"
    clin = pd.read_excel(clinical_path)
    # flexible id / event columns
    id_col = next(c for c in clin.columns if str(c).lower() in {"s_record_id", "id"})
    clin = clin.rename(columns={id_col: "id"})
    clin["id"] = clin["id"].astype(int)

    event_col = next(
        (c for c in clin.columns if str(c).lower() in {"pd_event", "evento_pd", "event"}),
        None,
    )
    obs_col = next(
        (c for c in clin.columns if str(c).lower() in {"obs_time", "tiempo_obs", "obs_time_final"}),
        None,
    )
    start_col = next(
        (c for c in clin.columns if "start" in str(c).lower() and "hdm" in str(c).lower()),
        None,
    )

    day = emb_df[["id", "embedding_ids", "n_days", "n_observed"]].copy()
    keep = ["id"]
    if event_col:
        clin["pd_event"] = clin[event_col]
        keep.append("pd_event")
    if obs_col:
        clin["obs_time_final"] = clin[obs_col]
        keep.append("obs_time_final")
    if start_col:
        clin["date_start_hdm"] = pd.to_datetime(clin[start_col], errors="coerce")
        keep.append("date_start_hdm")

    # Official rangos windows
    rangos = pd.read_excel(REV / "rangos_fechas.xlsx").rename(
        columns={
            "S_RECORD_id": "id",
            "Inicio_datos_DB1": "inicio_oficial",
            "Fin_datos_DB1": "fin_oficial",
        }
    )
    rangos["id"] = rangos["id"].astype(int)

    day = day.merge(clin[keep].drop_duplicates("id"), on="id", how="left")
    day = day.merge(rangos[["id", "inicio_oficial", "fin_oficial"]], on="id", how="left")

    # pd_event_diff ≈ obs_time for alarm builders when missing
    if "pd_event" not in day.columns:
        day["pd_event"] = 0
    if "obs_time_final" not in day.columns:
        day["obs_time_final"] = day["n_days"]
    day["pd_event_diff"] = day["obs_time_final"]

    day_csv = REV / "PD_cutoff_topics_revision_a0.csv"
    day.to_csv(day_csv, index=False)
    print(f"Wrote {emb_csv} ({len(emb_df)} users)")
    print(f"Wrote {day_csv}")
    return day_csv


def step_lda_assign(day_csv: Path) -> Path:
    print(f"\n{'=' * 70}\n3/5 Frozen LDA — whole-doc topics + hypertopic\n{'=' * 70}", flush=True)
    lda = LdaModel.load(str(MODELS_LDA / "LDA_model_vdec25.gensim"))
    dictionary = corpora.Dictionary.load(str(MODELS_LDA / "dictionary_LDA_vdec25.dict"))

    df = pd.read_csv(day_csv)
    rows = []
    for _, r in df.iterrows():
        emb = parse_embedding_ids(r["embedding_ids"])
        # Prefer observed-only tokens for whole-document mixture if some_observed present
        # (day CSV may not store it row-wise; use full seq for consistency with monthly).
        bow = dictionary.doc2bow([str(t) for t in emb])
        topics = lda.get_document_topics(bow, minimum_probability=0.0)
        probs = {i: 0.0 for i in range(lda.num_topics)}
        for tid, p in topics:
            probs[tid] = float(p)
        pred = max(probs, key=probs.get)
        p_good = probs[0] + probs[1] + probs[2]
        p_bad = probs[3] + probs[4] + probs[5]
        rows.append(
            {
                "id": int(r["id"]),
                "predominant_topic": int(pred),
                "topic_0": probs[0],
                "topic_1": probs[1],
                "topic_2": probs[2],
                "topic_3": probs[3],
                "topic_4": probs[4],
                "topic_5": probs[5],
                "p_favourable": p_good,
                "p_unfavourable": p_bad,
                "hypertopic_bueno": int(pred in (0, 1, 2)),  # 1=bueno, 0=malo
            }
        )

    topics_df = pd.DataFrame(rows)
    out = REV / "user_topics_revision_a0.csv"
    merged = df.merge(topics_df, on="id", how="left")
    # refresh day csv with topic cols
    merged.to_csv(day_csv, index=False)
    topics_df.to_csv(out, index=False)

    # quick hypertopic summary
    summary = (
        topics_df.groupby("hypertopic_bueno")
        .size()
        .rename({1: "bueno_0_1_2", 0: "malo_3_4_5"})
    )
    print("Hypertopic counts:\n", topics_df["hypertopic_bueno"].value_counts().to_string())
    print(f"Wrote {out}")
    return day_csv


def step_monthly_topics_probs(day_csv: Path) -> Path:
    print(f"\n{'=' * 70}\n4/5 Monthly topic probs → topics_probs.xlsx\n{'=' * 70}", flush=True)
    lda = LdaModel.load(str(MODELS_LDA / "LDA_model_vdec25.gensim"))
    dictionary = corpora.Dictionary.load(str(MODELS_LDA / "dictionary_LDA_vdec25.dict"))
    df = pd.read_csv(day_csv)

    # normalize columns expected by monthly helper
    if "pd_event" not in df.columns:
        df["pd_event"] = 0
    if "pd_event_diff" not in df.columns:
        df["pd_event_diff"] = df.get("obs_time_final", df.get("n_days", 0))

    records = []
    max_months = 0
    for pid in df["id"].tolist():
        dists, predominants = get_topic_distributions_and_predominant_topics_relaxedwindow(
            df,
            pid,
            lda,
            dictionary,
            window_size=30,
            min_embeddings=15,
            verbose=False,
        )
        row = df[df["id"] == pid].iloc[0]
        rec = {
            "id": int(pid),
            "Event eb2 ": int(row.get("pd_event", 0) or 0),
            "t_evento_eb2": row.get("pd_event_diff", np.nan),
            "topic_first_month": predominants[0] if predominants else np.nan,
            "topic_all": int(row["predominant_topic"])
            if "predominant_topic" in df.columns and pd.notna(row.get("predominant_topic"))
            else (predominants[-1] if predominants else np.nan),
        }
        # hypertopic flags
        if predominants:
            rec["Hypertopic mes 1 (1: bueno;  0: malo)."] = int(predominants[0] in (0, 1, 2))
        else:
            rec["Hypertopic mes 1 (1: bueno;  0: malo)."] = np.nan
        if "hypertopic_bueno" in df.columns:
            rec["Hypertopic General  (1: bueno 0-1-2; 0:malo 3-4-5)"] = int(
                row["hypertopic_bueno"]
            )
        else:
            rec["Hypertopic General  (1: bueno 0-1-2; 0:malo 3-4-5)"] = (
                int(rec["topic_all"] in (0, 1, 2)) if pd.notna(rec["topic_all"]) else np.nan
            )

        for m_i, probs in enumerate(dists, start=1):
            for k, p in enumerate(probs):
                rec[f"mes{m_i}_topic{k}"] = float(p)
            rec[f"topic_predominante_mes{m_i}"] = int(predominants[m_i - 1])
        max_months = max(max_months, len(dists))
        records.append(rec)

    master = pd.DataFrame(records)

    # ensure all mes columns exist up to max_months (NaN if missing — never 0-fill)
    for m in range(1, max_months + 1):
        for k in range(6):
            col = f"mes{m}_topic{k}"
            if col not in master.columns:
                master[col] = np.nan

    out_xlsx = REV / "topics_probs_revision.xlsx"
    master.to_excel(out_xlsx, index=False)
    master.to_csv(REV / "topics_probs_revision.csv", index=False)
    print(f"Wrote {out_xlsx}  patients={len(master)}  max_months={max_months}")

    # Install as alarm input (backup previous)
    dest = PROJECT_ROOT / "data/processed/DCABPs/topics_probs.xlsx"
    backup = PROJECT_ROOT / "data/processed/DCABPs/topics_probs_pre_revision_backup.xlsx"
    if dest.exists() and not backup.exists():
        shutil.copy2(dest, backup)
        print(f"Backup → {backup}")
    shutil.copy2(out_xlsx, dest)
    print(f"Installed alarm master → {dest}")

    # Also refresh day csv path used by sliding pUF
    dest_day = PROJECT_ROOT / "data/processed/lda/PD_cutoff_topics_per_day.csv"
    shutil.copy2(day_csv, dest_day)
    # Prefer local fallback over CNIO path by also writing the dic_2025 name in processed
    dest_day2 = (
        PROJECT_ROOT
        / "data/processed/lda/PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv"
    )
    # keep embedding_ids + clinical cols expected by builders
    day_out = pd.read_csv(day_csv)
    day_out.to_csv(dest_day2, index=False)
    print(f"Installed day table → {dest_day2}")
    return out_xlsx


def step_alarms(args) -> None:
    day_csv = REV / "PD_cutoff_topics_revision_a0.csv"
    tables = PROJECT_ROOT / "results/alarm/tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Explicit rebuild with revision day table (avoid stale CNIO path).
    _run(
        [
            PY,
            str(PROJECT_ROOT / "scripts/03_analysis/alarm/build_daily_sliding_puf.py"),
            "--topics-day-csv",
            str(day_csv),
            "--out-csv",
            str(tables / "daily_sliding_puf.csv"),
        ],
        "5a/5 Rebuild daily sliding pUF",
    )
    _run(
        [
            PY,
            str(PROJECT_ROOT / "scripts/03_analysis/alarm/build_collapsed_puf.py"),
            "--topics-day-csv",
            str(day_csv),
            "--block-size",
            "7",
            "14",
            "--out-dir",
            str(tables),
        ],
        "5b/5 Rebuild collapsed weekly/biweekly pUF",
    )

    cmd = [
        PY,
        str(PROJECT_ROOT / "scripts/03_analysis/alarm/run_alarm_pipeline.py"),
        "--skip-lda",  # pUF just rebuilt above from revision embeddings
    ]
    if args.monthly_only:
        cmd = [
            PY,
            str(PROJECT_ROOT / "scripts/03_analysis/alarm/run_alarm_pipeline.py"),
            "--monthly-only",
        ]
    if args.with_trajectories and not args.monthly_only:
        cmd.append("--with-trajectories")
    _run(cmd, "5c/5 Alarm granularity sweep + report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Revision frozen VQ-VAE→LDA→alarm pipeline")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=REV / "oncology_daily_summary_model_input_rangos.csv",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["a0"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-vqvae", action="store_true")
    parser.add_argument("--from-step", type=int, default=1, help="1=vqvae … 5=alarms")
    parser.add_argument("--monthly-only", action="store_true")
    parser.add_argument("--with-trajectories", action="store_true")
    args = parser.parse_args()

    day_csv = REV / "PD_cutoff_topics_revision_a0.csv"
    pkl = REV / "profiles_per_sample_oncology_revision.pkl"

    if args.from_step <= 1:
        pkl = step_vqvae(args)
    if args.from_step <= 2:
        day_csv = step_embeddings(pkl)
    if args.from_step <= 3:
        day_csv = step_lda_assign(day_csv)
    if args.from_step <= 4:
        step_monthly_topics_probs(day_csv)
    if args.from_step <= 5:
        step_alarms(args)

    print("\n✓ Revision pipeline finished.")
    print(f"Artifacts under {REV}")
    print(f"Alarm outputs under {PROJECT_ROOT / 'results/alarm'}")


if __name__ == "__main__":
    main()
