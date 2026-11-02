#!/usr/bin/env python3
"""Monthly LDA topics for DB2/DB3 — fixed 30-day blocks (alarm-compatible).

Same definition as DB1 / paper alarms / 178 audit:
  - densified daily calendar (real days)
  - non-overlapping windows of 30 embedding days from first densified day
  - trailing ``n % 30`` **kept** as last (possibly incomplete) month with LDA
    probs if there is at least one observed day (like 178 audit)
  - landmark month m ↔ observation day ~m×30

DB2 labels: ``mes_1`` = block that contains 2025-09-11 (full or short).
``mes_-3,-2,-1`` = three previous complete blocks. Never fall back to
sequential labeling from day 0 when cutoff is missing.
Clinical Excel = fechas_cnio base + summary preds + monthly topics.

DB2 summary: ``topic_predominante_DB2`` = LDA predominant over densified days
from 2025-09-11 → fin (extension window only, not full DB1∪DB2 span).
DB3 summary: ``topic_predominante_mes_1`` + ``topic_predominante_global``.

DB3: ``mes_1…N`` sequential; incomplete last month kept.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS_LDA = PROJECT_ROOT / "models" / "lda"
LDA_PATH = MODELS_LDA / "LDA_model_vdec25.gensim"
DICT_PATH = MODELS_LDA / "dictionary_LDA_vdec25.dict"

CUTOFF_DB2 = pd.Timestamp("2025-09-11")
WINDOW = 30
MIN_EMB = 15
MIN_OBS_IN_WINDOW = 15
TOPIC_K = list(range(6))

DB_CFG = {
    "DB2": {
        "pkl": PROJECT_ROOT
        / "data/processed/vq-vae/DB2/profiles_per_sample_DB2_span_real_a0.pkl",
        "clinical_base": PROJECT_ROOT
        / "data/processed/clinical/DB2/fechas_cnio_oct_2_2026.xlsx",
        "clinical_out": PROJECT_ROOT
        / "data/processed/clinical/DB2/fechas_cnio_oct_2_2026_con_topics.xlsx",
        "out_dir": PROJECT_ROOT / "results/by_db/DB2",
        "mode": "relative_cutoff",
        "paste_from_mes": -3,
    },
    "DB3": {
        "pkl": PROJECT_ROOT
        / "data/processed/vq-vae/DB3/profiles_per_sample_DB3_span_real_a0.pkl",
        "clinical_base": PROJECT_ROOT
        / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026.xlsx",
        "clinical_out": PROJECT_ROOT
        / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026_con_topics.xlsx",
        "out_dir": PROJECT_ROOT / "results/by_db/DB3",
        "mode": "sequential",
        "paste_from_mes": 1,
    },
}


def load_day_table_from_pkl(pkl_path: Path, model_type: str = "a0", n: int = 30) -> pd.DataFrame:
    with pkl_path.open("rb") as f:
        embedding_counts = pickle.load(f)
    users = embedding_counts[model_type][n]
    rows = []
    for user_id, payload in users.items():
        length, seq, _mapped, dates, some_obs, *_rest = payload[0]
        emb = np.asarray(seq).tolist()
        dates = [pd.Timestamp(d).normalize() for d in pd.to_datetime(dates)]
        some = np.asarray(some_obs).astype(int)
        rows.append(
            {
                "id": int(user_id),
                "embedding_ids": emb,
                "dates": dates,
                "some_observed": some.tolist(),
                "n_days": int(length),
                "n_observed": int(some.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def iter_windows(n: int, window: int) -> list[tuple[int, int]]:
    """Inclusive-end slices as (start, end_exclusive), including a possible short tail."""
    out = []
    for start in range(0, n, window):
        end = min(start + window, n)
        out.append((start, end))
    return out


def find_cutoff_block_idx(
    dates: list, windows: list[tuple[int, int]], cutoff: pd.Timestamp
) -> int | None:
    """Index of the window whose date span contains ``cutoff`` (incl. short last)."""
    for i, (s, e) in enumerate(windows):
        if dates[s] <= cutoff <= dates[e - 1]:
            return i
    return None


def block_mes_label(block_idx: int, cutoff_idx: int, mode: str) -> int:
    if mode != "relative_cutoff":
        return block_idx + 1
    off = block_idx - cutoff_idx
    return off if off < 0 else off + 1


def lda_window(
    emb: list,
    some: np.ndarray,
    start: int,
    end: int,
    lda: LdaModel,
    dictionary,
    min_obs: int = MIN_OBS_IN_WINDOW,
) -> tuple[list[float] | None, int | None, int]:
    idxs = [i for i in range(start, end) if int(some[i]) == 1]
    n_obs = len(idxs)
    if n_obs < min_obs:
        return None, None, n_obs
    bow = dictionary.doc2bow([str(emb[i]) for i in idxs])
    topics = lda.get_document_topics(bow, minimum_probability=0.0)
    probs = [float(p) for _, p in sorted(topics, key=lambda x: x[0])]
    pred = int(max(topics, key=lambda x: x[1])[0])
    return probs, pred, n_obs


def compute_long_for_patient(
    row: pd.Series,
    lda: LdaModel,
    dictionary,
    mode: str,
    cutoff: pd.Timestamp | None,
) -> list[dict]:
    """Fixed 30-day blocks from first densified day.

    Complete months stay unchanged. Trailing incomplete block (``len < 30``)
    is **kept** (178-audit style) with LDA probs if ≥1 observed day.
    DB2: ``mes_1`` = block (full or short) containing 2025-09-11.
    """
    pid = int(row["id"])
    dates = list(row["dates"])
    emb = list(row["embedding_ids"])
    some = np.asarray(row["some_observed"], dtype=int)
    n = len(dates)
    # Allow short-only patients (incomplete single block), like 178 audit
    if n < 1:
        return []

    windows = iter_windows(n, WINDOW)
    cutoff_idx = None
    if mode == "relative_cutoff" and cutoff is not None:
        cutoff_idx = find_cutoff_block_idx(dates, windows, cutoff)
        if cutoff_idx is None:
            # Cutoff not in densified span → cannot define relative months
            return []

    keep: list[tuple[int, int, int]] = []  # (orig_idx, start, end)
    for i, (s, e) in enumerate(windows):
        length = e - s
        if length == WINDOW:
            keep.append((i, s, e))
        elif length > 0:
            # incomplete trailing month kept (178 audit style)
            keep.append((i, s, e))

    if not keep:
        return []

    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"
    rows = []
    for abs_i, (orig_i, start, end) in enumerate(keep, start=1):
        d0, d1 = dates[start], dates[end - 1]
        block_len = int(end - start)
        # full months: require ≥15 observed; short last month: ≥1 (178 style)
        min_obs = MIN_OBS_IN_WINDOW if block_len == WINDOW else 1
        probs, pred, n_obs = lda_window(
            emb, some, start, end, lda, dictionary, min_obs=min_obs
        )
        ok = probs is not None
        if mode == "relative_cutoff":
            mes = block_mes_label(orig_i, cutoff_idx, mode)
        else:
            mes = abs_i
        rec = {
            "id": pid,
            mes_col: int(mes),
            "abs_month": abs_i,
            "anchor": dates[0].date().isoformat(),
            "fin_real": dates[n - 1].date().isoformat(),
            "block_start": d0.date().isoformat(),
            "block_end": d1.date().isoformat(),
            "block_len": block_len,
            "n_observed_days": int(n_obs),
            "topic_ok": bool(ok),
            "topic_predominante": pred if ok else np.nan,
            "decision_day": orig_i * WINDOW + (end - start),  # end day index (1-based-ish)
        }
        for k in TOPIC_K:
            rec[f"topic{k}"] = float(probs[k]) if ok else np.nan
        if mode == "relative_cutoff":
            rec["cutoff_block_idx"] = cutoff_idx
            rec["is_cutoff_block"] = bool(orig_i == cutoff_idx)
            cut_s, cut_e = windows[cutoff_idx]
            rec["cutoff_in_incomplete_tail"] = bool((cut_e - cut_s) < WINDOW)
        rows.append(rec)
    return rows


def ordered_meses(values, from_mes: int) -> list[int]:
    vals = sorted({int(v) for v in values if int(v) != 0 and int(v) >= from_mes})
    if from_mes < 0:
        for m in [-3, -2, -1, 1]:
            if m >= from_mes and m not in vals:
                vals.append(m)
        vals = sorted(set(vals))
    return [m for m in vals if m < 0] + [m for m in vals if m > 0]


def predominant_over_indices(
    emb: list, some: np.ndarray, idxs: list[int], lda: LdaModel, dictionary
) -> tuple[float | None, int]:
    """Predominant topic on selected day indices; returns (pred, n_obs)."""
    use = [i for i in idxs if int(some[i]) == 1]
    n_obs = len(use)
    if n_obs < 1:
        return None, 0
    bow = dictionary.doc2bow([str(emb[i]) for i in use])
    topics = lda.get_document_topics(bow, minimum_probability=0.0)
    pred = int(max(topics, key=lambda x: x[1])[0])
    return pred, n_obs


def compute_summary_topics(
    day: pd.DataFrame,
    long: pd.DataFrame,
    lda: LdaModel,
    dictionary,
    mode: str,
    cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """Extra summary columns for clinical Excel.

    DB2: predominant topic over densified days from extension cutoff → fin
         (not the whole DB1∪DB2 span).
    DB3: predominant of mes_1 + predominant over the full observed span.
    """
    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"
    rows = []
    day_by_id = day.set_index("id")
    for pid, g in long.groupby("id"):
        pid = int(pid)
        if pid not in day_by_id.index:
            continue
        r = day_by_id.loc[pid]
        emb = list(r["embedding_ids"])
        some = np.asarray(r["some_observed"], dtype=int)
        dates = [pd.Timestamp(d).normalize() for d in r["dates"]]
        n = len(dates)
        rec: dict = {"S_RECORD_id": pid}
        if mode == "relative_cutoff" and cutoff is not None:
            idxs = [i for i in range(n) if dates[i] >= cutoff]
            pred, n_obs = predominant_over_indices(emb, some, idxs, lda, dictionary)
            rec["topic_predominante_DB2"] = pred if pred is not None else np.nan
            rec["n_observed_days_DB2"] = int(n_obs)
        else:
            # DB3: mes_1 predominant (from monthly long) + global over full span
            m1 = g[g[mes_col] == 1]
            if len(m1) and bool(m1.iloc[0]["topic_ok"]):
                rec["topic_predominante_mes_1"] = int(m1.iloc[0]["topic_predominante"])
            else:
                rec["topic_predominante_mes_1"] = np.nan
            pred_g, n_obs = predominant_over_indices(
                emb, some, list(range(n)), lda, dictionary
            )
            rec["topic_predominante_global"] = pred_g if pred_g is not None else np.nan
            rec["n_observed_days_global"] = int(n_obs)
        rows.append(rec)
    return pd.DataFrame(rows)


def attach_clinical(
    base_path: Path,
    long: pd.DataFrame,
    out_path: Path,
    mode: str,
    from_mes: int,
    summary: pd.DataFrame | None = None,
) -> None:
    base = pd.read_excel(base_path)
    base["S_RECORD_id"] = pd.to_numeric(base["S_RECORD_id"], errors="coerce")
    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"
    meses = ordered_meses(long[mes_col].dropna().unique(), from_mes=from_mes)

    wide_rows = []
    for pid, g in long.groupby("id"):
        rec = {"S_RECORD_id": int(pid)}
        for _, r in g.iterrows():
            m = int(r[mes_col])
            if m not in meses:
                continue
            for k in TOPIC_K:
                rec[f"mes_{m}_topic{k}"] = r[f"topic{k}"]
            # only topic0..5 in monthly block (summary preds added separately)
        wide_rows.append(rec)
    wide = pd.DataFrame(wide_rows)
    topic_cols = []
    for m in meses:
        topic_cols += [f"mes_{m}_topic{k}" for k in TOPIC_K]
    for c in topic_cols:
        if c not in wide.columns:
            wide[c] = np.nan

    # Summary columns right after CNIO base, before monthly topics
    if summary is not None and len(summary):
        if mode == "relative_cutoff":
            sum_cols = ["topic_predominante_DB2"]
        else:
            sum_cols = ["topic_predominante_mes_1", "topic_predominante_global"]
        keep = ["S_RECORD_id"] + [c for c in sum_cols if c in summary.columns]
        wide = summary[keep].merge(wide, on="S_RECORD_id", how="right")
        ordered = ["S_RECORD_id"] + [c for c in sum_cols if c in wide.columns] + topic_cols
        wide = wide[ordered]
    else:
        wide = wide[["S_RECORD_id"] + topic_cols]

    out = base.merge(wide, on="S_RECORD_id", how="left")
    out.to_excel(out_path, index=False)


def _span_prefix_cols(long: pd.DataFrame, clinical_base: Path, db: str) -> pd.DataFrame:
    """Four leading span columns: clinical inicio/fin + daily inicio/fin used for topics."""
    span = (
        long.groupby("id", as_index=False)
        .agg(fecha_inicio_daily=("anchor", "first"), fecha_fin_daily=("fin_real", "first"))
    )
    clin = pd.read_excel(clinical_base)
    clin["S_RECORD_id"] = pd.to_numeric(clin["S_RECORD_id"], errors="coerce")
    if db == "DB2":
        # fecha_inicio_DB1, fecha_fin_DB2, daily start (from DB1), daily end (eB2 span)
        keep = clin[
            [
                "S_RECORD_id",
                "Fecha_inicio_reconciliada",
                "Fecha_Fin_DB2",
            ]
        ].rename(
            columns={
                "S_RECORD_id": "id",
                "Fecha_inicio_reconciliada": "fecha_inicio_DB1",
                "Fecha_Fin_DB2": "fecha_fin_DB2",
            }
        )
        span = span.rename(
            columns={
                "fecha_inicio_daily": "fecha_inicio_daily_DB1",
                "fecha_fin_daily": "fecha_fin_daily_eB2",
            }
        )
        prefix = [
            "fecha_inicio_DB1",
            "fecha_fin_DB2",
            "fecha_inicio_daily_DB1",
            "fecha_fin_daily_eB2",
        ]
    else:
        fin_col = "Fecha_fin_DB3" if "Fecha_fin_DB3" in clin.columns else "Fecha_fin"
        keep = clin[["S_RECORD_id", "Fecha_inicio_reconciliada", fin_col]].rename(
            columns={
                "S_RECORD_id": "id",
                "Fecha_inicio_reconciliada": "fecha_inicio",
                fin_col: "fecha_fin_DB3",
            }
        )
        span = span.rename(
            columns={
                "fecha_inicio_daily": "fecha_inicio_daily",
                "fecha_fin_daily": "fecha_fin_daily",
            }
        )
        prefix = ["fecha_inicio", "fecha_fin_DB3", "fecha_inicio_daily", "fecha_fin_daily"]

    out = keep.merge(span, on="id", how="right")
    for c in prefix:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[["id"] + prefix], prefix


def write_rangos_aux(
    long: pd.DataFrame,
    out_dir: Path,
    mode: str,
    from_mes: int,
    clinical_base: Path,
    db: str,
) -> None:
    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"
    # Rangos aux: full lookback (all months present), not truncated at paste_from_mes
    mes_vals = long[mes_col].dropna()
    full_from = int(mes_vals.min()) if len(mes_vals) else from_mes
    meses = ordered_meses(mes_vals.unique(), from_mes=full_from)
    long = long.copy()
    long["rango"] = long["block_start"].astype(str) + " → " + long["block_end"].astype(str)
    wide = (
        long.pivot_table(index="id", columns=mes_col, values="rango", aggfunc="first")
        .reindex(columns=meses)
    )
    wide.columns = [f"mes_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()
    starts = long.pivot_table(
        index="id", columns=mes_col, values="block_start", aggfunc="first"
    ).reindex(columns=meses)
    ends = long.pivot_table(
        index="id", columns=mes_col, values="block_end", aggfunc="first"
    ).reindex(columns=meses)
    starts.columns = [f"mes_{int(c)}_inicio" for c in starts.columns]
    ends.columns = [f"mes_{int(c)}_fin" for c in ends.columns]
    se = starts.reset_index().merge(ends.reset_index(), on="id")
    prefix_df, prefix_cols = _span_prefix_cols(long, clinical_base, db)
    wide = prefix_df.merge(wide, on="id", how="right")
    se = prefix_df.merge(se, on="id", how="right")
    ordered = ["id"] + prefix_cols
    for m in meses:
        ordered += [f"mes_{m}_inicio", f"mes_{m}_fin"]
    se = se[[c for c in ordered if c in se.columns]]
    wide = wide[["id"] + prefix_cols + [c for c in wide.columns if c not in (["id"] + prefix_cols)]]
    path = out_dir / "topics_meses_rangos_fechas.xlsx"
    bloques = build_bloques_rangos_long(long, mode=mode, db=db, cutoff=CUTOFF_DB2)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        wide.to_excel(w, sheet_name="rangos", index=False)
        se.to_excel(w, sheet_name="inicio_fin", index=False)
        bloques.to_excel(w, sheet_name="bloques_long", index=False)
    wide.to_csv(out_dir / "topics_meses_rangos_fechas.csv", index=False)
    bloques.to_csv(out_dir / "topics_bloques_rangos_long.csv", index=False)


def build_bloques_rangos_long(
    long: pd.DataFrame,
    mode: str,
    db: str,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per block: dates, length, and window-type flags (traceability).

    tipo_ventana:
      - regular_30d: non-overlapping full block (len=30)
      - cola_corta: incomplete last block (len<30), non-overlapping (not a
        sliding/overlapping tail). Kept with LDA probs if ≥1 observed day.
    There are **no overlapping final windows** in this pipeline.
    """
    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"
    rows: list[dict] = []
    for pid, g in long.groupby("id"):
        g = g.sort_values(mes_col).copy()
        last_idx = g.index[-1]
        for _, r in g.iterrows():
            blen = int(r["block_len"])
            is_last = r.name == last_idx
            if blen >= WINDOW:
                tipo = "regular_30d"
            else:
                tipo = "cola_corta"
            mes = int(r[mes_col])
            rec = {
                "id": int(pid),
                "db": db,
                "n_bloque_mes": mes,
                "block_start": pd.Timestamp(r["block_start"]).strftime("%Y-%m-%d"),
                "block_end": pd.Timestamp(r["block_end"]).strftime("%Y-%m-%d"),
                "block_len_days": blen,
                "n_observed_days": int(r["n_observed_days"]),
                "tipo_ventana": tipo,
                "es_ultimo_bloque": bool(is_last),
                "es_ventana_regular": blen >= WINDOW,
                "es_cola_corta": blen < WINDOW,
                "es_final_solapada": False,  # pipeline never uses overlapping tails
                "topic_ok": bool(r["topic_ok"]),
                "topic_predominante_bloque": (
                    int(r["topic_predominante"])
                    if bool(r["topic_ok"]) and pd.notna(r["topic_predominante"])
                    else np.nan
                ),
            }
            if mode == "relative_cutoff":
                rec["es_mes_1_extension"] = mes == 1
                rec["es_lookback"] = mes < 0
                rec["es_periodo_extension_DB2"] = mes >= 1
                if cutoff is not None:
                    b0 = pd.Timestamp(r["block_start"]).normalize()
                    b1 = pd.Timestamp(r["block_end"]).normalize()
                    rec["contiene_cutoff_2025_09_11"] = bool(b0 <= cutoff <= b1)
                    rec["cutoff_in_incomplete_tail"] = bool(
                        r.get("cutoff_in_incomplete_tail", False)
                    )
            else:
                rec["es_mes_1"] = mes == 1
            rows.append(rec)
    out = pd.DataFrame(rows)
    # stable column order
    front = [
        "id",
        "db",
        "n_bloque_mes",
        "block_start",
        "block_end",
        "block_len_days",
        "n_observed_days",
        "tipo_ventana",
        "es_ventana_regular",
        "es_cola_corta",
        "es_final_solapada",
        "es_ultimo_bloque",
        "topic_ok",
        "topic_predominante_bloque",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def run_db(db: str) -> None:
    cfg = DB_CFG[db]
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg["mode"]
    cutoff = CUTOFF_DB2 if mode == "relative_cutoff" else None
    mes_col = "mes_rel" if mode == "relative_cutoff" else "mes"

    print(f"\n{'=' * 70}\n{db}: fixed 30-day blocks (alarm-compatible)\n{'=' * 70}")
    day = load_day_table_from_pkl(cfg["pkl"])
    lda = LdaModel.load(str(LDA_PATH))
    dictionary = corpora.Dictionary.load(str(DICT_PATH))

    all_rows: list[dict] = []
    for _, row in day.iterrows():
        all_rows.extend(
            compute_long_for_patient(row, lda, dictionary, mode=mode, cutoff=cutoff)
        )
    long = pd.DataFrame(all_rows)
    print(f"  months={len(long)} patients={long['id'].nunique()}")
    n_short = int((long["block_len"] < WINDOW).sum()) if len(long) else 0
    n_short_ok = (
        int(((long["block_len"] < WINDOW) & long["topic_ok"]).sum()) if len(long) else 0
    )
    print(f"  incomplete last months kept: {n_short} (with probs: {n_short_ok})")
    if mode == "relative_cutoff" and len(long):
        n_with_m1 = int((long[mes_col] == 1).sum())
        n_m1_short = int(
            ((long[mes_col] == 1) & (long["block_len"] < WINDOW)).sum()
        )
        print(f"  mes_1 filled: {n_with_m1} (of which short/incomplete: {n_m1_short})")

    long_path = out_dir / "topics_mensuales_anclados_long.csv"
    clin_out = cfg["clinical_out"]
    # backup before switching to keep-incomplete-last rule
    if long_path.exists():
        bak = out_dir / "topics_mensuales_anclados_long.csv.pre_keep_incomplete_last_backup"
        if not bak.exists():
            shutil.copy2(long_path, bak)
    if clin_out.exists():
        bak_c = clin_out.with_name(clin_out.stem + "_pre_keep_incomplete_last_backup.xlsx")
        if not bak_c.exists():
            shutil.copy2(clin_out, bak_c)

    long.to_csv(long_path, index=False)
    summary = compute_summary_topics(
        day, long, lda, dictionary, mode=mode, cutoff=cutoff
    )
    summary.to_csv(out_dir / "topics_summary_predominante.csv", index=False)
    if mode == "relative_cutoff":
        n_pred = int(summary["topic_predominante_DB2"].notna().sum())
        print(f"  topic_predominante_DB2 filled: {n_pred}/{len(summary)}")
    else:
        n1 = int(summary["topic_predominante_mes_1"].notna().sum())
        ng = int(summary["topic_predominante_global"].notna().sum())
        print(f"  topic_predominante_mes_1: {n1}/{len(summary)}")
        print(f"  topic_predominante_global: {ng}/{len(summary)}")

    attach_clinical(
        cfg["clinical_base"],
        long,
        clin_out,
        mode=mode,
        from_mes=cfg["paste_from_mes"],
        summary=summary,
    )
    write_rangos_aux(
        long,
        out_dir,
        mode=mode,
        from_mes=cfg["paste_from_mes"],
        clinical_base=cfg["clinical_base"],
        db=db,
    )
    # delivery copies used in results/by_db
    delivery = out_dir / f"{db} oct 2 2026_topics.xlsx"
    shutil.copy2(clin_out, delivery)
    print(f"  → {delivery}")

    por = (
        long.groupby("id")
        .agg(
            n_meses=(mes_col, "count"),
            n_meses_ok=("topic_ok", "sum"),
            anchor=("anchor", "first"),
            fin_real=("fin_real", "first"),
        )
        .reset_index()
    )
    resumen = pd.DataFrame(
        [
            {
                "dataset": db,
                "block_rule": "non-overlapping 30 densified days from first day (alarm m×30)",
                "drop_incomplete_tail": False,
                "keep_incomplete_last": True,
                "mes_labeling": (
                    "mes_1 = block containing 2025-09-11 (full or short); "
                    "lookback -3,-2,-1; incomplete last month kept (178 style)"
                    if mode == "relative_cutoff"
                    else "mes_1..N sequential; incomplete last month kept (178 style)"
                ),
                "n_patients": long["id"].nunique(),
                "n_months": len(long),
                "n_months_ok": int(long["topic_ok"].sum()),
                "models": "frozen VQ-VAE a0 + LDA vdec25",
            }
        ]
    )
    qc = out_dir / "topics_mensuales_anclados_QC.xlsx"
    with pd.ExcelWriter(qc, engine="openpyxl") as w:
        por.to_excel(w, sheet_name="por_paciente", index=False)
        resumen.to_excel(w, sheet_name="resumen", index=False)
        if db == "DB2" and (long["id"] == 31001).any():
            long[long["id"] == 31001].sort_values(mes_col).to_excel(
                w, sheet_name="ejemplo_31001", index=False
            )

    if db == "DB2" and (long["id"] == 31001).any():
        ej = long[long["id"] == 31001].set_index("mes_rel")
        print("  31001 (fixed-30; dates may drift vs calendar anniversary):")
        for m in [-3, -2, -1, 1]:
            if m in ej.index:
                print(
                    f"    mes_{m}: {ej.loc[m, 'block_start']} → {ej.loc[m, 'block_end']} "
                    f"(len={ej.loc[m, 'block_len']})"
                )

    print(f"  → {long_path}")
    print(f"  → {clin_out}")
    print(f"  → {out_dir / 'topics_meses_rangos_fechas.xlsx'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", choices=["DB2", "DB3", "both"], default="both")
    args = parser.parse_args()
    for db in (["DB2", "DB3"] if args.db == "both" else [args.db]):
        run_db(db)


if __name__ == "__main__":
    main()
