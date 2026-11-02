#!/usr/bin/env python3
"""Drop incomplete trailing calendar month from DB2/DB3 anchored monthly topics.

Keeps all other DB2/DB3 criteria unchanged (anniversary blocks, mes_1 around
2025-09-11 for DB2, topic_ok if n_observed_days >= 15, etc.).

Rule: while the patient's **last** month has ``block_len < 30`` (calendar
window cut by fin_real), drop that month. Do **not** drop full-length months
that merely have ``n_observed_days < 30`` due to gaps (those keep topic_ok).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DB_CFG = {
    "DB2": {
        "long": PROJECT_ROOT / "results/by_db/DB2/topics_mensuales_anclados_long.csv",
        "clinical": PROJECT_ROOT
        / "data/processed/clinical/DB2/fechas_cnio_oct_2_2026_con_topics.xlsx",
        "qc": PROJECT_ROOT / "results/by_db/DB2/topics_mensuales_anclados_QC.xlsx",
        "mes_col": "mes_rel",
    },
    "DB3": {
        "long": PROJECT_ROOT / "results/by_db/DB3/topics_mensuales_anclados_long.csv",
        "clinical": PROJECT_ROOT
        / "data/processed/clinical/DB3/fechas_cnio_oct_2_2026_con_topics.xlsx",
        "qc": PROJECT_ROOT / "results/by_db/DB3/topics_mensuales_anclados_QC.xlsx",
        "mes_col": "mes",
    },
}

MIN_DAYS_COMPLETE_MONTH = 30
TOPIC_SUFFIXES = (
    "topic0",
    "topic1",
    "topic2",
    "topic3",
    "topic4",
    "topic5",
    "topic_predominante",
    "n_observed_days",
    "block_start",
    "block_end",
)


def _block_len(df: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(df["block_start"])
    end = pd.to_datetime(df["block_end"])
    return (end - start).dt.days + 1


def drop_incomplete_last_months(
    long: pd.DataFrame, mes_col: str, min_days: int = MIN_DAYS_COMPLETE_MONTH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop last month if calendar block_len < min_days (usually one pass)."""
    out = long.copy()
    out["_block_len"] = _block_len(out)
    out = out.sort_values(["id", mes_col]).reset_index(drop=True)

    last_idx = out.groupby("id", sort=False).tail(1).index
    last = out.loc[last_idx]
    drop_mask = last["_block_len"] < min_days
    drop_rows = last.loc[drop_mask].copy()

    kept = out.drop(index=last_idx[drop_mask]).drop(columns=["_block_len"])
    audit = drop_rows.drop(columns=["_block_len"], errors="ignore")
    audit["reason"] = f"last_month_block_len<{min_days}"
    return kept.reset_index(drop=True), audit.reset_index(drop=True)


def clear_clinical_months(
    clinical: pd.DataFrame, audit: pd.DataFrame, mes_col_name: str
) -> pd.DataFrame:
    """NaN topic columns for dropped (id, mes) pairs."""
    out = clinical.copy()
    id_col = "S_RECORD_id"
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce")
    for _, row in audit.iterrows():
        pid = int(row["id"])
        mes = int(row[mes_col_name])
        mask = out[id_col] == pid
        for suf in TOPIC_SUFFIXES:
            col = f"mes_{mes}_{suf}"
            if col in out.columns:
                out.loc[mask, col] = np.nan
    return out


def update_qc_db2(qc_path: Path, long: pd.DataFrame, audit: pd.DataFrame) -> None:
    por = pd.read_excel(qc_path, sheet_name="por_paciente")
    resumen = pd.read_excel(qc_path, sheet_name="resumen")
    try:
        ejemplo = pd.read_excel(qc_path, sheet_name="ejemplo_31001")
    except ValueError:
        ejemplo = None

    # refresh n_mes_pos_ok from long (mes_rel > 0 & topic_ok)
    g = (
        long[(long["mes_rel"] > 0) & (long["topic_ok"])]
        .groupby("id")
        .size()
        .rename("n_mes_pos_ok")
    )
    por = por.copy()
    por["n_mes_pos_ok"] = por["id"].map(g).fillna(0).astype(int)

    resumen = resumen.copy()
    resumen["drop_incomplete_last_month"] = True
    resumen["n_last_months_dropped"] = len(audit)
    resumen["rule_last_month"] = (
        f"drop trailing months while block_len<{MIN_DAYS_COMPLETE_MONTH} "
        "(calendar incomplete; gaps alone do not drop)"
    )

    with pd.ExcelWriter(qc_path, engine="openpyxl") as w:
        por.to_excel(w, sheet_name="por_paciente", index=False)
        resumen.to_excel(w, sheet_name="resumen", index=False)
        if ejemplo is not None:
            # re-filter ejemplo from new long if id present
            ej = long[long["id"] == 31001].copy()
            if len(ej):
                ej.to_excel(w, sheet_name="ejemplo_31001", index=False)
            else:
                ejemplo.to_excel(w, sheet_name="ejemplo_31001", index=False)
        audit.to_excel(w, sheet_name="dropped_last_months", index=False)


def update_qc_db3(qc_path: Path, long: pd.DataFrame, audit: pd.DataFrame) -> None:
    por = pd.read_excel(qc_path, sheet_name="por_paciente")
    resumen = pd.read_excel(qc_path, sheet_name="resumen")

    g_ok = long[long["topic_ok"]].groupby("id").size().rename("n_meses_ok")
    g_n = long.groupby("id").size().rename("n_meses")
    por = por.copy()
    por["n_meses"] = por["id"].map(g_n).fillna(0).astype(int)
    por["n_meses_ok"] = por["id"].map(g_ok).fillna(0).astype(int)

    resumen = resumen.copy()
    resumen["drop_incomplete_last_month"] = True
    resumen["n_last_months_dropped"] = len(audit)
    resumen["rule_last_month"] = (
        f"drop trailing months while block_len<{MIN_DAYS_COMPLETE_MONTH} "
        "(calendar incomplete; gaps alone do not drop)"
    )

    with pd.ExcelWriter(qc_path, engine="openpyxl") as w:
        por.to_excel(w, sheet_name="por_paciente", index=False)
        resumen.to_excel(w, sheet_name="resumen", index=False)
        audit.to_excel(w, sheet_name="dropped_last_months", index=False)


def process_db(db: str, dry_run: bool = False) -> None:
    cfg = DB_CFG[db]
    mes_col = cfg["mes_col"]
    long_path: Path = cfg["long"]
    clinical_path: Path = cfg["clinical"]
    qc_path: Path = cfg["qc"]

    long = pd.read_csv(long_path)
    n_before = len(long)

    # backups once from pre-drop state
    bak_long = long_path.with_suffix(".csv.pre_drop_incomplete_last_backup")
    bak_clin = clinical_path.with_name(
        clinical_path.stem + "_pre_drop_incomplete_last_backup.xlsx"
    )
    if not dry_run:
        if not bak_long.exists():
            shutil.copy2(long_path, bak_long)
        if not bak_clin.exists():
            shutil.copy2(clinical_path, bak_clin)

    audits: list[pd.DataFrame] = []
    round_i = 0
    while True:
        round_i += 1
        kept, audit = drop_incomplete_last_months(long, mes_col=mes_col)
        if len(audit) == 0:
            break
        audit = audit.copy()
        audit["drop_round"] = round_i
        audits.append(audit)
        long = kept
        if round_i > 20:
            raise RuntimeError(f"{db}: too many drop rounds")

    audit_all = (
        pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    )

    print(f"\n===== {db} =====")
    print(
        f"long before={n_before} after={len(long)} "
        f"dropped_last={len(audit_all)} rounds={max(round_i - 1, 0)}"
    )
    if len(audit_all):
        print(
            audit_all[
                ["id", mes_col, "block_start", "block_end", "n_observed_days", "topic_ok"]
            ]
            .head(5)
            .to_string(index=False)
        )

    if dry_run:
        return

    long.to_csv(long_path, index=False)

    clinical = pd.read_excel(
        bak_clin if bak_clin.exists() else clinical_path
    )
    clinical2 = clear_clinical_months(clinical, audit_all, mes_col_name=mes_col)
    clinical2.to_excel(clinical_path, index=False)

    audit_path = long_path.parent / "dropped_incomplete_last_months.csv"
    audit_all.to_csv(audit_path, index=False)

    if db == "DB2":
        update_qc_db2(qc_path, long, audit_all)
    else:
        update_qc_db3(qc_path, long, audit_all)

    print(f"wrote {long_path}")
    print(f"wrote {clinical_path}")
    print(f"wrote {audit_path}")
    print(f"updated {qc_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop incomplete last month (n<30) from DB2/DB3 topics."
    )
    parser.add_argument(
        "--db",
        choices=["DB2", "DB3", "both"],
        default="both",
        help="Which DB to process (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts; do not write",
    )
    args = parser.parse_args()
    dbs = ["DB2", "DB3"] if args.db == "both" else [args.db]
    for db in dbs:
        process_db(db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
