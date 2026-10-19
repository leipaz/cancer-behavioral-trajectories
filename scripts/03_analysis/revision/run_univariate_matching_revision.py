#!/usr/bin/env python3
"""Univariate matched analysis — revision (new CNIO date filter + 1:5 matching).

Uses:
  - data/processed/univariate/Base_Matching_matched_1to5_followup.xlsx
  - data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv
  - enriched extras left-joined from dic25_enriched (7 vars)

Writes:
  - cohorts under data/processed/univariate/cohorts/
  - stats tables under results/univariate/tables/revision/
  - matching_ventanas_fechas_auxiliar (structure + concrete V1/V2 dates)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/02_univariate_analysis/lda"))

from config import SEL_COLS, columns_present  # noqa: E402
from stats_utils import (  # noqa: E402
    build_summary_table,
    compute_mann_whitney_p_values,
    filter_by_response,
)

MATCHING = PROJECT_ROOT / "data/processed/univariate/Base_Matching_matched_1to5_followup.xlsx"
# Canonical CNIO daily (git); local intermediates may exist under data/revision/ (ignored)
DAILY_VQCLIP = (
    PROJECT_ROOT
    / "data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv"
)
DAILY_VALIDITY = (
    PROJECT_ROOT
    / "data/revision/daily_summary_oficial_filtrado_rangos_CNIO_dedup_validity.csv"
)
DAILY_DEDUP = PROJECT_ROOT / "data/revision/daily_summary_oficial_filtrado_rangos_CNIO_dedup.csv"
DAILY = PROJECT_ROOT / "data/revision/daily_summary_oficial_filtrado_rangos_CNIO.csv"
DAILY_ENRICHED_OUT = (
    PROJECT_ROOT
    / "data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_enriched_vqclip.csv"
)
ENRICHED_SRC = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/data_dic2025/"
    "daily_summary_eb2prod_dic25_enriched.csv"
)
OUT_DIR = PROJECT_ROOT / "results/univariate/tables/revision"
COHORT_DIR = PROJECT_ROOT / "data/processed/univariate/cohorts"

ID_COL = "ID"
FECHA_COL = "date_time"
LABEL_EPD = "E.PD"
LABEL_NO_EPD = "no-E.PD"

EXTRA_FROM_ENRICHED = [
    "total_unlocks",
    "total_communication",
    "total_social_networks",
    "total_others",
    "deep_time",
    "light_time",
    "rem_time",
]

# Origin validity (same as scripts/01_preprocess/lda/preprocess_daily_summaries.py)
COLS_HR_SENTINEL = ["heart_rate_hr_mean", "heart_rate_hr_min", "heart_rate_hr_max"]
SLEEP_START_MAX_SECONDS = 24 * 60 * 60  # 86_400


def apply_origin_validity(df: pd.DataFrame) -> pd.DataFrame:
    """HR −1 → NaN; sleep_start > 24h → NaN (do not invent other corrections)."""
    out = df.copy()
    present_hr = [c for c in COLS_HR_SENTINEL if c in out.columns]
    n_hr = 0
    if present_hr:
        before = out[present_hr].eq(-1).sum().sum()
        out[present_hr] = out[present_hr].replace(-1.0, np.nan)
        n_hr = int(before)
    n_sleep = 0
    if "sleep_start" in out.columns:
        s = pd.to_numeric(out["sleep_start"], errors="coerce")
        mask = s > SLEEP_START_MAX_SECONDS
        n_sleep = int(mask.sum())
        out.loc[mask, "sleep_start"] = np.nan
    print(
        f"Origin validity: HR −1→NaN in {n_hr} cells "
        f"({', '.join(present_hr) or 'none'}); "
        f"sleep_start >{SLEEP_START_MAX_SECONDS}s → NaN in {n_sleep} rows"
    )
    return out


def load_matching() -> pd.DataFrame:
    m = pd.read_excel(MATCHING)
    m = m.rename(
        columns={
            "S_RECORD_id": ID_COL,
            "Fechas inicio ventana 1": "Fecha_entrada_v1",
            "Fecha fin ventana 1": "Fecha_final_simulada_v1",
            "Fecha inicio ventana 2": "Fecha_entrada_v2",
            "Fecha fin ventana 2": "Fecha_final_simulada_v2",
        }
    )
    m[ID_COL] = pd.to_numeric(m[ID_COL], errors="coerce").astype("Int64")
    for c in (
        "Fecha_entrada_v1",
        "Fecha_final_simulada_v1",
        "Fecha_entrada_v2",
        "Fecha_final_simulada_v2",
    ):
        m[c] = pd.to_datetime(m[c], errors="coerce").dt.normalize()
    m["Early PD"] = pd.to_numeric(m["Early PD"], errors="coerce").astype(int)
    return m


def merge_enriched_onto_filtered(daily: pd.DataFrame) -> pd.DataFrame:
    """Left-join the 7 enriched-only columns onto the CNIO-filtered daily."""
    if not ENRICHED_SRC.is_file():
        print(f"[warn] Enriched not found ({ENRICHED_SRC}); continuing without extras.")
        return daily

    need = [c for c in EXTRA_FROM_ENRICHED if c not in daily.columns]
    if not need:
        print("Enriched columns already present in daily.")
        return daily

    enr = pd.read_csv(ENRICHED_SRC, usecols=["user", "date_time"] + EXTRA_FROM_ENRICHED, low_memory=False)
    enr = enr.rename(columns={"user": ID_COL})
    enr[ID_COL] = pd.to_numeric(enr[ID_COL], errors="coerce").astype("Int64")
    enr[FECHA_COL] = pd.to_datetime(enr[FECHA_COL], errors="coerce").dt.normalize()
    enr = enr.dropna(subset=[ID_COL, FECHA_COL]).drop_duplicates([ID_COL, FECHA_COL], keep="first")

    # drop extras if partially present
    daily = daily.drop(columns=[c for c in EXTRA_FROM_ENRICHED if c in daily.columns], errors="ignore")
    merged = daily.merge(enr, on=[ID_COL, FECHA_COL], how="left")
    assert len(merged) == len(daily)
    for c in EXTRA_FROM_ENRICHED:
        nn = int(merged[c].notna().sum())
        print(f"  enriched merge {c}: {nn}/{len(merged)} non-null ({nn/len(merged):.1%})")
    merged.to_csv(DAILY_ENRICHED_OUT, index=False)
    print(f"Wrote {DAILY_ENRICHED_OUT}")
    return merged


def load_daily() -> pd.DataFrame:
    # Prefer VQ-clip daily (includes origin validity); else validity; else dedup/filtrado
    if DAILY_VQCLIP.is_file():
        src = DAILY_VQCLIP
        already_clean = True
        clean_label = "origin validity + VQ CLIP_INFO"
    elif DAILY_VALIDITY.is_file():
        src = DAILY_VALIDITY
        already_clean = True
        clean_label = "origin validity"
    elif DAILY_DEDUP.is_file():
        src = DAILY_DEDUP
        already_clean = False
        clean_label = ""
    else:
        src = DAILY
        already_clean = False
        clean_label = ""
    df = pd.read_csv(src, low_memory=False)
    if "user" in df.columns and ID_COL not in df.columns:
        df = df.rename(columns={"user": ID_COL})
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")
    df[FECHA_COL] = pd.to_datetime(df[FECHA_COL], errors="coerce").dt.normalize()
    before = len(df)
    df = df.drop_duplicates([ID_COL, FECHA_COL], keep="first")
    print(
        f"Daily ({src.name}): {before} rows → {len(df)} unique (ID, date); "
        f"users={df[ID_COL].nunique()}"
    )
    if already_clean:
        print(f"Cleaning: already in daily ({clean_label}; skip re-clean).")
    else:
        df = apply_origin_validity(df)
    df = merge_enriched_onto_filtered(df)
    return df


def write_matching_fechas_auxiliar(matching: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """One row per matching participant: structure + concrete V1/V2 date ranges + coverage."""
    cov_v1 = cov[cov["window"] == "v1"][
        ["ID", "match_set_id", "expected_days", "days_in_daily", "coverage"]
    ].rename(
        columns={
            "expected_days": "v1_dias_esperados",
            "days_in_daily": "v1_dias_en_daily",
            "coverage": "v1_cobertura",
        }
    )
    cov_v2 = cov[cov["window"] == "v2"][
        ["ID", "match_set_id", "expected_days", "days_in_daily", "coverage"]
    ].rename(
        columns={
            "expected_days": "v2_dias_esperados",
            "days_in_daily": "v2_dias_en_daily",
            "coverage": "v2_cobertura",
        }
    )

    aux = matching[
        [
            ID_COL,
            "match_set_id",
            "match_role",
            "index_case_id",
            "Early PD",
            "Ventana 1",
            "Ventana 2",
            "V2_i",
            "V2_e",
            "Fecha_entrada_v1",
            "Fecha_final_simulada_v1",
            "Fecha_entrada_v2",
            "Fecha_final_simulada_v2",
        ]
    ].copy()
    # optional clinical labels if present
    for c in ("Cohorte", "Group tumor", "Tiempo Obs Final", "Evento PD"):
        if c in matching.columns:
            aux[c] = matching[c]

    aux = aux.rename(
        columns={
            ID_COL: "ID",
            "Early PD": "Early_PD",
            "Ventana 1": "ventana1_etiqueta",
            "Ventana 2": "ventana2_etiqueta",
            "Fecha_entrada_v1": "v1_fecha_inicio",
            "Fecha_final_simulada_v1": "v1_fecha_fin",
            "Fecha_entrada_v2": "v2_fecha_inicio",
            "Fecha_final_simulada_v2": "v2_fecha_fin",
        }
    )
    aux["v1_fecha_inicio"] = pd.to_datetime(aux["v1_fecha_inicio"]).dt.strftime("%Y-%m-%d")
    aux["v1_fecha_fin"] = pd.to_datetime(aux["v1_fecha_fin"]).dt.strftime("%Y-%m-%d")
    aux["v2_fecha_inicio"] = pd.to_datetime(aux["v2_fecha_inicio"]).dt.strftime("%Y-%m-%d")
    aux["v2_fecha_fin"] = pd.to_datetime(aux["v2_fecha_fin"]).dt.strftime("%Y-%m-%d")
    aux["v1_n_dias_calendario"] = (
        pd.to_datetime(aux["v1_fecha_fin"]) - pd.to_datetime(aux["v1_fecha_inicio"])
    ).dt.days + 1
    aux["v2_n_dias_calendario"] = (
        pd.to_datetime(aux["v2_fecha_fin"]) - pd.to_datetime(aux["v2_fecha_inicio"])
    ).dt.days + 1

    aux = aux.merge(cov_v1, on=["ID", "match_set_id"], how="left")
    aux = aux.merge(cov_v2, on=["ID", "match_set_id"], how="left")
    aux = aux.sort_values(["match_set_id", "match_role", "ID"], ascending=[True, False, True])

    out_csv = OUT_DIR / "matching_ventanas_fechas_auxiliar.csv"
    out_xlsx = OUT_DIR / "matching_ventanas_fechas_auxiliar.xlsx"
    # also under univariate data for visibility next to matching Excel
    out_csv2 = COHORT_DIR.parent / "matching_ventanas_fechas_auxiliar.csv"
    out_xlsx2 = COHORT_DIR.parent / "matching_ventanas_fechas_auxiliar.xlsx"
    for p in (out_csv, out_csv2):
        aux.to_csv(p, index=False)
    for p in (out_xlsx, out_xlsx2):
        aux.to_excel(p, index=False)
    print(f"Wrote auxiliar fechas → {out_xlsx2}")
    return aux


def build_window_cohorts(
    daily: pd.DataFrame,
    matching: pd.DataFrame,
    sel_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build V1/V2 day cohorts from matching absolute dates.

    Both windows keep ``match_set_id`` so a control reused across sets is
    counted once per set (fairer means in matched analysis). Within a set,
    ``(ID, date)`` is unique; across sets the same basal day may repeat.
    """
    keep_m = matching[
        [
            ID_COL,
            "match_set_id",
            "Early PD",
            "Fecha_entrada_v1",
            "Fecha_final_simulada_v1",
            "Fecha_entrada_v2",
            "Fecha_final_simulada_v2",
        ]
    ].copy()
    merged = daily[[ID_COL, FECHA_COL] + sel_cols].merge(keep_m, on=ID_COL, how="inner")
    merged["Respuesta"] = merged["Early PD"].map({1: LABEL_EPD, 0: LABEL_NO_EPD})

    mask_v1 = (merged[FECHA_COL] >= merged["Fecha_entrada_v1"]) & (
        merged[FECHA_COL] <= merged["Fecha_final_simulada_v1"]
    )
    mask_v2 = (merged[FECHA_COL] >= merged["Fecha_entrada_v2"]) & (
        merged[FECHA_COL] <= merged["Fecha_final_simulada_v2"]
    )

    base_cols = [ID_COL, FECHA_COL, "match_set_id", "Early PD", "Respuesta"]
    df_first = (
        merged.loc[mask_v1, base_cols + sel_cols]
        .drop_duplicates(subset=[ID_COL, FECHA_COL, "match_set_id"], keep="first")
        .sort_values(by=["match_set_id", ID_COL, FECHA_COL])
        .copy()
    )
    df_previous = (
        merged.loc[mask_v2, base_cols + sel_cols]
        .drop_duplicates(subset=[ID_COL, FECHA_COL, "match_set_id"], keep="first")
        .sort_values(by=["match_set_id", ID_COL, FECHA_COL])
        .copy()
    )
    print(
        f"First 15 days: {df_first[ID_COL].nunique()} IDs, {len(df_first)} day-rows "
        f"(unique ID+date={df_first.drop_duplicates([ID_COL, FECHA_COL]).shape[0]}; "
        f"E.PD={int((df_first['Early PD']==1).sum())}, "
        f"no-E.PD={int((df_first['Early PD']==0).sum())})"
    )
    print(
        f"Previous 15 days: {df_previous[ID_COL].nunique()} IDs, {len(df_previous)} day-rows "
        f"(unique ID+date={df_previous.drop_duplicates([ID_COL, FECHA_COL]).shape[0]}; "
        f"E.PD={int((df_previous['Early PD']==1).sum())}, "
        f"no-E.PD={int((df_previous['Early PD']==0).sum())})"
    )
    return df_first, df_previous


def period_comparison_table(
    df_first: pd.DataFrame,
    df_previous: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Within-group First15 vs Previous15 (Mann–Whitney), as in dic2025 table."""
    rows = []
    for var in variables:
        for group in (LABEL_EPD, LABEL_NO_EPD):
            a = pd.to_numeric(
                df_first.loc[df_first["Respuesta"] == group, var], errors="coerce"
            ).dropna()
            b = pd.to_numeric(
                df_previous.loc[df_previous["Respuesta"] == group, var], errors="coerce"
            ).dropna()
            if len(a) and len(b):
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                q1a, q3a = np.percentile(a, [25, 75])
                q1b, q3b = np.percentile(b, [25, 75])
                rows.append(
                    {
                        "Variable": var,
                        "Group": group,
                        "Comparison": "First15 vs Previous15",
                        "N_first": len(a),
                        "N_prev": len(b),
                        "Mean_first": round(float(a.mean()), 3),
                        "Mean_prev": round(float(b.mean()), 3),
                        "Median [Q1–Q3] First": f"{np.median(a):.3f} [{q1a:.3f}–{q3a:.3f}]",
                        "Median [Q1–Q3] Prev": f"{np.median(b):.3f} [{q1b:.3f}–{q3b:.3f}]",
                        "p-value": f"{p:.2e}",
                    }
                )
            else:
                rows.append(
                    {
                        "Variable": var,
                        "Group": group,
                        "Comparison": "First15 vs Previous15",
                        "N_first": len(a),
                        "N_prev": len(b),
                        "Mean_first": np.nan,
                        "Mean_prev": np.nan,
                        "Median [Q1–Q3] First": "NaN",
                        "Median [Q1–Q3] Prev": "NaN",
                        "p-value": "NaN",
                    }
                )
    return pd.DataFrame(rows)


def coverage_report(matching: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    ids_m = set(matching[ID_COL].dropna().astype(int))
    ids_d = set(daily[ID_COL].dropna().astype(int))
    rows = []
    for _, r in matching.iterrows():
        pid = int(r[ID_COL])
        dsub = daily.loc[daily[ID_COL] == pid, FECHA_COL]
        for label, s, e in (
            ("v1", r["Fecha_entrada_v1"], r["Fecha_final_simulada_v1"]),
            ("v2", r["Fecha_entrada_v2"], r["Fecha_final_simulada_v2"]),
        ):
            n = int(((dsub >= s) & (dsub <= e)).sum()) if pid in ids_d else 0
            expected = (pd.Timestamp(e) - pd.Timestamp(s)).days + 1
            rows.append(
                {
                    "ID": pid,
                    "match_set_id": int(r["match_set_id"]),
                    "match_role": r["match_role"],
                    "index_case_id": int(r["index_case_id"]),
                    "Early_PD": int(r["Early PD"]),
                    "window": label,
                    "start": s,
                    "end": e,
                    "expected_days": expected,
                    "days_in_daily": n,
                    "coverage": n / expected if expected else np.nan,
                    "id_in_daily": pid in ids_d,
                }
            )
    cov = pd.DataFrame(rows)
    print(
        f"Matching IDs: {len(ids_m)}; in daily: {len(ids_m & ids_d)}; "
        f"missing from daily: {sorted(ids_m - ids_d)}"
    )
    print(
        f"Window coverage mean v1={cov.loc[cov.window=='v1','coverage'].mean():.2%}, "
        f"v2={cov.loc[cov.window=='v2','coverage'].mean():.2%}"
    )
    return cov


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_DIR.mkdir(parents=True, exist_ok=True)

    matching = load_matching()
    daily = load_daily()
    sel_cols = columns_present(daily.columns, SEL_COLS)
    print(f"Using {len(sel_cols)} variables: {sel_cols}")
    missing = [c for c in SEL_COLS if c not in sel_cols]
    if missing:
        print(f"Missing in filtered daily (skipped): {missing}")

    cov = coverage_report(matching, daily)
    cov.to_csv(OUT_DIR / "matching_window_coverage.csv", index=False)
    write_matching_fechas_auxiliar(matching, cov)

    df_first, df_previous = build_window_cohorts(daily, matching, sel_cols)

    df_first.to_csv(COHORT_DIR / "cohort_ventana1_basal.csv", index=False)
    df_previous.to_csv(COHORT_DIR / "cohort_ventana2_prePD_matching.csv", index=False)
    df_first.to_excel(COHORT_DIR / "cohort_ventana1_basal.xlsx", index=False)
    df_previous.to_excel(COHORT_DIR / "cohort_ventana2_prePD_matching.xlsx", index=False)

    # For stats_utils Early PD filter
    df_first_f = filter_by_response(df_first)
    df_previous_f = filter_by_response(df_previous)
    p_first, p_previous = compute_mann_whitney_p_values(
        df_first, df_previous, sel_cols
    )

    full = build_summary_table(
        df_first_f,
        df_previous_f,
        sel_cols,
        p_first,
        p_previous,
        include_min_max=True,
    )
    full.to_csv(OUT_DIR / "variable_summary_full_stats_EPD.csv", index=False)

    compact = full.drop(columns=["Min", "Max"], errors="ignore")
    compact.to_csv(
        OUT_DIR / "variable_summary_mean_medianIQR_var_EPDvsnoEPD_2dates.csv",
        index=False,
    )

    intra = period_comparison_table(df_first_f, df_previous_f, sel_cols)
    intra.to_csv(OUT_DIR / "variable_period_comparison_intra_group.csv", index=False)

    enriched_note = (
        f" + enriched left-join ({', '.join(EXTRA_FROM_ENRICHED)}) → "
        "`daily_summary_oficial_filtrado_rangos_CNIO_enriched.csv`"
        if all(c in daily.columns for c in EXTRA_FROM_ENRICHED)
        else " (enriched extras missing)"
    )
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Univariate tables — revision (Evento_DB1 / rangos CNIO)",
                "",
                "Matching: `data/processed/univariate/Base_Matching_matched_1to5_followup.xlsx` (1:5, same pathology, at-risk ≥ case FU).",
                "Daily: `data/daily_summaries/daily_summary_oficial_filtrado_rangos_CNIO_dedup_vqclip.csv` "
                "(origin validity + VQ-VAE CLIP_INFO → NaN; from dedup CNIO)."
                + enriched_note
                + ".",
                "",
                "Windows per matching row (absolute calendar dates from the matching Excel):",
                "- **Ventana 1 (basal / first ~15d)** = `Fechas inicio ventana 1` → `Fecha fin ventana 1`",
                "- **Ventana 2 (pre-PD / matched)** = `Fecha inicio ventana 2` → `Fecha fin ventana 2`",
                "",
                "Controls reused across sets keep each set's own ventana 2.",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `variable_summary_full_stats_EPD.csv` | Mean, median IQR, min/max, var, MW p (E.PD vs no-E.PD) |",
                "| `variable_summary_mean_medianIQR_var_EPDvsnoEPD_2dates.csv` | Same without min/max |",
                "| `variable_period_comparison_intra_group.csv` | First vs Previous within E.PD / no-E.PD |",
                "| `matching_window_coverage.csv` | Days available in daily per window (long) |",
                "| `matching_ventanas_fechas_auxiliar.csv/.xlsx` | Matching structure + concrete V1/V2 dates + coverage |",
                "",
                "Also copied to `data/processed/univariate/matching_ventanas_fechas_auxiliar.xlsx`.",
                "",
                f"Variables used ({len(sel_cols)}): " + ", ".join(sel_cols),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nWrote tables → {OUT_DIR}")
    print(full.groupby(["Period", "Group"])["N"].sum().to_string())


if __name__ == "__main__":
    main()
