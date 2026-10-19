#!/usr/bin/env python3
"""Full alarm report reproduction with Evento_DB1 + Fecha_Fin clock.

Same artefacts as the original ``alarm_granularity_report`` (monthly, granularity,
W×H, missingness, trajectories, analysis figures with bootstrap CIs), but:
  - behavioural series from revision daily / topics
  - event clock from audit BBDD (Evento_DB1, Fin−Inicio)

Temporarily overlays ``Subjects_data.xlsx`` PD_event/Obs_time so existing alarm
scripts keep working unchanged; restores the backup afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
ALARM = PROJECT_ROOT / "scripts/03_analysis/alarm"
REV = PROJECT_ROOT / "data/revision"
CLIN = PROJECT_ROOT / "data/processed/clinical/Subjects_data.xlsx"
CLIN_BAK = PROJECT_ROOT / "data/processed/clinical/Subjects_data_PRE_Evento_DB1_overlay_backup.xlsx"
AUDIT = REV / "178 audit oct 2 2026 copia.xlsx"
OUT = PROJECT_ROOT / "results/alarm"
TABLES = OUT / "tables"
DB1 = OUT / "Evento_DB1_Fin"


def _run(script: str, *args: str, label: str) -> None:
    cmd = [PY, str(ALARM / script), *args]
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def load_audit_clock() -> pd.DataFrame:
    a = pd.read_excel(AUDIT)
    a = a[pd.to_numeric(a["id"], errors="coerce").notna()].copy()
    a["id"] = a["id"].astype(int)
    a["Evento_DB1"] = pd.to_numeric(a["Evento_DB1"], errors="coerce")
    a = a[a["Evento_DB1"].isin([0, 1])].copy()
    a["Evento_DB1"] = a["Evento_DB1"].astype(int)
    a["Fecha_inicio_actualizada"] = pd.to_datetime(a["Fecha_inicio_actualizada"])
    a["Fecha_Fin_DB1_Actualizado"] = pd.to_datetime(a["Fecha_Fin_DB1_Actualizado"])
    a["t_evento_DB1"] = (
        a["Fecha_Fin_DB1_Actualizado"] - a["Fecha_inicio_actualizada"]
    ).dt.days.astype(float)
    return a[
        [
            "id",
            "Evento_DB1",
            "Fecha_inicio_actualizada",
            "Fecha_Fin_DB1_Actualizado",
            "t_evento_DB1",
        ]
    ]


def overlay_clinical(clock: pd.DataFrame) -> None:
    """Write Evento_DB1 / t_evento into Subjects_data PD_event / Obs_time."""
    if not CLIN_BAK.exists():
        shutil.copy2(CLIN, CLIN_BAK)
        print(f"Clinical backup → {CLIN_BAK}")
    clin = pd.read_excel(CLIN_BAK if CLIN_BAK.exists() else CLIN)
    id_col = "S_RECORD_id" if "S_RECORD_id" in clin.columns else "id"
    clin[id_col] = clin[id_col].astype(int)
    m = clin.merge(
        clock.rename(columns={"id": id_col}),
        on=id_col,
        how="left",
        suffixes=("", "_db1"),
    )
    # only overwrite where audit has a row
    has = m["Evento_DB1"].notna()
    m.loc[has, "PD_event"] = m.loc[has, "Evento_DB1"].astype(int)
    m.loc[has, "Obs_time"] = m.loc[has, "t_evento_DB1"].astype(float)
    # drop merge helper cols
    drop = [c for c in m.columns if c.endswith("_db1") or c in {
        "Evento_DB1", "Fecha_inicio_actualizada", "Fecha_Fin_DB1_Actualizado", "t_evento_DB1"
    }]
    m = m.drop(columns=drop, errors="ignore")
    m.to_excel(CLIN, index=False)
    print(f"Overlay Evento_DB1 → {CLIN.name} ({int(has.sum())} ids updated)")


def restore_clinical() -> None:
    if CLIN_BAK.exists():
        shutil.copy2(CLIN_BAK, CLIN)
        print(f"Restored clinical from {CLIN_BAK.name}")


def install_reclocked_puf(clock: pd.DataFrame) -> None:
    """Ensure tables use Evento_DB1 times (reuse Evento_DB1_Fin if present)."""
    TABLES.mkdir(parents=True, exist_ok=True)
    DB1.mkdir(parents=True, exist_ok=True)

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        amap = clock.set_index("id")
        out = df.copy()
        out = out[out["id"].isin(amap.index)].copy()
        out["Evento PD"] = out["id"].map(amap["Evento_DB1"]).astype(int)
        out["t_evento_eb2"] = out["id"].map(amap["t_evento_DB1"]).astype(float)
        out["follow_up_days"] = out["id"].map(amap["t_evento_DB1"]).astype(float)
        return out

    daily_src = TABLES / "daily_sliding_puf.csv"
    if (DB1 / "daily_sliding_puf_Evento_DB1_Fin.csv").exists():
        daily = pd.read_csv(DB1 / "daily_sliding_puf_Evento_DB1_Fin.csv")
    else:
        daily = apply(pd.read_csv(daily_src))
    daily.to_csv(TABLES / "daily_sliding_puf.csv", index=False)
    daily.to_csv(DB1 / "daily_sliding_puf_Evento_DB1_Fin.csv", index=False)

    for name in ("weekly", "biweekly"):
        src = TABLES / f"collapsed_puf_{name}.csv"
        dest_db1 = DB1 / f"collapsed_puf_{name}_Evento_DB1_Fin.csv"
        if dest_db1.exists():
            cdf = pd.read_csv(dest_db1)
        else:
            cdf = apply(pd.read_csv(src))
        cdf.to_csv(src, index=False)
        cdf.to_csv(dest_db1, index=False)

    print("Installed Evento_DB1 pUF tables under results/alarm/tables/")


def main() -> None:
    clock = load_audit_clock()
    clock.to_csv(DB1 / "audit_Evento_DB1_clock.csv", index=False)

    try:
        overlay_clinical(clock)
        install_reclocked_puf(clock)

        # Full pipeline (reuse LDA windows; clinical overlay drives labels)
        _run(
            "run_granularity_sweep.py",
            "--monthly-only",
            label="1/5 Monthly paper W=3 H=4",
        )
        _run(
            "run_granularity_sweep.py",
            "--skip-lda",
            "--w-days",
            "60,90,120",
            "--h-days",
            "90,120,180",
            label="2/5 Granularity × W×H sweep",
        )
        _run(
            "analyze_missingness_and_wh_grid.py",
            "--w-days",
            "60,90,120",
            "--h-days",
            "90,120,180",
            label="3/5 Missingness funnel + W×H grid",
        )
        _run(
            "plot_patient_trajectories.py",
            "--patients",
            "62004,23003,31002,41006",
            label="4/5 Monthly patient trajectories",
        )
        _run(
            "plot_granularity_trajectories.py",
            "--patients",
            "62004,23003,31002,41006",
            "--W",
            "90",
            label="4b/5 Granularity trajectories",
        )
        _run(
            "plot_analysis_figures.py",
            label="5/5 Analysis delivery figures (bootstrap CIs)",
        )
    finally:
        restore_clinical()

    print("\n✓ Full alarm pipeline finished with Evento_DB1 clock.")
    print(f"Tables/figures under {OUT}")
    print("Next: regenerate report MD/HTML/PDF from these outputs.")


if __name__ == "__main__":
    main()
