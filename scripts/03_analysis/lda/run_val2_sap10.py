#!/usr/bin/env python3
"""Ten repeats of the stratified nested VAL-2 design, SAP section 12.

Seeds are three uint32 draws per repeat from numpy Generator(20261031):
the outer split, the inner-split base, and the LDA initialisation. The inner
split of outer fold f uses inner_base + f. Everything else stays as in the
single stratified run: five outer folds, four inner folds, hospital then
event then tumour, K=6, the 3/3 training-only F/U rule, W=3, H=120, and the
1013 pairs in anidada_estratificada/objetivo.csv.

The manual clinical maps are not reapplied. They label the 25 models already
in anidada_estratificada and do not transfer to a refit.

    python scripts/03_analysis/lda/run_val2_sap10.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import PROJECT_ROOT  # noqa: E402

SAP_BASE = 20261031
N_REPEATS = 10
N_WORKERS = 4
OUT_ROOT = PROJECT_ROOT / "results/lda/val_2/anidada_estratificada_sap10"
PYTHON = PROJECT_ROOT / "env/bin/python"
OBJETIVO = PROJECT_ROOT / "results/lda/val_2/anidada_estratificada/objetivo.csv"


def sap_seeds() -> pd.DataFrame:
    rng = np.random.default_rng(SAP_BASE)
    draws = rng.integers(0, 2**32, size=(N_REPEATS, 3), dtype=np.uint32)
    return pd.DataFrame({
        "repeat": np.arange(N_REPEATS, dtype=int),
        "outer_seed": draws[:, 0].astype(np.int64),
        "inner_seed": draws[:, 1].astype(np.int64),
        "lda_seed": draws[:, 2].astype(np.int64),
    })


def finish_repeat(out: Path) -> None:
    from run_val2_anticipation import anticipation
    from run_val2_intervals import intervals_for

    decisions_path = out / "alarma" / "decisiones.csv"
    reference = pd.read_csv(out / "alarma" / "rendimiento_por_vuelta.csv")
    external = reference[reference["role"] == "external_test"]
    if int(external["n_decisions"].sum()) != 1013:
        raise SystemExit(f"{out.name}: external decisions sum to {int(external['n_decisions'].sum())}, expected 1013")
    table = intervals_for(decisions_path, ["outer_fold", "role"])
    merged = reference.merge(table, on=["outer_fold", "role"])
    for name in ("sensitivity", "specificity", "PPV", "NPV", "AUROC"):
        gap = (merged[f"{name}_x"] - merged[f"{name}_y"]).abs().max()
        if gap > 1e-9:
            raise SystemExit(f"{out.name}: {name} differs from the bootstrap point by {gap}")
    table.to_csv(out / "alarma" / "intervalos_bootstrap.csv", index=False)

    decisions = pd.read_csv(decisions_path)
    decisions["alarm"] = decisions["alarm"].map({"True": True, "False": False, True: True, False: False})
    by_patient, summary = anticipation(decisions, "outer_fold")
    by_patient.to_csv(out / "alarma" / "anticipacion_por_paciente.csv", index=False)
    summary.to_csv(out / "alarma" / "anticipacion_resumen.csv", index=False)
    leads = by_patient["first_tp_lead"].dropna()
    if (leads > 120).any() or (leads <= 0).any():
        raise SystemExit(f"{out.name}: a lead falls outside 1 to 120 days")


def run_repeat(row: pd.Series) -> str:
    repeat = int(row["repeat"])
    out = OUT_ROOT / f"rep_{repeat:02d}"
    if (out / "alarma" / "intervalos_bootstrap.csv").exists() and (
        out / "alarma" / "anticipacion_resumen.csv"
    ).exists():
        return f"rep_{repeat:02d} already complete"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    commands = [
        [
            str(PYTHON),
            str(_SCRIPT_DIR / "run_val2_stratified.py"),
            "--out", str(out),
            "--outer-seed", str(int(row.outer_seed)),
            "--inner-seed", str(int(row.inner_seed)),
            "--lda-seed", str(int(row.lda_seed)),
            "--repeat", str(repeat),
            "--sap-base", str(SAP_BASE),
        ],
        [str(PYTHON), str(_SCRIPT_DIR / "run_val2_blocks.py"), "--out", str(out)],
        [str(PYTHON), str(_SCRIPT_DIR / "run_val2_fu.py"), "--out", str(out)],
    ]
    log_path = OUT_ROOT / "logs" / f"rep_{repeat:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        for command in commands:
            handle.write("\n$ " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            if completed.returncode != 0:
                raise SystemExit(f"rep_{repeat:02d} failed at {Path(command[1]).name}; see {log_path}")
        handle.write("\n$ intervals and anticipation\n")
        handle.flush()
    finish_repeat(out)
    return f"rep_{repeat:02d} done"


def summarize() -> None:
    frames = []
    for repeat in range(N_REPEATS):
        path = OUT_ROOT / f"rep_{repeat:02d}" / "alarma" / "rendimiento_por_vuelta.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "repeat", repeat)
        frames.append(frame)
    per_fold = pd.concat(frames, ignore_index=True)
    per_fold.to_csv(OUT_ROOT / "rendimiento_por_vuelta.csv", index=False)
    external = per_fold[per_fold["role"] == "external_test"].copy()
    metrics = ["AUROC", "sensitivity", "specificity", "PPV", "NPV"]
    per_repeat = external.groupby("repeat", sort=True)[metrics].mean().reset_index()
    per_repeat.to_csv(OUT_ROOT / "rendimiento_medio_por_repeticion.csv", index=False)
    summary = {
        "n_repeats": N_REPEATS,
        "n_outer": 5,
        "n_inner": 4,
        "sap_base_seed": SAP_BASE,
        "seed_derivation": (
            "numpy.random.Generator(20261031).integers(0, 2**32, size=(10, 3)): "
            "outer_seed, inner_seed, lda_seed. Inner fold f uses inner_seed + f."
        ),
        "comparison": str(OBJETIVO.relative_to(PROJECT_ROOT)),
        "n_decisions": 1013,
        "fu_rule": "3/3 Kaplan-Meier at 365 days on training patients only",
        "clinical_maps": (
            "not applied: alarma_clinica/mapeos_clinicos_25_modelos.txt labels the "
            "single stratified delivery and does not transfer to refitted topics"
        ),
        "window_blocks": 3,
        "horizon_days": 120,
        "bootstrap_seed": 20261020,
        "external_mean_of_repeat_means": {name: float(per_repeat[name].mean()) for name in metrics},
        "external_sd_of_repeat_means": {name: float(per_repeat[name].std(ddof=1)) for name in metrics},
        "external_mean_of_50_folds": {name: float(external[name].mean()) for name in metrics},
        "external_sd_of_50_folds": {name: float(external[name].std(ddof=1)) for name in metrics},
    }
    (OUT_ROOT / "resumen.json").write_text(json.dumps(summary, indent=2))
    print(per_repeat.to_string(index=False))
    print(json.dumps(summary["external_mean_of_repeat_means"], indent=2))


def main() -> None:
    if not PYTHON.exists():
        raise SystemExit(f"missing {PYTHON}")
    if not OBJETIVO.exists():
        raise SystemExit(f"missing {OBJETIVO}")
    from run_val2_anticipation import anticipation  # noqa: F401
    from run_val2_intervals import intervals_for  # noqa: F401
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = sap_seeds()
    seeds.to_csv(OUT_ROOT / "seeds.csv", index=False)
    failures = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_repeat, row): int(row["repeat"]) for _, row in seeds.iterrows()}
        for future in as_completed(futures):
            repeat = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((repeat, str(exc)))
                print(f"rep_{repeat:02d} FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} repeats failed")
    summarize()


if __name__ == "__main__":
    main()
