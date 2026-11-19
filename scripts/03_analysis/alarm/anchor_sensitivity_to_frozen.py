#!/usr/bin/env python3
"""Anchor the missingness sensitivity runs to the frozen alarm.

A mask needs the VQ-VAE and the LDA to run again, so every scenario is a
re-encoding. Re-encoding without any mask does not give back the stored load
exactly (MAE 0.0019: a few days get other codes, and the stored table was made
from a slightly earlier set of codes). To keep every alarm metric on the frozen
alarm, each masked load is anchored:

    anchored load = stored frozen load (AUC_W) + (masked load - unmasked re-run load)

The baseline is then the frozen alarm exactly, and each scenario carries only
the change caused by its mask. The alarm is the anchored load >= threshold.
Topic, pUF and load-change metrics do not depend on the anchor and are unchanged.

Reads the saved per-decision loads of results/missingness/ramp_db1_175_sport_keep1
(ramp and fixed gaps) and writes *_anchored.csv next to them. Nothing is re-encoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_missingness_ramp as ramp  # noqa: E402
import run_sensitivity_db1_objetivo as base  # noqa: E402

ROOT = ramp.ROOT
RUN = base.OUT
GAPS = RUN / "fixed_gaps"
STORED = ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv"
THR = 1.0596384140662845
KEYS = ["id", "study_day", "y"]
# Columns of metrics_by_seed that depend on the load or the alarm.
ALARM_COLS = ["load_pearson", "load_mae", "alarm_agreement", "flip_off_to_on", "flip_on_to_off"]


def frozen_load() -> pd.DataFrame:
    st = pd.read_csv(STORED)
    st["study_day"] = st["decision_day"].astype(int)
    return st[["id", "study_day", "AUC_W"]]


def anchor(scored: pd.DataFrame, rerun_ref: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    m = scored.merge(rerun_ref[KEYS + ["load"]].rename(columns={"load": "load_rerun"}), on=KEYS, how="left", validate="one_to_one")
    m = m.merge(frozen, on=["id", "study_day"], how="left", validate="one_to_one")
    if m[["load_rerun", "AUC_W"]].isna().any().any():
        raise SystemExit("anchor: missing re-run or frozen load for some decisions")
    out = scored.copy()
    out["load"] = (m["AUC_W"] + (m["load"] - m["load_rerun"])).to_numpy()
    out["alarm"] = (out["load"] >= THR).astype(int)
    return out


def reanchor_table(metrics: pd.DataFrame, landmark_dir: Path, rerun_ref: pd.DataFrame, frozen: pd.DataFrame,
                   ref_anchored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in metrics.to_dict("records"):
        name = rec["scenario"]
        scored = pd.read_csv(landmark_dir / f"{name}.csv") if name != "reference" else rerun_ref
        anchored = anchor(scored, rerun_ref, frozen)
        # summarize needs the topic inputs only for columns we keep from the original row.
        dummy = np.ones(6)
        miss = pd.DataFrame({"day_missing_rate": [rec["mean_day_missing"]], "variable_missing_rate": [rec["mean_variable_missing"]]})
        s = ramp.summarize(name, rec["mode"], rec["added_rate"], rec["seed"], anchored, ref_anchored,
                           dummy, dummy, dummy, dummy, miss)
        new = dict(rec)
        for col in ALARM_COLS:
            new[col] = s[col]
        for col in s:
            if col.startswith("emitted_") or col.startswith("valid_"):
                new[col] = s[col]
        rows.append(new)
    return pd.DataFrame(rows)


def main() -> None:
    frozen = frozen_load()
    rerun_ref = pd.read_csv(RUN / "landmarks" / "reference.csv")
    ref_anchored = anchor(rerun_ref, rerun_ref, frozen)
    if not np.allclose(ref_anchored["load"].to_numpy(), rerun_ref[["id", "study_day"]].merge(frozen, how="left")["AUC_W"].to_numpy()):
        raise SystemExit("anchored reference is not the frozen load")

    ramp_metrics = pd.read_csv(RUN / "metrics_by_seed.csv")
    out = reanchor_table(ramp_metrics, RUN / "landmarks", rerun_ref, frozen, ref_anchored)
    out.to_csv(RUN / "metrics_by_seed_anchored.csv", index=False)
    gaps = pd.read_csv(GAPS / "gaps_3_7_14_metrics_by_seed.csv")
    gout = reanchor_table(gaps, GAPS / "landmarks", rerun_ref, frozen, ref_anchored)
    gout.to_csv(GAPS / "gaps_3_7_14_metrics_by_seed_anchored.csv", index=False)

    b = out[out.scenario == "reference"].iloc[0]
    print("baseline (anchored):", {k: round(b["emitted_" + k], 4) for k in ("AUROC", "sensitivity", "specificity", "PPV", "NPV")},
          "alarm agreement", b["alarm_agreement"])


if __name__ == "__main__":
    main()
