#!/usr/bin/env python3
"""Monthly rolling-burden alarm (manuscript definition).

``pUF(m)`` = P(topic 3) + P(topic 4) + P(topic 5) in month *m*
``AUC_W(m)`` = sum of ``pUF`` over the last *W* months (requires *W* complete months)
Alarm if ``AUC_W(m) ≥ θ``; target = progression within the next *H* months.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

UNFAVOURABLE_TOPICS = (3, 4, 5)
DAYS_PER_MONTH = 30.0
PAPER_THRESHOLD = 1.61


def project_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[3]


def default_master_table_path(root: Path | None = None) -> Path:
    """Tabla maestra = monthly topic probabilities (``topics_probs.xlsx``)."""
    root = project_root_from(root)
    return root / "data/processed/DCABPs/topics_probs.xlsx"


def default_clinical_path(root: Path | None = None) -> Path:
    root = project_root_from(root)
    return root / "data/processed/clinical/Subjects_data.xlsx"


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def extract_monthly_puf_from_master(
    master: pd.DataFrame,
    clinical: pd.DataFrame | None = None,
    unfavourable_topics: Iterable[int] = UNFAVOURABLE_TOPICS,
    root: Path | None = None,
) -> pd.DataFrame:
    """Build long monthly ``pUF`` table from ``topics_probs.xlsx`` (+ clinical FU)."""
    master = _strip_columns(master)
    if "id" not in master.columns:
        raise ValueError("Master table must contain an 'id' column")

    event_col = next(
        (c for c in master.columns if c.lower().startswith("event")),
        None,
    )
    if clinical is None:
        clin_path = default_clinical_path(root)
        if clin_path.exists():
            clinical = pd.read_excel(clin_path)

    if clinical is not None:
        clinical = _strip_columns(clinical)
        id_clin = "S_RECORD_id" if "S_RECORD_id" in clinical.columns else "id"
        keep = [id_clin]
        for c in ("PD_event", "Obs_time", "Evento PD", "t_evento_eb2"):
            if c in clinical.columns:
                keep.append(c)
        clinical = clinical[keep].drop_duplicates(id_clin)
        master = master.merge(
            clinical, left_on="id", right_on=id_clin, how="left", suffixes=("", "_clin")
        )

    if "PD_event" in master.columns:
        event = master["PD_event"].astype(int)
    elif event_col is not None:
        event = master[event_col].fillna(0).astype(int)
    else:
        raise ValueError("Could not find an event column (PD_event / Event eb2)")

    if "Obs_time" in master.columns:
        follow_up = master["Obs_time"].astype(float)
    elif "t_evento_eb2" in master.columns:
        follow_up = master["t_evento_eb2"].astype(float)
    else:
        raise ValueError("Need Obs_time or t_evento_eb2 for follow-up")

    t_event = master["t_evento_eb2"].astype(float) if "t_evento_eb2" in master.columns else follow_up.copy()
    t_event = np.where(event == 1, t_event.fillna(follow_up), follow_up).astype(float)

    months = sorted(
        {
            int(re.findall(r"mes(\d+)_", c)[0])
            for c in master.columns
            if re.match(r"mes\d+_topic[0-5]$", str(c))
        }
    )
    unf = list(unfavourable_topics)
    rows: list[dict] = []
    for i, row in master.iterrows():
        pid = int(row["id"])
        ev = int(event.loc[i])
        te = float(t_event[i])
        fu = float(follow_up.loc[i])
        for m in months:
            cols_all = [f"mes{m}_topic{k}" for k in range(6)]
            if any(c not in master.columns for c in cols_all):
                continue
            if not all(pd.notna(row[c]) for c in cols_all):
                continue
            puf = float(sum(row[f"mes{m}_topic{k}"] for k in unf))
            rows.append(
                {
                    "id": pid,
                    "month": m,
                    "decision_day": m * DAYS_PER_MONTH,
                    "pUF": puf,
                    "Evento PD": ev,
                    "t_evento_eb2": te,
                    "follow_up_days": fu,
                }
            )
    return pd.DataFrame(rows).sort_values(["id", "month"]).reset_index(drop=True)


def build_decision_points(
    monthly_puf: pd.DataFrame,
    W: int = 3,
    H: int = 4,
    days_per_month: float = DAYS_PER_MONTH,
) -> pd.DataFrame:
    """Landmark decision points with ``AUC_W`` = sum of last *W* monthly ``pUF``."""
    rows: list[dict] = []
    h_days = H * days_per_month
    for pid, g in monthly_puf.groupby("id"):
        g = g.sort_values("month")
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        fu = float(g["follow_up_days"].iloc[0]) if "follow_up_days" in g.columns else t_ev
        pufs = g["pUF"].to_numpy(dtype=float)
        months = g["month"].to_numpy(dtype=int)
        for i in range(W - 1, len(g)):
            window = pufs[i - W + 1 : i + 1]
            if np.isnan(window).any():
                continue
            month = int(months[i])
            t = month * days_per_month
            if not (t < t_ev):
                continue
            auc_w = float(window.sum())
            if event == 1:
                target = int(t_ev <= t + h_days)
            else:
                if fu < t + h_days:
                    continue
                target = 0
            rows.append(
                {
                    "id": int(pid),
                    "month": month,
                    "decision_day": t,
                    "W": W,
                    "H": H,
                    "AUC_W": auc_w,
                    "AUC_W_per_month": auc_w / W,
                    "pUF": float(pufs[i]),
                    "target": target,
                    "Evento PD": event,
                }
            )
    return pd.DataFrame(rows)


@dataclass
class AlarmMetrics:
    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    roc_auc: float | None = None


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def metrics_at_threshold(decision_points: pd.DataFrame, threshold: float, score_col: str = "AUC_W") -> AlarmMetrics:
    y = decision_points["target"].to_numpy(dtype=int)
    s = decision_points[score_col].to_numpy(dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return AlarmMetrics(
        threshold=float(threshold),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        sensitivity=_safe_div(tp, tp + fn),
        specificity=_safe_div(tn, tn + fp),
        ppv=_safe_div(tp, tp + fp),
        npv=_safe_div(tn, tn + fn),
        roc_auc=float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan"),
    )


def youden_optimal_threshold(decision_points: pd.DataFrame, score_col: str = "AUC_W") -> AlarmMetrics:
    y = decision_points["target"].to_numpy(dtype=int)
    s = decision_points[score_col].to_numpy(dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    # sklearn thr has length n_thresholds; last ROC point has no thr — align
    j = tpr - fpr
    # thr length is len(tpr)-1 typically when drop_intermediate=True... use paired
    if len(thr) == len(j) - 1:
        j = j[:-1]
    best = int(np.argmax(j))
    threshold = float(thr[best]) if len(thr) else float(np.median(s))
    return metrics_at_threshold(decision_points, threshold, score_col=score_col)


def sweep_thresholds(
    decision_points: pd.DataFrame,
    score_col: str = "AUC_W",
    n_quantiles: int = 50,
) -> pd.DataFrame:
    s = decision_points[score_col]
    qs = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    thresholds = sorted(set(float(x) for x in s.quantile(qs).tolist()))
    rows = []
    for th in thresholds:
        m = metrics_at_threshold(decision_points, th, score_col=score_col)
        rows.append(
            {
                "threshold": m.threshold,
                "sensitivity": m.sensitivity,
                "specificity": m.specificity,
                "PPV": m.ppv,
                "NPV": m.npv,
                "youden": m.sensitivity + m.specificity - 1,
                "TP": m.tp,
                "FN": m.fn,
                "FP": m.fp,
                "TN": m.tn,
            }
        )
    return pd.DataFrame(rows)
