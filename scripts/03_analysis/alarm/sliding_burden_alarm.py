#!/usr/bin/env python3
"""Sliding (simple) and collapsed burden alarms at multiple granularities.

Two constructions of ``pUF``:

1. **Simple / sliding** — for each study day ``d ≥ 30``, run LDA on the previous
   30 days → ``pUF(d)``. Evaluation can then use every day, or subsample every
   7 / 14 / 30 days (``weekly`` / ``biweekly`` / ``monthly_sample``).
2. **Collapsed** — non-overlapping blocks of 7 / 14 / 30 days; one LDA topic
   mixture (or predominant-topic proxy) per block → ``pUF(period)``.

Lookback **W** and horizon **H** are in days. For collapsed series they are
converted to an integer number of blocks via ``round(days / block_size)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from rolling_burden_alarm import (
    DAYS_PER_MONTH,
    metrics_at_threshold,
)

GRANULARITY_STEPS = {
    "daily": 1,
    "weekly": 7,
    "biweekly": 14,
    "monthly_sample": 30,
}

COLLAPSED_BLOCK_DAYS = {
    "weekly_collapsed": 7,
    "biweekly_collapsed": 14,
    "monthly_collapsed": 30,
}


def add_rolling_burden(
    daily_puf: pd.DataFrame,
    w_load_days: int,
    min_periods: int = 1,
    score: str = "mean",
) -> pd.DataFrame:
    """Attach ``burden_score`` = rolling mean/sum of daily ``pUF`` over *W* days."""
    parts: list[pd.DataFrame] = []
    for _, g in daily_puf.groupby("id"):
        g = g.sort_values("study_day").copy()
        roll = g["pUF"].rolling(w_load_days, min_periods=min_periods)
        g["burden_score"] = roll.mean() if score == "mean" else roll.sum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _label_at_landmark(
    event: int,
    t_ev: float,
    fu: float,
    t: float,
    h_horizon_days: float,
) -> int | None:
    """Return target 0/1, or ``None`` if the landmark is not adjudicable."""
    if not (t < t_ev):
        return None
    if event == 1:
        return int(t_ev <= t + h_horizon_days)
    if fu < t + h_horizon_days:
        return None
    return 0


def build_sliding_decision_points(
    daily_puf: pd.DataFrame,
    granularity: str = "daily",
    w_load_days: int = 90,
    h_horizon_days: int = 120,
    sample_base_day: int = 30,
    score: str = "mean",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Decision points from daily sliding ``pUF`` at a chosen evaluation step."""
    if granularity not in GRANULARITY_STEPS:
        raise ValueError(f"Unknown granularity {granularity!r}; expected {list(GRANULARITY_STEPS)}")
    step = GRANULARITY_STEPS[granularity]
    scored = add_rolling_burden(daily_puf, w_load_days, min_periods=min_periods, score=score)

    rows: list[dict] = []
    for pid, g in scored.groupby("id"):
        g = g.sort_values("study_day")
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        fu = float(g["follow_up_days"].iloc[0]) if "follow_up_days" in g.columns else t_ev
        days = g["study_day"].to_numpy(dtype=int)
        scores = g["burden_score"].to_numpy(dtype=float)
        keep = ((days - sample_base_day) % step == 0) & ~np.isnan(scores)
        for day, burden in zip(days[keep], scores[keep]):
            target = _label_at_landmark(event, t_ev, fu, float(day), h_horizon_days)
            if target is None:
                continue
            rows.append(
                {
                    "id": int(pid),
                    "study_day": int(day),
                    "decision_day": float(day),
                    "granularity": granularity,
                    "w_load_days": w_load_days,
                    "h_horizon_days": h_horizon_days,
                    "burden_score": float(burden),
                    "target": int(target),
                    "Evento PD": event,
                }
            )
    return pd.DataFrame(rows)


def build_collapsed_decision_points(
    collapsed_puf: pd.DataFrame,
    block_size_days: int,
    w_load_days: int = 90,
    h_horizon_days: int = 120,
    score: str = "mean",
    granularity: str | None = None,
) -> pd.DataFrame:
    """Decision points from non-overlapping block ``pUF`` series."""
    w_periods = max(1, int(round(w_load_days / block_size_days)))
    h_periods = max(1, int(round(h_horizon_days / block_size_days)))
    h_days = h_periods * block_size_days
    name = granularity or f"collapsed_{block_size_days}d"

    rows: list[dict] = []
    for pid, g in collapsed_puf.groupby("id"):
        g = g.sort_values("period")
        event = int(g["Evento PD"].iloc[0])
        t_ev = float(g["t_evento_eb2"].iloc[0])
        fu = float(g["follow_up_days"].iloc[0]) if "follow_up_days" in g.columns else t_ev
        vals = g["pUF"].to_numpy(dtype=float)
        days = g["decision_day"].to_numpy(dtype=float)
        periods = g["period"].to_numpy(dtype=int)
        for i in range(w_periods - 1, len(g)):
            window = vals[i - w_periods + 1 : i + 1]
            if np.isnan(window).any():
                continue
            t = float(days[i])
            target = _label_at_landmark(event, t_ev, fu, t, h_days)
            if target is None:
                continue
            burden = float(window.mean() if score == "mean" else window.sum())
            rows.append(
                {
                    "id": int(pid),
                    "period": int(periods[i]),
                    "decision_day": t,
                    "granularity": name,
                    "block_size_days": block_size_days,
                    "w_periods": w_periods,
                    "h_periods": h_periods,
                    "w_load_days": w_load_days,
                    "h_horizon_days": h_horizon_days,
                    "burden_score": burden,
                    "target": int(target),
                    "Evento PD": event,
                }
            )
    return pd.DataFrame(rows)


def build_monthly_collapsed_decision_points(
    monthly_puf: pd.DataFrame,
    w_months: int = 3,
    h_months: int = 4,
) -> pd.DataFrame:
    """Paper monthly collapsed landmarks; ``burden_score`` = ``AUC_W`` (sum)."""
    from rolling_burden_alarm import build_decision_points

    dp = build_decision_points(monthly_puf, W=w_months, H=h_months)
    out = dp.rename(columns={"AUC_W": "burden_score"}).copy()
    out["granularity"] = "monthly_collapsed"
    out["w_load_days"] = int(w_months * DAYS_PER_MONTH)
    out["h_horizon_days"] = int(h_months * DAYS_PER_MONTH)
    return out


def summarize_decision_points(
    decision_points: pd.DataFrame,
    score_col: str = "burden_score",
) -> dict:
    y = decision_points["target"].to_numpy(dtype=int)
    s = decision_points[score_col].to_numpy(dtype=float)
    auc = float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    if len(thr) == len(j) - 1:
        j_use, thr_use = j[:-1], thr
    else:
        j_use, thr_use = j, thr
    best = int(np.argmax(j_use)) if len(j_use) else 0
    threshold = float(thr_use[best]) if len(thr_use) else float(np.median(s))
    m = metrics_at_threshold(decision_points, threshold, score_col=score_col)
    return {
        "n_decision_points": int(len(decision_points)),
        "n_positive_targets": int(y.sum()),
        "roc_auc": auc,
        "youden_threshold": m.threshold,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "PPV": m.ppv,
        "NPV": m.npv,
        "TP": m.tp,
        "FN": m.fn,
        "FP": m.fp,
        "TN": m.tn,
    }


def metrics_row_from_dp(
    granularity: str,
    w_load_days: int,
    h_horizon_days: int,
    decision_points: pd.DataFrame,
) -> dict:
    base = summarize_decision_points(decision_points)
    return {
        "granularity": granularity,
        "w_load_days": w_load_days,
        "h_horizon_days": h_horizon_days,
        "w_months_equiv": w_load_days / DAYS_PER_MONTH,
        "h_months_equiv": h_horizon_days / DAYS_PER_MONTH,
        **base,
    }
