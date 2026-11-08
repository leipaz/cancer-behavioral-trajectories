#!/usr/bin/env python3
"""Favourable/unfavourable labels for the 25 nested VAL-2 models.

Each topic is ranked by the Kaplan-Meier PFS at 365 days of the patients who
trained that model. The three highest are favourable and the three lowest are
unfavourable. Train rows label an inner model. Refit rows label a refit.
Held-out and external patients do not enter the curves.

Time is Tiempo_obs_DB1, counted from the treatment start. A patient with fewer
than 28 days of that clinical observation does not enter the curve. The alarm
keeps Tiempo_obs_digital. An exact tie between the third and fourth topic is
broken by the restricted mean survival time at 365 days. A model that still
cannot be split is left unlabelled.

The block score then uses the map of the model that produced the probabilities.
The decision day is month times 30. The cut and the metrics use the 1013
(id, month) pairs of the every-30-day series in decision_points.csv that are
included in the main alarm. The other decision rows stay in the file and do
not enter the cut. The time to progression stays Tiempo_obs_digital.
One Youden cut per outer fold is chosen on the held-out decisions and kept for
the 35 external patients of the refit.

    python scripts/03_analysis/lda/run_val2_fu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_SCRIPT_DIR = Path(__file__).resolve().parent
_ALARM_DIR = _SCRIPT_DIR.parent / "alarm"
for _path in (_SCRIPT_DIR, _ALARM_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from run_val2_nested import AUDIT, N_TOPICS, OUT_DIR  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DECISION_POINTS = _SCRIPT_DIR.parents[2] / "results/alarm/base_analysis/decision_points.csv"

TAU = 365.0
MIN_CLINICAL_DAYS = 28.0
Z95 = 1.959963984540054
FU_DIR = OUT_DIR / "fu"
ALARM_DIR = OUT_DIR / "alarma"
CURVE_DIR = FU_DIR / "curvas"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def _check_rmst() -> None:
    one = km_rows(np.array([100.0]), np.array([1]))
    assert abs(rmst(one, TAU) - 100.0) < 1e-9
    assert abs(survival_at(one, TAU)[0] - 0.0) < 1e-12
    two = km_rows(np.array([100.0, 400.0]), np.array([1, 0]))
    assert abs(survival_at(two, TAU)[0] - 0.5) < 1e-12
    assert abs(rmst(two, TAU) - 232.5) < 1e-9


def km_rows(times: np.ndarray, events: np.ndarray) -> list[tuple[float, int, int, float, float]]:
    """Event-time rows. Survival after each event time is P(T > t)."""
    order = np.argsort(times, kind="mergesort")
    t = np.asarray(times, dtype=float)[order]
    e = np.asarray(events, dtype=int)[order]
    rows = []
    i = 0
    n_left = len(t)
    survival = 1.0
    while i < len(t):
        ti = float(t[i])
        j = i
        deaths = 0
        while j < len(t) and t[j] == ti:
            deaths += int(e[j] == 1)
            j += 1
        n_risk = n_left
        if deaths:
            if n_risk == deaths:
                variance_term = np.inf
            else:
                variance_term = deaths / (n_risk * (n_risk - deaths))
            survival *= 1.0 - deaths / n_risk
            rows.append((ti, n_risk, deaths, survival, variance_term))
        n_left -= j - i
        i = j
    return rows


def survival_at(rows, tau: float) -> tuple[float, float, bool]:
    survival = 1.0
    greenwood = 0.0
    defined = True
    for ti, _n_risk, _deaths, survival_after, variance_term in rows:
        if ti > tau:
            break
        survival = survival_after
        if not np.isfinite(variance_term):
            defined = False
        else:
            greenwood += variance_term
    return survival, greenwood, defined


def rmst(rows, tau: float) -> float:
    survival = 1.0
    previous = 0.0
    area = 0.0
    for ti, _n_risk, _deaths, survival_after, _variance_term in rows:
        if ti >= tau:
            break
        area += survival * (ti - previous)
        survival = survival_after
        previous = ti
    area += survival * (tau - previous)
    return float(area)


def interval_at(survival: float, greenwood: float, defined: bool) -> tuple[float, float]:
    if (not defined) or survival <= 0.0 or survival >= 1.0:
        if survival in (0.0, 1.0) and greenwood == 0.0:
            return survival, survival
        return np.nan, np.nan
    theta = np.log(-np.log(survival))
    se = np.sqrt(greenwood) / abs(np.log(survival))
    low = float(np.exp(-np.exp(theta + Z95 * se)))
    high = float(np.exp(-np.exp(theta - Z95 * se)))
    return low, high


def median_pfs(rows) -> float:
    for ti, _n_risk, _deaths, survival_after, _variance_term in rows:
        if survival_after <= 0.5:
            return float(ti)
    return np.nan


def topic_summary(frame: pd.DataFrame, topic: int) -> dict:
    sub = frame[frame["topic"] == topic]
    times = sub["t_pfs"].to_numpy(float)
    events = sub["event"].to_numpy(int)
    n = int(len(sub))
    n_events = int(events.sum()) if n else 0
    if n == 0 or not np.isfinite(times).all():
        return {
            "topic": topic,
            "n": n,
            "events": n_events,
            "pfs_365": np.nan,
            "pfs_365_plateau": np.nan,
            "pfs_365_low": np.nan,
            "pfs_365_high": np.nan,
            "n_risk_365": 0,
            "max_time": np.nan,
            "supported_365": False,
            "median_pfs": np.nan,
            "rmst_365": np.nan,
            "estimable": False,
        }
    rows = km_rows(times, events)
    survival, greenwood, defined = survival_at(rows, TAU)
    low, high = interval_at(survival, greenwood, defined)
    n_risk = int((times >= TAU).sum())
    # A positive value past the last observation is the last step carried forward,
    # not a one-year Kaplan-Meier estimate. Reaching 0 at an observed event is.
    supported = n_risk > 0 or survival == 0.0
    return {
        "topic": topic,
        "n": n,
        "events": n_events,
        "pfs_365": survival if supported else np.nan,
        "pfs_365_plateau": survival,
        "pfs_365_low": low if supported else np.nan,
        "pfs_365_high": high if supported else np.nan,
        "n_risk_365": n_risk,
        "max_time": float(np.max(times)),
        "supported_365": supported,
        "median_pfs": median_pfs(rows),
        "rmst_365": rmst(rows, TAU),
        "estimable": bool(supported and np.isfinite(survival)),
        "rows": rows,
        "times": times,
        "events_arr": events,
    }


def label_by_pfs(summaries: list[dict]) -> tuple[dict[int, str] | None, str]:
    ranked = sorted(summaries, key=lambda item: (-round(item["pfs_365"], 10), item["topic"]))
    boundary = round(ranked[2]["pfs_365"], 10)
    if round(ranked[3]["pfs_365"], 10) != boundary:
        return {item["topic"]: ("F" if i < 3 else "U") for i, item in enumerate(ranked)}, "pfs_365"
    above = [item for item in ranked if round(item["pfs_365"], 10) > boundary]
    tied = [item for item in ranked if round(item["pfs_365"], 10) == boundary]
    below = [item for item in ranked if round(item["pfs_365"], 10) < boundary]
    slots = 3 - len(above)
    tied = sorted(tied, key=lambda item: (-round(item["rmst_365"], 10), item["topic"]))
    if round(tied[slots - 1]["rmst_365"], 10) == round(tied[slots]["rmst_365"], 10):
        return None, "empate_rmst"
    labels = {item["topic"]: "F" for item in above}
    labels.update({item["topic"]: "U" for item in below})
    for i, item in enumerate(tied):
        labels[item["topic"]] = "F" if i < slots else "U"
    return labels, "rmst_365"


def assign_labels(summaries: list[dict]) -> tuple[dict[int, str] | None, str]:
    """Top 3 favourable, bottom 3 unfavourable.

    A topic whose follow-up ends before day 365 with the curve still above 0 has
    no one-year estimate. That last step is a ceiling: the curve can stay or
    fall, and it cannot rise. When the third estimable topic is strictly above
    that ceiling, the topic is unfavourable and the three topics above it are
    favourable. Two such topics, or a ceiling that still reaches the third,
    leave the model unlabelled.
    """
    if any(item.get("n", 0) == 0 for item in summaries):
        return None, "no estimable"
    unsupported = [item for item in summaries if not item.get("supported_365", False)]
    if len(unsupported) > 1:
        return None, "sin soporte a 365"
    if len(unsupported) == 1:
        known = [item for item in summaries if item["estimable"]]
        ceiling = unsupported[0].get("pfs_365_plateau")
        if len(known) != 5 or ceiling is None or not np.isfinite(ceiling):
            return None, "sin soporte a 365"
        labels, rule = label_by_pfs(known)
        if labels is None:
            return None, rule
        third = min(item["pfs_365"] for item in known if labels[item["topic"]] == "F")
        if round(third, 10) <= round(float(ceiling), 10):
            return None, "sin soporte a 365"
        labels[unsupported[0]["topic"]] = "U"
        return labels, "techo_365"
    if any(not item["estimable"] for item in summaries):
        return None, "no estimable"
    return label_by_pfs(summaries)


def curve_points(summary: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = summary["rows"]
    last = float(np.max(summary["times"])) if len(summary["times"]) else TAU
    xs = [0.0]
    ys = [1.0]
    for ti, _n_risk, _deaths, survival_after, _variance_term in rows:
        xs.append(ti)
        ys.append(survival_after)
    if not xs or xs[-1] < last:
        xs.append(last)
        ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def draw_model(model_name: str, title: str, summaries: list[dict], labels: dict[int, str] | None, pdf) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for topic, summary in enumerate(summaries):
        if summary["n"] == 0:
            continue
        xs, ys = curve_points(summary)
        label = "sin etiqueta" if labels is None else labels[topic]
        ax.step(
            xs,
            ys,
            where="post",
            color=COLORS[topic],
            linewidth=1.8,
            label=f"Topic {topic} · {label} · n={summary['n']}",
        )
    ax.axvline(TAU, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, max(TAU, ax.get_xlim()[1]))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Días desde el inicio del tratamiento")
    ax.set_ylabel("Supervivencia libre de progresión")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(CURVE_DIR / f"{model_name}.png", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def draw_grouped(title: str, frame: pd.DataFrame, labels: dict[int, str], pdf) -> None:
    """Two training curves: patients pooled into favourable and unfavourable topics."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = {"F": "#0072B2", "U": "#D55E00"}
    names = {"F": "Favorable", "U": "Desfavorable"}
    last_day = TAU
    for etiqueta in ("F", "U"):
        topics = sorted(topic for topic, label in labels.items() if label == etiqueta)
        sub = frame[frame["topic"].isin(topics)]
        times = sub["t_pfs"].to_numpy(float)
        events = sub["event"].to_numpy(int)
        rows = km_rows(times, events)
        survival, greenwood, defined = survival_at(rows, TAU)
        summary = {"rows": rows, "times": times}
        xs, ys = curve_points(summary)
        last_day = max(last_day, float(xs[-1]) if len(xs) else TAU)
        topic_list = ", ".join(str(topic) for topic in topics)
        ax.step(
            xs,
            ys,
            where="post",
            color=colors[etiqueta],
            linewidth=2.0,
            label=(
                f"{names[etiqueta]} · topics {topic_list} · n={len(sub)} · "
                f"{int(events.sum())} eventos · PFS a 365 días {survival:.2f}".replace(".", ",")
            ),
        )
    ax.axvline(TAU, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, last_day)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Días desde el inicio del tratamiento")
    ax.set_ylabel("Supervivencia libre de progresión")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def comparison_blocks() -> set[tuple[int, int]]:
    """(id, month) pairs of the main alarm's every-30-day evaluable points."""
    points = pd.read_csv(DECISION_POINTS)
    monthly = points[(points["cadence"] == "Every 30 days") & (points["included_in_metrics"] == 1)]
    pairs = set(zip(monthly["id"].astype(int), (monthly["study_day"] // 30).astype(int)))
    if len(pairs) != 1013:
        raise SystemExit(f"expected 1013 comparison blocks, found {len(pairs)}")
    return pairs


def mark_comparison(decision_df: pd.DataFrame, pairs: set[tuple[int, int]]) -> pd.DataFrame:
    marked = decision_df.copy()
    marked["in_comparison"] = [
        int((int(pid), int(block)) in pairs)
        for pid, block in zip(marked["id"], marked["block"])
    ]
    return marked


def drop_outside_comparison(decision_df: pd.DataFrame) -> pd.DataFrame:
    """Drop evaluable decisions that are not one of the 1013 landmarks."""
    outside = decision_df["outcome"].isin(["0", "1"]) & decision_df["in_comparison"].eq(0)
    return decision_df.loc[~outside].copy()


def cut_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["in_comparison"].eq(1)
        & frame["outcome"].isin(["0", "1"])
        & np.isfinite(frame["score"])
    ].copy()


def rendimiento(decision_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    usable = cut_rows(decision_df)
    usable["y"] = usable["outcome"].astype(int)
    usable["hit"] = usable["alarm"].astype(bool)
    rows = []
    for key, frame in usable.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        y = frame["y"].to_numpy()
        alarm = frame["hit"].to_numpy()
        tp = int(((y == 1) & alarm).sum())
        fn = int(((y == 1) & ~alarm).sum())
        fp = int(((y == 0) & alarm).sum())
        tn = int(((y == 0) & ~alarm).sum())
        patients = []
        for _, group in frame.groupby("id"):
            patients.append({
                "event": int(group["event"].iloc[0]),
                "detected": bool(((group["y"] == 1) & group["hit"]).any()),
                "false_alarm": bool(((group["y"] == 0) & group["hit"]).any()),
            })
        patient_df = pd.DataFrame(patients)
        row = {name: value for name, value in zip(group_cols, key)}
        row.update({
            "n_decisions": int(len(frame)),
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "AUROC": float(roc_auc_score(y, frame["score"])) if len(np.unique(y)) > 1 else np.nan,
            "patients_with_event": int((patient_df["event"] == 1).sum()),
            "patients_detected": int(((patient_df["event"] == 1) & patient_df["detected"]).sum()),
            "patients_false_alarm": int(((patient_df["event"] == 0) & patient_df["false_alarm"]).sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def youden_cut(scores: np.ndarray, outcomes: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    x = scores[order]
    y = outcomes[order]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    last = np.r_[np.flatnonzero(np.diff(x)), len(x) - 1]
    tp = np.r_[0, np.cumsum(y)[last]]
    fp = np.r_[0, last + 1 - np.cumsum(y)[last]]
    threshold = np.r_[np.inf, x[last]]
    j_num = tp * n_neg - fp * n_pos
    tied = np.flatnonzero(j_num == j_num.max())
    theta = float(np.min(threshold[tied]))
    sens = float(tp[tied][np.argmin(threshold[tied])] / n_pos)
    spec = float(1.0 - fp[tied][np.argmin(threshold[tied])] / n_neg)
    return {
        "threshold": theta,
        "n_tied": int(len(tied)),
        "n_decisions": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "sensitivity": sens,
        "specificity": spec,
        "youden": sens + spec - 1.0,
    }


def main() -> None:
    global OUT_DIR, FU_DIR, ALARM_DIR, CURVE_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is not None:
        OUT_DIR = args.out
        ALARM_DIR = OUT_DIR / "alarma"
        # The stratified folder keeps the shared LDA at the root and the
        # three-and-three labels with that variant's alarm.
        FU_DIR = ALARM_DIR / "fu" if OUT_DIR.name == "anidada_estratificada" else OUT_DIR / "fu"
        CURVE_DIR = FU_DIR / "curvas"
    _check_rmst()
    FU_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    ALARM_DIR.mkdir(parents=True, exist_ok=True)

    topics = pd.read_csv(OUT_DIR / "topic_probabilities.csv")
    topics["inner_fold"] = topics["inner_fold"].astype(str)
    training = topics[topics["role"].isin(["train", "refit"])].copy()
    audit = pd.read_excel(AUDIT).dropna(subset=["id"]).copy()
    audit["id"] = audit["id"].astype(int)
    clinical = audit.set_index("id")[["Evento_DB1", "Tiempo_obs_DB1"]]
    training = training.join(clinical, on="id")
    if training["Tiempo_obs_DB1"].isna().any() or training["Evento_DB1"].isna().any():
        missing = training.loc[training["Tiempo_obs_DB1"].isna() | training["Evento_DB1"].isna(), "id"].unique()
        raise SystemExit(f"Falta Tiempo_obs_DB1 o el evento en {missing.tolist()}")
    training["t_pfs"] = training["Tiempo_obs_DB1"].astype(float)
    training["event"] = training["Evento_DB1"].astype(int)
    short = training["t_pfs"] < MIN_CLINICAL_DAYS
    print(f"excluded from F/U curves, clinical time < {MIN_CLINICAL_DAYS:.0f} days: {training.loc[short, 'id'].nunique()} patients")
    training = training.loc[~short].copy()

    table_rows = []
    maps = {}
    reserved = []
    pdf_path = CURVE_DIR / "curvas_entrenamiento.pdf"
    grouped_path = CURVE_DIR / "supervivencia_por_fu.pdf"
    with PdfPages(pdf_path) as pdf, PdfPages(grouped_path) as grouped_pdf:
        for (outer_fold, inner_fold), frame in training.groupby(["outer_fold", "inner_fold"], sort=True):
            model_name = f"outer_{int(outer_fold)}_{inner_fold if inner_fold == 'refit' else 'inner_' + inner_fold}"
            summaries = [topic_summary(frame, topic) for topic in range(N_TOPICS)]
            labels, rule = assign_labels(summaries)
            if labels is None:
                reserved.append({"model": model_name, "outer_fold": int(outer_fold), "inner_fold": inner_fold, "reason": rule})
            else:
                maps[(int(outer_fold), inner_fold)] = labels
            title_bit = "reajuste, 140" if inner_fold == "refit" else f"interno {inner_fold}, {len(frame)} train"
            draw_model(
                model_name,
                f"Vuelta externa {int(outer_fold)} · {title_bit} · solo entrenamiento",
                summaries,
                labels,
                pdf,
            )
            if labels is not None:
                draw_grouped(
                    f"Vuelta externa {int(outer_fold)} · {title_bit} · topics agrupados",
                    frame,
                    labels,
                    grouped_pdf,
                )
            for summary in summaries:
                table_rows.append({
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "inner_fold": inner_fold,
                    "n_train_model": int(len(frame)),
                    "topic": summary["topic"],
                    "n": summary["n"],
                    "events": summary["events"],
                    "pfs_365": summary["pfs_365"],
                    "pfs_365_plateau": summary["pfs_365_plateau"],
                    "pfs_365_low": summary["pfs_365_low"],
                    "pfs_365_high": summary["pfs_365_high"],
                    "n_risk_365": summary["n_risk_365"],
                    "max_time": summary["max_time"],
                    "supported_365": summary["supported_365"],
                    "median_pfs": summary["median_pfs"],
                    "rmst_365": summary["rmst_365"],
                    "etiqueta": "" if labels is None else labels[summary["topic"]],
                    "criterio": rule,
                    "time": "Tiempo_obs_DB1",
                })

    table = pd.DataFrame(table_rows)
    table_path = FU_DIR / "km_por_topic.csv"
    table.to_csv(table_path, index=False)
    pd.DataFrame(reserved, columns=["model", "outer_fold", "inner_fold", "reason"]).to_csv(
        FU_DIR / "modelos_reservados.csv", index=False
    )

    blocks = pd.read_csv(OUT_DIR / "block_probabilities.csv")
    blocks["inner_fold"] = blocks["inner_fold"].astype(str)
    blocks["map_fold"] = blocks["inner_fold"].replace({"external": "refit"})
    if reserved:
        blocked = {(int(item["outer_fold"]), str(item["inner_fold"])) for item in reserved}
    else:
        blocked = set()

    puf = []
    for row in blocks.itertuples(index=False):
        key = (int(row.outer_fold), row.map_fold)
        labels = maps.get(key)
        if labels is None:
            value = np.nan
        else:
            value = float(sum(getattr(row, f"topic_{topic}") for topic, label in labels.items() if label == "U"))
        puf.append(value)
    blocks["pUF"] = puf
    blocks.to_csv(ALARM_DIR / "bloques_puf.csv", index=False)

    clinical_full = audit.set_index("id")[["Evento_DB1", "Tiempo_obs_digital", "Fecha_Fin_DB1_Actualizado"]]
    decisions = []
    grouped = blocks.groupby(["id", "outer_fold", "inner_fold", "role", "model"], sort=True)
    for (pid, outer_fold, inner_fold, role, model_name), frame in grouped:
        frame = frame.sort_values("block")
        if len(frame) < 3:
            continue
        t_digital = float(clinical_full.loc[int(pid), "Tiempo_obs_digital"])
        event = int(clinical_full.loc[int(pid), "Evento_DB1"])
        values = frame["pUF"].to_numpy(float)
        block_no = frame["block"].to_numpy(int)
        for i in range(2, len(frame)):
            window = values[i - 2 : i + 1]
            day = float(block_no[i]) * 30.0
            outcome = label_landmark(day, t_digital, event, HORIZON_DAYS)
            decisions.append({
                "id": int(pid),
                "outer_fold": int(outer_fold),
                "inner_fold": inner_fold,
                "role": role,
                "model": model_name,
                "block": int(block_no[i]),
                "decision_day": day,
                "puf_1": float(window[0]),
                "puf_2": float(window[1]),
                "puf_3": float(window[2]),
                "score": float(np.sum(window)),
                "outcome": outcome,
                "event": event,
                "t_digital": t_digital,
                "fecha_fin": str(clinical_full.loc[int(pid), "Fecha_Fin_DB1_Actualizado"])[:10],
            })
    decision_df = drop_outside_comparison(mark_comparison(pd.DataFrame(decisions), comparison_blocks()))

    cuts = []
    threshold_of = {}
    for outer_fold, frame in decision_df[decision_df["role"] == "held_out"].groupby("outer_fold"):
        usable = cut_rows(frame)
        found = youden_cut(usable["score"].to_numpy(float), usable["outcome"].astype(int).to_numpy())
        if found is None:
            cuts.append({"outer_fold": int(outer_fold), "threshold": np.nan, "reason": "sin positivos o sin negativos"})
            continue
        found["outer_fold"] = int(outer_fold)
        found["tie_rule"] = "menor corte"
        found["time"] = "Tiempo_obs_digital"
        found["horizon_days"] = HORIZON_DAYS
        found["comparison"] = "1013 every 30 days"
        cuts.append(found)
        threshold_of[int(outer_fold)] = found["threshold"]
    cut_df = pd.DataFrame(cuts)
    cut_df.to_csv(ALARM_DIR / "cortes.csv", index=False)

    decision_df["threshold"] = decision_df["outer_fold"].map(threshold_of)
    decision_df["alarm"] = decision_df["score"] >= decision_df["threshold"]
    decision_df.loc[decision_df["threshold"].isna() | ~np.isfinite(decision_df["score"]), "alarm"] = pd.NA
    decision_df.to_csv(ALARM_DIR / "decisiones.csv", index=False)
    rendimiento(decision_df.dropna(subset=["threshold"]), ["outer_fold", "role"]).to_csv(
        ALARM_DIR / "rendimiento_por_vuelta.csv", index=False
    )

    manifest = {
        "time": "Tiempo_obs_DB1, desde el inicio del tratamiento",
        "min_clinical_days": MIN_CLINICAL_DAYS,
        "alarm_time": "Tiempo_obs_digital",
        "pfs_rule": "Kaplan-Meier at 365 days. A positive last step before day 365 is a ceiling: if the third estimable topic is above it, that topic is unfavourable and the three above it are favourable. Otherwise the model is reserved. Top 3 favourable, bottom 3 unfavourable",
        "tie_break": "restricted mean survival time at 365 days; otherwise the model is reserved",
        "interval": "95 percent log-log Greenwood",
        "training_only_curves": str(pdf_path),
        "n_models_labelled": len(maps),
        "n_models_reserved": len(reserved),
        "reserved": reserved,
        "score": "sum of three consecutive block pUF values",
        "cut": "one Youden cut per outer fold, held-out decisions among the 1013 every-30-day pairs, lowest cut if tied",
        "horizon_days": HORIZON_DAYS,
        "comparison_blocks": "1013 (id, month) pairs, decision_points.csv cadence Every 30 days, included_in_metrics 1",
        "landmark": "decision day is month x 30; event study day is Tiempo_obs_digital + 1",
        "blocked_models_not_in_curves_package": sorted(f"{a}:{b}" for a, b in blocked),
    }
    (FU_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"labelled {len(maps)} models, reserved {len(reserved)}")
    print(f"table {table_path}")
    print(cut_df.to_string(index=False))
    print(f"decisions {len(decision_df)} -> {ALARM_DIR / 'decisiones.csv'}")


if __name__ == "__main__":
    main()
