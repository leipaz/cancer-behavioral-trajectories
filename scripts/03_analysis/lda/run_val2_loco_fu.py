#!/usr/bin/env python3
"""F/U labels, pUF and one Youden cut per left-out centre.

Uses the 45 saved centre models. Train rows label each inner model. Refit rows
label the refit. Held-out decisions choose the cut. The left-out centre receives
that cut and does not enter the label or the threshold.

The F/U curve uses Tiempo_obs_DB1, from the treatment start. A patient with
fewer than 28 days of that clinical observation does not enter it. The decision
day stays month times 30 and the alarm time stays Tiempo_obs_digital. The cut
and the metrics use the 1013 (id, month) pairs of the every-30-day series.

    python scripts/03_analysis/lda/run_val2_loco_fu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_SCRIPT_DIR = Path(__file__).resolve().parent
_ALARM_DIR = _SCRIPT_DIR.parent / "alarm"
for _path in (_SCRIPT_DIR, _ALARM_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_val2_fu as fu  # noqa: E402
from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from run_val2_fu import (  # noqa: E402
    N_TOPICS,
    assign_labels,
    comparison_blocks,
    cut_rows,
    drop_outside_comparison,
    draw_grouped,
    draw_model,
    mark_comparison,
    rendimiento,
    topic_summary,
    youden_cut,
)
from run_val2_loco import LOCO_DIR  # noqa: E402

FU_DIR = LOCO_DIR / "fu"
ALARM_DIR = LOCO_DIR / "alarma"
CURVE_DIR = FU_DIR / "curvas"
fu.CURVE_DIR = CURVE_DIR


def main() -> None:
    FU_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    ALARM_DIR.mkdir(parents=True, exist_ok=True)

    topics = pd.read_csv(LOCO_DIR / "topic_probabilities.csv")
    topics["inner_fold"] = topics["inner_fold"].astype(str)
    topics["t_pfs"] = topics["tiempo_obs"].astype(float)
    topics["event"] = topics["event"].astype(int)
    training = topics[topics["role"].isin(["train", "refit"])].copy()
    short = training["t_pfs"] < fu.MIN_CLINICAL_DAYS
    print(
        f"excluded from F/U curves, clinical time < {fu.MIN_CLINICAL_DAYS:.0f} days: "
        f"{training.loc[short, 'id'].nunique()} patients"
    )
    training = training.loc[~short].copy()

    table_rows = []
    maps = {}
    reserved = []
    pdf_path = CURVE_DIR / "curvas_entrenamiento.pdf"
    grouped_path = CURVE_DIR / "supervivencia_por_fu.pdf"
    with PdfPages(pdf_path) as pdf, PdfPages(grouped_path) as grouped_pdf:
        grouped = training.groupby(["left_out_centre", "inner_fold", "model"], sort=True)
        for (centre, inner_fold, model_name), frame in grouped:
            summaries = [topic_summary(frame, topic) for topic in range(N_TOPICS)]
            labels, rule = assign_labels(summaries)
            if labels is None:
                reserved.append({
                    "model": model_name,
                    "left_out_centre": int(centre),
                    "inner_fold": inner_fold,
                    "reason": rule,
                })
            else:
                maps[(int(centre), inner_fold)] = labels
            who = "reajuste" if inner_fold == "refit" else f"interno {inner_fold}"
            title = f"Centro {int(centre)} fuera · {who} · {len(frame)} entrenamiento"
            draw_model(model_name, title, summaries, labels, pdf)
            if labels is not None:
                draw_grouped(title, frame, labels, grouped_pdf)
            for summary in summaries:
                table_rows.append({
                    "model": model_name,
                    "left_out_centre": int(centre),
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
    table.to_csv(FU_DIR / "km_por_topic.csv", index=False)
    pd.DataFrame(reserved, columns=["model", "left_out_centre", "inner_fold", "reason"]).to_csv(
        FU_DIR / "modelos_reservados.csv", index=False
    )

    blocks = pd.read_csv(LOCO_DIR / "block_probabilities.csv")
    blocks["inner_fold"] = blocks["inner_fold"].astype(str)
    blocks["map_fold"] = blocks["inner_fold"].replace({"external": "refit"})
    puf = []
    for row in blocks.itertuples(index=False):
        labels = maps.get((int(row.left_out_centre), row.map_fold))
        if labels is None:
            puf.append(np.nan)
        else:
            puf.append(float(sum(getattr(row, f"topic_{topic}") for topic, label in labels.items() if label == "U")))
    blocks["pUF"] = puf
    blocks.to_csv(ALARM_DIR / "bloques_puf.csv", index=False)

    decisions = []
    grouped_blocks = blocks.groupby(
        ["id", "left_out_centre", "inner_fold", "role", "model"], sort=True
    )
    for (pid, centre, inner_fold, role, model_name), frame in grouped_blocks:
        frame = frame.sort_values("block")
        if len(frame) < 3:
            continue
        t_digital = float(frame["tiempo_obs_digital"].iloc[0])
        event = int(frame["event"].iloc[0])
        values = frame["pUF"].to_numpy(float)
        block_no = frame["block"].to_numpy(int)
        for i in range(2, len(frame)):
            window = values[i - 2 : i + 1]
            if np.isnan(window).any():
                continue
            day = float(block_no[i]) * 30.0
            decisions.append({
                "id": int(pid),
                "left_out_centre": int(centre),
                "inner_fold": inner_fold,
                "role": role,
                "model": model_name,
                "block": int(block_no[i]),
                "decision_day": day,
                "puf_1": float(window[0]),
                "puf_2": float(window[1]),
                "puf_3": float(window[2]),
                "score": float(np.sum(window)),
                "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
                "event": event,
                "t_digital": t_digital,
                "fecha_fin": str(frame["fecha_fin"].iloc[0])[:10],
            })
    decision_df = drop_outside_comparison(mark_comparison(pd.DataFrame(decisions), comparison_blocks()))

    cuts = []
    threshold_of = {}
    held = decision_df[decision_df["role"] == "held_out"]
    for centre, frame in held.groupby("left_out_centre"):
        usable = cut_rows(frame)
        found = youden_cut(usable["score"].to_numpy(float), usable["outcome"].astype(int).to_numpy())
        if found is None:
            cuts.append({
                "left_out_centre": int(centre),
                "threshold": np.nan,
                "reason": "sin positivos o sin negativos",
            })
            continue
        found["left_out_centre"] = int(centre)
        found["tie_rule"] = "menor corte"
        found["time"] = "Tiempo_obs_digital"
        found["horizon_days"] = HORIZON_DAYS
        found["comparison"] = "1013 every 30 days"
        cuts.append(found)
        threshold_of[int(centre)] = found["threshold"]
    cut_df = pd.DataFrame(cuts)
    cut_df.to_csv(ALARM_DIR / "cortes.csv", index=False)
    decision_df["threshold"] = decision_df["left_out_centre"].map(threshold_of)
    decision_df["alarm"] = decision_df["score"] >= decision_df["threshold"]
    decision_df.loc[decision_df["threshold"].isna() | ~np.isfinite(decision_df["score"]), "alarm"] = pd.NA
    decision_df.to_csv(ALARM_DIR / "decisiones.csv", index=False)
    metrics = rendimiento(decision_df.dropna(subset=["threshold"]), ["left_out_centre", "role"])
    metrics.to_csv(ALARM_DIR / "rendimiento_por_vuelta.csv", index=False)

    labelled = table[table["etiqueta"].isin(["F", "U"])]
    fav = labelled[labelled["etiqueta"] == "F"].groupby("model")["topic"].apply(
        lambda values: tuple(sorted(int(v) for v in values))
    )
    manifest = {
        "time": "Tiempo_obs_DB1, desde el inicio del tratamiento; pacientes con menos de 28 dias no entran en la curva",
        "alarm_time": "Tiempo_obs_digital",
        "pfs_rule": "Kaplan-Meier at 365 days. A positive last step before day 365 is a ceiling: if the third estimable topic is above it, that topic is unfavourable and the three above it are favourable. Otherwise the model is reserved",
        "tie_break": "restricted mean survival time at 365 days; otherwise the model is reserved",
        "n_models_labelled": len(maps),
        "n_models_reserved": len(reserved),
        "reserved": reserved,
        "criterio_counts": table.groupby("model")["criterio"].first().value_counts().to_dict(),
        "models_with_F_012": int((fav == (0, 1, 2)).sum()),
        "score": "sum of three consecutive block pUF values",
        "cut": "one Youden cut per left-out centre, held-out decisions among the 1013 every-30-day pairs, lowest cut if tied",
        "comparison_blocks": "1013 (id, month) pairs, decision_points.csv cadence Every 30 days, included_in_metrics 1",
        "landmark": "decision day is month x 30; event study day is Tiempo_obs_digital + 1",
        "curves": str(pdf_path),
        "grouped_curves": str(grouped_path),
    }
    (FU_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    loco_manifest = json.loads((LOCO_DIR / "manifest.json").read_text())
    loco_manifest["fu_and_cut"] = "computed; see fu/ and alarma/"
    (LOCO_DIR / "manifest.json").write_text(json.dumps(loco_manifest, indent=2, ensure_ascii=False))

    print(f"labelled {len(maps)} reserved {len(reserved)}")
    if reserved:
        print(pd.DataFrame(reserved).to_string(index=False))
    print(cut_df.to_string(index=False))
    print(f"decisions {len(decision_df)}")
    print(f"F is 0-1-2 in {manifest['models_with_F_012']} of {len(fav)} models")


if __name__ == "__main__":
    main()
