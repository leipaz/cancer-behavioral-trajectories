#!/usr/bin/env python3
"""Clinical F/U adjudication for the stratified nested models.

Keeps the 3/3 alarm untouched. Writes a second alarm under alarma_clinica
from the 25 maps in alarma_clinica/mapeos_clinicos_25_modelos.txt. pUF is the sum of the
unfavourable topic probabilities, with no extra scaling. The Youden cut is
chosen again on the held-out series and is not copied from the 3/3 variant.
External patients use the refit map and that frozen cut. Decisions are the
1013 every-30-day landmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_ALARM_DIR = _SCRIPT_DIR.parent / "alarm"
for _path in (_SCRIPT_DIR, _ALARM_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from digital_followup import HORIZON_DAYS, label_landmark  # noqa: E402
from run_val2_anticipation import anticipation  # noqa: E402
from run_val2_fu import (  # noqa: E402
    comparison_blocks,
    cut_rows,
    drop_outside_comparison,
    mark_comparison,
    rendimiento,
    youden_cut,
)
from run_val2_intervals import intervals_for  # noqa: E402
from run_val2_nested import AUDIT  # noqa: E402

ROOT = _SCRIPT_DIR.parents[2]
OUT = ROOT / "results/lda/val_2/anidada_estratificada"
MAPS_PATH = OUT / "alarma_clinica/mapeos_clinicos_25_modelos.txt"
ALARM = OUT / "alarma_clinica"
SHORT = {
    ("1", "1", 4),
    ("2", "3", 3),
    ("4", "1", 4),
    ("4", "2", 3),
}


def parse_maps(path: Path) -> dict[tuple[int, str], dict[int, str]]:
    maps = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("outer_"):
            continue
        model = parts[0]
        favourable = {int(token[1:]) for token in parts[1].split(",") if token.startswith("T")}
        unfavourable = {int(token[1:]) for token in parts[2].split(",") if token.startswith("T")}
        if favourable | unfavourable != set(range(6)) or favourable & unfavourable:
            raise SystemExit(f"mapa incompleto: {model}")
        outer, kind = model.split("_", 1)[1].split("_", 1)
        fold = "refit" if kind == "refit" else kind.split("_")[1]
        maps[(int(outer), fold)] = {topic: ("F" if topic in favourable else "U") for topic in range(6)}
    if len(maps) != 25:
        raise SystemExit(f"esperaba 25 mapas, hay {len(maps)}")
    return maps


def main() -> None:
    maps = parse_maps(MAPS_PATH)
    blocks = pd.read_csv(OUT / "block_probabilities.csv")
    blocks["inner_fold"] = blocks["inner_fold"].astype(str)
    blocks["map_fold"] = blocks["inner_fold"].replace({"external": "refit"})
    puf = []
    n_u = []
    for row in blocks.itertuples(index=False):
        labels = maps[(int(row.outer_fold), row.map_fold)]
        unfavourable = [topic for topic, label in labels.items() if label == "U"]
        n_u.append(len(unfavourable))
        puf.append(float(sum(getattr(row, f"topic_{topic}") for topic in unfavourable)))
    blocks["pUF"] = puf
    blocks["n_U"] = n_u

    audit = pd.read_excel(AUDIT).dropna(subset=["id"])
    audit["id"] = audit["id"].astype(int)
    clinical = audit.set_index("id")[["Evento_DB1", "Tiempo_obs_digital", "Fecha_Fin_DB1_Actualizado"]]
    decisions = []
    grouped = blocks.groupby(["id", "outer_fold", "inner_fold", "role", "model"], sort=True)
    for (pid, outer_fold, inner_fold, role, model_name), frame in grouped:
        frame = frame.sort_values("block")
        if len(frame) < 3:
            continue
        t_digital = float(clinical.loc[int(pid), "Tiempo_obs_digital"])
        event = int(clinical.loc[int(pid), "Evento_DB1"])
        values = frame["pUF"].to_numpy(float)
        block_no = frame["block"].to_numpy(int)
        n_unfavourable = int(frame["n_U"].iloc[0])
        for index in range(2, len(frame)):
            window = values[index - 2 : index + 1]
            day = float(block_no[index]) * 30.0
            decisions.append({
                "id": int(pid),
                "outer_fold": int(outer_fold),
                "inner_fold": inner_fold,
                "role": role,
                "model": model_name,
                "block": int(block_no[index]),
                "decision_day": day,
                "puf_1": float(window[0]),
                "puf_2": float(window[1]),
                "puf_3": float(window[2]),
                "score": float(np.sum(window)),
                "n_U": n_unfavourable,
                "outcome": label_landmark(day, t_digital, event, HORIZON_DAYS),
                "event": event,
                "t_digital": t_digital,
                "fecha_fin": str(clinical.loc[int(pid), "Fecha_Fin_DB1_Actualizado"])[:10],
            })
    decision_df = drop_outside_comparison(mark_comparison(pd.DataFrame(decisions), comparison_blocks()))

    cuts = []
    threshold_of = {}
    for outer_fold, frame in decision_df[decision_df["role"] == "held_out"].groupby("outer_fold"):
        usable = cut_rows(frame)
        found = youden_cut(usable["score"].to_numpy(float), usable["outcome"].astype(int).to_numpy())
        if found is None:
            raise SystemExit(f"vuelta {outer_fold} sin corte")
        found["outer_fold"] = int(outer_fold)
        found["tie_rule"] = "menor corte"
        found["time"] = "Tiempo_obs_digital"
        found["horizon_days"] = HORIZON_DAYS
        found["comparison"] = "1013 every 30 days"
        found["map"] = "adjudicacion clinica"
        cuts.append(found)
        threshold_of[int(outer_fold)] = found["threshold"]
    cut_df = pd.DataFrame(cuts)
    decision_df["threshold"] = decision_df["outer_fold"].map(threshold_of)
    decision_df["alarm"] = decision_df["score"] >= decision_df["threshold"]
    decision_df.loc[~np.isfinite(decision_df["score"]), "alarm"] = pd.NA

    ALARM.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(ALARM / "bloques_puf.csv", index=False)
    cut_df.to_csv(ALARM / "cortes.csv", index=False)
    decision_df.to_csv(ALARM / "decisiones.csv", index=False)
    metrics = rendimiento(decision_df.dropna(subset=["threshold"]), ["outer_fold", "role"])
    metrics.to_csv(ALARM / "rendimiento_por_vuelta.csv", index=False)

    intervals = intervals_for(ALARM / "decisiones.csv", ["outer_fold", "role"])
    merged = metrics.merge(intervals, on=["outer_fold", "role"], suffixes=("", "_i"))
    for name in ("sensitivity", "specificity", "PPV", "NPV", "AUROC"):
        gap = (merged[name] - merged[f"{name}_i"]).abs().max()
        if gap > 1e-9:
            raise SystemExit(f"{name} difiere del intervalo en {gap}")
    intervals.to_csv(ALARM / "intervalos_bootstrap.csv", index=False)

    by_patient, summary = anticipation(decision_df, "outer_fold")
    by_patient.to_csv(ALARM / "anticipacion_por_paciente.csv", index=False)
    summary.to_csv(ALARM / "anticipacion_resumen.csv", index=False)

    usable = cut_rows(decision_df)
    rows = []
    for (outer_fold, role, n_unfavourable), frame in usable.groupby(["outer_fold", "role", "n_U"], sort=True):
        scores = frame["score"].to_numpy(float)
        rows.append({
            "outer_fold": int(outer_fold),
            "role": role,
            "n_U": int(n_unfavourable),
            "n_decisions": int(len(scores)),
            "min": float(scores.min()),
            "p25": float(np.percentile(scores, 25)),
            "median": float(np.median(scores)),
            "p75": float(np.percentile(scores, 75)),
            "max": float(scores.max()),
        })
    pd.DataFrame(rows).to_csv(ALARM / "distribucion_scores.csv", index=False)

    km = pd.read_csv(OUT / "alarma/fu/km_por_topic.csv")
    clinical_rows = []
    for _, item in km.iterrows():
        fold = "refit" if item["inner_fold"] == "refit" else str(item["inner_fold"])
        labels = maps[(int(item["outer_fold"]), fold)]
        topic = int(item["topic"])
        short = (str(item["outer_fold"]), fold, topic) in SHORT
        clinical_rows.append({
            "model": item["model"],
            "outer_fold": int(item["outer_fold"]),
            "inner_fold": item["inner_fold"],
            "topic": topic,
            "etiqueta_3_3": item["etiqueta"],
            "etiqueta_clinica": labels[topic],
            "pfs_365": item["pfs_365"],
            "supported_365": item["supported_365"],
            "motivo": (
                "adjudicacion clinica; sin soporte a 365; permanece en U"
                if short
                else "adjudicacion clinica"
            ),
        })
    pd.DataFrame(clinical_rows).to_csv(ALARM / "mapeo_clinico_por_topic.csv", index=False)

    print(cut_df[["outer_fold", "threshold", "n_decisions", "sensitivity", "specificity", "youden"]].to_string(index=False))
    external = metrics[metrics.role == "external_test"]
    print(external[["outer_fold", "n_decisions", "sensitivity", "specificity", "PPV", "AUROC", "patients_with_event", "patients_detected", "patients_false_alarm"]].to_string(index=False))
    print(summary.to_string(index=False))
    print("decisions", len(decision_df))


if __name__ == "__main__":
    main()
