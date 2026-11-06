"""Patient-clustered bootstrap intervals for the VAL-2 alarm metrics.

Same rule as the DB1 alarm: 2000 patient resamples, percentile 2.5 and 97.5,
seed 20261020, and the Youden cut of that fold stays fixed. Only the 1013
comparison blocks enter the interval. A patient drawn
twice contributes her decisions twice. Draws with one class do not enter the
AUROC percentile. The detection interval resamples patients with an event;
the false-alarm interval resamples patients without one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
N_BOOT = 2000
SEED = 20261020
MIN_FINITE = 100

SOURCES = (
    (
        ROOT / "results/lda/val_2/anidada_estratificada/alarma/decisiones.csv",
        ["outer_fold", "role"],
    ),
    (
        ROOT / "results/lda/val_2/loco/alarma/decisiones.csv",
        ["left_out_centre", "role"],
    ),
)


def auroc(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    pos, neg = w * y, w * (1.0 - y)
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    _, index = np.unique(x, return_inverse=True)
    pos_group = np.bincount(index, weights=pos)
    neg_group = np.bincount(index, weights=neg)
    return float(np.sum(pos_group * (np.cumsum(neg_group) - neg_group / 2.0)) / (pos.sum() * neg.sum()))


def clasif(y: np.ndarray, x: np.ndarray, threshold: float, w: np.ndarray) -> dict:
    predicted = x >= threshold
    pos, neg = w * y, w * (1.0 - y)
    true_pos, false_neg = float(pos[predicted].sum()), float(pos[~predicted].sum())
    false_pos, true_neg = float(neg[predicted].sum()), float(neg[~predicted].sum())

    def ratio(num: float, den: float) -> float:
        return num / den if den else np.nan

    return {
        "AUROC": auroc(y, x, w),
        "sensitivity": ratio(true_pos, true_pos + false_neg),
        "specificity": ratio(true_neg, true_neg + false_pos),
        "PPV": ratio(true_pos, true_pos + false_pos),
        "NPV": ratio(true_neg, true_neg + false_neg),
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_FINITE:
        return np.nan, np.nan, int(len(finite))
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def one_group(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    work = frame.copy()
    work["outcome"] = pd.to_numeric(work["outcome"], errors="coerce")
    evaluable = work[
        work["in_comparison"].eq(1) & work["outcome"].isin([0, 1]) & np.isfinite(work["score"])
    ].copy()
    outcome = evaluable["outcome"].to_numpy(float)
    score = evaluable["score"].to_numpy(float)
    threshold = float(evaluable["threshold"].iloc[0])
    patient = evaluable["id"].to_numpy()
    patients = np.unique(patient)
    index_of = {pid: i for i, pid in enumerate(patients)}
    patient_index = np.array([index_of[pid] for pid in patient])
    point = clasif(outcome, score, threshold, np.ones(len(outcome)))

    flags = []
    for pid, sub in evaluable.groupby("id"):
        labels = sub["outcome"].to_numpy(int)
        alarm = sub["score"].to_numpy(float) >= threshold
        event = int(sub["event"].iloc[0])
        flags.append(
            (
                pid,
                event,
                int(event == 1 and np.any((labels == 1) & alarm)),
                int(event == 0 and np.any((labels == 0) & alarm)),
            )
        )
    by_patient = pd.DataFrame(flags, columns=["id", "event", "detected", "false_alarm"])
    with_event = by_patient[by_patient.event == 1]
    without_event = by_patient[by_patient.event == 0]

    draws = rng.integers(0, len(patients), size=(N_BOOT, len(patients)))
    replicated = np.empty((N_BOOT, 5))
    for draw_index in range(N_BOOT):
        copies = np.bincount(draws[draw_index], minlength=len(patients)).astype(float)
        metrics = clasif(outcome, score, threshold, copies[patient_index])
        replicated[draw_index] = [
            metrics["AUROC"],
            metrics["sensitivity"],
            metrics["specificity"],
            metrics["PPV"],
            metrics["NPV"],
        ]

    result = dict(point)
    result["n_decisions"] = int(len(evaluable))
    result["n_patients"] = int(len(patients))
    result["threshold"] = threshold
    for column, name in enumerate(["AUROC", "sensitivity", "specificity", "PPV", "NPV"]):
        low, high, n_finite = percentile_interval(replicated[:, column])
        result[f"{name}_lo"] = low
        result[f"{name}_hi"] = high
        result[f"{name}_nfin"] = n_finite

    def proportion(sub: pd.DataFrame, column: str) -> tuple[float, float, float, int, int, int]:
        if len(sub) == 0:
            return np.nan, np.nan, np.nan, 0, 0, 0
        flag = sub[column].to_numpy(float)
        drawn = rng.integers(0, len(sub), size=(N_BOOT, len(sub)))
        low, high, n_finite = percentile_interval(flag[drawn].mean(axis=1))
        return float(flag.mean()), low, high, n_finite, int(len(sub)), int(flag.sum())

    detection, detection_lo, detection_hi, detection_nfin, n_event, n_detected = proportion(with_event, "detected")
    false_alarm, false_alarm_lo, false_alarm_hi, false_alarm_nfin, n_nonevent, n_false = proportion(
        without_event, "false_alarm"
    )
    result.update(
        detection=detection,
        detection_lo=detection_lo,
        detection_hi=detection_hi,
        detection_nfin=detection_nfin,
        n_event_patients=n_event,
        n_detected=n_detected,
        false_alarm_rate=false_alarm,
        false_alarm_lo=false_alarm_lo,
        false_alarm_hi=false_alarm_hi,
        false_alarm_nfin=false_alarm_nfin,
        n_nonevent_patients=n_nonevent,
        n_false_alarm=n_false,
        n_boot=N_BOOT,
        seed=SEED,
        min_finite=MIN_FINITE,
    )
    return result


def intervals_for(path: Path, keys: list[str]) -> pd.DataFrame:
    decisions = pd.read_csv(path)
    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in decisions.groupby(keys, sort=True):
        record = one_group(group, rng)
        for name, value in zip(keys, key):
            record[name] = value
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    for path, keys in SOURCES:
        table = intervals_for(path, keys)
        reference = pd.read_csv(path.with_name("rendimiento_por_vuelta.csv"))
        merged = reference.merge(table, on=keys)
        for left, right in (
            ("sensitivity", "sensitivity"),
            ("specificity", "specificity"),
            ("PPV", "PPV"),
            ("NPV", "NPV"),
            ("AUROC", "AUROC"),
        ):
            gap = (merged[f"{left}_x"] - merged[f"{right}_y"]).abs().max()
            if gap > 1e-9:
                raise SystemExit(f"{path.name} {left} differs from rendimiento by {gap}")
        if (merged["patients_detected"] - merged["n_detected"]).abs().max() != 0:
            raise SystemExit(f"{path.name} detected patients do not match")
        if (merged["patients_false_alarm"] - merged["n_false_alarm"]).abs().max() != 0:
            raise SystemExit(f"{path.name} false-alarm patients do not match")
        destination = path.with_name("intervalos_bootstrap.csv")
        table.to_csv(destination, index=False)
        print(f"{destination} {len(table)} rows")


if __name__ == "__main__":
    main()
