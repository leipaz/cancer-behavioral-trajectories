#!/usr/bin/env python3
"""Word report for missingness point 1 after the sport-zero rule.

Reads the rerun in results/missingness/ramp_db1_175_sport_keep1/ and writes
report.md, report.docx and the ramp figures into
results/missingness/Sensitivity_to_artificial_missingness/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Cm
from scipy import stats
from sklearn.metrics import roc_auc_score

import build_punto2_informe as style
import run_missingness_ramp as ramp
from sport_missing import ALARM_SPORT_RULE as SPORT_RULE, load_daily_for_alarm

ROOT = ramp.ROOT
RUN = ROOT / "results/missingness/ramp_db1_175_sport_keep1"
GAPS = RUN / "fixed_gaps"
OUT = ROOT / "results/missingness/Sensitivity_to_artificial_missingness"
EXAMPLE_ID = 62004
FROZEN = OUT / "alarm_scored_objetivo_baseline.csv"


THR = 1.0596384140662845
STORED = ROOT / "results/alarm/base_analysis/decision_points_collapsed.csv"
KEYS = ("AUROC", "sensitivity", "specificity", "PPV", "NPV")


def frozen_scored() -> pd.DataFrame:
    """The 1013 decisions with the stored load of the original pipeline (the frozen alarm)."""
    f = pd.read_csv(FROZEN)[["id", "study_day", "y", "load"]].rename(columns={"load": "rerun_load"})
    st = pd.read_csv(STORED)
    st["study_day"] = st["decision_day"].astype(int)
    f = f.merge(st[["id", "study_day", "AUC_W"]], on=["id", "study_day"], how="left", validate="one_to_one")
    if len(f) != 1013 or f["AUC_W"].isna().any():
        raise SystemExit("stored load missing for some of the 1013 decisions")
    return f


def frozen_metrics() -> dict[str, float]:
    """Frozen alarm: stored load and threshold on the 1013 decisions."""
    f = frozen_scored()
    y = f["y"].astype(int).to_numpy()
    a = (f["AUC_W"] >= THR).astype(int).to_numpy()
    tp, fn = int(((a == 1) & (y == 1)).sum()), int(((a == 0) & (y == 1)).sum())
    tn, fp = int(((a == 0) & (y == 0)).sum()), int(((a == 1) & (y == 0)).sum())
    return {
        "AUROC": float(roc_auc_score(y, f["AUC_W"])),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
    }


def check_frozen_baseline(table: pd.DataFrame) -> None:
    """The anchored baseline must be the frozen alarm exactly."""
    fro = frozen_metrics()
    base = lookup(table, "initial", 0.0)
    for k in KEYS:
        if not np.isclose(base["emitted_" + k], fro[k]):
            raise SystemExit(f"anchored baseline {k} {base['emitted_' + k]:.4f} is not the frozen {fro[k]:.4f}")


def baseline_sentences(table_rerun: pd.DataFrame) -> tuple[str, str]:
    """Baseline text: the frozen alarm, and how the unmasked re-run relates to it."""
    rerun = lookup(table_rerun, "initial", 0.0)
    fro = frozen_metrics()
    f = frozen_scored()
    mae = float((f["rerun_load"] - f["AUC_W"]).abs().mean())
    crossed = f[(f["rerun_load"] >= THR) != (f["AUC_W"] >= THR)]
    crossed_txt = "one decision crosses" if len(crossed) == 1 else f"{len(crossed)} decisions cross"
    frozen_txt = ", ".join(f"{k} {fro[k]:.3f}" for k in KEYS)
    rerun_txt = ", ".join(f"{k} {rerun['emitted_' + k]:.3f}" for k in KEYS)
    text = (
        f"The threshold stays at 1.0596384140662845. The baseline is the frozen alarm: the stored load of the "
        f"original pipeline on the 1,013 decisions, with {frozen_txt}. A mask changes the input, so every scenario "
        f"is re-encoded with VQ-VAE a0 and the LDA. Re-encoding without any mask does not give back the stored load "
        f"exactly (MAE {mae:.4f}; {crossed_txt} the threshold), because a few days get other codes. So each masked "
        "load is anchored to the frozen one: frozen load + (load with the mask − load re-encoded without it). "
        "The baseline is then the frozen alarm exactly, every alarm metric below refers to it, and each scenario "
        "carries only the change caused by its mask. Topic and load-change metrics do not depend on the anchor. "
        "The sport rule does not change the model input: every stored 1 is kept, and a0 reads a missing sport "
        "cell as 0, which is what was stored there."
    )
    return text, text


ORDER = (
    ("initial", 0.0, "baseline"),
    ("extra_days", 0.10, "+10% days"),
    ("extra_days", 0.20, "+20% days"),
    ("extra_days", 0.30, "+30% days"),
    ("extra_days", 0.40, "+40% days"),
    ("extra_days", 0.50, "+50% days"),
    ("extra_variables", 0.10, "+10% variables"),
    ("extra_variables", 0.20, "+20% variables"),
    ("extra_variables", 0.30, "+30% variables"),
    ("extra_variables", 0.40, "+40% variables"),
    ("extra_variables", 0.50, "+50% variables"),
    ("consecutive_days", 0.10, "+10% consecutive days"),
    ("consecutive_days", 0.20, "+20% consecutive days"),
    ("consecutive_days", 0.30, "+30% consecutive days"),
    ("consecutive_days", 0.40, "+40% consecutive days"),
    ("consecutive_days", 0.50, "+50% consecutive days"),
)
SERIES = (
    ("extra_days", "Scattered days", "#C47B3A"),
    ("extra_variables", "Variables", "#2F5D50"),
    ("consecutive_days", "Consecutive days", "#3D5A80"),
)


def _t_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    t = float(stats.t.ppf(0.975, len(x) - 1))
    return m, m - t * se, m + t * se


def aggregate(frame: pd.DataFrame, keys: list[tuple[str, float, str]]) -> pd.DataFrame:
    rows = []
    numeric = [c for c in frame.columns if c not in {"scenario", "mode", "seed"} and pd.api.types.is_numeric_dtype(frame[c])]
    for mode, rate, label in keys:
        if "gap_days" in frame.columns:
            part = frame.loc[np.isclose(frame["gap_days"].astype(float), rate)]
        else:
            part = frame.loc[frame["mode"].eq(mode) & np.isclose(frame["added_rate"].astype(float), rate)]
        rec = {"mode": mode, "added_rate": rate, "label": label, "n": int(len(part))}
        for col in numeric:
            m, lo, hi = _t_ci(part[col].to_numpy(float))
            rec[col] = m
            rec[f"{col}_lo"] = lo
            rec[f"{col}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt_value(m: float, lo: float, hi: float, digits: int = 3, counts: bool = False) -> str:
    if not np.isfinite(m):
        return "—"
    if not np.isfinite(lo):
        if abs(m) < 1e-9:
            return "0"
        if abs(m - 1) < 1e-9:
            return "1"
        return f"{m:.{digits}f}"
    if counts:
        point = int(np.rint(m))
        lo_r = max(0, int(np.rint(lo)))
        hi_r = max(0, int(np.rint(hi)))
        if lo_r == hi_r == point:
            return str(point)
        return f"{point} [{lo_r}, {hi_r}]"
    if m < 0.0005 and digits >= 3:
        return "<0.001"
    point = f"{m:.{digits}f}"
    lo_s, hi_s = f"{max(lo, 0) if False else lo:.{digits}f}", f"{hi:.{digits}f}"
    if lo_s == hi_s:
        return point
    return f"{point} [{lo_s}, {hi_s}]"


def cell(row: pd.Series, col: str, digits: int = 3, counts: bool = False) -> str:
    return fmt_value(float(row[col]), float(row[f"{col}_lo"]), float(row[f"{col}_hi"]), digits, counts)


def pct(row: pd.Series, col: str) -> str:
    return f"{100 * float(row[col]):.1f}%"


def lookup(table: pd.DataFrame, mode: str, rate: float) -> pd.Series:
    hit = table.loc[table["mode"].eq(mode) & np.isclose(table["added_rate"], rate)]
    if len(hit) != 1:
        raise KeyError((mode, rate))
    return hit.iloc[0]


def save_fig(fig, name: str) -> Path:
    png = OUT / f"{name}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return png


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_ramps(table: pd.DataFrame, panels: list[tuple[str, str, tuple[float, float] | None, float | None]], name: str, ncols: int) -> Path:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10, "figure.facecolor": "white"})
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.15 * nrows), squeeze=False)
    rates = np.array([0, 10, 20, 30, 40, 50], float)
    for ax, (title, col, ylim, chance) in zip(axes.ravel(), panels):
        for mode, label, color in SERIES:
            ys, lo, hi = [], [], []
            base = lookup(table, "initial", 0.0)
            ys.append(float(base[col]))
            lo.append(float(base[col]))
            hi.append(float(base[col]))
            for rate in (0.1, 0.2, 0.3, 0.4, 0.5):
                row = lookup(table, mode, rate)
                ys.append(float(row[col]))
                lo.append(float(row[f"{col}_lo"]))
                hi.append(float(row[f"{col}_hi"]))
            ys, lo, hi = map(np.asarray, (ys, lo, hi))
            ax.plot(rates, ys, color=color, marker="o", linewidth=1.6, markersize=4.5, label=label)
            ax.fill_between(rates, lo, hi, color=color, alpha=0.18, linewidth=0)
        if chance is not None:
            ax.axhline(chance, color="#B0B0B0", linestyle="--", linewidth=0.9, label="Random alarm" if title == "AUROC" else None)
        ax.set_title(title)
        ax.set_xlabel("Added missingness (%)")
        ax.set_xticks(rates)
        if ylim:
            ax.set_ylim(*ylim)
        style_ax(ax)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Keep one legend, including the chance line when present on AUROC.
    if any(p[0] == "AUROC" for p in panels):
        h2, l2 = axes.ravel()[0].get_legend_handles_labels()
        handles, labels = h2, l2
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return save_fig(fig, name)


def _calendar(frame: pd.DataFrame, uid: int) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    g = frame.loc[frame["user"].astype(int) == uid].copy()
    dates = pd.to_datetime(g["date_time"]).dt.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    indexed = g.copy()
    indexed["date_time"] = dates.to_numpy()
    indexed = indexed.drop_duplicates("date_time", keep="first").set_index("date_time")
    return calendar, indexed.reindex(calendar)[ramp.MEASURE_COLS]


def plot_example(daily: pd.DataFrame) -> Path:
    masks = {
        "Baseline": daily,
        "Scattered days, +40%": ramp.mask_extra_days(daily, 0.40, 101),
        "Variables, +40%": ramp.mask_extra_cells(daily, 0.40, 101),
        "Consecutive days, 20%": ramp.mask_consecutive_gap(daily, 0.20, 101)[0],
    }
    scores = {}
    files = {
        "Baseline": RUN / "landmarks" / "reference.csv",
        "Scattered days, +40%": RUN / "landmarks" / "days_plus_40_s101.csv",
        "Variables, +40%": RUN / "landmarks" / "variables_plus_40_s101.csv",
        "Consecutive days, 20%": RUN / "landmarks" / "consecutive_plus_20_s101.csv",
    }
    colors = {
        "Baseline": "#1C1B19",
        "Scattered days, +40%": "#C47B3A",
        "Variables, +40%": "#2F5D50",
        "Consecutive days, 20%": "#3D5A80",
    }
    for name, path in files.items():
        sc = pd.read_csv(path)
        sc = sc.loc[sc["id"].astype(int) == EXAMPLE_ID].sort_values("study_day")
        scores[name] = sc
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_axes([0.12, 0.58, 0.84, 0.34])
    for name, sc in scores.items():
        x = sc["study_day"].to_numpy(float) / 30.0
        y = sc["load"].to_numpy(float)
        kw = {"color": colors[name], "linewidth": 1.7, "marker": "o", "markersize": 4.5, "label": name}
        if name.startswith("Consecutive"):
            kw["linestyle"] = "--"
        ax.plot(x, y, **kw)
    ax.axvspan(7, 10, color="#E7B3B3", alpha=0.45, zorder=0)
    ax.axhline(1.0596384140662845, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Study month")
    ax.set_ylabel("Score")
    ax.set_title("Patient 62004. Red band: alarm horizon (months 7–10).")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax)

    def strip(ax, items: list[tuple[str, str]], mode: str) -> None:
        rows = []
        labels = []
        for key, label in items:
            labels.append(label)
            calendar, vals = _calendar(masks[key], EXAMPLE_ID)
            present = vals.notna()
            if mode == "day":
                rows.append((~present.any(axis=1)).to_numpy(float))
            else:
                rows.append(1.0 - present.to_numpy().mean(axis=1))
        arr = np.vstack(rows)
        months = len(calendar) / 30.0
        cmap = "Oranges" if mode == "day" else "Greens"
        ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1, extent=[0, months, len(labels), 0], interpolation="nearest")
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Study month")
        ax.set_xlim(0, 10)

    ax_l = fig.add_axes([0.14, 0.08, 0.36, 0.36])
    ax_r = fig.add_axes([0.62, 0.08, 0.34, 0.36])
    strip(
        ax_l,
        [("Baseline", "Baseline"), ("Scattered days, +40%", "+40% days"), ("Consecutive days, 20%", "20% block")],
        "day",
    )
    strip(ax_r, [("Baseline", "Baseline"), ("Variables, +40%", "+40% variables")], "cell")
    ax_l.set_title("Whole-day missingness")
    ax_r.set_title("Variable missingness")
    fig.text(0.08, 0.01, "Left: colour marks a fully missing day. Right: darker means more of the nine measures are missing that day.", fontsize=8, color="#555555")
    return save_fig(fig, "example_62004")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Alarm metrics anchored to the frozen alarm (anchor_sensitivity_to_frozen.py).
    by_seed = pd.read_csv(RUN / "metrics_by_seed_anchored.csv")
    table = aggregate(by_seed, ORDER)
    check_frozen_baseline(table)
    table_rerun = aggregate(pd.read_csv(RUN / "metrics_by_seed.csv"), ORDER)
    gaps = pd.read_csv(GAPS / "gaps_3_7_14_metrics_by_seed_anchored.csv")
    gap_table = aggregate(gaps, [("fixed_gap", 3, "3 days"), ("fixed_gap", 7, "7 days"), ("fixed_gap", 14, "14 days")])
    base_miss = pd.read_csv(RUN / "reference_missingness.csv")
    q33, q66 = base_miss["variable_missing_rate"].quantile([1 / 3, 2 / 3])
    n_none = int((base_miss["day_missing_rate"] == 0).sum())
    n_some = int((base_miss["day_missing_rate"] > 0).sum())
    day_pct = 100 * float(base_miss["day_missing_rate"].mean())
    var_pct = 100 * float(base_miss["variable_missing_rate"].mean())
    var_med = 100 * float(base_miss["variable_missing_rate"].median())
    fixed_pct = 100 * float(lookup(table, "initial", 0.0)["mean_variable_missing_fixed_calendar"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
    topics_png = plot_ramps(
        table,
        [
            ("Day-type JS", "daytype_js", None, None),
            ("Topic JS", "topic_js", None, None),
            ("Topic MAE", "topic_mae", None, None),
            ("Topic cosine", "topic_cosine", (0.4, 1.02), None),
            ("Dominant", "dominant_agreement", (0.4, 1.02), None),
        ],
        "ramp_topics",
        3,
    )
    load_png = plot_ramps(
        table,
        [
            ("pUF correlation", "pUF_pearson", (0.4, 1.02), None),
            ("pUF MAE", "pUF_mae", None, None),
            ("Load correlation", "load_pearson", (0.4, 1.02), None),
            ("Load MAE", "load_mae", None, None),
        ],
        "ramp_load",
        3,
    )
    metrics_png = plot_ramps(
        table,
        [
            ("AUROC", "emitted_AUROC", (0.50, 0.74), 0.5),
            ("Sensitivity", "emitted_sensitivity", (0.30, 0.98), 0.5),
            ("Specificity", "emitted_specificity", (0.25, 0.85), 0.5),
            ("PPV", "emitted_PPV", (0.15, 0.27), 155 / 1013),
            ("NPV", "emitted_NPV", (0.85, 0.97), 858 / 1013),
        ],
        "ramp_metrics",
        3,
    )
    daily = load_daily_for_alarm(ramp.DAILY)
    example_png = plot_example(daily)
    baseline_doc, baseline_md = baseline_sentences(table_rerun)

    def rows_for(cols: list[tuple[str, int, bool]]) -> list[list[str]]:
        out = []
        for mode, rate, label in ORDER:
            row = lookup(table, mode, rate)
            out.append([label] + [cell(row, col, digits, counts) for col, digits, counts in cols])
        return out

    d50 = lookup(table, "extra_days", 0.5)
    v50 = lookup(table, "extra_variables", 0.5)
    c10 = lookup(table, "consecutive_days", 0.1)
    c20 = lookup(table, "consecutive_days", 0.2)
    c50 = lookup(table, "consecutive_days", 0.5)
    d20 = lookup(table, "extra_days", 0.2)
    d40 = lookup(table, "extra_days", 0.4)
    v40 = lookup(table, "extra_variables", 0.4)
    s20 = lookup(table, "extra_days", 0.2)
    s50 = lookup(table, "extra_days", 0.5)

    doc = Document()
    section = doc.sections[0]
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, side, Cm(1.5))
    style.add_heading(doc, "Sensitivity to artificial missingness", 1)
    style.add_text(
        doc,
        "DB1 daily patients (n=175): masking, then VQ-VAE a0 and LDA. Alarm metrics use the 1,013 decisions "
        "in objetivo.csv (114 patients, 155 positive / 858 negative). Every decision keeps a load. Six end "
        "after the encoded profile and take the last 30 profile days.",
    )
    style.add_text(
        doc,
        "practiced_sport: a stored 1 is always kept. A stored 0 counts only when activity_walking survived the "
        "filter of 120–15000 seconds; otherwise it is missing and it is not in the pool of observed cells. The extra mask is applied after that rule and before VQ-VAE imputation. "
        "VQ-VAE a0, the LDA, "
        "the F/U grouping (topics 3–5) and the threshold 1.0596384140662845 stay fixed. Seeds 101–105.",
    )
    style.add_heading(doc, "Initial missingness and the three masks", 2)
    style.add_text(
        doc,
        f"Fully missing days: patient-mean {day_pct:.1f}% of each patient’s expected calendar (first to last day), "
        f"then the unweighted mean across the 175 patients. {n_none} of 175 patients have none. {n_some} have at "
        "least one. The same patient-mean definition is used for every row of the empty-days table.",
    )
    style.add_text(
        doc,
        f"Missing variables within a day that still has some measurement: patient-mean {var_pct:.1f}%, "
        f"median {var_med:.1f}%. Tertiles at {100 * float(q33):.1f}% and {100 * float(q66):.1f}%. Counted over the "
        f"whole calendar, with a fully empty day as nine empty variables, the patient-mean is {fixed_pct:.1f}%.",
    )
    style.add_text(
        doc,
        "From there, an extra 10% to 50% of what is still observed is hidden. Scattered days remove whole days "
        "at random. Variables remove cells among the nine measures. Consecutive days hide one block of that share "
        "of the calendar, at least 7 days.",
    )
    miss_rows = []
    for mode, rate, label in ORDER:
        row = lookup(table, mode, rate)
        miss_rows.append([label, pct(row, "mean_day_missing"), pct(row, "mean_variable_missing_fixed_calendar")])
    style.add_table(doc, ["Added", "Empty days", "Empty variables"], miss_rows, [3.2, 2.2, 2.4])

    style.add_heading(doc, "Topics", 2)
    style.add_text(
        doc,
        "Each value below is the mean of seeds 101–105. The bracket is the 95% confidence interval of that "
        "seed-mean (variability of the deletion), not a patient bootstrap. The baseline is one run, so it has no interval.",
    )
    style.add_text(
        doc,
        f"A contiguous block moves the topics more than the same share removed at random. Topic cosine is "
        f"{c20['topic_cosine']:.3f} at +20% consecutive and {c50['topic_cosine']:.3f} at +50%, against "
        f"{s20['topic_cosine']:.3f} and {s50['topic_cosine']:.3f} for scattered days.",
    )
    style.add_table(
        doc,
        ["Added", "Day-type JS", "Topic JS", "Topic MAE", "Topic cosine", "Dominant"],
        rows_for(
            [
                ("daytype_js", 3, False),
                ("topic_js", 3, False),
                ("topic_mae", 3, False),
                ("topic_cosine", 3, False),
                ("dominant_agreement", 3, False),
            ]
        ),
        [2.4, 1.5, 1.5, 1.8, 1.6, 1.6],
        size=7,
    )
    style.add_figure(doc, topics_png, topics_png.with_suffix(".svg"))

    style.add_heading(doc, "Monthly pUF and three-block load", 2)
    style.add_text(
        doc,
        f"Scattered days and variables keep the load correlation above 0.90 through +40% "
        f"({d40['load_pearson']:.3f} and {v40['load_pearson']:.3f}), and at +50% it is "
        f"{d50['load_pearson']:.3f} and {v50['load_pearson']:.3f}. Consecutive days pull it down earlier: "
        f"{c20['load_pearson']:.3f} at +20% and {c50['load_pearson']:.3f} at +50%. The absolute difference of the "
        f"load at +50% is {d50['load_mae']:.3f} for scattered days, {v50['load_mae']:.3f} for variables and "
        f"{c50['load_mae']:.3f} for a consecutive block.",
    )
    style.add_table(
        doc,
        ["Added", "pUF corr.", "pUF MAE", "Load corr.", "Load MAE"],
        rows_for([("pUF_pearson", 3, False), ("pUF_mae", 3, False), ("load_pearson", 3, False), ("load_mae", 3, False)]),
        [2.6, 2.0, 1.8, 2.0, 1.8],
        size=8,
    )
    style.add_figure(doc, load_png, load_png.with_suffix(".svg"))

    style.add_heading(doc, "Alarm", 2)
    style.add_text(
        doc,
        baseline_doc,
    )
    style.add_text(
        doc,
        f"For scattered days and for variables, AUROC stays near 0.70 through +30%, and at +50% it is "
        f"{d50['emitted_AUROC']:.3f} and {v50['emitted_AUROC']:.3f}. When variables are missing, sensitivity at "
        f"+50% is {v50['emitted_sensitivity']:.3f} while specificity falls to {v50['emitted_specificity']:.3f}. "
        f"When scattered days are missing, sensitivity falls to {d50['emitted_sensitivity']:.3f} and specificity "
        f"is {d50['emitted_specificity']:.3f}.",
    )
    style.add_text(
        doc,
        f"Consecutive days lower the load, so alarms turn off. AUROC at +10% is {c10['emitted_AUROC']:.3f} and at "
        f"+50% is {c50['emitted_AUROC']:.3f}. Sensitivity falls to {c50['emitted_sensitivity']:.3f} and specificity "
        f"rises to {c50['emitted_specificity']:.3f}. At +20% consecutive, AUROC is {c20['emitted_AUROC']:.3f} versus "
        f"{d20['emitted_AUROC']:.3f} when those days are removed scattered, and "
        f"{int(np.rint(c20['flip_on_to_off']))} alarms turn off with none turning on.",
    )
    style.add_table(
        doc,
        ["Added", "AUROC", "Sensitivity", "Specificity", "PPV", "NPV"],
        rows_for(
            [
                ("emitted_AUROC", 3, False),
                ("emitted_sensitivity", 3, False),
                ("emitted_specificity", 3, False),
                ("emitted_PPV", 3, False),
                ("emitted_NPV", 3, False),
            ]
        ),
        [2.3, 1.9, 1.9, 1.9, 1.7, 1.7],
        size=7,
    )
    style.add_figure(doc, metrics_png, metrics_png.with_suffix(".svg"))
    style.add_table(
        doc,
        ["Added", "F/U agreement", "Alarm agreement", "Turns on", "Turns off"],
        rows_for(
            [
                ("fu_agreement", 3, False),
                ("alarm_agreement", 3, False),
                ("flip_off_to_on", 0, True),
                ("flip_on_to_off", 0, True),
            ]
        ),
        [2.6, 1.8, 1.9, 1.5, 1.6],
        size=8,
    )
    style.add_heading(doc, "One patient example", 2)
    style.add_text(
        doc,
        "Patient 62004, seed 101. The score is the three-block load. The red band is months 7–10. "
        "The strips use the same masks: a coloured day on the left has no remaining measure; on the right, "
        "darker means more of the nine measures are missing.",
    )
    style.add_figure(doc, example_png, example_png.with_suffix(".svg"))

    style.add_heading(doc, "Contiguous gaps of 3, 7 and 14 days", 2)
    style.add_text(
        doc,
        "Separate from the 10–50% ramp. On each of the 175 patients one run of 3, 7 or 14 days is hidden, "
        "placed at random with seeds 101–105. Two patients have a calendar shorter than 14 days. Every decision "
        "still has a load. The placement does not depend on which cells were observed. Agreement and flips "
        "refer to the frozen alarm, with the same anchoring as the ramp.",
    )
    gap_rows = []
    for _, row in gap_table.iterrows():
        gap_rows.append(
            [
                row["label"],
                cell(row, "topic_js"),
                cell(row, "topic_mae"),
                cell(row, "dominant_agreement"),
                cell(row, "alarm_agreement"),
                cell(row, "flip_on_to_off", counts=True),
                cell(row, "emitted_AUROC"),
                cell(row, "emitted_sensitivity"),
                cell(row, "emitted_specificity"),
            ]
        )
    style.add_table(
        doc,
        ["Gap", "Posterior JS", "Posterior MAE", "Dominant", "Threshold", "Turns off", "AUROC", "Sensitivity", "Specificity"],
        gap_rows,
        [1.1, 1.3, 1.6, 1.6, 1.6, 1.2, 1.7, 1.6, 1.6],
        size=7,
    )

    dest = OUT / "report.docx"
    doc.save(dest)

    shutil.copyfile(RUN / "metrics_by_seed_anchored.csv", OUT / "metrics_by_seed.csv")
    shutil.copyfile(GAPS / "gaps_3_7_14_metrics_by_seed_anchored.csv", OUT / "gaps_3_7_14_metrics_by_seed.csv")
    shutil.copyfile(RUN / "reference_missingness.csv", OUT / "initial_missingness_by_patient.csv")
    mean_cols = [c for c in table.columns if not c.endswith("_lo") and not c.endswith("_hi") and c not in {"label", "n"}]
    table[mean_cols].to_csv(OUT / "metrics_mean.csv", index=False)
    prov_path = OUT / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    ref_json = json.loads((RUN / "reference_vs_collapsed.json").read_text(encoding="utf-8"))
    if ref_json.get("sport_rule") != SPORT_RULE:
        raise SystemExit(f"{RUN} was built with {ref_json.get('sport_rule')}, not {SPORT_RULE}")
    prov.update({
        "sport_rule": SPORT_RULE,
        "run_directory": str(RUN.relative_to(ROOT)),
        "baseline_mae_vs_collapsed": round(float(ref_json["mae_vs_collapsed_AUC_W"]), 4),
        "report": "scripts/03_analysis/alarm/build_punto1_informe.py",
    })
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (OUT / "PROVENANCE.md").write_text(
        "# Provenance — sensitivity missingness\n\n"
        "- Pipeline cohort (mask → VQ-VAE → LDA): all DB1 daily patients (n=175)\n"
        f"- Sport rule: {SPORT_RULE}. Applied before the baseline missingness count and before the extra mask.\n"
        "- A stored 1 is always kept. A stored 0 stays observed only when activity_walking survived 120–15000 seconds.\n"
        "- Mask applied before VQ-VAE imputation; learned parameters frozen\n"
        "- Alarm / landmark metrics: the 1013 decisions in objetivo.csv (114 / 155+ / 858−)\n"
        "- Alarm denominator in every scenario/seed: 1013 loads\n"
        "- Baseline = frozen alarm (stored AUC_W in results/alarm/base_analysis/decision_points_collapsed.csv). Masked loads are anchored: AUC_W + (masked re-encoding − unmasked re-encoding); alarm = anchored load >= threshold. Script: scripts/03_analysis/alarm/anchor_sensitivity_to_frozen.py. The CSVs here are the anchored ones; the raw re-encoding metrics stay in the run directory (metrics_by_seed.csv).\n"
        "- Seed CIs: deletion-experiment variability (Student t, 5 seeds), not a patient bootstrap\n"
        "- Empty-days/variables table: patient-mean rates on each expected calendar\n"
        "- Threshold: 1.0596384140662845\n"
        "- Seeds: [101, 102, 103, 104, 105]\n"
        "- Rates: [0.1, 0.2, 0.3, 0.4, 0.5]\n"
        "- Script: `scripts/03_analysis/alarm/run_sensitivity_db1_objetivo.py`\n"
        "- Report: `scripts/03_analysis/alarm/build_punto1_informe.py`\n"
        f"- Run directory: `{RUN.relative_to(ROOT)}`\n"
    )
    lines = [
        "# Sensitivity to artificial missingness",
        "",
        "Masks, VQ-VAE a0 and LDA run on all 175 DB1 patients. Topics, pUF, the three-block load, F/U and the alarm use the 1013 decisions in `objetivo.csv`: 114 patients, 155 positive and 858 negative.",
        "",
        "practiced_sport: a stored 1 is always kept. A stored 0 counts only when activity_walking survived 120–15000 seconds; otherwise it is missing, so the extra mask cannot select it.",
        "",
        "The extra mask is applied before VQ-VAE imputation. The calendar, origin, windows, frozen VQ-VAE a0, LDA, F/U (topics 3–5) and the full threshold **1.0596384140662845** stay fixed. No parameters are re-learned.",
        "",
        baseline_md,
        "",
        f"On the 175 patients the starting missingness uses the same patient-mean definition in every row. Fully empty days {day_pct:.1f}% ({n_none} with none, {n_some} with at least one). Variables inside a day that still has a measurement {var_pct:.1f}% (median {var_med:.1f}%, tertiles {100 * float(q33):.1f}% and {100 * float(q66):.1f}%). Over the whole calendar, with a fully empty day as nine empty variables, the mean is {fixed_pct:.1f}%.",
        "",
        "Each masked value below is the mean of seeds 101–105. The bracket is the 95% confidence interval of that seed-mean. It is not a patient-resampling bootstrap.",
        "",
        "| Added | AUROC | Sensitivity | Specificity | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, rate, label in ORDER:
        row = lookup(table, mode, rate)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    cell(row, "emitted_AUROC"),
                    cell(row, "emitted_sensitivity"),
                    cell(row, "emitted_specificity"),
                    cell(row, "emitted_PPV"),
                    cell(row, "emitted_NPV"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"For scattered days and for variables, AUROC stays near 0.70 through +30%, and at +50% it is {d50['emitted_AUROC']:.3f} and {v50['emitted_AUROC']:.3f}. Consecutive days lower the load: at +20% the AUROC is {c20['emitted_AUROC']:.3f}, against {d20['emitted_AUROC']:.3f} when those days are scattered, and {int(np.rint(c20['flip_on_to_off']))} alarms turn off with none turning on. At +50% a consecutive block turns {int(np.rint(c50['flip_on_to_off']))} alarms off.",
        "",
        "## Contiguous gaps of 3, 7 and 14 days",
        "",
        "One extra block, separate from the 10–50% ramp. On each of the 175 DB1 patients a single run of 3, 7 or 14 days is hidden, placed at random with seeds 101–105. Every one of the 1013 decisions still has a load.",
        "",
        "| Gap | Posterior JS | Posterior MAE | Dominant | Threshold | Turns off | AUROC | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in gap_table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    cell(row, "topic_js"),
                    cell(row, "topic_mae"),
                    cell(row, "dominant_agreement"),
                    cell(row, "alarm_agreement"),
                    cell(row, "flip_on_to_off", counts=True),
                    cell(row, "emitted_AUROC"),
                    cell(row, "emitted_sensitivity"),
                    cell(row, "emitted_specificity"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Per-seed file: `results/missingness/Sensitivity_to_artificial_missingness/gaps_3_7_14_metrics_by_seed.csv`.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("wrote", dest)
    print(f"empty days {day_pct:.1f}%  within-day {var_pct:.1f}%  calendar {fixed_pct:.1f}%")


if __name__ == "__main__":
    build()
