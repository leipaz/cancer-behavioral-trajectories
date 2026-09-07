"""Raw behavioral comparison between cohorts (before VQ-VAE encoding).

Aggregates daily features to the patient level, tests oncology vs mental blocks,
and plots standardized signatures for side-by-side comparison with VQ-VAE decode results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import constants as c
from .compare_cohort_profiles import (
    BEHAVIORAL_FEATURES,
    COHORT_LABELS,
    COHORT_ORDER,
    DEFAULT_FIG_DIR,
    DEFAULT_OUT_DIR,
)

DEFAULT_CSV_DIR = c.PROJECT_ROOT / "data" / "daily_summaries" / "revision_cohort_samples"

ONCOLOGY = ["CNIO", "PMP"]
MENTAL = ["CI", "das", "ED1", "ED2", "SR", "TMC", "GM"]


def load_cohort_days(csv_dir: Path = DEFAULT_CSV_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(csv_dir.glob("model_input_*.csv")):
        code = path.stem.replace("model_input_", "")
        df = pd.read_csv(path, usecols=["user", "date_time", *c.FEATURE_COLS])
        df["cohort"] = code
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No model_input_*.csv in {csv_dir}")
    out = pd.concat(frames, ignore_index=True)
    for col in BEHAVIORAL_FEATURES + ["weekend", "practiced_sport"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def patient_means(days: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """One row per patient: mean of observed daily features."""
    features = features or BEHAVIORAL_FEATURES
    g = (
        days.groupby(["cohort", "user"], as_index=False)[features]
        .mean(numeric_only=True)
    )
    g["block"] = np.where(g["cohort"].isin(ONCOLOGY), "oncology", "mental")
    g.loc[~g["cohort"].isin(ONCOLOGY + MENTAL), "block"] = "other"
    return g


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size: P(x>y) - P(x<y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # efficient approximate via rank of concatenated
    nx, ny = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    rank_x = ranks[:nx]
    # Mann-Whitney U = sum(ranks_x) - nx*(nx+1)/2
    u = rank_x.sum() - nx * (nx + 1) / 2.0
    return float((2 * u) / (nx * ny) - 1)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def oncology_vs_mental_tests(
    patients: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or BEHAVIORAL_FEATURES
    onc = patients[patients["block"] == "oncology"]
    ment = patients[patients["block"] == "mental"]
    rows = []
    for feat in features:
        x = onc[feat].dropna().to_numpy()
        y = ment[feat].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            continue
        u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "feature": feat,
                "n_oncology": int(len(x)),
                "n_mental": int(len(y)),
                "mean_oncology": float(np.mean(x)),
                "mean_mental": float(np.mean(y)),
                "median_oncology": float(np.median(x)),
                "median_mental": float(np.median(y)),
                "delta_mean_onc_minus_mental": float(np.mean(x) - np.mean(y)),
                "cohens_d": cohens_d(x, y),
                "cliffs_delta": cliffs_delta(x, y),
                "mannwhitney_U": float(u_stat),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    # BH-FDR
    p = out["p_value"].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out["p_fdr"] = np.empty(m)
    out.loc[out.index[order], "p_fdr"] = adj
    out["significant_fdr_05"] = out["p_fdr"] < 0.05
    return out.sort_values("p_value").reset_index(drop=True)


def cohort_feature_summary(
    patients: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or BEHAVIORAL_FEATURES
    rows = []
    for cohort, sub in patients.groupby("cohort"):
        row = {
            "cohort": cohort,
            "cohort_label": COHORT_LABELS.get(cohort, cohort),
            "n_patients": int(sub["user"].nunique()),
            "block": sub["block"].iloc[0],
        }
        for feat in features:
            row[feat] = float(sub[feat].mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_raw_cohort_means_z(
    summary: pd.DataFrame,
    out_path: Path,
    features: list[str] | None = None,
) -> Path:
    features = features or BEHAVIORAL_FEATURES
    order = [c for c in COHORT_ORDER if c in set(summary["cohort"])]
    mat = summary.set_index("cohort").loc[order, features]
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0, np.nan)

    melted = z.reset_index().melt(id_vars="cohort", var_name="Feature", value_name="Z-score")
    melted["cohort_label"] = melted["cohort"].map(lambda x: COHORT_LABELS.get(x, x))

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(
        data=melted,
        x="Feature",
        y="Z-score",
        hue="cohort_label",
        ax=ax,
        palette="tab10",
    )
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(
        "Raw behavioral patterns by cohort (patient-mean features, z across cohorts)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Behavioral features")
    ax.set_ylabel("Z-score across cohorts")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_oncology_vs_mental_boxplots(
    patients: pd.DataFrame,
    out_path: Path,
    features: list[str] | None = None,
) -> Path:
    features = features or BEHAVIORAL_FEATURES
    sub = patients[patients["block"].isin(["oncology", "mental"])].copy()
    melted = sub.melt(
        id_vars=["block", "cohort", "user"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )
    # standardize within feature for shared y-scale readability
    melted["Z"] = melted.groupby("Feature")["Value"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.boxplot(
        data=melted,
        x="Feature",
        y="Z",
        hue="block",
        showfliers=False,
        ax=ax,
        palette={"oncology": "#6b8e23", "mental": "#4c72b0"},
    )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("Raw patient-level features: oncology vs mental (z within feature)")
    ax.set_xlabel("Behavioral features")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Block")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_raw_vs_vqvae_deltas(
    tests: pd.DataFrame,
    vqvae_weighted_csv: Path,
    out_path: Path,
    features: list[str] | None = None,
) -> Path:
    """Compare oncology−mental deltas: raw patient means vs VQ-VAE usage-weighted decode."""
    features = features or BEHAVIORAL_FEATURES
    vw = pd.read_csv(vqvae_weighted_csv).set_index("cohort")
    onc_v = vw.loc[[c for c in ONCOLOGY if c in vw.index], features].mean()
    ment_v = vw.loc[[c for c in MENTAL if c in vw.index], features].mean()
    v_delta = onc_v - ment_v

    raw = tests.set_index("feature").loc[features]
    # z-score deltas within each method for shape comparison
    raw_d = raw["delta_mean_onc_minus_mental"]
    raw_z = (raw_d - raw_d.mean()) / (raw_d.std(ddof=0) or 1)
    v_z = (v_delta - v_delta.mean()) / (v_delta.std(ddof=0) or 1)

    plot_df = pd.DataFrame(
        {
            "Feature": features + features,
            "Z-scored delta (onc − mental)": list(raw_z) + list(v_z),
            "Source": ["Raw features"] * len(features) + ["VQ-VAE decoded"] * len(features),
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=plot_df,
        x="Feature",
        y="Z-scored delta (onc − mental)",
        hue="Source",
        ax=ax,
        palette=["#4c72b0", "#c44e52"],
    )
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title("Oncology − mental contrast: raw features vs VQ-VAE decoded signature")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_raw_comparison(
    csv_dir: Path = DEFAULT_CSV_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> dict[str, Path]:
    days = load_cohort_days(csv_dir)
    patients = patient_means(days)
    tests = oncology_vs_mental_tests(patients)
    summary = cohort_feature_summary(patients)

    table_dir = out_dir / "comparison_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["patient_means_csv"] = table_dir / "raw_patient_feature_means.csv"
    patients.to_csv(paths["patient_means_csv"], index=False)
    paths["cohort_summary_csv"] = table_dir / "raw_cohort_feature_means.csv"
    summary.to_csv(paths["cohort_summary_csv"], index=False)
    paths["tests_csv"] = table_dir / "raw_oncology_vs_mental_tests.csv"
    tests.to_csv(paths["tests_csv"], index=False)

    paths["fig_raw_cohort_z"] = plot_raw_cohort_means_z(
        summary, fig_dir / "raw_behavioral_patterns_cohort_means.png"
    )
    paths["fig_raw_box"] = plot_oncology_vs_mental_boxplots(
        patients, fig_dir / "raw_oncology_vs_mental_boxplots.png"
    )

    vqvae_csv = table_dir / "usage_weighted_decoded_by_cohort.csv"
    if vqvae_csv.exists():
        paths["fig_raw_vs_vqvae"] = plot_raw_vs_vqvae_deltas(
            tests, vqvae_csv, fig_dir / "raw_vs_vqvae_oncology_mental_deltas.png"
        )

    summary_json = {
        "n_patients": int(patients["user"].nunique()),
        "n_oncology": int((patients["block"] == "oncology").sum()),
        "n_mental": int((patients["block"] == "mental").sum()),
        "n_features_significant_fdr_05": int(tests["significant_fdr_05"].sum()),
        "features_significant": tests.loc[
            tests["significant_fdr_05"], "feature"
        ].tolist(),
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"] = table_dir / "raw_comparison_summary.json"
    paths["summary_json"].write_text(json.dumps(summary_json, indent=2))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()
    paths = run_raw_comparison(args.csv_dir, args.out_dir, args.fig_dir)
    print("Wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
