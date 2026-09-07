"""Compare VQ-VAE day-type occupancy and decoded behavioral signatures by cohort.

Two complementary views:
1. Occupancy — which codebook profiles are used more in each cohort.
2. Decode — usage-weighted mean of decoded behavioral features per cohort,
   plus decoded vectors of each cohort's top-K day-types.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import constants as c

DEFAULT_OUT_DIR = c.PROJECT_ROOT / "data" / "output_vq_vae" / "revision_cohorts"
DEFAULT_FIG_DIR = c.PROJECT_ROOT / "results" / "vq-vae" / "figures" / "revision_cohorts"
DEFAULT_DECODED_PKL = (
    c.PROJECT_ROOT / "data" / "output_vq_vae" / "decoded_embedding_vectors_a0.pkl"
)
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "sampling_manifest.json"

FEATURE_NAMES = [
    "sleep_start",
    "location_distance",
    "location_time_home",
    "sleep_duration",
    "activity_walking",
    "app_usage_total",
    "location_clusters_count",
    "steps_steps_total",
    "weekend",
    "practiced_sport",
]
BEHAVIORAL_FEATURES = [f for f in FEATURE_NAMES if f != "weekend"]

# Display order for revision figures
COHORT_ORDER = ["CI", "das", "ED1", "ED2", "GM", "PMP", "SR", "TMC", "CNIO"]
COHORT_LABELS = {
    "CI": "Cognitive impairment",
    "das": "das",
    "ED1": "Eating disorder 1",
    "ED2": "Eating disorder 2",
    "GM": "Gregorio Marañón",
    "PMP": "PMP (train)",
    "SR": "Suicide risk",
    "TMC": "Common mental disorder",
    "CNIO": "CNIO oncology",
}


def load_decoded_embeddings(path: Path = DEFAULT_DECODED_PKL) -> dict[int, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return {int(k): np.asarray(v, dtype=float) for k, v in data.items()}


def load_profile_counts(
    pkl_path: Path,
    mode: str = "a0",
    top_n: int = 30,
) -> Counter:
    with pkl_path.open("rb") as handle:
        profiles = pickle.load(handle)
    counts: Counter = Counter()
    for _user_id, entries in profiles[mode][top_n].items():
        length, seq, *_ = entries[0]
        counts.update(int(x) for x in np.asarray(seq[:length]).tolist())
    return counts


def discover_cohort_pickles(out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Path]:
    paths = {}
    for path in sorted(out_dir.glob("profiles_per_sample_*.pkl")):
        code = path.stem.replace("profiles_per_sample_", "")
        paths[code] = path
    return paths


def occupancy_frame(
    cohort_pickles: dict[str, Path],
    mode: str = "a0",
    top_n: int = 30,
    n_codes: int = 256,
) -> pd.DataFrame:
    """Rows = day-type id, columns = cohorts; values = usage probability."""
    cols = {}
    for code, path in cohort_pickles.items():
        counts = load_profile_counts(path, mode=mode, top_n=top_n)
        total = sum(counts.values()) or 1
        probs = np.zeros(n_codes, dtype=float)
        for embed_id, n in counts.items():
            if 0 <= embed_id < n_codes:
                probs[embed_id] = n / total
        cols[code] = probs
    return pd.DataFrame(cols, index=pd.RangeIndex(n_codes, name="day_type"))


def top_profiles_table(
    occ: pd.DataFrame,
    decoded: dict[int, np.ndarray],
    top_k: int = 15,
) -> pd.DataFrame:
    rows = []
    for cohort in occ.columns:
        series = occ[cohort].sort_values(ascending=False).head(top_k)
        for rank, (day_type, prob) in enumerate(series.items(), start=1):
            vec = decoded.get(int(day_type))
            row = {
                "cohort": cohort,
                "cohort_label": COHORT_LABELS.get(cohort, cohort),
                "rank": rank,
                "day_type": int(day_type),
                "occupancy": float(prob),
            }
            if vec is not None:
                for name, value in zip(FEATURE_NAMES, vec):
                    row[name] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def usage_weighted_decoded(
    occ: pd.DataFrame,
    decoded: dict[int, np.ndarray],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Mean decoded feature vector per cohort, weighted by day-type occupancy."""
    feature_names = feature_names or BEHAVIORAL_FEATURES
    feat_idx = [FEATURE_NAMES.index(f) for f in feature_names]
    rows = []
    for cohort in occ.columns:
        probs = occ[cohort].to_numpy()
        acc = np.zeros(len(feat_idx), dtype=float)
        weight = 0.0
        for day_type, p in enumerate(probs):
            if p <= 0:
                continue
            vec = decoded.get(day_type)
            if vec is None:
                continue
            acc += p * vec[feat_idx]
            weight += p
        if weight > 0:
            acc /= weight
        row = {"cohort": cohort, "cohort_label": COHORT_LABELS.get(cohort, cohort)}
        row.update({f: float(v) for f, v in zip(feature_names, acc)})
        rows.append(row)
    return pd.DataFrame(rows)


def enrichment_vs_mean(occ: pd.DataFrame) -> pd.DataFrame:
    """Lift = p(code|cohort) / mean_cohort p(code)."""
    baseline = occ.mean(axis=1).replace(0, np.nan)
    return occ.div(baseline, axis=0)


def plot_top_occupancy_heatmap(
    occ: pd.DataFrame,
    out_path: Path,
    top_k: int = 25,
    cohort_order: list[str] | None = None,
) -> Path:
    order = [c for c in (cohort_order or COHORT_ORDER) if c in occ.columns]
    # union of top-k day-types across cohorts
    top_ids = set()
    for col in order:
        top_ids.update(occ[col].nlargest(top_k).index.tolist())
    top_ids = sorted(top_ids)
    mat = occ.loc[top_ids, order]
    mat = mat.rename(columns={c: COHORT_LABELS.get(c, c) for c in mat.columns})

    fig, ax = plt.subplots(figsize=(1.2 * len(order) + 2, max(6, 0.28 * len(top_ids))))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="mako",
        cbar_kws={"label": "occupancy P(day-type | cohort)"},
    )
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Day-type (VQ-VAE codebook id)")
    ax.set_title(f"Most-used day-types by cohort (union of top-{top_k})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_enrichment_heatmap(
    lift: pd.DataFrame,
    occ: pd.DataFrame,
    out_path: Path,
    top_k: int = 15,
    cohort_order: list[str] | None = None,
) -> Path:
    order = [c for c in (cohort_order or COHORT_ORDER) if c in lift.columns]
    rows = []
    for col in order:
        # enriched among reasonably used codes
        mask = occ[col] >= occ[col].quantile(0.7)
        cand = lift.loc[mask, col].sort_values(ascending=False).head(top_k)
        rows.extend(cand.index.tolist())
    # unique preserve order
    seen = set()
    top_ids = []
    for i in rows:
        if i not in seen:
            seen.add(i)
            top_ids.append(i)

    mat = np.log2(lift.loc[top_ids, order].clip(lower=1e-6))
    mat = mat.rename(columns={c: COHORT_LABELS.get(c, c) for c in mat.columns})
    fig, ax = plt.subplots(figsize=(1.2 * len(order) + 2, max(6, 0.28 * len(top_ids))))
    sns.heatmap(
        mat,
        ax=ax,
        center=0,
        cmap="vlag",
        cbar_kws={"label": "log2 lift vs mean cohort"},
    )
    ax.set_title(f"Day-types enriched per cohort (top-{top_k} lifts)")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Day-type id")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decoded_cohort_heatmap(
    weighted: pd.DataFrame,
    out_path: Path,
    feature_names: list[str] | None = None,
    cohort_order: list[str] | None = None,
) -> Path:
    feature_names = feature_names or BEHAVIORAL_FEATURES
    order = [c for c in (cohort_order or COHORT_ORDER) if c in set(weighted["cohort"])]
    mat = weighted.set_index("cohort").loc[order, feature_names]
    # z-score across cohorts for readability
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0, np.nan)
    z.index = [COHORT_LABELS.get(i, i) for i in z.index]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(order))))
    sns.heatmap(z, ax=ax, center=0, cmap="vlag", cbar_kws={"label": "z across cohorts"})
    ax.set_title("Usage-weighted decoded behavioral signature (z-scored)")
    ax.set_xlabel("Decoded feature")
    ax.set_ylabel("Cohort")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def codebook_feature_frame(
    decoded: dict[int, np.ndarray],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """All codebook day-types as a feature table (raw decoded values)."""
    feature_names = feature_names or BEHAVIORAL_FEATURES
    feat_idx = [FEATURE_NAMES.index(f) for f in feature_names]
    rows = []
    for day_type, vec in sorted(decoded.items()):
        row = {"day_type": int(day_type)}
        row.update({f: float(vec[i]) for f, i in zip(feature_names, feat_idx)})
        rows.append(row)
    return pd.DataFrame(rows)


def zscore_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Z-score columns like the manuscript decode step (population = codebook)."""
    out = df.copy()
    values = out[feature_names]
    out[feature_names] = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    return out


def plot_top_decoded_grid(
    top_table: pd.DataFrame,
    out_path: Path,
    n_top: int = 5,
    feature_names: list[str] | None = None,
    cohort_order: list[str] | None = None,
) -> Path:
    """One small heatmap per cohort: top-N day-types × decoded features (z within cohort)."""
    feature_names = feature_names or BEHAVIORAL_FEATURES
    order = [c for c in (cohort_order or COHORT_ORDER) if c in set(top_table["cohort"])]
    n = len(order)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for ax, cohort in zip(axes.ravel(), order):
        sub = top_table[top_table["cohort"] == cohort].nsmallest(n_top, "rank")
        mat = sub.set_index("day_type")[feature_names]
        z = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0, np.nan)
        sns.heatmap(z, ax=ax, center=0, cmap="vlag", cbar=False)
        ax.set_title(f"{COHORT_LABELS.get(cohort, cohort)}\n(top {n_top} day-types)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle("Decoded features of most-occupied day-types per cohort", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_behavioral_pattern_bars(
    top_table: pd.DataFrame,
    decoded_z: pd.DataFrame,
    out_path: Path,
    n_top: int = 5,
    feature_names: list[str] | None = None,
    cohort_order: list[str] | None = None,
) -> Path:
    """Manuscript-style bar grid: z-scored behavioral features of top day-types per cohort.

    Z-scores use the full codebook as reference (same scale in every panel), matching
    the decode notebooks' standardized behavioral patterns.
    """
    feature_names = feature_names or BEHAVIORAL_FEATURES
    order = [c for c in (cohort_order or COHORT_ORDER) if c in set(top_table["cohort"])]
    z_lookup = decoded_z.set_index("day_type")

    n = len(order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, cohort) in enumerate(zip(axes, order)):
        ids = (
            top_table[top_table["cohort"] == cohort]
            .nsmallest(n_top, "rank")["day_type"]
            .astype(int)
            .tolist()
        )
        rows = []
        for day_type in ids:
            if day_type not in z_lookup.index:
                continue
            for feat in feature_names:
                rows.append(
                    {
                        "Feature": feat,
                        "Z-score": float(z_lookup.loc[day_type, feat]),
                        "Day-Type ID": str(day_type),
                    }
                )
        melted = pd.DataFrame(rows)
        sns.barplot(
            data=melted,
            x="Feature",
            y="Z-score",
            hue="Day-Type ID",
            ax=ax,
            palette="flare",
        )
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_title(COHORT_LABELS.get(cohort, cohort), fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Standardized value (Z-score)" if i % ncols == 0 else "")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(title="Day-Type", fontsize="x-small", ncol=2, loc="best")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Predominant day-types per cohort: behavioral feature deviations",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    # also SVG like manuscript decode figures
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cohort_mean_behavioral_bars(
    weighted: pd.DataFrame,
    decoded_z: pd.DataFrame,
    occ: pd.DataFrame,
    out_path: Path,
    feature_names: list[str] | None = None,
    cohort_order: list[str] | None = None,
) -> Path:
    """One behavioral pattern bar per cohort: occupancy-weighted mean z over the codebook."""
    feature_names = feature_names or BEHAVIORAL_FEATURES
    order = [c for c in (cohort_order or COHORT_ORDER) if c in occ.columns]
    z_mat = decoded_z.set_index("day_type")[feature_names]

    rows = []
    for cohort in order:
        probs = occ[cohort]
        # align to codebook ids present in z_mat
        common = z_mat.index.intersection(probs.index)
        p = probs.loc[common].to_numpy()
        p = p / (p.sum() if p.sum() else 1.0)
        mean_z = (z_mat.loc[common].to_numpy() * p[:, None]).sum(axis=0)
        for feat, val in zip(feature_names, mean_z):
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS.get(cohort, cohort),
                    "Feature": feat,
                    "Z-score": float(val),
                }
            )
    melted = pd.DataFrame(rows)

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
        "Cohort behavioral patterns (usage-weighted mean of decoded day-types)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Behavioral features")
    ax.set_ylabel("Standardized value (Z-score vs codebook)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def pairwise_js_divergence(occ: pd.DataFrame) -> pd.DataFrame:
    """Jensen–Shannon divergence between cohort occupancy distributions."""
    cols = list(occ.columns)
    m = np.zeros((len(cols), len(cols)))
    for i, a in enumerate(cols):
        pa = occ[a].to_numpy()
        for j, b in enumerate(cols):
            pb = occ[b].to_numpy()
            mid = 0.5 * (pa + pb)

            def _kl(p, q):
                mask = p > 0
                return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))

            m[i, j] = 0.5 * _kl(pa, mid) + 0.5 * _kl(pb, mid)
    return pd.DataFrame(m, index=cols, columns=cols)


def plot_js_heatmap(js: pd.DataFrame, out_path: Path) -> Path:
    labels = [COHORT_LABELS.get(c, c) for c in js.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        js,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cmap="rocket_r",
        cbar_kws={"label": "JS divergence"},
    )
    ax.set_title("Occupancy distribution distance between cohorts")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_comparison(
    out_dir: Path = DEFAULT_OUT_DIR,
    fig_dir: Path = DEFAULT_FIG_DIR,
    decoded_pkl: Path = DEFAULT_DECODED_PKL,
    mode: str = "a0",
    top_n: int = 30,
    top_k: int = 15,
) -> dict[str, Path]:
    cohort_pickles = discover_cohort_pickles(out_dir)
    if not cohort_pickles:
        raise FileNotFoundError(f"No profile pickles in {out_dir}")

    decoded = load_decoded_embeddings(decoded_pkl)
    occ = occupancy_frame(cohort_pickles, mode=mode, top_n=top_n)
    # stable column order
    ordered = [c for c in COHORT_ORDER if c in occ.columns] + [
        c for c in occ.columns if c not in COHORT_ORDER
    ]
    occ = occ[ordered]

    top_table = top_profiles_table(occ, decoded, top_k=top_k)
    weighted = usage_weighted_decoded(occ, decoded)
    lift = enrichment_vs_mean(occ)
    js = pairwise_js_divergence(occ)
    codebook_raw = codebook_feature_frame(decoded)
    codebook_z = zscore_features(codebook_raw, BEHAVIORAL_FEATURES)

    table_dir = out_dir / "comparison_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["occupancy_csv"] = table_dir / "daytype_occupancy_by_cohort.csv"
    occ.to_csv(paths["occupancy_csv"])
    paths["top_profiles_csv"] = table_dir / "top_daytypes_decoded_by_cohort.csv"
    top_table.to_csv(paths["top_profiles_csv"], index=False)
    paths["weighted_decoded_csv"] = table_dir / "usage_weighted_decoded_by_cohort.csv"
    weighted.to_csv(paths["weighted_decoded_csv"], index=False)
    paths["codebook_z_csv"] = table_dir / "codebook_decoded_zscore.csv"
    codebook_z.to_csv(paths["codebook_z_csv"], index=False)
    paths["js_csv"] = table_dir / "occupancy_js_divergence.csv"
    js.to_csv(paths["js_csv"])

    paths["fig_occupancy"] = plot_top_occupancy_heatmap(
        occ, fig_dir / "occupancy_top_daytypes_heatmap.png", top_k=top_k
    )
    paths["fig_enrichment"] = plot_enrichment_heatmap(
        lift, occ, fig_dir / "enrichment_top_daytypes_heatmap.png", top_k=top_k
    )
    paths["fig_decoded"] = plot_decoded_cohort_heatmap(
        weighted, fig_dir / "decoded_signature_by_cohort.png"
    )
    paths["fig_top_decoded"] = plot_top_decoded_grid(
        top_table, fig_dir / "top_daytypes_decoded_grid.png", n_top=5
    )
    paths["fig_behavioral_bars"] = plot_behavioral_pattern_bars(
        top_table,
        codebook_z,
        fig_dir / "behavioral_patterns_top_daytypes_by_cohort.png",
        n_top=5,
    )
    paths["fig_cohort_mean_bars"] = plot_cohort_mean_behavioral_bars(
        weighted,
        codebook_z,
        occ,
        fig_dir / "behavioral_patterns_cohort_means.png",
    )
    paths["fig_js"] = plot_js_heatmap(js, fig_dir / "occupancy_js_divergence.png")

    summary = {
        "cohorts": ordered,
        "n_day_types": int(occ.shape[0]),
        "mode": mode,
        "top_n_window": top_n,
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"] = table_dir / "comparison_summary.json"
    paths["summary_json"].write_text(json.dumps(summary, indent=2))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--decoded-pkl", type=Path, default=DEFAULT_DECODED_PKL)
    parser.add_argument("--mode", default="a0")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    paths = run_comparison(
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        decoded_pkl=args.decoded_pkl,
        mode=args.mode,
        top_n=args.top_n,
        top_k=args.top_k,
    )
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
