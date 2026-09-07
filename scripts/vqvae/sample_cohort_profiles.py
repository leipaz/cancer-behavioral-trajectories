"""Sample users per clinical cohort group and encode with VQ-VAE.

Produces the same profile pickle layout as daily_to_profiles.ipynb:
  profiles[mode][top_n][user_id] -> list of (length, seq, mapped, dates, ...)

one pickle per cohort group under data/output_vq_vae/revision_cohorts/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as c
from .inference import generate_profiles

DEFAULT_SOURCE = (
    c.PROJECT_ROOT / "data" / "daily_summaries" / "daily_summary_eb2prod_may26.csv"
)
DEFAULT_OUT_DIR = c.PROJECT_ROOT / "data" / "output_vq_vae" / "revision_cohorts"
DEFAULT_CSV_DIR = c.PROJECT_ROOT / "data" / "daily_summaries" / "revision_cohort_samples"

# Clinical cohort groups (label -> services). CNIO kept for reporting only (too small to sample).
COHORT_GROUPS: dict[str, dict] = {
    "CI": {
        "name": "Cognitive impairment",
        "services": [
            "DC_vizcaya",
            "DC_huelva",
            "DC_madrid",
            "DC_lleida",
            "DC_alicante",
            "DC_barcelona",
            "DC_albacete",
        ],
        "sample": True,
    },
    "CNIO": {
        "name": "CNIO",
        "services": ["cnio_caspar"],
        "sample": False,  # aparte; too few users
    },
    "das": {
        "name": "das",
        "services": ["das"],
        "sample": True,
    },
    "DM": {
        "name": "DM",
        "services": ["DM"],
        "sample": True,  # skipped automatically if n_users < n_per_cohort
    },
    "ED1": {
        "name": "Eating disorder 1",
        "services": [
            "ita_urgell",
            "ita_vidrio",
            "ita_general",
            "ita_moscatelar",
            "ita_malaga",
            "ita_jerez",
            "ita_sabadell",
            "ita_angli",
            "ita_mirasierra",
            "ita_argentona",
            "ita_acude",
            "ita_alcala",
            "ita_p_adicciones",
            "ita_galicia",
        ],
        "sample": True,
    },
    "ED2": {
        "name": "Eating disorder 2",
        "services": ["adalmed"],
        "sample": True,
    },
    "GM": {
        "name": "Gregorio Marañón",
        "services": ["GM_colon", "GM_MCEI"],
        "sample": True,
    },
    "PMP": {
        "name": "PMP",
        "services": [
            "PMP_IDIBELL",
            "PMP_CHUAC",
            "PMP_FUENLABRADA",
            "PMP_HUVM",
            "PMP_PRINCESA",
            "PMP_BALEARES",
            "PMP_NAVARRA",
            "PMP_VALENCIA",
            "PMP_CACERES",
        ],
        "sample": True,
    },
    "SR": {
        "name": "Suicide risk",
        "services": ["afsp", "FJD_smartcrisis_2_0"],
        "sample": True,
    },
    "TMC": {
        "name": "Common mental disorder",
        "services": ["TMC", "SJD"],
        "sample": True,
    },
}


def users_per_service(source_csv: Path) -> pd.Series:
    df = pd.read_csv(source_csv, usecols=["user", "service"])
    return df.groupby("service")["user"].nunique().sort_values(ascending=False)


def cohort_group_counts(source_csv: Path) -> pd.DataFrame:
    """Unique users per clinical cohort group."""
    df = pd.read_csv(source_csv, usecols=["user", "service"])
    rows = []
    for code, meta in COHORT_GROUPS.items():
        services = meta["services"]
        n = df.loc[df["service"].isin(services), "user"].nunique()
        rows.append(
            {
                "cohort": code,
                "name": meta["name"],
                "n_services": len(services),
                "n_users": int(n),
                "sample": bool(meta["sample"]),
                "services": " ".join(services),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)


def sample_users_by_cohort_group(
    source_csv: Path,
    n_per_cohort: int = 90,
    cohorts: list[str] | None = None,
    seed: int = c.DEFAULT_SEED,
    skip_undersized: bool = True,
) -> dict[str, list[int]]:
    """Return {cohort_code: [user_id, ...]} with equal n_per_cohort when possible.

    CNIO (sample=False) is never included. Groups with fewer than n_per_cohort
    users are skipped if skip_undersized else raise.
    """
    df = pd.read_csv(source_csv, usecols=["user", "service"])
    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {}

    if cohorts is None:
        codes = [code for code, meta in COHORT_GROUPS.items() if meta["sample"]]
    else:
        unknown = [c for c in cohorts if c not in COHORT_GROUPS]
        if unknown:
            raise ValueError(f"Unknown cohort codes: {unknown}")
        codes = cohorts

    for code in codes:
        meta = COHORT_GROUPS[code]
        if not meta["sample"]:
            print(f"  skip {code} ({meta['name']}): marked aparte / not sampled")
            continue
        users = (
            df.loc[df["service"].isin(meta["services"]), "user"]
            .drop_duplicates()
            .to_numpy()
        )
        if len(users) < n_per_cohort:
            msg = f"{code} ({meta['name']}) has {len(users)} users; need {n_per_cohort}."
            if skip_undersized:
                print(f"  skip {msg}")
                continue
            raise ValueError(msg)
        pick = rng.choice(users, size=n_per_cohort, replace=False)
        selected[code] = sorted(int(u) for u in pick)
        print(f"  {code}: sampled {n_per_cohort} / {len(users)} users")

    return selected


def write_model_input_csvs(
    source_csv: Path,
    selected: dict[str, list[int]],
    csv_dir: Path,
    clean: bool = True,
) -> dict[str, Path]:
    """Filter source rows for selected users; write one model-input CSV per cohort."""
    if clean and csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_users = sorted({u for ids in selected.values() for u in ids})
    wanted = set(all_users)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_csv, usecols=c.COLS, chunksize=200_000):
        part = chunk[chunk["user"].isin(wanted)]
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError("No rows found for sampled users.")
    filtered = pd.concat(chunks, ignore_index=True)

    paths: dict[str, Path] = {}
    for cohort, ids in selected.items():
        id_set = set(ids)
        out = filtered[filtered["user"].isin(id_set)].copy()
        out = out.sort_values(["user", "date_time"]).reset_index(drop=True)
        path = csv_dir / f"model_input_{cohort}.csv"
        out.to_csv(path, index=False)
        paths[cohort] = path
        print(
            f"  {cohort}: {out['user'].nunique()} users, {len(out)} rows -> {path.name}"
        )
    return paths


def encode_cohort_csvs(
    csv_paths: dict[str, Path],
    out_dir: Path,
    models_dir: Path = c.DEFAULT_MODELS_DIR,
    scaler_path: Path = c.DEFAULT_SCALER,
    modes: tuple[str, ...] = c.DEFAULT_MODES,
    device: str = "cpu",
    batch_size: int = c.DEFAULT_BATCH_SIZE,
    seed: int = c.DEFAULT_SEED,
    clean_pickles: bool = False,
) -> dict[str, Path]:
    """Run generate_profiles per cohort; save oncology-style pickles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean_pickles:
        for old in out_dir.glob("profiles_per_sample_*.pkl"):
            old.unlink()

    written: dict[str, Path] = {}
    for cohort, csv_path in csv_paths.items():
        out_pkl = out_dir / f"profiles_per_sample_{cohort}.pkl"
        print(f"Encoding {cohort} ...")
        generate_profiles(
            data_csv=csv_path,
            models_dir=models_dir,
            scaler_path=scaler_path,
            output_path=out_pkl,
            modes=modes,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        written[cohort] = out_pkl
        print(f"  wrote {out_pkl}")
    return written


def save_manifest(
    selected: dict[str, list[int]],
    out_dir: Path,
    n_per_cohort: int,
    seed: int,
    source_csv: Path,
    group_counts: pd.DataFrame | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sampling_manifest.json"
    payload = {
        "source_csv": str(source_csv),
        "n_per_cohort": n_per_cohort,
        "seed": seed,
        "grouping": "clinical_cohort_groups",
        "cohort_groups": {
            code: {
                "name": meta["name"],
                "services": meta["services"],
                "sample": meta["sample"],
            }
            for code, meta in COHORT_GROUPS.items()
        },
        "sampled": {
            k: {
                "n_users": len(v),
                "users": v,
                "name": COHORT_GROUPS[k]["name"],
                "services": COHORT_GROUPS[k]["services"],
            }
            for k, v in selected.items()
        },
    }
    if group_counts is not None:
        payload["available_counts"] = group_counts.to_dict(orient="records")
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Sample clinical cohort groups and encode VQ-VAE day-type profiles."
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-cohort", type=int, default=90)
    parser.add_argument("--seed", type=int, default=c.DEFAULT_SEED)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Optional subset of cohort codes (e.g. CI das ED1 PMP). Default: all sampleable.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only write sampled model-input CSVs; skip VQ-VAE encoding.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=c.DEFAULT_BATCH_SIZE)
    parser.add_argument("--modes", nargs="+", default=["a0"])
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="Print user counts per clinical cohort group and exit.",
    )
    parser.add_argument(
        "--list-services",
        action="store_true",
        help="Print user counts per raw service and exit.",
    )
    args = parser.parse_args()

    if args.list_services:
        print(users_per_service(args.source_csv).to_string())
        return

    counts = cohort_group_counts(args.source_csv)
    if args.list_groups:
        print(counts.to_string(index=False))
        ok = counts[(counts["sample"]) & (counts["n_users"] >= args.n_per_cohort)]
        print(f"\nEqual-sample eligible (>= {args.n_per_cohort}): {', '.join(ok['cohort'])}")
        return

    print("Clinical cohort group sizes:")
    print(counts.to_string(index=False))

    print("\nSampling equal users per group ...")
    selected = sample_users_by_cohort_group(
        args.source_csv,
        n_per_cohort=args.n_per_cohort,
        cohorts=args.cohorts,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No cohorts sampled.")
    print(f"Sampled cohorts: {list(selected)}")

    manifest = save_manifest(
        selected,
        args.out_dir,
        args.n_per_cohort,
        args.seed,
        args.source_csv,
        group_counts=counts,
    )
    print(f"Manifest: {manifest}")

    print("Writing model-input CSVs ...")
    csv_paths = write_model_input_csvs(args.source_csv, selected, args.csv_dir, clean=True)

    if args.sample_only:
        print("Done (sample-only).")
        return

    encode_cohort_csvs(
        csv_paths,
        args.out_dir,
        modes=tuple(args.modes),
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        clean_pickles=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
