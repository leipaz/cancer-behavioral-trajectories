"""Per-variable reconstruction on the Table S3 programs present in the May 2026 daily.

The frozen VQ-VAE a0 is applied in evaluation mode. Error uses only continuous
components that were observed. This is not a refit, and it does not use the
comparison cap from the HDO adequacy report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

from scripts.vqvae import constants as c
from scripts.vqvae.data import load_scaler_params
from scripts.vqvae.inference import load_models
from scripts.vqvae.run_external_pretraining_adequacy import (
    DICT_PATH,
    LDA_PATH,
    MENTAL_CSV,
    SPLITS_CSV,
    code_topic_map,
    encode_cohort,
    feature_table,
    pooled_mae,
    program_of_service,
    read_daily,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "vq-vae" / "reconstruction_s3_may26"

# Published Table S3 N, where this extract has a confirmed program.
S3_PROGRAMS = [
    ("eating", "Eating disorders", 2464),
    ("cognitive", "Cognitive impairment", 197),
    ("suicide", "Suicide prevention", 2555),
    ("tmc", "Depression / Anxiety disorders", 149),
    ("cancer_gm", "Cancer Patients", 93),
    ("dm", "HIV Patients", 24),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS_CSV)
    splits["user"] = splits["user"].astype(int)
    splits["program"] = splits["service"].map(program_of_service)
    selected = splits[splits["program"].isin([item[0] for item in S3_PROGRAMS])].copy()

    print("Reading the May 2026 daily", flush=True)
    mental = read_daily(MENTAL_CSV)
    mental["user"] = mental["user"].astype(int)
    mental = mental[mental["user"].isin(set(selected["user"]))]

    scaler = load_scaler_params(c.DEFAULT_SCALER)
    model = load_models(c.DEFAULT_MODELS_DIR, modes=("a0",), device="cpu")["a0"]
    model.eval()
    dictionary = corpora.Dictionary.load(str(DICT_PATH))
    lda = LdaModel.load(str(LDA_PATH))
    code_topic = code_topic_map(dictionary, lda)

    feature_rows = []
    cohort_rows = []
    for program, label, published_n in S3_PROGRAMS:
        ids = set(selected.loc[selected["program"].eq(program), "user"])
        frame = mental[mental["user"].isin(ids)]
        print(f"Encoding {label}: {frame['user'].nunique()} patients", flush=True)
        result = encode_cohort(program, frame, scaler, model, dictionary, lda, code_topic, "cpu", 16, None)
        features = feature_table(program, result["stats"], result["orig_ae"])
        features = features.rename(columns={"label": "feature_label"})
        features["cohort_label"] = label
        feature_rows.append(features)
        patients = result["patients"]
        cohort_rows.append(
            {
                "program": program,
                "label": label,
                "published_n": published_n,
                "patients_in_extract": int(frame["user"].nunique()),
                "patients_with_an_observed_day": int((patients["n_observed_days"] > 0).sum()),
                "observed_days": int(patients["n_observed_days"].sum()),
                "mae_scaled": pooled_mae(result["stats"]),
            }
        )
        patients.to_csv(OUT / f"patients_{program}.csv", index=False)

    features_out = pd.concat(feature_rows, ignore_index=True)
    cohorts_out = pd.DataFrame(cohort_rows)
    features_out.to_csv(OUT / "reconstruction_by_variable.csv", index=False)
    cohorts_out.to_csv(OUT / "cohort_summary.csv", index=False)
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "daily": str(MENTAL_CSV),
                "model": "models/vq-vae/vqvae_a0.pt",
                "scaler": "models/vq-vae/preprocessing_scaler_vqvae.npz",
                "mode": "eval",
                "objective_in_repository": "The module returns a reconstruction of the input channels and a quantization term. No training loop and no diagnostic loss are in this repository.",
                "error": "Mean absolute error on continuous components that were observed, after the frozen clip, log1p and scaler.",
                "cap": "None. Every patient of these programs in the May 2026 extract is included.",
                "bipolar": "Table S3 lists Bipolar disorder, 51 participants. This extract has no service under that name.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(cohorts_out.to_string(index=False))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
