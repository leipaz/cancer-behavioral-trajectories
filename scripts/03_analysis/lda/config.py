"""Paths and defaults for LDA analysis."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# VQ-VAE / profile PKL (input)
DEFAULT_PROFILES_PKL = (
    PROJECT_ROOT
    / "data/processed/vq-vae/profiles_per_sample_oncology_28_12_2025.pkl"
)

# LDA intermediates and models (local)
PROCESSED_LDA_DIR = PROJECT_ROOT / "data/processed/lda"
MODELS_LDA_DIR = PROJECT_ROOT / "models/lda"
USER_EMBEDDINGS_CSV = PROCESSED_LDA_DIR / "user_embeddings_from_pkl.csv"
DICTIONARY_PATH = MODELS_LDA_DIR / "dictionary_LDA_vdec25.dict"
LDA_MODEL_PATH = MODELS_LDA_DIR / "LDA_model_vdec25.gensim"

USER_TOPIC_CLUSTER_CSV = PROCESSED_LDA_DIR / "user_topic_cluster_6topics_lda.csv"
PATIENT_TOPICS_CLINICAL_CSV = PROCESSED_LDA_DIR / "patient_topics_clinical.csv"
MERGED_TOPICS_CLINICAL_CSV = PROCESSED_LDA_DIR / "merged_topics_clinical.csv"

# Clinical covariates
CLINICAL_XLSX = PROJECT_ROOT / "data/processed/clinical/Subjects_data.xlsx"

# LDA results (versioned in git)
RESULTS_LDA_FIGURES = PROJECT_ROOT / "results/lda/figures"
RESULTS_LDA_TABLES = PROJECT_ROOT / "results/lda/tables"

FIGURE_TOP_TERMS_GRID = RESULTS_LDA_FIGURES / "lda_top_terms_grid.png"
FIGURE_TOPICS_BY_EVENT = RESULTS_LDA_FIGURES / "topics_by_event.png"
FIGURE_AVG_TOPIC_BY_EVENT = RESULTS_LDA_FIGURES / "avg_topic_distribution_by_event.png"

TABLE_TOP_TERMS = RESULTS_LDA_TABLES / "lda_topics_top10.csv"

# Monthly predominant topics (entropy / variability notebook 6_1)
DEFAULT_TOPICS_DAY_CSV = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/daily_&_targets/"
    "PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv"
)
FALLBACK_TOPICS_DAY_CSV = PROCESSED_LDA_DIR / "PD_cutoff_topics_per_day.csv"

DEFAULT_MONTH_PREDOM_CSV = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/daily_&_targets/"
    "PD_cutoff_dic_2025s_eB2_MonthPredom.csv"
)
FALLBACK_MONTH_PREDOM_CSV = PROCESSED_LDA_DIR / "monthly_predominant_topics.csv"

CNIO_LDA_MODEL_DEC25 = Path(
    "/export/gts_usuarios/lparbaiza/cnio/2nd_phase/lda_models/"
    "LDA_model_vdec25.gensim"
)
CNIO_DICTIONARY_DEC25 = Path(
    "/export/gts_usuarios/lparbaiza/cnio/2nd_phase/lda_models/"
    "dictionary_LDA_vdec25.dict"
)

FIGURE_MONTHLY_TOPICS_HEATMAP = (
    RESULTS_LDA_FIGURES / "predominant_topics_per_month.png"
)
FIGURE_MONTHLY_TOPICS_HEATMAP_SVG = (
    RESULTS_LDA_FIGURES / "predominant_topics_per_month.svg"
)

# Discrete colors for monthly-topic heatmap (notebook 6_1, cell 32)
HEATMAP_TOPIC_COLORS = {
    3: "#b22222",
    5: "#ff6347",
    1: "#228b22",
    0: "#32cd32",
    4: "#ffd700",
    2: "#f0e68c",
}

MIN_EMBEDDINGS_FOR_MONTHLY = 15
DEFAULT_WINDOW_SIZE = 30

# VQ-VAE decoded embedding vectors (profile id → behavioral features)
DEFAULT_DECODED_EMBEDDINGS_PKL = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/eb2_profiles/"
    "decoded_embedding_vectors_a0.pkl"
)
FALLBACK_DECODED_EMBEDDINGS_PKL = (
    PROJECT_ROOT / "data/processed/vq-vae/decoded_embedding_vectors_a0.pkl"
)
ALTERNATE_DECODED_EMBEDDINGS_PKL = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/eb2_profiles/"
    "decoded_embedding_vectors_finetune_selected.pkl"
)

DECODED_PROFILES_CSV = PROCESSED_LDA_DIR / "decoded_profiles_top10.csv"
DECODED_PROFILES_SCALED_CSV = PROCESSED_LDA_DIR / "decoded_profiles_top10_scaled.csv"

RESULTS_DECODED_TOPIC_PLOTS_DIR = RESULTS_LDA_FIGURES / "decoded_topic_plots"

FIGURE_FINAL_PATTERN_GRID = (
    RESULTS_DECODED_TOPIC_PLOTS_DIR / "final_pattern_analysis_grid.svg"
)
FIGURE_PATTERN_0_INDIVIDUAL = (
    RESULTS_DECODED_TOPIC_PLOTS_DIR / "pattern_0_individual.svg"
)
FIGURE_PATTERN_3_INDIVIDUAL = (
    RESULTS_DECODED_TOPIC_PLOTS_DIR / "pattern_3_individual.svg"
)
FIGURE_DECODED_FEATURES_BOXPLOT = (
    RESULTS_LDA_FIGURES / "behavioral_features_by_pattern_boxplot.svg"
)

# Seaborn palette per pattern (notebook final grid)
PATTERN_BAR_PALETTE = {0: "Greens_d", 3: "Blues_d"}
PATTERN_BAR_DEFAULT_PALETTE = "flare"
PATTERN_INDIVIDUAL_TOPICS = (0, 3)

PROFILE_FEATURE_NAMES = [
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

# Features used in manuscript plots (weekend excluded after decode)
BEHAVIORAL_FEATURE_COLS = [c for c in PROFILE_FEATURE_NAMES if c != "weekend"]

TOPIC_COLORS = {
    0: "#2e7d32",
    1: "#4caf50",
    2: "#81c784",
    3: "#8b0000",
    4: "#ef9a9a",
    5: "#e53935",
}

DEFAULT_MODEL_TYPE = "a0"
DEFAULT_N = 30
DEFAULT_NUM_TOPICS = 6
DEFAULT_PASSES = 10_000
DEFAULT_TOP_N_TERMS = 10

CLINICAL_COLS = ["id", "Date_start_HDM", "PD_event", "Obs_time"]


def resolve_topics_day_csv() -> Path:
    if DEFAULT_TOPICS_DAY_CSV.is_file():
        return DEFAULT_TOPICS_DAY_CSV
    if FALLBACK_TOPICS_DAY_CSV.is_file():
        return FALLBACK_TOPICS_DAY_CSV
    raise FileNotFoundError(
        "Per-day topics CSV not found. Place "
        f"'PD_cutoff_dic_2025s_eB2_Topics_a0_6topics100000.csv' at "
        f"{DEFAULT_TOPICS_DAY_CSV} or {FALLBACK_TOPICS_DAY_CSV}."
    )


def resolve_month_predom_csv() -> Path:
    if DEFAULT_MONTH_PREDOM_CSV.is_file():
        return DEFAULT_MONTH_PREDOM_CSV
    if FALLBACK_MONTH_PREDOM_CSV.is_file():
        return FALLBACK_MONTH_PREDOM_CSV
    raise FileNotFoundError(
        "Monthly predominant topics CSV not found. Run build_monthly_predominant_topics.py "
        f"or place the file at {DEFAULT_MONTH_PREDOM_CSV} or {FALLBACK_MONTH_PREDOM_CSV}."
    )


def resolve_lda_model_for_monthly() -> Path:
    if LDA_MODEL_PATH.is_file():
        return LDA_MODEL_PATH
    if CNIO_LDA_MODEL_DEC25.is_file():
        return CNIO_LDA_MODEL_DEC25
    raise FileNotFoundError(
        f"LDA model not found at {LDA_MODEL_PATH} or {CNIO_LDA_MODEL_DEC25}."
    )


def resolve_dictionary_for_monthly() -> Path:
    if DICTIONARY_PATH.is_file():
        return DICTIONARY_PATH
    if CNIO_DICTIONARY_DEC25.is_file():
        return CNIO_DICTIONARY_DEC25
    raise FileNotFoundError(
        f"LDA dictionary not found at {DICTIONARY_PATH} or {CNIO_DICTIONARY_DEC25}."
    )


def resolve_decoded_embeddings_pkl(model_type: str = "a0") -> Path:
    if model_type == "finetune":
        if ALTERNATE_DECODED_EMBEDDINGS_PKL.is_file():
            return ALTERNATE_DECODED_EMBEDDINGS_PKL
        raise FileNotFoundError(
            f"Finetune decoded embeddings not found at {ALTERNATE_DECODED_EMBEDDINGS_PKL}"
        )
    if DEFAULT_DECODED_EMBEDDINGS_PKL.is_file():
        return DEFAULT_DECODED_EMBEDDINGS_PKL
    if FALLBACK_DECODED_EMBEDDINGS_PKL.is_file():
        return FALLBACK_DECODED_EMBEDDINGS_PKL
    raise FileNotFoundError(
        "Decoded embedding pickle not found. Place "
        f"'decoded_embedding_vectors_a0.pkl' at {FALLBACK_DECODED_EMBEDDINGS_PKL} "
        f"or {DEFAULT_DECODED_EMBEDDINGS_PKL}."
    )


def resolve_clinical_xlsx() -> Path:
    if CLINICAL_XLSX.is_file():
        return CLINICAL_XLSX
    raise FileNotFoundError(
        f"Clinical Excel not found. Place Subjects_data.xlsx at {CLINICAL_XLSX}."
    )
