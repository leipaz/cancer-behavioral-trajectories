"""Paths and constants for univariate analysis (LDA pipeline)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Inputs
DEFAULT_DAILY_FILTERED = (
    PROJECT_ROOT / "data/raw/daily_summary_eb2prod_dic25_enriched_filtered.csv"
)
DEFAULT_DAILY_ENRICHED = (
    PROJECT_ROOT / "data/raw/daily_summary_eb2prod_dic25_enriched.csv"
)
DEFAULT_MATCHING = Path(
    "/export/gts_usuarios/lparbaiza/cnio/data/data_aug2025/"
    "PD_noPD_Matching_fechas_finales_dic25.csv"
)
FALLBACK_MATCHING = (
    PROJECT_ROOT / "data/raw/PD_noPD_Matching_fechas_finales_dic25.csv"
)

# Intermediate cohort tables (local, not versioned with patient data policy)
PROCESSED_DIR = PROJECT_ROOT / "data/processed/lda"
COHORT_WINDOW1 = PROCESSED_DIR / "df_15_primeros_EB2_prog.csv"
COHORT_WINDOW2 = PROCESSED_DIR / "df_15_previos_HDM_prog.csv"

# Univariate outputs
RESULTS_FIGURES = PROJECT_ROOT / "results/univariate/figures"
RESULTS_TABLES = PROJECT_ROOT / "results/univariate/tables"

FIGURE_DENSITY = RESULTS_FIGURES / "univariate_matched_dens_2dates.png"
FIGURE_HISTOGRAM = RESULTS_FIGURES / "univariate_matched_hist_absolute_2dates.png"
TABLE_SUMMARY_FULL = RESULTS_TABLES / "variable_summary_full_stats_EPD.csv"

ID_COL = "ID"
FECHA_COL_EB2 = "date_time"

WINDOW_DATE_COLS = [
    "Fecha_entrada_v1",
    "Fecha_final_simulada_v1",
    "Fecha_entrada_v2",
    "Fecha_final_simulada_v2",
]

GRUPO_EPD = 1
GRUPO_NO_EPD = 0
LABEL_EPD = "E.PD"
LABEL_NO_EPD = "no-E.PD"
RESPONSE_MAP = {GRUPO_EPD: LABEL_EPD, GRUPO_NO_EPD: LABEL_NO_EPD}

SEL_COLS = [
    "sleep_start",
    "location_distance",
    "location_time_home",
    "sleep_duration",
    "activity_walking",
    "app_usage_total",
    "location_clusters_count",
    "steps_steps_total",
    "practiced_sport",
    "heart_rate_hr_mean",
    "heart_rate_hr_baseline",
    "heart_rate_hr_min",
    "heart_rate_hr_max",
    "heart_rate_sleep_min",
    "heart_rate_sleep_max",
    "heart_rate_sleep_mean",
    "oxygen_saturation_max_spo2",
    "oxygen_saturation_min_spo2",
    "oxygen_saturation_mean_spo2",
    "total_unlocks",
    "total_communication",
    "total_social_networks",
    "total_others",
    "deep_time",
    "light_time",
    "rem_time",
    "emotions_valence_pred",
]


def resolve_matching_path() -> Path:
    if DEFAULT_MATCHING.is_file():
        return DEFAULT_MATCHING
    if FALLBACK_MATCHING.is_file():
        return FALLBACK_MATCHING
    raise FileNotFoundError(
        "Matching file not found. Place PD_noPD_Matching_fechas_finales_dic25.csv "
        f"at {FALLBACK_MATCHING} or {DEFAULT_MATCHING}."
    )


def columns_present(df_columns, candidates=None):
    candidates = candidates or SEL_COLS
    return [c for c in candidates if c in df_columns]
