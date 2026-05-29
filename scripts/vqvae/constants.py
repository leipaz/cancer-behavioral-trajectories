from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_CSV = PROJECT_ROOT / "data" / "daily_summaries" / "oncology_daily_summary_model_input.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "vq-vae"
DEFAULT_SCALER = DEFAULT_MODELS_DIR / "preprocessing_scaler_vqvae.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vq-vae" / "profiles_per_sample_oncology.pkl"

COMPLETE = ["user", "date_time", "service", "practiced_sport", "weekend"]
UNINFORMATIVE = ["user", "date_time", "service"]

COLS = [
    "user",
    "date_time",
    "service",
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

FEATURE_COLS = [col for col in COLS if col not in UNINFORMATIVE]

CONTINUOUS_REAL_VALUED_IDX = [0]
CONTINUOUS_POSITIVE_IDX = range(1, 8)
BINARY_IDX = [8, 9]

CONTINUOUS_REAL_VALUED_COLS = ["sleep_start"]
CONTINUOUS_POSITIVE_COLS = [
    "location_distance",
    "location_time_home",
    "sleep_duration",
    "activity_walking",
    "app_usage_total",
    "location_clusters_count",
    "steps_steps_total",
]
BINARY_COLS = ["weekend", "practiced_sport"]

CLIP_INFO = {
    "sleep_start": (-22500, 25000),
    "location_distance": (20, 95000),
    "location_time_home": (120, None),
    "sleep_duration": (3600, 54000),
    "activity_walking": (120, 15000),
    "app_usage_total": (180, 35000),
    "location_clusters_count": (1, 15),
    "steps_steps_total": (150, 25000),
}

MODEL_CONFIG = {
    "num_features": 10,
    "embed_dim": 80,
    "num_embed": 256,
    "num_layers": 4,
    "conv_dims": (16, 64, 128),
    "kernel_sizes": (4, 4, 4, 4),
    "strides": (1, 1, 1, 1),
    "dropout": 0.5,
    "decay": 0.99,
    "threshold": 0.1,
}

DEFAULT_MODES = ("a0", "a1", "a2")
DEFAULT_TOP_N = (5, 10, 15, 20, 25, 30)
DEFAULT_BATCH_SIZE = 16
DEFAULT_MISSINGNESS_MODE = "none"
DEFAULT_MISSING_RATE = 0.0
DEFAULT_SEED = 123
