from pathlib import Path

TARGET = "Dengue_fever_rates"
DATE_COLS = ["year", "month"]
PROVINCE_COL = "province"

WEATHER_VARS = [
    "Total_Evaporation",
    "Total_Rainfall",
    "Max_Daily_Rainfall",
    "n_raining_days",
    "Average_temperature",
    "Average_Humidity",
    "n_hours_sunshine",
]

SOCIAL_VARS = [
    "poverty_rate",
    "clean_water_rate_all",
    "urban_water_usage_rate",
    "toilet_rate",
    "population_density",
]

LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3]
SEQ_LEN = 12
HORIZONS = [1, 2, 3]

TRAIN_END = "2014-12-31"
TEST_START = "2016-01-01"

DATA_FOLDER = Path("data/raw")
PROCESSED_FOLDER = Path("data/processed")
OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"
SHAP_DIR = OUTPUT_DIR / "shap"

RANDOM_STATE = 42
