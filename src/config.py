from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = BASE_DIR / "images"

RANDOM_SEED = 42

START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
FREQ = "h"   # ✅ FIXED

TARGET_COLUMN = "energy_kwh"
TEST_SIZE_RATIO = 0.2
FORECAST_HORIZON = 168  # 7 days

LAGS = [1, 2, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]

RAW_DATA_FILE = DATA_DIR / "energy_data.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
FUTURE_FORECAST_FILE = OUTPUTS_DIR / "forecast_output.csv"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"
MODEL_FILE = MODELS_DIR / "best_model.joblib"