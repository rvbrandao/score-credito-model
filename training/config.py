from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "credit_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_model.pkl"

FEATURE_COLUMNS = (
    "age",
    "income",
    "number_of_loans",
    "payment_delays",
)
TARGET_COLUMN = "default"

TRAIN_TEST_SIZE = 0.25
RANDOM_STATE = 42
LOGISTIC_MAX_ITER = 1000
