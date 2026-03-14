from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "credit_model.pkl"
)
MODEL_FEATURES = ["age", "income", "number_of_loans", "payment_delays"]


def load_model(model_path: Path | None = None) -> Any:
    selected_path = model_path or MODEL_PATH

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {selected_path}. "
            "Run training/train_model.py before starting the API."
        )

    return joblib.load(selected_path)
