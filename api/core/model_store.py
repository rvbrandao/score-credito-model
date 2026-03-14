from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI

from api.core.config import MODEL_PATH


def load_model_artifact(model_path: Path = MODEL_PATH) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Run training/train_model.py before starting the API."
        )

    return joblib.load(model_path)


def setup_model_store(app: FastAPI) -> None:
    app.state.model = load_model_artifact()


def get_model_from_app(app: FastAPI) -> Any:
    model = getattr(app.state, "model", None)
    if model is None:
        raise RuntimeError("Model is not initialized in application state.")

    return model
