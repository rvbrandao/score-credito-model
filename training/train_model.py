from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["age", "income", "number_of_loans", "payment_delays"]
TARGET_COLUMN = "default"


def get_project_root() -> Path:
    # training/train_model.py -> project root is two levels up from this file.
    return Path(__file__).resolve().parents[1]


def train_model() -> None:
    project_root = get_project_root()
    data_path = project_root / "data" / "credit_dataset.csv"
    model_path = project_root / "models" / "credit_model.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model trained successfully. AUC: {auc:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
