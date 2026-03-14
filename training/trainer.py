from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from training.config import MODEL_PATH
from training.config import RANDOM_STATE
from training.config import TRAIN_TEST_SIZE
from training.data_loader import load_training_data
from training.pipeline import build_training_pipeline


def train_and_save_model() -> tuple[float, Path]:
    features, target = load_training_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_training_pipeline()
    pipeline.fit(x_train, y_train)

    y_probabilities = pipeline.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_probabilities)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return auc, MODEL_PATH
